"""Unit tests for TrainiumPulse core autograd, 1F1B knapsack, FP8 SR, compiler cache & API."""

from __future__ import annotations

import json
import random
import unittest

from trainium_pulse.checkpoint_1f1b import (
    _operator_catalog,
    solve_selective_checkpoint_knapsack,
    verify_1f1b_and_checkpointing,
)
from trainium_pulse.dist_compile_cache import run_compile_cache_and_bisect
from trainium_pulse.fp8_stochastic import (
    cast_bf16,
    cast_fp8_e4m3,
    cast_fp8_e5m2,
    compare_outlier_microscaling,
    float_to_fp8_e4m3_byte,
    float_to_fp8_e5m2_byte,
    fp8_e4m3_byte_to_float,
    fp8_e5m2_byte_to_float,
    run_fp8_convergence_lab,
)
from trainium_pulse.server import lambda_handler
from trainium_pulse.sharded_autograd import (
    Tensor,
    cross_entropy_loss,
    layernorm_3d,
    linear_3d,
    reshape_tensor,
    scaled_dot_product_attention,
    swiglu_mlp,
    verify_sharded_autograd,
)


class TrainiumPulseUnitTests(unittest.TestCase):
    def test_01_autograd_finite_difference_jacobian(self) -> None:
        """Verifies analytical backward pass against central finite differences (< 1e-6)."""
        rng = random.Random(11)
        b, s, d, h, ff, v = 1, 4, 8, 2, 16, 6
        x_vals = [rng.uniform(-0.4, 0.4) for _ in range(b * s * d)]
        g_vals = [1.0 + rng.uniform(-0.1, 0.1) for _ in range(d)]
        b_vals = [rng.uniform(-0.1, 0.1) for _ in range(d)]
        wq_vals = [rng.uniform(-0.2, 0.2) for _ in range(d * d)]
        wk_vals = [rng.uniform(-0.2, 0.2) for _ in range(d * d)]
        wv_vals = [rng.uniform(-0.2, 0.2) for _ in range(d * d)]
        wg_vals = [rng.uniform(-0.2, 0.2) for _ in range(d * ff)]
        wu_vals = [rng.uniform(-0.2, 0.2) for _ in range(d * ff)]
        wd_vals = [rng.uniform(-0.2, 0.2) for _ in range(ff * d)]
        wh_vals = [rng.uniform(-0.2, 0.2) for _ in range(d * v)]
        targets = [1, 3, 0, 5]

        def forward_loss(xv: list[float]) -> tuple[Tensor, Tensor]:
            xt = Tensor(xv, (b, s, d), requires_grad=True)
            ln = layernorm_3d(xt, Tensor(g_vals, (d,), True), Tensor(b_vals, (d,), True))
            q = reshape_tensor(linear_3d(ln, Tensor(wq_vals, (d, d), True)), (b, s, h, d // h))
            k = reshape_tensor(linear_3d(ln, Tensor(wk_vals, (d, d), True)), (b, s, h, d // h))
            val = reshape_tensor(linear_3d(ln, Tensor(wv_vals, (d, d), True)), (b, s, h, d // h))
            attn = reshape_tensor(scaled_dot_product_attention(q, k, val, causal=True), (b, s, d))
            mlp = swiglu_mlp(
                attn,
                Tensor(wg_vals, (d, ff), True),
                Tensor(wu_vals, (d, ff), True),
                Tensor(wd_vals, (ff, d), True),
            )
            logits = linear_3d(mlp, Tensor(wh_vals, (d, v), True))
            return cross_entropy_loss(logits, targets), xt

        loss, xt = forward_loss(x_vals)
        loss.backward()
        assert xt.grad is not None

        eps = 1e-6
        for idx in (0, 5, 13, 27, 31):
            xp = list(x_vals)
            xm = list(x_vals)
            xp[idx] += eps
            xm[idx] -= eps
            lp, _ = forward_loss(xp)
            lm, _ = forward_loss(xm)
            num_grad = (lp.data[0] - lm.data[0]) / (2.0 * eps)
            self.assertLess(abs(xt.grad[idx] - num_grad), 1e-6)

    def test_02_sharded_autograd_llama_causal_parity(self) -> None:
        """Proves R=4 TP + Ulysses SP + ZeRO-1 L_inf gradient parity < 1e-10 on causal LLM."""
        res = verify_sharded_autograd(world_size=4, architecture="llama_causal", seed=42)
        self.assertTrue(res["parity_verified"])
        self.assertLess(float(res["max_abs_grad_diff"]), 1e-10)
        self.assertLess(float(res["max_abs_param_diff"]), 1e-10)
        self.assertLess(float(res["input_dx_max_diff"]), 1e-10)

    def test_03_sharded_autograd_flux_mmdit_bidirectional_and_zero1_ram(self) -> None:
        """Proves R=4 bidirectional DiT parity and real tracemalloc ZeRO-1 RAM reduction."""
        res = verify_sharded_autograd(
            world_size=4, architecture="flux_mmdit_bidirectional", seq_len=16, seed=99
        )
        self.assertTrue(res["parity_verified"])
        self.assertLess(float(res["max_abs_grad_diff"]), 1e-10)
        mem = res["memory_tracemalloc"]
        self.assertGreater(float(mem["per_rank_reduction_pct"]), 70.0)
        self.assertGreaterEqual(float(mem["compression_ratio_x"]), 3.8)

    def test_04_knapsack_dp_matches_exhaustive_bruteforce(self) -> None:
        """Verifies exact 0/1 Knapsack DP matches brute-force search across all 2^8 subsets."""
        ops = _operator_catalog(batch_size=2, seq_len=32, d_model=32, n_heads=4, d_ff=64)
        dp_sol = solve_selective_checkpoint_knapsack(ops, budget_ratio=0.45)
        budget = int(dp_sol["budget_bytes"])
        total_flops = sum(int(o["recompute_flops"]) for o in ops)

        best_avoided = -1
        n = len(ops)
        for mask in range(1 << n):
            b_used = 0
            f_saved = 0
            for i in range(n):
                if (mask >> i) & 1:
                    b_used += int(ops[i]["bytes"])
                    f_saved += int(ops[i]["recompute_flops"])
            if b_used <= budget and f_saved > best_avoided:
                best_avoided = f_saved

        expected_recompute = total_flops - best_avoided
        self.assertEqual(int(dp_sol["recompute_flops"]), expected_recompute)
        self.assertLessEqual(int(dp_sol["saved_bytes"]), budget)
        # Quadratic S^2 attention scores should be recomputed while heavy linear projections stay in RAM
        saved_set = set(dp_sol["saved_op_ids"])
        self.assertIn("swiglu_gate_up_proj", saved_set)
        self.assertNotIn("attn_softmax_probs", saved_set)

    def test_05_pipeline_1f1b_vs_gpipe_memory_and_grad_parity(self) -> None:
        """Verifies 1F1B + Knapsack cuts live activation bytes while matching GPipe gradients."""
        res = verify_1f1b_and_checkpointing(num_stages=4, num_microbatches=8, seq_len=16)
        sm = res["summary_metrics"]
        self.assertTrue(sm["parity_verified"])
        self.assertLess(float(sm["max_grad_diff_knapsack_vs_gpipe"]), 1e-10)
        self.assertLess(int(sm["p1f1b_peak_stash_bytes"]), int(sm["gpipe_peak_stash_bytes"]))
        self.assertLess(
            int(sm["p1f1b_knapsack_peak_stash_bytes"]), int(sm["p1f1b_peak_stash_bytes"])
        )
        self.assertGreater(float(sm["combined_1f1b_knapsack_ram_reduction_pct"]), 70.0)
        self.assertAlmostEqual(float(sm["theoretical_bubble_fraction"]), 3.0 / 11.0, places=4)

    def test_06_fp8_e4m3_e5m2_and_bf16_bit_exactness(self) -> None:
        """Verifies IEEE/OCP bit packing, max normal saturation, and subnormal precision."""
        # E4M3FN max finite normal is 448.0 (0x7E)
        self.assertEqual(float_to_fp8_e4m3_byte(448.0), 0x7E)
        self.assertEqual(fp8_e4m3_byte_to_float(0x7E), 448.0)
        self.assertEqual(fp8_e4m3_byte_to_float(float_to_fp8_e4m3_byte(-600.0)), -448.0)
        # E4M3FN smallest subnormal is 2^-9 = 0.001953125 (0x01)
        self.assertEqual(float_to_fp8_e4m3_byte(2.0 ** -9), 0x01)
        self.assertEqual(fp8_e4m3_byte_to_float(0x01), 2.0 ** -9)

        # E5M2 max finite normal is 57344.0 (0x7B)
        self.assertEqual(float_to_fp8_e5m2_byte(57344.0), 0x7B)
        self.assertEqual(fp8_e5m2_byte_to_float(0x7B), 57344.0)
        # Exact power-of-two values round-trip with 0 error
        for v in (-3.5, -1.25, 0.0, 0.5, 1.5, 12.0):
            self.assertEqual(cast_bf16(v), v)
            self.assertEqual(cast_fp8_e4m3(v), v)
            self.assertEqual(cast_fp8_e5m2(v), v)

    def test_07_stochastic_rounding_is_statistically_unbiased(self) -> None:
        """Proves E[SR(x)] == x for sub-ULP values that deterministic RNE swamps to 1.0."""
        rng = random.Random(2026)
        # In BF16, 1.0 has ULP = 2^-7 = 0.0078125. 1.002 is strictly < 1.0 + 0.5*ULP (1.00390625)
        target = 1.0023
        self.assertEqual(cast_bf16(target, stochastic=False), 1.0)
        sr_mean = sum(cast_bf16(target, stochastic=True, rng=rng) for _ in range(8000)) / 8000.0
        self.assertLess(abs(sr_mean - target), 0.0004)

    def test_08_mxfp8_outlier_isolation_vs_per_tensor_fp8(self) -> None:
        """Verifies 32-element MXFP8 prevents small-signal underflow under outlier spikes."""
        comp = compare_outlier_microscaling(outlier_magnitude=1024.0, seed=19)
        pt = comp["per_tensor_fp8_e4m3"]
        mx = comp["mxfp8_block32_e4m3"]
        self.assertGreater(int(pt["underflow_zero_count"]), 10)
        self.assertEqual(int(mx["underflow_zero_count"]), 0)
        self.assertLess(float(mx["non_outlier_rel_rmse_pct"]), float(pt["non_outlier_rel_rmse_pct"]) * 0.25)

    def test_09_fp8_40_step_swamping_vs_stochastic_rounding_convergence(self) -> None:
        """Runs real 40-step MLP training: RNE stalls (swamped) while SR & MXFP8 converge."""
        lab = run_fp8_convergence_lab(steps=40, learning_rate=1e-4, seed=123)
        rne = lab["regimes"]["bf16_rne"]
        sr = lab["regimes"]["bf16_sr"]
        mxfp8 = lab["regimes"]["mxfp8_sr_master"]
        ref = lab["regimes"]["fp64_reference"]

        self.assertGreater(float(rne["swamped_update_pct"]), 95.0)
        self.assertLess(float(rne["loss_reduction_pct"]), 5.0)
        self.assertGreater(float(sr["loss_reduction_pct"]), 50.0)
        self.assertGreater(float(mxfp8["loss_reduction_pct"]), 50.0)
        self.assertLess(abs(float(mxfp8["final_loss"]) - float(ref["final_loss"])), 0.12)

    def test_10_alpha_canonical_24_layer_compile_cache_hits(self) -> None:
        """Proves 24 identical Transformer blocks compile once and hit SHA-256 cache 23/24 times."""
        res = run_compile_cache_and_bisect(num_layers=24, reset_cache_first=True)
        self.assertEqual(int(res["naive_unique_hashes"]), 24)
        self.assertEqual(int(res["canonical_unique_subgraphs"]), 1)
        self.assertEqual(int(res["cache_hits"]), 23)
        self.assertEqual(int(res["cache_misses"]), 1)
        self.assertAlmostEqual(float(res["cache_hit_rate_pct"]), 95.8, places=1)
        self.assertEqual(float(res["max_diff_cached_vs_uncached"]), 0.0)
        self.assertLess(float(res["max_diff_optimized_vs_reference"]), 1e-12)

    def test_11_automated_pass_bisection_and_2_op_bugpoint_reducer(self) -> None:
        """Verifies binary search isolates BrokenFastExpPass and ddmin shrinks IR to 2 SSA ops."""
        res = run_compile_cache_and_bisect(num_layers=8, reset_cache_first=True)
        bp = res["bugpoint_bisector"]
        self.assertEqual(bp["isolated_pass_name"], "BrokenFastExpPass")
        self.assertLessEqual(int(bp["bisect_oracle_calls"]), 4)
        self.assertEqual(int(bp["minimal_reproducer_op_count"]), 2)
        self.assertGreater(float(bp["minimal_reproducer_max_diff"]), 1.0)

    def test_12_lambda_handler_api_routes(self) -> None:
        """Exercises Lambda HTTP payload v2.0 routes end-to-end."""
        h = lambda_handler({"rawPath": "/api/health", "requestContext": {"http": {"method": "GET"}}})
        self.assertEqual(h["statusCode"], 200)
        self.assertEqual(json.loads(h["body"])["status"], "ok")

        ag = lambda_handler(
            {
                "rawPath": "/api/autograd/verify",
                "requestContext": {"http": {"method": "POST"}},
                "body": json.dumps({"world_size": 4, "seq_len": 16}),
            }
        )
        self.assertEqual(ag["statusCode"], 200)
        self.assertTrue(json.loads(ag["body"])["parity_verified"])

        rst = lambda_handler({"rawPath": "/api/reset", "requestContext": {"http": {"method": "POST"}}, "body": "{}"})
        self.assertEqual(rst["statusCode"], 200)


if __name__ == "__main__":
    unittest.main()
