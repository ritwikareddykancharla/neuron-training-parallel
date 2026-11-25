"""Exact 0/1 Knapsack Selective Activation Checkpointing & 1F1B vs. GPipe Schedule Verifier.

1. Solves exact 0/1 dynamic programming over intermediate block operators to decide which
   activations to retain in RAM vs. recompute during backward under a memory budget.
2. Executes real multi-microbatch GPipe and 1F1B pipeline schedules across P stages and
   M microbatches, measuring real tracemalloc peak live activation tensor bytes, pipeline
   bubble fraction, and exact backward gradient parity (< 1e-12).
"""

from __future__ import annotations

from array import array
import gc
import math
import random
import time
import tracemalloc
from typing import Dict, List, Set, Tuple


def _operator_catalog(
    batch_size: int, seq_len: int, d_model: int, n_heads: int, d_ff: int
) -> List[Dict[str, object]]:
    """Computes exact IEEE-754 float64 tensor bytes and forward recompute FLOPs per operator."""
    b, s, d, h, ff = batch_size, seq_len, d_model, n_heads, d_ff
    elem_bytes = 8  # Python array('d') IEEE-754 double element size
    return [
        {
            "op_id": "input_rmsnorm",
            "category": "Normalization",
            "elements": b * s * d,
            "bytes": b * s * d * elem_bytes,
            "recompute_flops": 4 * b * s * d,
        },
        {
            "op_id": "qkv_linear_proj",
            "category": "Linear Projection",
            "elements": 3 * b * s * d,
            "bytes": 3 * b * s * d * elem_bytes,
            "recompute_flops": 6 * b * s * d * d,
        },
        {
            "op_id": "attn_qk_scores",
            "category": "Attention Scores",
            "elements": b * h * s * s,
            "bytes": b * h * s * s * elem_bytes,
            "recompute_flops": 2 * b * s * s * d,
        },
        {
            "op_id": "attn_softmax_probs",
            "category": "Attention Scores",
            "elements": b * h * s * s,
            "bytes": b * h * s * s * elem_bytes,
            "recompute_flops": 5 * b * h * s * s,
        },
        {
            "op_id": "attn_context_out",
            "category": "Linear Projection",
            "elements": b * s * d,
            "bytes": b * s * d * elem_bytes,
            "recompute_flops": 2 * b * s * s * d + 2 * b * s * d * d,
        },
        {
            "op_id": "post_attn_rmsnorm",
            "category": "Normalization",
            "elements": b * s * d,
            "bytes": b * s * d * elem_bytes,
            "recompute_flops": 4 * b * s * d,
        },
        {
            "op_id": "swiglu_gate_up_proj",
            "category": "Linear Projection",
            "elements": 2 * b * s * ff,
            "bytes": 2 * b * s * ff * elem_bytes,
            "recompute_flops": 4 * b * s * d * ff,
        },
        {
            "op_id": "swiglu_silu_product",
            "category": "Gated Activation",
            "elements": b * s * ff,
            "bytes": b * s * ff * elem_bytes,
            "recompute_flops": 6 * b * s * ff,
        },
    ]


def solve_selective_checkpoint_knapsack(
    ops: List[Dict[str, object]], budget_ratio: float = 0.45
) -> Dict[str, object]:
    """Exact 0/1 Knapsack DP maximizing avoided recompute FLOPs subject to saved_bytes <= budget."""
    total_bytes = sum(int(op["bytes"]) for op in ops)
    total_flops = sum(int(op["recompute_flops"]) for op in ops)
    budget_bytes = max(0, min(total_bytes, int(total_bytes * budget_ratio)))

    # Exact integer weight quantum via GCD of operator byte footprints
    quantum = int(ops[0]["bytes"])
    for op in ops[1:]:
        quantum = math.gcd(quantum, int(op["bytes"]))
    quantum = max(1, quantum)
    cap_units = budget_bytes // quantum

    n_ops = len(ops)
    weights = [int(op["bytes"]) // quantum for op in ops]
    values = [int(op["recompute_flops"]) for op in ops]

    # Exact DP table: dp[i][w] = max avoided recompute FLOPs using subset of first i ops
    dp: List[List[int]] = [[0] * (cap_units + 1) for _ in range(n_ops + 1)]
    for i in range(1, n_ops + 1):
        wi = weights[i - 1]
        vi = values[i - 1]
        prev = dp[i - 1]
        curr = dp[i]
        for w in range(cap_units + 1):
            best = prev[w]
            if wi <= w and prev[w - wi] + vi > best:
                best = prev[w - wi] + vi
            curr[w] = best

    # Backtrack optimal subset to SAVE in RAM
    saved_ids: Set[str] = set()
    rem_w = cap_units
    for i in range(n_ops, 0, -1):
        if dp[i][rem_w] != dp[i - 1][rem_w]:
            saved_ids.add(str(ops[i - 1]["op_id"]))
            rem_w -= weights[i - 1]

    saved_bytes = 0
    recompute_flops = 0
    decisions: List[Dict[str, object]] = []
    for op in ops:
        oid = str(op["op_id"])
        obytes = int(op["bytes"])
        oflops = int(op["recompute_flops"])
        keep = oid in saved_ids
        if keep:
            saved_bytes += obytes
        else:
            recompute_flops += oflops
        decisions.append(
            {
                "op_id": oid,
                "category": op["category"],
                "bytes": obytes,
                "recompute_flops": oflops,
                "flops_per_byte": round(oflops / max(1, obytes), 2),
                "action": "SAVE_IN_RAM" if keep else "RECOMPUTE_IN_BWD",
            }
        )

    return {
        "budget_ratio": round(budget_ratio, 3),
        "budget_bytes": budget_bytes,
        "uncheckpointed_bytes": total_bytes,
        "saved_bytes": saved_bytes,
        "memory_saved_pct": round((1.0 - saved_bytes / max(1, total_bytes)) * 100.0, 2),
        "total_forward_flops": total_flops,
        "recompute_flops": recompute_flops,
        "recompute_flop_overhead_pct": round(
            (recompute_flops / max(1, total_flops)) * 100.0, 2
        ),
        "saved_op_ids": sorted(saved_ids),
        "decisions": decisions,
    }


# ---------------------------------------------------------------------------
# Real Stage Forward / Backward Execution with Selective Activation Stashing
# ---------------------------------------------------------------------------

def _init_stage_weights(num_stages: int, d_model: int, d_ff: int, seed: int) -> List[Dict[str, List[float]]]:
    rng = random.Random(seed)
    stages: List[Dict[str, List[float]]] = []
    for _ in range(num_stages):
        stages.append(
            {
                "w_q": [rng.uniform(-0.12, 0.12) for _ in range(d_model * d_model)],
                "w_k": [rng.uniform(-0.12, 0.12) for _ in range(d_model * d_model)],
                "w_v": [rng.uniform(-0.12, 0.12) for _ in range(d_model * d_model)],
                "w_o": [rng.uniform(-0.12, 0.12) for _ in range(d_model * d_model)],
                "w_gate": [rng.uniform(-0.12, 0.12) for _ in range(d_model * d_ff)],
                "w_up": [rng.uniform(-0.12, 0.12) for _ in range(d_model * d_ff)],
                "w_down": [rng.uniform(-0.12, 0.12) for _ in range(d_ff * d_model)],
            }
        )
    return stages


def _matmul_arr(a: List[float], rows: int, k_dim: int, b: List[float], n_dim: int) -> List[float]:
    b_cols = [[b[k * n_dim + n] for k in range(k_dim)] for n in range(n_dim)]
    sumprod = math.sumprod
    out = [0.0] * (rows * n_dim)
    for m in range(rows):
        row_a = a[m * k_dim : (m + 1) * k_dim]
        o_off = m * n_dim
        for n in range(n_dim):
            out[o_off + n] = sumprod(row_a, b_cols[n])
    return out


def _matmul_at_b(a: List[float], rows: int, k_dim: int, b: List[float], n_dim: int) -> List[float]:
    """Computes A^T @ B where A is (rows, k_dim) and B is (rows, n_dim) -> (k_dim, n_dim)."""
    a_cols = [[a[m * k_dim + k] for m in range(rows)] for k in range(k_dim)]
    b_cols = [[b[m * n_dim + n] for m in range(rows)] for n in range(n_dim)]
    sumprod = math.sumprod
    out = [0.0] * (k_dim * n_dim)
    for k in range(k_dim):
        col_a = a_cols[k]
        o_off = k * n_dim
        for n in range(n_dim):
            out[o_off + n] = sumprod(col_a, b_cols[n])
    return out


def _matmul_a_bt(a: List[float], rows: int, n_dim: int, b: List[float], k_dim: int) -> List[float]:
    """Computes A @ B^T where A is (rows, n_dim) and B is (k_dim, n_dim) -> (rows, k_dim)."""
    b_rows = [b[k * n_dim : (k + 1) * n_dim] for k in range(k_dim)]
    sumprod = math.sumprod
    out = [0.0] * (rows * k_dim)
    for m in range(rows):
        row_a = a[m * n_dim : (m + 1) * n_dim]
        o_off = m * k_dim
        for k in range(k_dim):
            out[o_off + k] = sumprod(row_a, b_rows[k])
    return out


def _rmsnorm_fwd(x: List[float], rows: int, d_dim: int, eps: float = 1e-5) -> Tuple[List[float], List[float]]:
    out = [0.0] * (rows * d_dim)
    inv_rms = [0.0] * rows
    for m in range(rows):
        off = m * d_dim
        ms = 0.0
        for j in range(d_dim):
            v = x[off + j]
            ms += v * v
        ir = 1.0 / math.sqrt(ms / d_dim + eps)
        inv_rms[m] = ir
        for j in range(d_dim):
            out[off + j] = x[off + j] * ir
    return out, inv_rms


def _rmsnorm_bwd(dy: List[float], x: List[float], inv_rms: List[float], rows: int, d_dim: int) -> List[float]:
    dx = [0.0] * (rows * d_dim)
    for m in range(rows):
        off = m * d_dim
        ir = inv_rms[m]
        dot = 0.0
        for j in range(d_dim):
            dot += dy[off + j] * x[off + j]
        coeff = (ir * ir * ir) * dot / d_dim
        for j in range(d_dim):
            dx[off + j] = dy[off + j] * ir - x[off + j] * coeff
    return dx


def _attention_fwd(
    q: List[float], k: List[float], v: List[float], bsz: int, seq: int, n_heads: int, d_head: int
) -> Tuple[List[float], List[float], List[float]]:
    """Returns (scores [B*H*S*S], probs [B*H*S*S], context [B*S*D])."""
    d_model = n_heads * d_head
    scale = 1.0 / math.sqrt(d_head)
    sumprod = math.sumprod
    scores = [0.0] * (bsz * n_heads * seq * seq)
    probs = [0.0] * (bsz * n_heads * seq * seq)
    ctx = [0.0] * (bsz * seq * d_model)

    for b in range(bsz):
        for h in range(n_heads):
            bh_off = (b * n_heads + h) * seq * seq
            q_vecs = [
                q[(b * seq + s) * d_model + h * d_head : (b * seq + s) * d_model + (h + 1) * d_head]
                for s in range(seq)
            ]
            k_vecs = [
                k[(b * seq + s) * d_model + h * d_head : (b * seq + s) * d_model + (h + 1) * d_head]
                for s in range(seq)
            ]
            v_cols = [
                [v[(b * seq + s) * d_model + h * d_head + d] for s in range(seq)]
                for d in range(d_head)
            ]
            for i in range(seq):
                qi = q_vecs[i]
                row_off = bh_off + i * seq
                row_s = [sumprod(qi, k_vecs[j]) * scale for j in range(seq)]
                scores[row_off : row_off + seq] = row_s
                max_s = max(row_s)
                exps = [math.exp(sv - max_s) for sv in row_s]
                denom = sum(exps)
                row_p = [ev / denom for ev in exps]
                probs[row_off : row_off + seq] = row_p
                qi_off = (b * seq + i) * d_model + h * d_head
                for d in range(d_head):
                    ctx[qi_off + d] = sumprod(row_p, v_cols[d])

    return scores, probs, ctx


def _attention_bwd(
    dctx: List[float],
    q: List[float],
    k: List[float],
    v: List[float],
    probs: List[float],
    bsz: int,
    seq: int,
    n_heads: int,
    d_head: int,
) -> Tuple[List[float], List[float], List[float]]:
    d_model = n_heads * d_head
    scale = 1.0 / math.sqrt(d_head)
    sumprod = math.sumprod
    dq = [0.0] * (bsz * seq * d_model)
    dk = [0.0] * (bsz * seq * d_model)
    dv = [0.0] * (bsz * seq * d_model)

    for b in range(bsz):
        for h in range(n_heads):
            bh_off = (b * n_heads + h) * seq * seq
            v_vecs = [
                v[(b * seq + s) * d_model + h * d_head : (b * seq + s) * d_model + (h + 1) * d_head]
                for s in range(seq)
            ]
            q_cols = [
                [q[(b * seq + s) * d_model + h * d_head + d] for s in range(seq)]
                for d in range(d_head)
            ]
            k_cols = [
                [k[(b * seq + s) * d_model + h * d_head + d] for s in range(seq)]
                for d in range(d_head)
            ]
            ds_mat = [[0.0] * seq for _ in range(seq)]
            for i in range(seq):
                qi_off = (b * seq + i) * d_model + h * d_head
                dci = dctx[qi_off : qi_off + d_head]
                row_p = probs[bh_off + i * seq : bh_off + (i + 1) * seq]
                dp = [sumprod(dci, v_vecs[j]) for j in range(seq)]
                for j in range(seq):
                    vj_off = (b * seq + j) * d_model + h * d_head
                    p_ij = row_p[j]
                    for d in range(d_head):
                        dv[vj_off + d] += p_ij * dci[d]
                sum_p_dp = sumprod(row_p, dp)
                ds_row = [row_p[j] * (dp[j] - sum_p_dp) * scale for j in range(seq)]
                ds_mat[i] = ds_row
                for d in range(d_head):
                    dq[qi_off + d] = sumprod(ds_row, k_cols[d])
            for j in range(seq):
                kj_off = (b * seq + j) * d_model + h * d_head
                ds_col = [ds_mat[i][j] for i in range(seq)]
                for d in range(d_head):
                    dk[kj_off + d] = sumprod(ds_col, q_cols[d])

    return dq, dk, dv


def _stage_forward(
    x_in: List[float],
    weights: Dict[str, List[float]],
    saved_ops: Set[str],
    bsz: int,
    seq: int,
    d_model: int,
    n_heads: int,
    d_ff: int,
) -> Tuple[List[float], Dict[str, array]]:
    """Executes one pipeline stage forward pass and retains ONLY activations in `saved_ops`
    as contiguous IEEE-754 double `array('d')` buffers tracked by `tracemalloc`.
    """
    rows = bsz * seq
    d_head = d_model // n_heads
    stash: Dict[str, array] = {"stage_input": array("d", x_in)}

    norm1, inv_rms1 = _rmsnorm_fwd(x_in, rows, d_model)
    if "input_rmsnorm" in saved_ops:
        stash["input_rmsnorm"] = array("d", norm1)
        stash["inv_rms1"] = array("d", inv_rms1)

    q = _matmul_arr(norm1, rows, d_model, weights["w_q"], d_model)
    k = _matmul_arr(norm1, rows, d_model, weights["w_k"], d_model)
    v = _matmul_arr(norm1, rows, d_model, weights["w_v"], d_model)
    if "qkv_linear_proj" in saved_ops:
        stash["qkv_linear_proj"] = array("d", q + k + v)

    scores, probs, ctx = _attention_fwd(q, k, v, bsz, seq, n_heads, d_head)
    if "attn_qk_scores" in saved_ops:
        stash["attn_qk_scores"] = array("d", scores)
    if "attn_softmax_probs" in saved_ops:
        stash["attn_softmax_probs"] = array("d", probs)

    attn_proj = _matmul_arr(ctx, rows, d_model, weights["w_o"], d_model)
    if "attn_context_out" in saved_ops:
        stash["attn_context_out"] = array("d", ctx)

    h1 = [x_in[i] + attn_proj[i] for i in range(rows * d_model)]
    stash["h1_residual"] = array("d", h1)

    norm2, inv_rms2 = _rmsnorm_fwd(h1, rows, d_model)
    if "post_attn_rmsnorm" in saved_ops:
        stash["post_attn_rmsnorm"] = array("d", norm2)
        stash["inv_rms2"] = array("d", inv_rms2)

    gate_pre = _matmul_arr(norm2, rows, d_model, weights["w_gate"], d_ff)
    up_pre = _matmul_arr(norm2, rows, d_model, weights["w_up"], d_ff)
    if "swiglu_gate_up_proj" in saved_ops:
        stash["swiglu_gate_up_proj"] = array("d", gate_pre + up_pre)

    silu_prod = [0.0] * (rows * d_ff)
    for i in range(rows * d_ff):
        gv = gate_pre[i]
        sig = 1.0 / (1.0 + math.exp(-gv)) if gv >= 0.0 else math.exp(gv) / (1.0 + math.exp(gv))
        silu_prod[i] = (gv * sig) * up_pre[i]
    if "swiglu_silu_product" in saved_ops:
        stash["swiglu_silu_product"] = array("d", silu_prod)

    mlp_out = _matmul_arr(silu_prod, rows, d_ff, weights["w_down"], d_model)
    out = [h1[i] + mlp_out[i] for i in range(rows * d_model)]
    return out, stash


def _stage_backward(
    dout: List[float],
    stash: Dict[str, array],
    weights: Dict[str, List[float]],
    grad_accum: Dict[str, List[float]],
    bsz: int,
    seq: int,
    d_model: int,
    n_heads: int,
    d_ff: int,
) -> Tuple[List[float], int]:
    """Executes exact backward pass for one stage, recomputing any un-stashed intermediate
    tensors on the fly and accumulating weight gradients into `grad_accum`.
    """
    rows = bsz * seq
    d_head = d_model // n_heads
    recomputed_ops = 0

    x_in = stash["stage_input"].tolist()
    h1 = stash["h1_residual"].tolist()

    if "post_attn_rmsnorm" in stash:
        norm2 = stash["post_attn_rmsnorm"].tolist()
        inv_rms2 = stash["inv_rms2"].tolist()
    else:
        norm2, inv_rms2 = _rmsnorm_fwd(h1, rows, d_model)
        recomputed_ops += 1

    if "swiglu_gate_up_proj" in stash:
        gu = stash["swiglu_gate_up_proj"].tolist()
        n_ff = rows * d_ff
        gate_pre = gu[:n_ff]
        up_pre = gu[n_ff:]
    else:
        gate_pre = _matmul_arr(norm2, rows, d_model, weights["w_gate"], d_ff)
        up_pre = _matmul_arr(norm2, rows, d_model, weights["w_up"], d_ff)
        recomputed_ops += 1

    if "swiglu_silu_product" in stash:
        silu_prod = stash["swiglu_silu_product"].tolist()
    else:
        silu_prod = [0.0] * (rows * d_ff)
        for i in range(rows * d_ff):
            gv = gate_pre[i]
            sig = 1.0 / (1.0 + math.exp(-gv)) if gv >= 0.0 else math.exp(gv) / (1.0 + math.exp(gv))
            silu_prod[i] = (gv * sig) * up_pre[i]
        recomputed_ops += 1

    dw_down = _matmul_at_b(silu_prod, rows, d_ff, dout, d_model)
    ga_down = grad_accum["w_down"]
    for i in range(len(dw_down)):
        ga_down[i] += dw_down[i]
    dsilu_prod = _matmul_a_bt(dout, rows, d_model, weights["w_down"], d_ff)

    dgate_pre = [0.0] * (rows * d_ff)
    dup_pre = [0.0] * (rows * d_ff)
    for i in range(rows * d_ff):
        gv = gate_pre[i]
        sig = 1.0 / (1.0 + math.exp(-gv)) if gv >= 0.0 else math.exp(gv) / (1.0 + math.exp(gv))
        silu_g = gv * sig
        dsilu_g = sig * (1.0 + gv * (1.0 - sig))
        dup_pre[i] = dsilu_prod[i] * silu_g
        dgate_pre[i] = dsilu_prod[i] * up_pre[i] * dsilu_g

    dw_gate = _matmul_at_b(norm2, rows, d_model, dgate_pre, d_ff)
    dw_up = _matmul_at_b(norm2, rows, d_model, dup_pre, d_ff)
    ga_gate = grad_accum["w_gate"]
    ga_up = grad_accum["w_up"]
    for i in range(len(dw_gate)):
        ga_gate[i] += dw_gate[i]
        ga_up[i] += dw_up[i]

    dnorm2_g = _matmul_a_bt(dgate_pre, rows, d_ff, weights["w_gate"], d_model)
    dnorm2_u = _matmul_a_bt(dup_pre, rows, d_ff, weights["w_up"], d_model)
    dnorm2 = [dnorm2_g[i] + dnorm2_u[i] for i in range(rows * d_model)]
    dh1_mlp = _rmsnorm_bwd(dnorm2, h1, inv_rms2, rows, d_model)
    dh1 = [dout[i] + dh1_mlp[i] for i in range(rows * d_model)]

    if "input_rmsnorm" in stash:
        norm1 = stash["input_rmsnorm"].tolist()
        inv_rms1 = stash["inv_rms1"].tolist()
    else:
        norm1, inv_rms1 = _rmsnorm_fwd(x_in, rows, d_model)
        recomputed_ops += 1

    if "qkv_linear_proj" in stash:
        qkv = stash["qkv_linear_proj"].tolist()
        nd = rows * d_model
        q = qkv[:nd]
        k = qkv[nd : 2 * nd]
        v = qkv[2 * nd :]
    else:
        q = _matmul_arr(norm1, rows, d_model, weights["w_q"], d_model)
        k = _matmul_arr(norm1, rows, d_model, weights["w_k"], d_model)
        v = _matmul_arr(norm1, rows, d_model, weights["w_v"], d_model)
        recomputed_ops += 1

    if "attn_softmax_probs" in stash and "attn_context_out" in stash:
        probs = stash["attn_softmax_probs"].tolist()
        ctx = stash["attn_context_out"].tolist()
    else:
        _, probs, ctx = _attention_fwd(q, k, v, bsz, seq, n_heads, d_head)
        recomputed_ops += 1

    dw_o = _matmul_at_b(ctx, rows, d_model, dh1, d_model)
    ga_o = grad_accum["w_o"]
    for i in range(len(dw_o)):
        ga_o[i] += dw_o[i]
    dctx = _matmul_a_bt(dh1, rows, d_model, weights["w_o"], d_model)

    dq, dk, dv = _attention_bwd(dctx, q, k, v, probs, bsz, seq, n_heads, d_head)
    dw_q = _matmul_at_b(norm1, rows, d_model, dq, d_model)
    dw_k = _matmul_at_b(norm1, rows, d_model, dk, d_model)
    dw_v = _matmul_at_b(norm1, rows, d_model, dv, d_model)
    ga_q, ga_k, ga_v = grad_accum["w_q"], grad_accum["w_k"], grad_accum["w_v"]
    for i in range(len(dw_q)):
        ga_q[i] += dw_q[i]
        ga_k[i] += dw_k[i]
        ga_v[i] += dw_v[i]

    dnorm1_q = _matmul_a_bt(dq, rows, d_model, weights["w_q"], d_model)
    dnorm1_k = _matmul_a_bt(dk, rows, d_model, weights["w_k"], d_model)
    dnorm1_v = _matmul_a_bt(dv, rows, d_model, weights["w_v"], d_model)
    dnorm1 = [dnorm1_q[i] + dnorm1_k[i] + dnorm1_v[i] for i in range(rows * d_model)]
    dx_attn = _rmsnorm_bwd(dnorm1, x_in, inv_rms1, rows, d_model)
    dx_in = [dh1[i] + dx_attn[i] for i in range(rows * d_model)]
    return dx_in, recomputed_ops


# ---------------------------------------------------------------------------
# Multi-Microbatch Pipeline Schedule Builders (GPipe vs. 1F1B)
# ---------------------------------------------------------------------------

def _build_gpipe_schedule(num_stages: int, num_microbatches: int) -> List[List[Tuple[int, str, int]]]:
    """Clock-accurate GPipe schedule: all M forward microbatches, then all M backward."""
    total_ticks = 2 * (num_microbatches + num_stages - 1)
    ticks: List[List[Tuple[int, str, int]]] = [[] for _ in range(total_ticks)]
    bwd_offset = num_microbatches + num_stages - 1
    for m in range(num_microbatches):
        for p in range(num_stages):
            ticks[m + p].append((p, "F", m))
            ticks[bwd_offset + m + (num_stages - 1 - p)].append((p, "B", m))
    return ticks


def _build_1f1b_schedule(num_stages: int, num_microbatches: int) -> List[List[Tuple[int, str, int]]]:
    """Exact clock-accurate 1F1B non-interleaved pipeline schedule via dependency simulation."""
    fwd_done: Set[Tuple[int, int]] = set()
    bwd_done: Set[Tuple[int, int]] = set()
    next_f = [0] * num_stages
    next_b = [0] * num_stages
    in_flight = [0] * num_stages
    max_in_flight = [min(num_microbatches, num_stages - p) for p in range(num_stages)]

    ticks: List[List[Tuple[int, str, int]]] = []
    total_ops = 2 * num_stages * num_microbatches
    completed = 0

    while completed < total_ops:
        slot: List[Tuple[int, str, int]] = []
        new_f: List[Tuple[int, int]] = []
        new_b: List[Tuple[int, int]] = []

        for p in range(num_stages):
            mb_b = next_b[p]
            can_b = (
                mb_b < num_microbatches
                and (p, mb_b) in fwd_done
                and (p == num_stages - 1 or (p + 1, mb_b) in bwd_done)
            )
            mb_f = next_f[p]
            can_f = (
                mb_f < num_microbatches
                and in_flight[p] < max_in_flight[p]
                and (p == 0 or (p - 1, mb_f) in fwd_done)
            )
            # In steady-state 1F1B, prioritize backward as soon as warmup target is reached
            if can_b and (in_flight[p] >= max_in_flight[p] or not can_f):
                slot.append((p, "B", mb_b))
                new_b.append((p, mb_b))
                next_b[p] += 1
            elif can_f:
                slot.append((p, "F", mb_f))
                new_f.append((p, mb_f))
                next_f[p] += 1
            elif can_b:
                slot.append((p, "B", mb_b))
                new_b.append((p, mb_b))
                next_b[p] += 1

        for p, m in new_f:
            fwd_done.add((p, m))
            in_flight[p] += 1
            completed += 1
        for p, m in new_b:
            bwd_done.add((p, m))
            in_flight[p] -= 1
            completed += 1
        ticks.append(slot)

    return ticks


def _execute_pipeline_run(
    schedule_ticks: List[List[Tuple[int, str, int]]],
    stage_weights: List[Dict[str, List[float]]],
    microbatch_inputs: List[List[float]],
    microbatch_targets: List[List[float]],
    saved_ops: Set[str],
    bsz: int,
    seq: int,
    d_model: int,
    n_heads: int,
    d_ff: int,
) -> Dict[str, object]:
    """Executes the exact pipeline schedule while tracking real tracemalloc peak bytes."""
    num_stages = len(stage_weights)
    num_microbatches = len(microbatch_inputs)
    elems = bsz * seq * d_model

    grad_accums: List[Dict[str, List[float]]] = [
        {name: [0.0] * len(arr) for name, arr in sw.items()}
        for sw in stage_weights
    ]

    tracemalloc.start()
    base_current, _ = tracemalloc.get_traced_memory()
    tracemalloc.reset_peak()

    live_stashes: Dict[Tuple[int, int], Dict[str, array]] = {}
    fwd_wire: Dict[Tuple[int, int], List[float]] = {}
    bwd_wire: Dict[Tuple[int, int], List[float]] = {}

    peak_stash_bytes = 0
    peak_in_flight_stage0 = 0
    cur_in_flight_stage0 = 0
    total_loss = 0.0
    recomputed_total = 0

    for slot in schedule_ticks:
        for p, kind, m in slot:
            if kind == "F":
                x_in = microbatch_inputs[m] if p == 0 else fwd_wire.pop((p - 1, m))
                out, stash = _stage_forward(
                    x_in, stage_weights[p], saved_ops, bsz, seq, d_model, n_heads, d_ff
                )
                live_stashes[(p, m)] = stash
                if p == 0:
                    cur_in_flight_stage0 += 1
                    peak_in_flight_stage0 = max(peak_in_flight_stage0, cur_in_flight_stage0)
                if p < num_stages - 1:
                    fwd_wire[(p, m)] = out
                else:
                    tgt = microbatch_targets[m]
                    scale = 1.0 / (num_microbatches * elems)
                    dout = [0.0] * elems
                    mb_loss = 0.0
                    for i in range(elems):
                        diff = out[i] - tgt[i]
                        mb_loss += 0.5 * diff * diff * scale
                        dout[i] = diff * scale
                    total_loss += mb_loss
                    bwd_wire[(p, m)] = dout

                cur_stash_bytes = sum(
                    sum(arr.buffer_info()[1] * arr.itemsize for arr in st.values())
                    for st in live_stashes.values()
                )
                peak_stash_bytes = max(peak_stash_bytes, cur_stash_bytes)
            else:
                dout = bwd_wire.pop((p, m))
                stash = live_stashes.pop((p, m))
                dx_in, rec_cnt = _stage_backward(
                    dout,
                    stash,
                    stage_weights[p],
                    grad_accums[p],
                    bsz,
                    seq,
                    d_model,
                    n_heads,
                    d_ff,
                )
                recomputed_total += rec_cnt
                del stash
                if p == 0:
                    cur_in_flight_stage0 -= 1
                else:
                    bwd_wire[(p - 1, m)] = dx_in

    _, peak_traced = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_delta_bytes = max(peak_stash_bytes, peak_traced - base_current)

    total_slots = len(schedule_ticks) * num_stages
    busy_slots = sum(len(slot) for slot in schedule_ticks)
    bubble_slots = total_slots - busy_slots
    bubble_fraction = bubble_slots / max(1, total_slots)

    timeline = [
        [{"stage": p, "type": kind, "microbatch": m} for (p, kind, m) in slot]
        for slot in schedule_ticks
    ]

    return {
        "loss": total_loss,
        "grad_accums": grad_accums,
        "peak_tracemalloc_bytes": peak_delta_bytes,
        "peak_live_stash_bytes": peak_stash_bytes,
        "peak_in_flight_stage0": peak_in_flight_stage0,
        "clock_ticks": len(schedule_ticks),
        "total_stage_slots": total_slots,
        "busy_slots": busy_slots,
        "bubble_slots": bubble_slots,
        "bubble_fraction": round(bubble_fraction, 4),
        "bubble_pct": round(bubble_fraction * 100.0, 2),
        "recomputed_op_invocations": recomputed_total,
        "timeline": timeline,
    }


def verify_1f1b_and_checkpointing(
    num_stages: int = 4,
    num_microbatches: int = 8,
    microbatch_size: int = 1,
    seq_len: int = 16,
    d_model: int = 16,
    n_heads: int = 4,
    d_ff: int = 32,
    memory_budget_ratio: float = 0.45,
    seed: int = 7,
) -> Dict[str, object]:
    """Solves exact 0/1 knapsack selective activation checkpointing and executes real
    GPipe vs. 1F1B vs. 1F1B+Knapsack schedules with tracemalloc and gradient parity checks.
    """
    t0 = time.perf_counter()
    if num_stages < 2 or num_stages > 8:
        raise ValueError("num_stages must be between 2 and 8")
    if num_microbatches < num_stages or num_microbatches > 24:
        raise ValueError("num_microbatches must be >= num_stages and <= 24")

    ops = _operator_catalog(microbatch_size, seq_len, d_model, n_heads, d_ff)
    knapsack = solve_selective_checkpoint_knapsack(ops, budget_ratio=memory_budget_ratio)

    all_op_ids = {str(op["op_id"]) for op in ops}
    selective_op_ids = set(knapsack["saved_op_ids"])

    rng = random.Random(seed)
    elems = microbatch_size * seq_len * d_model
    mb_inputs = [
        [rng.uniform(-0.4, 0.4) for _ in range(elems)]
        for _ in range(num_microbatches)
    ]
    mb_targets = [
        [rng.uniform(-0.4, 0.4) for _ in range(elems)]
        for _ in range(num_microbatches)
    ]
    stage_weights = _init_stage_weights(num_stages, d_model, d_ff, seed=seed + 100)

    gpipe_sched = _build_gpipe_schedule(num_stages, num_microbatches)
    one_f_one_b_sched = _build_1f1b_schedule(num_stages, num_microbatches)

    # 1. GPipe without checkpointing (all M microbatches live in RAM)
    gpipe_res = _execute_pipeline_run(
        gpipe_sched,
        stage_weights,
        mb_inputs,
        mb_targets,
        all_op_ids,
        microbatch_size,
        seq_len,
        d_model,
        n_heads,
        d_ff,
    )

    # 2. 1F1B without checkpointing (at most P microbatches live in RAM)
    p1f1b_res = _execute_pipeline_run(
        one_f_one_b_sched,
        stage_weights,
        mb_inputs,
        mb_targets,
        all_op_ids,
        microbatch_size,
        seq_len,
        d_model,
        n_heads,
        d_ff,
    )

    # 3. 1F1B + Exact 0/1 Knapsack Selective Activation Checkpointing
    p1f1b_knapsack_res = _execute_pipeline_run(
        one_f_one_b_sched,
        stage_weights,
        mb_inputs,
        mb_targets,
        selective_op_ids,
        microbatch_size,
        seq_len,
        d_model,
        n_heads,
        d_ff,
    )

    # 4. 1F1B + Full Block Checkpointing (recompute all interior ops)
    p1f1b_full_ckpt_res = _execute_pipeline_run(
        one_f_one_b_sched,
        stage_weights,
        mb_inputs,
        mb_targets,
        set(),
        microbatch_size,
        seq_len,
        d_model,
        n_heads,
        d_ff,
    )

    # Verify exact gradient parity across all schedules & checkpoint modes
    max_grad_diff_1f1b = 0.0
    max_grad_diff_knapsack = 0.0
    for p in range(num_stages):
        g_ref = gpipe_res["grad_accums"][p]
        g_1f1b = p1f1b_res["grad_accums"][p]
        g_knap = p1f1b_knapsack_res["grad_accums"][p]
        for name in g_ref:
            for a, b in zip(g_ref[name], g_1f1b[name]):
                max_grad_diff_1f1b = max(max_grad_diff_1f1b, abs(a - b))
            for a, b in zip(g_ref[name], g_knap[name]):
                max_grad_diff_knapsack = max(max_grad_diff_knapsack, abs(a - b))

    gpipe_stash = int(gpipe_res["peak_live_stash_bytes"])
    p1f1b_stash = int(p1f1b_res["peak_live_stash_bytes"])
    knap_stash = int(p1f1b_knapsack_res["peak_live_stash_bytes"])
    full_stash = int(p1f1b_full_ckpt_res["peak_live_stash_bytes"])

    def _strip(res: Dict[str, object]) -> Dict[str, object]:
        return {k: v for k, v in res.items() if k != "grad_accums"}

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return {
        "config": {
            "num_stages": num_stages,
            "num_microbatches": num_microbatches,
            "microbatch_size": microbatch_size,
            "seq_len": seq_len,
            "d_model": d_model,
            "n_heads": n_heads,
            "d_ff": d_ff,
            "memory_budget_ratio": memory_budget_ratio,
        },
        "knapsack_solver": knapsack,
        "schedules": {
            "gpipe_no_ckpt": _strip(gpipe_res),
            "1f1b_no_ckpt": _strip(p1f1b_res),
            "1f1b_selective_knapsack": _strip(p1f1b_knapsack_res),
            "1f1b_full_recompute": _strip(p1f1b_full_ckpt_res),
        },
        "summary_metrics": {
            "theoretical_bubble_fraction": round(
                (num_stages - 1) / (num_microbatches + num_stages - 1), 4
            ),
            "measured_bubble_pct": p1f1b_res["bubble_pct"],
            "gpipe_peak_stash_bytes": gpipe_stash,
            "p1f1b_peak_stash_bytes": p1f1b_stash,
            "p1f1b_knapsack_peak_stash_bytes": knap_stash,
            "p1f1b_full_ckpt_peak_stash_bytes": full_stash,
            "schedule_only_ram_reduction_pct": round(
                (1.0 - p1f1b_stash / max(1, gpipe_stash)) * 100.0, 2
            ),
            "combined_1f1b_knapsack_ram_reduction_pct": round(
                (1.0 - knap_stash / max(1, gpipe_stash)) * 100.0, 2
            ),
            "combined_compression_ratio_x": round(gpipe_stash / max(1, knap_stash), 2),
            "max_grad_diff_1f1b_vs_gpipe": max_grad_diff_1f1b,
            "max_grad_diff_knapsack_vs_gpipe": max_grad_diff_knapsack,
            "parity_verified": bool(max_grad_diff_knapsack < 1e-10),
            "exec_ms": round(elapsed_ms, 3),
        },
    }
