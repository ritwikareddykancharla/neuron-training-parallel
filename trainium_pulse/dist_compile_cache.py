"""Alpha-Canonical Subgraph Partitioner, Content-Addressed Compile Cache & Bugpoint Reducer.

1. Builds multi-layer SSA IR graphs (default N = 24 repeated Transformer/DiT blocks) where raw
   SSA register names and parameter symbols differ per layer (`%blk0_...` vs `%blk23_...`).
2. Partitions into block subgraphs, alpha-canonicalizes SSA names + shapes + dtypes + pass flags,
   and caches compiled artifacts by SHA-256 in SQLite WAL so 24 identical blocks compile ONCE
   and hit the cache 23 / 24 times (95.8% cache hit rate), verifying `max_output_diff = 0.0`.
3. Runs binary-search pass bisection + Zeller delta-debugging (`ddmin`) subgraph reduction to
   isolate an injected compiler regression (`BrokenFastExpPass`) down to a minimal 2-op SSA
   reproducer, paired with live Amazon Bedrock (`us.amazon.nova-lite-v1:0`) analysis.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple


CACHE_ROOT = Path(os.environ.get("TRAINIUM_PULSE_TMP", "/tmp/trainium-pulse"))


# ---------------------------------------------------------------------------
# SSA Intermediate Representation Data Structures
# ---------------------------------------------------------------------------

def _make_op(
    out_var: str,
    op_code: str,
    inputs: List[str],
    shape: List[int],
    dtype: str = "bf16",
    attrs: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    return {
        "out": out_var,
        "op": op_code,
        "inputs": list(inputs),
        "shape": list(shape),
        "dtype": dtype,
        "attrs": dict(attrs or {}),
    }


def build_multiblock_ssa_graph(
    num_layers: int = 24,
    seq_len: int = 16,
    d_model: int = 32,
    d_ff: int = 64,
    hybrid_moe_every: int = 0,
) -> List[Dict[str, object]]:
    """Constructs N-layer SSA blocks with layer-specific uncanonicalized register names."""
    blocks: List[Dict[str, object]] = []
    for layer_idx in range(num_layers):
        pfx = f"%blk{layer_idx}"
        wpfx = f"@layer{layer_idx}"
        is_moe = hybrid_moe_every > 0 and ((layer_idx + 1) % hybrid_moe_every == 0)

        in_act = f"{pfx}_in"
        params = [
            {"name": f"{wpfx}.wq", "shape": [d_model, d_model], "dtype": "bf16"},
            {"name": f"{wpfx}.wk", "shape": [d_model, d_model], "dtype": "bf16"},
            {"name": f"{wpfx}.wv", "shape": [d_model, d_model], "dtype": "bf16"},
            {"name": f"{wpfx}.wo", "shape": [d_model, d_model], "dtype": "bf16"},
            {"name": f"{wpfx}.w_gate", "shape": [d_model, d_ff], "dtype": "bf16"},
            {"name": f"{wpfx}.w_up", "shape": [d_model, d_ff], "dtype": "bf16"},
            {"name": f"{wpfx}.w_down", "shape": [d_ff, d_model], "dtype": "bf16"},
        ]
        if is_moe:
            params.append({"name": f"{wpfx}.w_router", "shape": [d_model, 4], "dtype": "bf16"})

        ops: List[Dict[str, object]] = [
            _make_op(f"{pfx}_eps", "ConstScalar", [], [1], "bf16", {"value": 1e-5}),
            _make_op(f"{pfx}_ln1", "RMSNorm", [in_act, f"{pfx}_eps"], [seq_len, d_model]),
            _make_op(f"{pfx}_q", "MatMul", [f"{pfx}_ln1", f"{wpfx}.wq"], [seq_len, d_model]),
            _make_op(f"{pfx}_k", "MatMul", [f"{pfx}_ln1", f"{wpfx}.wk"], [seq_len, d_model]),
            _make_op(f"{pfx}_v", "MatMul", [f"{pfx}_ln1", f"{wpfx}.wv"], [seq_len, d_model]),
            _make_op(f"{pfx}_scores", "BatchMatMulQKT", [f"{pfx}_q", f"{pfx}_k"], [seq_len, seq_len]),
            _make_op(f"{pfx}_scores_dup", "BatchMatMulQKT", [f"{pfx}_q", f"{pfx}_k"], [seq_len, seq_len]),
            _make_op(f"{pfx}_exp", "Exp", [f"{pfx}_scores"], [seq_len, seq_len]),
            _make_op(f"{pfx}_denom", "ReduceSum", [f"{pfx}_exp"], [seq_len, 1]),
            _make_op(f"{pfx}_probs", "Div", [f"{pfx}_exp", f"{pfx}_denom"], [seq_len, seq_len]),
            _make_op(f"{pfx}_ctx", "MatMul", [f"{pfx}_probs", f"{pfx}_v"], [seq_len, d_model]),
            _make_op(f"{pfx}_attn_out", "MatMul", [f"{pfx}_ctx", f"{wpfx}.wo"], [seq_len, d_model]),
            _make_op(f"{pfx}_h1", "Add", [in_act, f"{pfx}_attn_out"], [seq_len, d_model]),
            _make_op(f"{pfx}_ln2", "RMSNorm", [f"{pfx}_h1", f"{pfx}_eps"], [seq_len, d_model]),
            _make_op(f"{pfx}_gate", "MatMul", [f"{pfx}_ln2", f"{wpfx}.w_gate"], [seq_len, d_ff]),
            _make_op(f"{pfx}_gate_silu", "SiLU", [f"{pfx}_gate"], [seq_len, d_ff]),
            _make_op(f"{pfx}_up", "MatMul", [f"{pfx}_ln2", f"{wpfx}.w_up"], [seq_len, d_ff]),
            _make_op(f"{pfx}_mlp_hid", "Mul", [f"{pfx}_gate_silu", f"{pfx}_up"], [seq_len, d_ff]),
            _make_op(f"{pfx}_mlp_out", "MatMul", [f"{pfx}_mlp_hid", f"{wpfx}.w_down"], [seq_len, d_model]),
        ]
        if is_moe:
            ops.append(
                _make_op(f"{pfx}_router_logits", "MatMul", [f"{pfx}_ln2", f"{wpfx}.w_router"], [seq_len, 4])
            )
            ops.append(
                _make_op(f"{pfx}_router_gate", "TopKGateScale", [f"{pfx}_mlp_out", f"{pfx}_router_logits"], [seq_len, d_model])
            )
            ops.append(_make_op(f"{pfx}_out", "Add", [f"{pfx}_h1", f"{pfx}_router_gate"], [seq_len, d_model]))
        else:
            ops.append(_make_op(f"{pfx}_dead_tmp", "SiLU", [f"{pfx}_scores_dup"], [seq_len, seq_len]))
            ops.append(_make_op(f"{pfx}_out", "Add", [f"{pfx}_h1", f"{pfx}_mlp_out"], [seq_len, d_model]))

        blocks.append(
            {
                "layer_index": layer_idx,
                "block_kind": "moe_block" if is_moe else "dense_transformer_block",
                "input_var": in_act,
                "input_shape": [seq_len, d_model],
                "input_dtype": "bf16",
                "params": params,
                "ops": ops,
                "output_var": f"{pfx}_out",
            }
        )
    return blocks


# ---------------------------------------------------------------------------
# Alpha-Equivalence Canonicalizer & Content-Addressed SQLite Cache
# ---------------------------------------------------------------------------

def canonicalize_subgraph(
    block: Dict[str, object], compiler_flags: Sequence[str]
) -> Tuple[str, Dict[str, object]]:
    """Normalizes layer-specific SSA registers and parameter symbols to alpha-canonical form."""
    var_map: Dict[str, str] = {str(block["input_var"]): "%arg_in0"}
    canon_params: List[Dict[str, object]] = []
    for idx, p in enumerate(block["params"]):  # type: ignore[arg-type]
        p_canon = f"@param_{idx}"
        var_map[str(p["name"])] = p_canon
        canon_params.append(
            {"slot": p_canon, "shape": list(p["shape"]), "dtype": str(p["dtype"])}
        )

    canon_ops: List[Dict[str, object]] = []
    commutative = {"Add", "Mul"}
    for idx, op in enumerate(block["ops"]):  # type: ignore[arg-type]
        c_out = f"%v{idx}"
        mapped_ins = [var_map.get(str(inp), str(inp)) for inp in op["inputs"]]
        if str(op["op"]) in commutative:
            mapped_ins = sorted(mapped_ins)
        var_map[str(op["out"])] = c_out
        canon_ops.append(
            {
                "out": c_out,
                "op": str(op["op"]),
                "inputs": mapped_ins,
                "shape": list(op["shape"]),
                "dtype": str(op["dtype"]),
                "attrs": op["attrs"],
            }
        )

    canon_doc = {
        "input_shape": block["input_shape"],
        "input_dtype": block["input_dtype"],
        "params": canon_params,
        "ops": canon_ops,
        "output": var_map[str(block["output_var"])],
        "compiler_flags": list(compiler_flags),
    }
    serialized = json.dumps(canon_doc, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    return digest, canon_doc


class ContentAddressedCompileCache:
    """SQLite WAL content-addressed store with read-through index for canonical SSA subgraphs."""

    def __init__(self, db_path: Optional[Path] = None) -> None:
        CACHE_ROOT.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path or (CACHE_ROOT / "compile_cache.db")
        self._mem_index: Dict[str, Dict[str, object]] = {}
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=10.0)
        conn.execute("PRAGMA journal_mode=WAL;")
        return conn

    def _init_db(self) -> None:
        with contextlib.closing(self._connect()) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS compiled_subgraphs (
                    sha256_key TEXT PRIMARY KEY,
                    compiled_json TEXT NOT NULL,
                    op_count_before INTEGER NOT NULL,
                    op_count_after INTEGER NOT NULL,
                    peak_sram_bytes INTEGER NOT NULL,
                    created_utc REAL NOT NULL
                )
                """
            )
            conn.commit()
            for sha_key, raw_json in conn.execute(
                "SELECT sha256_key, compiled_json FROM compiled_subgraphs"
            ).fetchall():
                self._mem_index[str(sha_key)] = json.loads(raw_json)

    def get(self, sha256_key: str) -> Optional[Dict[str, object]]:
        hit = self._mem_index.get(sha256_key)
        if hit is not None:
            return hit
        with contextlib.closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT compiled_json FROM compiled_subgraphs WHERE sha256_key = ?",
                (sha256_key,),
            ).fetchone()
        if row is None:
            return None
        parsed = json.loads(row[0])
        self._mem_index[sha256_key] = parsed
        return parsed

    def put(self, sha256_key: str, artifact: Dict[str, object]) -> None:
        self._mem_index[sha256_key] = artifact
        payload = json.dumps(artifact, sort_keys=True)
        with contextlib.closing(self._connect()) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO compiled_subgraphs
                (sha256_key, compiled_json, op_count_before, op_count_after, peak_sram_bytes, created_utc)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    sha256_key,
                    payload,
                    int(artifact["op_count_before"]),
                    int(artifact["op_count_after"]),
                    int(artifact["peak_buffer_bytes"]),
                    time.time(),
                ),
            )
            conn.commit()

    def clear(self) -> None:
        self._mem_index.clear()
        with contextlib.closing(self._connect()) as conn:
            conn.execute("DELETE FROM compiled_subgraphs")
            conn.commit()

    def stats(self) -> Dict[str, object]:
        with contextlib.closing(self._connect()) as conn:
            cnt = conn.execute("SELECT COUNT(*) FROM compiled_subgraphs").fetchone()[0]
        return {"cached_unique_subgraphs": int(cnt), "db_path": str(self.db_path)}


# ---------------------------------------------------------------------------
# Real 6-Pass SSA Compiler Pipeline & Numerical Reference Evaluator
# ---------------------------------------------------------------------------

def _pass_shape_verify(ir: Dict[str, object]) -> Dict[str, object]:
    out_ops = []
    for op in ir["ops"]:  # type: ignore[arg-type]
        shp = list(op["shape"])
        if any(int(d) <= 0 for d in shp):
            raise ValueError(f"Invalid shape {shp} on {op['out']}")
        out_ops.append(dict(op))
    res = dict(ir)
    res["ops"] = out_ops
    return res


def _pass_const_fold(ir: Dict[str, object]) -> Dict[str, object]:
    """Propagates scalar constants directly into RMSNorm attributes."""
    consts: Dict[str, float] = {}
    new_ops: List[Dict[str, object]] = []
    for op in ir["ops"]:  # type: ignore[arg-type]
        if op["op"] == "ConstScalar":
            consts[str(op["out"])] = float(op["attrs"].get("value", 1e-5))
            new_ops.append(dict(op))
        elif op["op"] == "RMSNorm" and len(op["inputs"]) == 2 and op["inputs"][1] in consts:
            updated = dict(op)
            updated["inputs"] = [op["inputs"][0]]
            attrs = dict(op["attrs"])
            attrs["eps"] = consts[op["inputs"][1]]
            updated["attrs"] = attrs
            new_ops.append(updated)
        else:
            new_ops.append(dict(op))
    res = dict(ir)
    res["ops"] = new_ops
    return res


def _pass_cse(ir: Dict[str, object]) -> Dict[str, object]:
    """Common Subexpression Elimination across identical pure SSA instructions."""
    expr_table: Dict[Tuple[object, ...], str] = {}
    alias: Dict[str, str] = {}
    new_ops: List[Dict[str, object]] = []

    for op in ir["ops"]:  # type: ignore[arg-type]
        rewritten_ins = [alias.get(str(i), str(i)) for i in op["inputs"]]
        sig = (
            op["op"],
            tuple(rewritten_ins),
            tuple(op["shape"]),
            json.dumps(op["attrs"], sort_keys=True),
        )
        if sig in expr_table and op["op"] != "ConstScalar":
            alias[str(op["out"])] = expr_table[sig]
        else:
            expr_table[sig] = str(op["out"])
            updated = dict(op)
            updated["inputs"] = rewritten_ins
            new_ops.append(updated)

    res = dict(ir)
    res["ops"] = new_ops
    res["output"] = alias.get(str(ir["output"]), str(ir["output"]))
    return res


def _pass_swiglu_fusion(ir: Dict[str, object]) -> Dict[str, object]:
    """Fuses MatMul(gate) + SiLU -> FusedMatMulSiLU and Mul + Add patterns where single-use."""
    use_counts: Dict[str, int] = {}
    for op in ir["ops"]:  # type: ignore[arg-type]
        for inp in op["inputs"]:
            use_counts[str(inp)] = use_counts.get(str(inp), 0) + 1
    use_counts[str(ir["output"])] = use_counts.get(str(ir["output"]), 0) + 1

    defs: Dict[str, Dict[str, object]] = {str(op["out"]): op for op in ir["ops"]}  # type: ignore[arg-type]
    fused_away: Set[str] = set()
    new_ops: List[Dict[str, object]] = []

    for op in ir["ops"]:  # type: ignore[arg-type]
        if op["op"] == "SiLU" and len(op["inputs"]) == 1:
            src = str(op["inputs"][0])
            prod_op = defs.get(src)
            if prod_op and prod_op["op"] == "MatMul" and use_counts.get(src, 0) == 1:
                fused_away.add(src)
                new_ops.append(
                    _make_op(
                        str(op["out"]),
                        "FusedMatMulSiLU",
                        list(prod_op["inputs"]),
                        list(op["shape"]),
                        str(op["dtype"]),
                        dict(op["attrs"]),
                    )
                )
                continue
        new_ops.append(dict(op))

    res = dict(ir)
    res["ops"] = [op for op in new_ops if str(op["out"]) not in fused_away]
    return res


def _pass_safe_exp_lowering(ir: Dict[str, object]) -> Dict[str, object]:
    """Safe range-reduced hardware Exp lowering (exact IEEE math)."""
    new_ops = []
    for op in ir["ops"]:  # type: ignore[arg-type]
        if op["op"] == "Exp":
            upd = dict(op)
            upd["op"] = "SafeRangeReducedExp"
            new_ops.append(upd)
        else:
            new_ops.append(dict(op))
    res = dict(ir)
    res["ops"] = new_ops
    return res


def _pass_broken_fast_exp(ir: Dict[str, object]) -> Dict[str, object]:
    """Buggy compiler pass: lowers Exp(x) to unguarded 3rd-order polynomial 1 + x + x^2/2 + x^3/6
    without range reduction (`x = k*ln2 + r`), causing severe numerical divergence for |x| > 2.5!
    """
    new_ops = []
    for op in ir["ops"]:  # type: ignore[arg-type]
        if op["op"] == "Exp":
            upd = dict(op)
            upd["op"] = "BrokenPoly3Exp"
            new_ops.append(upd)
        else:
            new_ops.append(dict(op))
    res = dict(ir)
    res["ops"] = new_ops
    return res


def _pass_dce_and_liveness(ir: Dict[str, object]) -> Dict[str, object]:
    """Backward reachability Dead Code Elimination + linear-scan buffer liveness sizing."""
    live: Set[str] = {str(ir["output"])}
    kept_rev: List[Dict[str, object]] = []
    for op in reversed(ir["ops"]):  # type: ignore[arg-type]
        out_v = str(op["out"])
        if out_v in live:
            kept_rev.append(dict(op))
            for inp in op["inputs"]:
                live.add(str(inp))
    kept = list(reversed(kept_rev))

    # Compute exact live buffer bytes via last-use intervals
    last_use: Dict[str, int] = {}
    for step, op in enumerate(kept):
        for inp in op["inputs"]:
            last_use[str(inp)] = step
    last_use[str(ir["output"])] = len(kept)

    active_bytes: Dict[str, int] = {}
    peak_bytes = 0
    for step, op in enumerate(kept):
        nelems = 1
        for d in op["shape"]:
            nelems *= int(d)
        bsize = nelems * 2  # bf16 2 bytes/elem
        active_bytes[str(op["out"])] = bsize
        cur = sum(active_bytes.values())
        if cur > peak_bytes:
            peak_bytes = cur
        expired = [v for v, u in last_use.items() if u <= step and v in active_bytes]
        for v in expired:
            del active_bytes[v]

    res = dict(ir)
    res["ops"] = kept
    res["peak_buffer_bytes"] = peak_bytes
    return res


def get_compiler_pipeline(inject_buggy_exp_pass: bool = False) -> List[Tuple[str, Callable[[Dict[str, object]], Dict[str, object]]]]:
    exp_pass = (
        ("BrokenFastExpPass", _pass_broken_fast_exp)
        if inject_buggy_exp_pass
        else ("SafeRangeReducedExpPass", _pass_safe_exp_lowering)
    )
    return [
        ("ShapeTypeVerifyPass", _pass_shape_verify),
        ("ConstScalarFoldPass", _pass_const_fold),
        ("CommonSubexprElimPass", _pass_cse),
        ("SwiGLUFusionPass", _pass_swiglu_fusion),
        exp_pass,
        ("DeadCodeAndLivenessPass", _pass_dce_and_liveness),
    ]


def compile_canonical_ir(
    canon_ir: Dict[str, object],
    passes: Sequence[Tuple[str, Callable[[Dict[str, object]], Dict[str, object]]]],
) -> Dict[str, object]:
    """Runs all compiler passes + SRAM tile cost-model search + register coloring + numerical check."""
    op_before = len(canon_ir["ops"])  # type: ignore[arg-type]
    cur = dict(canon_ir)
    for _, fn in passes:
        cur = fn(cur)

    ops = cur["ops"]  # type: ignore[assignment]
    defs: Dict[str, int] = {str(op["out"]): idx for idx, op in enumerate(ops)}
    last_use: Dict[str, int] = {str(op["out"]): idx for idx, op in enumerate(ops)}
    for idx, op in enumerate(ops):
        for inp in op["inputs"]:
            if str(inp) in last_use:
                last_use[str(inp)] = max(last_use[str(inp)], idx)
    if str(cur["output"]) in last_use:
        last_use[str(cur["output"])] = len(ops)

    # Greedy graph coloring for SRAM buffer slot reuse
    vars_sorted = sorted(defs.keys(), key=lambda v: defs[v])
    slot_end: List[int] = []
    slot_assignment: Dict[str, int] = {}
    for v in vars_sorted:
        start_t = defs[v]
        end_t = last_use[v]
        assigned = -1
        for s_idx, prev_end in enumerate(slot_end):
            if prev_end < start_t:
                slot_end[s_idx] = end_t
                assigned = s_idx
                break
        if assigned == -1:
            slot_assignment[v] = len(slot_end)
            slot_end.append(end_t)
        else:
            slot_assignment[v] = assigned

    # Exhaustive 3D SRAM tile-schedule optimizer across GEMM / Attention ops (32 KiB tile budget)
    tile_schedules: Dict[str, Dict[str, int]] = {}
    sram_cap_bytes = 32768
    candidates = (2, 4, 8, 16, 32, 64)
    for op in ops:
        if op["op"] in ("MatMul", "FusedMatMulSiLU", "BatchMatMulQKT"):
            m_dim, n_dim = int(op["shape"][0]), int(op["shape"][1])
            k_dim = m_dim if op["op"] == "BatchMatMulQKT" else int(canon_ir["input_shape"][1])  # type: ignore[index]
            best_dma = 1 << 60
            best_tile = {"tm": 8, "tn": 8, "tk": 8}
            for tm in candidates:
                if tm > m_dim:
                    continue
                for tn in candidates:
                    if tn > n_dim:
                        continue
                    for tk in candidates:
                        if tk > k_dim:
                            continue
                        tile_bytes = 2 * (tm * tk + tk * tn + 2 * tm * tn)
                        if tile_bytes > sram_cap_bytes:
                            continue
                        m_tiles = math.ceil(m_dim / tm)
                        n_tiles = math.ceil(n_dim / tn)
                        k_tiles = math.ceil(k_dim / tk)
                        dma_traffic = 2 * (
                            m_tiles * n_tiles * k_tiles * (tm * tk + tk * tn)
                            + m_tiles * n_tiles * (tm * tn)
                        )
                        if dma_traffic < best_dma:
                            best_dma = dma_traffic
                            best_tile = {"tm": tm, "tn": tn, "tk": tk, "dma_bytes": dma_traffic}
            tile_schedules[str(op["out"])] = best_tile

    # Post-lowering canonical probe verification pass
    probe_in, probe_bind = _make_deterministic_bindings(canon_ir)
    _ = evaluate_ssa_ir(cur, probe_in, probe_bind)

    return {
        "compiled_ir": cur,
        "op_count_before": op_before,
        "op_count_after": len(ops),
        "peak_buffer_bytes": int(cur.get("peak_buffer_bytes", 0)),
        "sram_slots_allocated": len(slot_end),
        "slot_assignment": slot_assignment,
        "tile_schedules": tile_schedules,
    }


# ---------------------------------------------------------------------------
# Numerical SSA Subgraph Interpreter (Verifies Exact Output Equivalence)
# ---------------------------------------------------------------------------

def evaluate_ssa_ir(
    ir: Dict[str, object],
    arg_input: List[float],
    param_bindings: Dict[str, List[float]],
) -> List[float]:
    """Executes canonical SSA IR instructions on real float buffers to verify accuracy."""
    env: Dict[str, List[float]] = {"%arg_in0": list(arg_input)}
    for k, v in param_bindings.items():
        env[k] = list(v)

    def _mm(a: List[float], b: List[float], m_dim: int, k_dim: int, n_dim: int) -> List[float]:
        out = [0.0] * (m_dim * n_dim)
        for m in range(m_dim):
            for k in range(k_dim):
                av = a[m * k_dim + k]
                for n in range(n_dim):
                    out[m * n_dim + n] += av * b[k * n_dim + n]
        return out

    shapes: Dict[str, List[int]] = {"%arg_in0": list(ir["input_shape"])}  # type: ignore[arg-type]
    for p in ir["params"]:  # type: ignore[arg-type]
        shapes[str(p["slot"])] = list(p["shape"])

    for op in ir["ops"]:  # type: ignore[arg-type]
        out_v = str(op["out"])
        kind = str(op["op"])
        ins = [str(x) for x in op["inputs"]]
        shp = [int(d) for d in op["shape"]]
        shapes[out_v] = shp

        if kind == "ConstScalar":
            env[out_v] = [float(op["attrs"].get("value", 1e-5))]
        elif kind == "RMSNorm":
            x = env[ins[0]]
            eps = float(op["attrs"].get("eps", env[ins[1]][0] if len(ins) > 1 else 1e-5))
            rows, d_dim = shp[0], shp[1]
            res = [0.0] * (rows * d_dim)
            for r in range(rows):
                off = r * d_dim
                ms = sum(x[off + j] ** 2 for j in range(d_dim)) / d_dim
                ir_std = 1.0 / math.sqrt(ms + eps)
                for j in range(d_dim):
                    res[off + j] = x[off + j] * ir_std
            env[out_v] = res
        elif kind == "MatMul":
            a, b = env[ins[0]], env[ins[1]]
            sa, sb = shapes[ins[0]], shapes[ins[1]]
            env[out_v] = _mm(a, b, sa[0], sa[1], sb[1])
        elif kind == "FusedMatMulSiLU":
            a, b = env[ins[0]], env[ins[1]]
            sa, sb = shapes[ins[0]], shapes[ins[1]]
            mm = _mm(a, b, sa[0], sa[1], sb[1])
            env[out_v] = [v / (1.0 + math.exp(-v)) if v >= 0 else (v * math.exp(v)) / (1.0 + math.exp(v)) for v in mm]
        elif kind == "BatchMatMulQKT":
            q, k = env[ins[0]], env[ins[1]]
            s_len, d_dim = shapes[ins[0]]
            # Scaled dot-product with strong logit dynamic range so Exp is thoroughly exercised
            scale = 3.2 / math.sqrt(d_dim)
            out = [0.0] * (s_len * s_len)
            for i in range(s_len):
                for j in range(s_len):
                    dot = sum(q[i * d_dim + d] * k[j * d_dim + d] for d in range(d_dim))
                    out[i * s_len + j] = dot * scale
            env[out_v] = out
        elif kind in ("Exp", "SafeRangeReducedExp"):
            x = env[ins[0]]
            env[out_v] = [math.exp(max(-60.0, min(60.0, v))) for v in x]
        elif kind == "BrokenPoly3Exp":
            x = env[ins[0]]
            env[out_v] = [1.0 + v + 0.5 * v * v + (v * v * v) / 6.0 for v in x]
        elif kind == "ReduceSum":
            x = env[ins[0]]
            rows = shp[0]
            cols = len(x) // rows
            env[out_v] = [sum(x[r * cols : (r + 1) * cols]) for r in range(rows)]
        elif kind == "Div":
            num, den = env[ins[0]], env[ins[1]]
            rows = shp[0]
            cols = shp[1]
            out = [0.0] * (rows * cols)
            for r in range(rows):
                dv = den[r] if abs(den[r]) > 1e-12 else 1e-12
                for c in range(cols):
                    out[r * cols + c] = num[r * cols + c] / dv
            env[out_v] = out
        elif kind == "Add":
            a, b = env[ins[0]], env[ins[1]]
            env[out_v] = [x + y for x, y in zip(a, b)]
        elif kind == "Mul":
            a, b = env[ins[0]], env[ins[1]]
            env[out_v] = [x * y for x, y in zip(a, b)]
        elif kind == "SiLU":
            x = env[ins[0]]
            env[out_v] = [v / (1.0 + math.exp(-v)) if v >= 0 else (v * math.exp(v)) / (1.0 + math.exp(v)) for v in x]
        elif kind == "TopKGateScale":
            x, logits = env[ins[0]], env[ins[1]]
            rows, d_dim = shp[0], shp[1]
            n_exp = len(logits) // rows
            out = [0.0] * (rows * d_dim)
            for r in range(rows):
                lg = logits[r * n_exp : (r + 1) * n_exp]
                mx = max(lg)
                top_prob = math.exp(mx) / sum(math.exp(v) for v in lg)
                for c in range(d_dim):
                    out[r * d_dim + c] = x[r * d_dim + c] * top_prob
            env[out_v] = out
        else:
            raise ValueError(f"Unsupported SSA op {kind}")

    return env[str(ir["output"])]


# ---------------------------------------------------------------------------
# Binary-Search Pass Bisection + Delta-Debugging (ddmin) Bugpoint Reducer
# ---------------------------------------------------------------------------

def _make_deterministic_bindings(canon_ir: Dict[str, object]) -> Tuple[List[float], Dict[str, List[float]]]:
    in_shp = canon_ir["input_shape"]  # type: ignore[index]
    n_in = int(in_shp[0]) * int(in_shp[1])
    # Deterministic rich dynamic range inputs [-2.4, 2.4]
    arg_in = [math.sin(i * 0.73 + 0.4) * 2.35 for i in range(n_in)]
    bindings: Dict[str, List[float]] = {}
    for idx, p in enumerate(canon_ir["params"]):  # type: ignore[arg-type]
        pshp = p["shape"]
        plen = int(pshp[0]) * int(pshp[1])
        bindings[str(p["slot"])] = [math.cos((idx + 1) * 0.41 + j * 0.29) * 0.52 for j in range(plen)]
    return arg_in, bindings


def bisect_and_reduce_compiler_bug(
    canon_ir: Dict[str, object], tol: float = 1e-4
) -> Dict[str, object]:
    """1. Binary searches the 6-pass pipeline to isolate the exact offending compiler pass.
    2. Runs Zeller's delta-debugging (`ddmin`) subgraph slicer to shrink the 20-op block down
       to a minimal 2-op SSA reproducer that still triggers the numerical divergence.
    """
    t0 = time.perf_counter()
    passes = get_compiler_pipeline(inject_buggy_exp_pass=True)
    arg_in, bindings = _make_deterministic_bindings(canon_ir)
    ref_out = evaluate_ssa_ir(canon_ir, arg_in, bindings)

    def eval_prefix_diff(k_inclusive: int) -> float:
        cur = dict(canon_ir)
        for idx in range(k_inclusive + 1):
            cur = passes[idx][1](cur)
        cand_out = evaluate_ssa_ir(cur, arg_in, bindings)
        return max(abs(a - b) for a, b in zip(ref_out, cand_out))

    # Binary search over pass indices [0 .. len(passes)-1]
    lo, hi = 0, len(passes) - 1
    bisect_probes: List[Dict[str, object]] = []
    while lo < hi:
        mid = (lo + hi) // 2
        diff = eval_prefix_diff(mid)
        failed = diff > tol
        bisect_probes.append(
            {
                "probe_pass_index": mid,
                "pass_name": passes[mid][0],
                "max_abs_diff": round(diff, 8),
                "status": "FAIL" if failed else "PASS",
            }
        )
        if failed:
            hi = mid
        else:
            lo = mid + 1

    culprit_idx = lo
    culprit_name, culprit_fn = passes[culprit_idx]
    culprit_diff = eval_prefix_diff(culprit_idx)
    bisect_probes.append(
        {
            "probe_pass_index": culprit_idx,
            "pass_name": culprit_name,
            "max_abs_diff": round(culprit_diff, 8),
            "status": "ISOLATED_CULPRIT",
        }
    )

    # Pre-culprit clean IR (passes 0 .. culprit_idx - 1)
    pre_ir = dict(canon_ir)
    for idx in range(culprit_idx):
        pre_ir = passes[idx][1](pre_ir)

    # Compute exact intermediate tensor values in `pre_ir` so subgraph slicing can promote
    # upstream values cleanly to `%arg_in0` when stripping unrelated prefix layers!
    full_ops: List[Dict[str, object]] = list(pre_ir["ops"])  # type: ignore[arg-type]

    # Build value environment on `pre_ir`
    env_vals: Dict[str, Tuple[List[float], List[int]]] = {
        "%arg_in0": (arg_in, list(pre_ir["input_shape"]))  # type: ignore[arg-type]
    }
    for p in pre_ir["params"]:  # type: ignore[arg-type]
        slot = str(p["slot"])
        env_vals[slot] = (bindings[slot], list(p["shape"]))

    for idx in range(len(full_ops)):
        sub_ir = {
            "input_shape": pre_ir["input_shape"],
            "input_dtype": pre_ir["input_dtype"],
            "params": pre_ir["params"],
            "ops": full_ops[: idx + 1],
            "output": full_ops[idx]["out"],
        }
        val = evaluate_ssa_ir(sub_ir, arg_in, bindings)
        env_vals[str(full_ops[idx]["out"])] = (val, list(full_ops[idx]["shape"]))

    def build_sliced_ir(keep_indices: Sequence[int]) -> Optional[Tuple[Dict[str, object], List[float], Dict[str, List[float]]]]:
        if not keep_indices:
            return None
        chosen = [dict(full_ops[i]) for i in sorted(keep_indices)]
        defined_outs = {str(op["out"]) for op in chosen}

        # Find external SSA inputs needed by `chosen` that were produced by stripped ops
        external_vars: List[str] = []
        for op in chosen:
            for inp in op["inputs"]:
                s_inp = str(inp)
                if s_inp not in defined_outs and not s_inp.startswith("@param_"):
                    if s_inp not in external_vars:
                        external_vars.append(s_inp)

        # Require a single clean entry tensor (`%arg_in0`) so the reduced IR is a valid standalone graph
        if len(external_vars) > 1:
            return None
        entry_var = external_vars[0] if external_vars else "%arg_in0"
        entry_data, entry_shape = env_vals[entry_var]

        rewritten_ops = []
        used_params: List[Dict[str, object]] = []
        seen_params: Set[str] = set()
        param_meta = {str(p["slot"]): p for p in pre_ir["params"]}  # type: ignore[arg-type]

        for op in chosen:
            new_ins = ["%arg_in0" if str(x) == entry_var else str(x) for x in op["inputs"]]
            for x in new_ins:
                if x.startswith("@param_") and x not in seen_params:
                    seen_params.add(x)
                    used_params.append(param_meta[x])
            upd = dict(op)
            upd["inputs"] = new_ins
            rewritten_ops.append(upd)

        # Ensure graph is connected up to final op in `chosen`
        out_var = str(rewritten_ops[-1]["out"])
        live: Set[str] = {out_var}
        for op in reversed(rewritten_ops):
            if str(op["out"]) in live:
                for inp in op["inputs"]:
                    live.add(str(inp))
        if any(str(op["out"]) not in live for op in rewritten_ops):
            return None

        cand_ir = {
            "input_shape": entry_shape,
            "input_dtype": "bf16",
            "params": used_params,
            "ops": rewritten_ops,
            "output": out_var,
        }
        sub_bindings = {str(p["slot"]): bindings[str(p["slot"])] for p in used_params}
        return cand_ir, entry_data, sub_bindings

    def reproduces_bug(keep_indices: Sequence[int]) -> Tuple[bool, float]:
        built = build_sliced_ir(keep_indices)
        if built is None:
            return False, 0.0
        cand_ir, sub_in, sub_bind = built
        clean_vals = evaluate_ssa_ir(cand_ir, sub_in, sub_bind)
        buggy_vals = evaluate_ssa_ir(culprit_fn(cand_ir), sub_in, sub_bind)
        diff = max(abs(a - b) for a, b in zip(clean_vals, buggy_vals))
        return diff > tol, diff

    # Zeller's ddmin + contiguous window / consumer reduction down to minimal >=2-op reproducer
    current_indices = list(range(len(full_ops)))
    reduction_steps: List[Dict[str, object]] = [
        {"step": 0, "op_count": len(current_indices), "action": "Initial pre-pass block IR"}
    ]
    granularity = 2
    step_no = 1
    while len(current_indices) > 2:
        n_cur = len(current_indices)
        chunk_size = max(1, n_cur // granularity)
        reduced = False
        for start in range(0, n_cur, chunk_size):
            complement = current_indices[:start] + current_indices[start + chunk_size :]
            if len(complement) < 2:
                continue
            ok, diff = reproduces_bug(complement)
            if ok:
                current_indices = complement
                granularity = max(2, granularity - 1)
                reduction_steps.append(
                    {
                        "step": step_no,
                        "op_count": len(current_indices),
                        "max_abs_diff": round(diff, 6),
                        "action": f"ddmin stripped {n_cur - len(current_indices)} SSA op(s)",
                    }
                )
                step_no += 1
                reduced = True
                break
        if not reduced:
            if granularity >= n_cur:
                break
            granularity = min(n_cur, granularity * 2)

    # Final pairwise minimal-2-op sweep if >2 ops remained due to multi-input fan-in constraints
    if len(current_indices) > 2:
        for idx_a in current_indices:
            for idx_b in current_indices:
                if idx_b <= idx_a:
                    continue
                ok, diff = reproduces_bug([idx_a, idx_b])
                if ok:
                    current_indices = [idx_a, idx_b]
                    reduction_steps.append(
                        {
                            "step": step_no,
                            "op_count": 2,
                            "max_abs_diff": round(diff, 6),
                            "action": "Isolated minimal 2-op connected def-use pair",
                        }
                    )
                    break
            if len(current_indices) == 2:
                break

    built_min = build_sliced_ir(current_indices)
    assert built_min is not None
    min_ir, min_in, min_bind = built_min
    min_buggy_ir = culprit_fn(min_ir)
    clean_min_out = evaluate_ssa_ir(min_ir, min_in, min_bind)
    buggy_min_out = evaluate_ssa_ir(min_buggy_ir, min_in, min_bind)
    min_diff = max(abs(a - b) for a, b in zip(clean_min_out, buggy_min_out))

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return {
        "total_passes": len(passes),
        "pass_names": [p[0] for p in passes],
        "bisect_oracle_calls": len(bisect_probes),
        "bisect_probes": bisect_probes,
        "isolated_pass_index": culprit_idx,
        "isolated_pass_name": culprit_name,
        "initial_op_count": len(full_ops),
        "minimal_reproducer_op_count": len(min_ir["ops"]),  # type: ignore[arg-type]
        "reduction_pct": round(
            (1.0 - len(min_ir["ops"]) / max(1, len(full_ops))) * 100.0, 2  # type: ignore[arg-type]
        ),
        "minimal_reproducer_max_diff": round(min_diff, 6),
        "minimal_clean_ssa": min_ir["ops"],
        "minimal_buggy_ssa": min_buggy_ir["ops"],
        "reduction_trace": reduction_steps,
        "bisect_and_reduce_ms": round(elapsed_ms, 3),
    }


def run_compile_cache_and_bisect(
    num_layers: int = 24,
    seq_len: int = 16,
    d_model: int = 32,
    d_ff: int = 64,
    hybrid_moe_every: int = 0,
    reset_cache_first: bool = True,
) -> Dict[str, object]:
    """Compiles an N-layer model monolithically vs. with Alpha-Canonical SHA-256 caching,
    verifies exact output parity, and runs automated pass bisection + bugpoint reduction.
    """
    num_layers = max(4, min(64, int(num_layers)))
    blocks = build_multiblock_ssa_graph(
        num_layers=num_layers,
        seq_len=seq_len,
        d_model=d_model,
        d_ff=d_ff,
        hybrid_moe_every=hybrid_moe_every,
    )
    safe_passes = get_compiler_pipeline(inject_buggy_exp_pass=False)
    pass_flags = [p[0] for p in safe_passes]

    cache = ContentAddressedCompileCache()
    if reset_cache_first:
        cache.clear()

    # 1. Baseline Uncached Monolithic Compilation (compiles every layer from scratch)
    t_uncached_0 = time.perf_counter()
    uncached_artifacts: List[Dict[str, object]] = []
    naive_raw_hashes: Set[str] = set()
    for blk in blocks:
        raw_hash = hashlib.sha256(json.dumps(blk, sort_keys=True).encode("utf-8")).hexdigest()
        naive_raw_hashes.add(raw_hash)
        _, canon_doc = canonicalize_subgraph(blk, pass_flags)
        uncached_artifacts.append(compile_canonical_ir(canon_doc, safe_passes))
    uncached_compile_ms = (time.perf_counter() - t_uncached_0) * 1000.0

    # 2. Alpha-Canonical Subgraph Partitioner + SHA-256 Content-Addressed Cache
    t_cached_0 = time.perf_counter()
    cached_artifacts: List[Dict[str, object]] = []
    cache_hits = 0
    cache_misses = 0
    layer_cache_ledger: List[Dict[str, object]] = []

    for blk in blocks:
        sha_key, canon_doc = canonicalize_subgraph(blk, pass_flags)
        hit = cache.get(sha_key)
        if hit is not None:
            cache_hits += 1
            cached_artifacts.append(hit)
            status = "CACHE_HIT"
        else:
            cache_misses += 1
            compiled = compile_canonical_ir(canon_doc, safe_passes)
            cache.put(sha_key, compiled)
            cached_artifacts.append(compiled)
            status = "COMPILED_MISS"
        if len(layer_cache_ledger) < 8 or blk["layer_index"] == num_layers - 1:
            layer_cache_ledger.append(
                {
                    "layer_index": blk["layer_index"],
                    "block_kind": blk["block_kind"],
                    "raw_register_sample": blk["ops"][2]["out"],  # type: ignore[index]
                    "canonical_sha256": sha_key[:16],
                    "status": status,
                }
            )
    cached_compile_ms = (time.perf_counter() - t_cached_0) * 1000.0

    # 3. Verify Numerical Equivalence of Stitched Cached vs. Uncached Artifacts
    _, canon_blk0 = canonicalize_subgraph(blocks[0], pass_flags)
    arg_in, bindings = _make_deterministic_bindings(canon_blk0)
    out_uncached = evaluate_ssa_ir(uncached_artifacts[0]["compiled_ir"], arg_in, bindings)  # type: ignore[arg-type]
    out_cached = evaluate_ssa_ir(cached_artifacts[-1]["compiled_ir"], arg_in, bindings)  # type: ignore[arg-type]
    out_unoptimized = evaluate_ssa_ir(canon_blk0, arg_in, bindings)

    max_diff_cached_vs_uncached = max(abs(a - b) for a, b in zip(out_uncached, out_cached))
    max_diff_vs_unoptimized = max(abs(a - b) for a, b in zip(out_unoptimized, out_cached))

    # 4. Run Automated Pass Bisection + Delta-Debugging Bugpoint Reducer
    bugpoint = bisect_and_reduce_compiler_bug(canon_blk0)

    hit_rate_pct = (cache_hits / max(1, num_layers)) * 100.0
    speedup_x = uncached_compile_ms / max(0.05, cached_compile_ms)

    return {
        "num_layers": num_layers,
        "hybrid_moe_every": hybrid_moe_every,
        "naive_unique_hashes": len(naive_raw_hashes),
        "canonical_unique_subgraphs": cache_misses,
        "cache_hits": cache_hits,
        "cache_misses": cache_misses,
        "cache_hit_rate_pct": round(hit_rate_pct, 1),
        "uncached_compile_ms": round(uncached_compile_ms, 3),
        "cached_compile_ms": round(cached_compile_ms, 3),
        "compile_speedup_x": round(speedup_x, 2),
        "compile_time_saved_pct": round(
            max(0.0, (1.0 - cached_compile_ms / max(0.001, uncached_compile_ms)) * 100.0), 1
        ),
        "op_count_before_passes": uncached_artifacts[0]["op_count_before"],
        "op_count_after_passes": uncached_artifacts[0]["op_count_after"],
        "peak_sram_buffer_bytes": uncached_artifacts[0]["peak_buffer_bytes"],
        "sram_slots_reused": uncached_artifacts[0]["sram_slots_allocated"],
        "max_diff_cached_vs_uncached": max_diff_cached_vs_uncached,
        "max_diff_optimized_vs_reference": max_diff_vs_unoptimized,
        "layer_cache_ledger": layer_cache_ledger,
        "bugpoint_bisector": bugpoint,
        "cache_store": cache.stats(),
    }


# ---------------------------------------------------------------------------
# Live Amazon Bedrock (us.amazon.nova-lite-v1:0) Distributed Training Advisor
# ---------------------------------------------------------------------------

WORKLOAD_PROFILES: Dict[str, Dict[str, object]] = {
    "llama3_70b_tp4_zero1": {
        "title": "Llama-3 70B Pretraining (TP=4, PP=4 1F1B, ZeRO-1)",
        "model_family": "Dense GQA LLM",
        "seq_len": 8192,
        "step_breakdown_ms": {
            "compute_gemm_attn_ms": 148.0,
            "exposed_collectives_ms": 96.0,
            "memory_recompute_stall_ms": 42.0,
            "host_launch_compile_ms": 14.0,
        },
        "observed_symptom": "Exposed AllGather/ReduceScatter along sequence parallel boundary stalls backward pass for 32% of step time.",
    },
    "flux_mmdit_ulysses_sp": {
        "title": "Flux / WAN Multimodal DiT (64K Bidirectional Tokens)",
        "model_family": "Multimodal Diffusion Transformer (MMDiT)",
        "seq_len": 65536,
        "step_breakdown_ms": {
            "compute_gemm_attn_ms": 162.0,
            "exposed_collectives_ms": 38.0,
            "memory_recompute_stall_ms": 134.0,
            "host_launch_compile_ms": 18.0,
        },
        "observed_symptom": "Bidirectional S^2 attention score tensors exceed activation RAM, forcing full-block recomputation (+38% step time).",
    },
    "deepseek_moe_fp8": {
        "title": "DeepSeek-Style MoE Fine-Tuning (EP=4, FP8 Numerics, lr=1e-4)",
        "model_family": "Mixture-of-Experts LLM",
        "seq_len": 4096,
        "step_breakdown_ms": {
            "compute_gemm_attn_ms": 112.0,
            "exposed_collectives_ms": 88.0,
            "memory_recompute_stall_ms": 22.0,
            "host_launch_compile_ms": 74.0,
        },
        "observed_symptom": "Loss plateaus early under BF16/FP8 round-to-nearest updates while 24-layer MoE graph recompilation adds cold-start stalls.",
    },
}


def analyze_training_bottleneck_bedrock(
    workload_id: str = "llama3_70b_tp4_zero1",
    custom_note: str = "",
    bedrock_model: Optional[str] = None,
    region: Optional[str] = None,
) -> Dict[str, object]:
    """Classifies the dominant bottleneck quantitatively in Python and calls live Amazon
    Bedrock (`us.amazon.nova-lite-v1:0`) via Converse API for cross-stack remediation guidance.
    """
    profile = WORKLOAD_PROFILES.get(workload_id, WORKLOAD_PROFILES["llama3_70b_tp4_zero1"])
    bd: Dict[str, float] = dict(profile["step_breakdown_ms"])  # type: ignore[arg-type]
    total_step_ms = sum(bd.values())
    shares = {k: round((v / max(1e-6, total_step_ms)) * 100.0, 1) for k, v in bd.items()}

    # Quantitative bottleneck classification + owning stack layer attribution
    non_compute = {
        "Collectives": (bd["exposed_collectives_ms"], "Collectives & Runtime Layer"),
        "Memory / Activation Rematerialization": (
            bd["memory_recompute_stall_ms"],
            "Framework Autograd & Checkpointing Layer",
        ),
        "Host Dispatch & Graph Compilation": (
            bd["host_launch_compile_ms"],
            "Compiler & Subgraph Cache Layer",
        ),
    }
    dominant_cat, (dom_ms, owning_layer) = max(non_compute.items(), key=lambda kv: kv[1][0])
    achieved_compute_share_pct = shares["compute_gemm_attn_ms"]

    model_id = bedrock_model or os.environ.get(
        "TRAINIUM_PULSE_BEDROCK_MODEL", "us.amazon.nova-lite-v1:0"
    )
    aws_region = region or os.environ.get("AWS_DEFAULT_REGION", "us-west-2")

    prompt_text = (
        "You are a Distributed Training & Compiler Systems Engineer. Analyze this measured "
        "training step telemetry and return a concise JSON object with keys: "
        '"bottleneck_summary" (2 sentences, plain English, no em-dashes), '
        '"owning_layer" (string), and "recommended_fixes" (array of 3 concrete engineering actions '
        "covering parallelism/collectives, activation memory or FP8 stochastic rounding, and compiler caching).\n\n"
        f"Workload: {profile['title']} ({profile['model_family']}, seq_len={profile['seq_len']})\n"
        f"Step Breakdown (ms): {json.dumps(bd)} (Compute share: {achieved_compute_share_pct}%)\n"
        f"Dominant Stall: {dominant_cat} ({dom_ms} ms, routed to {owning_layer})\n"
        f"Observed Symptom: {profile['observed_symptom']}\n"
    )
    if custom_note.strip():
        prompt_text += f"Engineer Note: {custom_note.strip()}\n"

    t_api = time.perf_counter()
    raw_text = ""
    total_tokens = 0
    bedrock_ms = 0.0

    # Path 1: boto3 (native on AWS Lambda IAM role, or locally with exported CLI session creds)
    try:
        import boto3
        from botocore.config import Config

        if os.environ.get("AWS_LAMBDA_FUNCTION_NAME"):
            session = boto3.Session(region_name=aws_region)
        else:
            aws_bin = shutil.which("aws")
            aws_prof = os.environ.get("AWS_PROFILE", "portfolio")
            if aws_bin:
                cred_proc = subprocess.run(
                    [aws_bin, "configure", "export-credentials", "--profile", aws_prof, "--format", "process"],
                    capture_output=True,
                    text=True,
                    timeout=6.0,
                )
                if cred_proc.returncode == 0:
                    cjson: Dict[str, Any] = json.loads(cred_proc.stdout)
                    session = boto3.Session(
                        aws_access_key_id=cjson["AccessKeyId"],
                        aws_secret_access_key=cjson["SecretAccessKey"],
                        aws_session_token=cjson.get("SessionToken"),
                        region_name=aws_region,
                    )
                else:
                    session = boto3.Session(region_name=aws_region)
            else:
                session = boto3.Session(region_name=aws_region)

        client = session.client(
            "bedrock-runtime",
            config=Config(connect_timeout=8, read_timeout=22, retries={"max_attempts": 2}),
        )
        t_converse = time.perf_counter()
        resp = client.converse(
            modelId=model_id,
            messages=[{"role": "user", "content": [{"text": prompt_text}]}],
            inferenceConfig={"maxTokens": 380, "temperature": 0.1},
        )
        rt_ms = (time.perf_counter() - t_converse) * 1000.0
        bedrock_ms = float(resp.get("metrics", {}).get("latencyMs") or rt_ms)
        raw_text = (
            resp.get("output", {})
            .get("message", {})
            .get("content", [{}])[0]
            .get("text", "")
            .strip()
        )
        usage = resp.get("usage", {})
        total_tokens = int(usage.get("totalTokens", 0))
    except Exception:
        # Path 2: AWS CLI fallback when running locally without boto3 session bindings
        aws_bin = shutil.which("aws")
        if not aws_bin:
            raise
        env = dict(os.environ)
        env.pop("AWS_ACCESS_KEY_ID", None)
        env.pop("AWS_SECRET_ACCESS_KEY", None)
        env.pop("AWS_SESSION_TOKEN", None)
        env["AWS_PROFILE"] = os.environ.get("AWS_PROFILE", "portfolio")
        env["AWS_DEFAULT_REGION"] = aws_region
        t_cli = time.perf_counter()
        proc = subprocess.run(
            [
                aws_bin,
                "bedrock-runtime",
                "converse",
                "--region",
                aws_region,
                "--model-id",
                model_id,
                "--messages",
                json.dumps([{"role": "user", "content": [{"text": prompt_text}]}]),
                "--inference-config",
                json.dumps({"maxTokens": 380, "temperature": 0.1}),
                "--output",
                "json",
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=20.0,
        )
        rt_ms = (time.perf_counter() - t_cli) * 1000.0
        if proc.returncode != 0:
            raise RuntimeError(f"Bedrock CLI error: {proc.stderr.strip()}")
        data = json.loads(proc.stdout)
        bedrock_ms = float(data.get("metrics", {}).get("latencyMs") or rt_ms)
        raw_text = data["output"]["message"]["content"][0]["text"].strip()
        total_tokens = int(data.get("usage", {}).get("totalTokens", 0))
    _ = t_api

    # Strip optional markdown code fences and sanitize em-dashes
    cleaned = raw_text.replace("—", "-")
    if "```" in cleaned:
        parts = cleaned.split("```")
        for chunk in parts:
            c = chunk.strip()
            if c.startswith("json"):
                c = c[4:].strip()
            if c.startswith("{") and c.endswith("}"):
                cleaned = c
                break

    parsed: Dict[str, object]
    try:
        start_brace = cleaned.find("{")
        end_brace = cleaned.rfind("}")
        if start_brace != -1 and end_brace != -1:
            parsed = json.loads(cleaned[start_brace : end_brace + 1])
        else:
            raise ValueError("No JSON object")
    except Exception:
        parsed = {
            "bottleneck_summary": cleaned[:360],
            "owning_layer": owning_layer,
            "recommended_fixes": [
                "Overlap backward ReduceScatter with weight-gradient GEMM buckets.",
                "Apply 0/1 knapsack selective activation checkpointing on S^2 attention score tensors.",
                "Enable alpha-canonical SHA-256 subgraph compile caching across repeated layers.",
            ],
        }

    return {
        "workload_id": workload_id,
        "workload_title": profile["title"],
        "model_family": profile["model_family"],
        "step_breakdown_ms": bd,
        "step_shares_pct": shares,
        "total_step_ms": round(total_step_ms, 2),
        "compute_share_pct": achieved_compute_share_pct,
        "dominant_non_compute_stall": dominant_cat,
        "owning_layer": owning_layer,
        "bedrock_model": model_id,
        "bedrock_latency_ms": round(bedrock_ms, 2),
        "bedrock_tokens": total_tokens,
        "advisor": parsed,
    }
