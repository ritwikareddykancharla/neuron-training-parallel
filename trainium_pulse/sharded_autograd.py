"""Real Multi-Rank Reverse-Mode Autograd, TP, ZeRO-1 & Ulysses Sequence Parallelism Verifier.

Executes 100% real forward and backward tensor passes in Python 3.12 on CPU/Lambda:
- Reverse-mode autograd engine: MatMul, LayerNorm, SwiGLU, ScaledDotProductAttention, CrossEntropy
- Megatron ColumnParallelLinear + RowParallelLinear (TP) with AllGather / ReduceScatter
- DeepSpeed-Ulysses Sequence Parallelism (AllToAll sequence-to-head and head-to-sequence transpose)
- ZeRO-1 AdamW optimizer state partitioning across R ranks with real tracemalloc verification
"""

from __future__ import annotations

from array import array
import math
import random
import time
import tracemalloc
from typing import Callable, Dict, List, Optional, Sequence, Tuple


def _prod(shape: Sequence[int]) -> int:
    out = 1
    for dim in shape:
        out *= dim
    return out


class Tensor:
    """Reverse-mode autograd tensor backed by flat IEEE-754 floats."""

    def __init__(
        self,
        data: Sequence[float],
        shape: Tuple[int, ...],
        requires_grad: bool = False,
        _children: Tuple["Tensor", ...] = (),
        _op: str = "",
        name: str = "",
    ) -> None:
        expected = _prod(shape)
        if len(data) != expected:
            raise ValueError(f"Shape {shape} expects {expected} elements, got {len(data)}")
        self.data: List[float] = list(data)
        self.shape: Tuple[int, ...] = tuple(shape)
        self.requires_grad: bool = requires_grad
        self.grad: Optional[List[float]] = [0.0] * expected if requires_grad else None
        self._prev: Tuple["Tensor", ...] = _children
        self._backward: Callable[[], None] = lambda: None
        self._op: str = _op
        self.name: str = name

    def zero_grad(self) -> None:
        if self.requires_grad:
            self.grad = [0.0] * len(self.data)

    def backward(self) -> None:
        if self.shape != (1,):
            raise ValueError("backward() requires a scalar (1,) loss tensor")
        topo: List[Tensor] = []
        visited: set[int] = set()

        def build(node: Tensor) -> None:
            nid = id(node)
            if nid not in visited:
                visited.add(nid)
                for child in node._prev:
                    build(child)
                topo.append(node)

        build(self)
        self.grad = [1.0]
        for node in reversed(topo):
            node._backward()


# ---------------------------------------------------------------------------
# Primitive Autograd Operators
# ---------------------------------------------------------------------------

def add_tensors(a: Tensor, b: Tensor) -> Tensor:
    if a.shape != b.shape:
        raise ValueError(f"add shape mismatch: {a.shape} vs {b.shape}")
    req = a.requires_grad or b.requires_grad
    out = Tensor(
        [x + y for x, y in zip(a.data, b.data)],
        a.shape,
        requires_grad=req,
        _children=(a, b),
        _op="Add",
    )

    def _backward() -> None:
        if out.grad is None:
            return
        g = out.grad
        if a.requires_grad and a.grad is not None:
            ag = a.grad
            for i in range(len(g)):
                ag[i] += g[i]
        if b.requires_grad and b.grad is not None:
            bg = b.grad
            for i in range(len(g)):
                bg[i] += g[i]

    out._backward = _backward
    return out


def mul_tensors(a: Tensor, b: Tensor) -> Tensor:
    if a.shape != b.shape:
        raise ValueError(f"mul shape mismatch: {a.shape} vs {b.shape}")
    req = a.requires_grad or b.requires_grad
    out = Tensor(
        [x * y for x, y in zip(a.data, b.data)],
        a.shape,
        requires_grad=req,
        _children=(a, b),
        _op="Mul",
    )

    def _backward() -> None:
        if out.grad is None:
            return
        g = out.grad
        ad = a.data
        bd = b.data
        if a.requires_grad and a.grad is not None:
            ag = a.grad
            for i in range(len(g)):
                ag[i] += g[i] * bd[i]
        if b.requires_grad and b.grad is not None:
            bg = b.grad
            for i in range(len(g)):
                bg[i] += g[i] * ad[i]

    out._backward = _backward
    return out


def silu(x: Tensor) -> Tensor:
    n = len(x.data)
    sigs = [0.0] * n
    out_data = [0.0] * n
    xd = x.data
    for i in range(n):
        v = xd[i]
        if v >= 0.0:
            s = 1.0 / (1.0 + math.exp(-v))
        else:
            ev = math.exp(v)
            s = ev / (1.0 + ev)
        sigs[i] = s
        out_data[i] = v * s

    out = Tensor(
        out_data,
        x.shape,
        requires_grad=x.requires_grad,
        _children=(x,),
        _op="SiLU",
    )

    def _backward() -> None:
        if out.grad is None or not x.requires_grad or x.grad is None:
            return
        g = out.grad
        xg = x.grad
        for i in range(n):
            s = sigs[i]
            xg[i] += g[i] * s * (1.0 + xd[i] * (1.0 - s))

    out._backward = _backward
    return out


def linear_3d(x: Tensor, w: Tensor) -> Tensor:
    """MatMul for X of shape (B, S, K) and W of shape (K, N) -> (B, S, N)."""
    if len(x.shape) != 3 or len(w.shape) != 2 or x.shape[2] != w.shape[0]:
        raise ValueError(f"linear_3d incompatible shapes: {x.shape} @ {w.shape}")
    bsz, seq, k_dim = x.shape
    _, n_dim = w.shape
    rows = bsz * seq
    xd = x.data
    wd = w.data

    out_data = [0.0] * (rows * n_dim)
    for m in range(rows):
        x_off = m * k_dim
        y_off = m * n_dim
        for k in range(k_dim):
            x_val = xd[x_off + k]
            w_off = k * n_dim
            for n in range(n_dim):
                out_data[y_off + n] += x_val * wd[w_off + n]

    req = x.requires_grad or w.requires_grad
    out = Tensor(
        out_data,
        (bsz, seq, n_dim),
        requires_grad=req,
        _children=(x, w),
        _op="MatMul",
    )

    def _backward() -> None:
        if out.grad is None:
            return
        dy = out.grad
        if x.requires_grad and x.grad is not None:
            xg = x.grad
            for m in range(rows):
                x_off = m * k_dim
                y_off = m * n_dim
                for k in range(k_dim):
                    w_off = k * n_dim
                    acc = 0.0
                    for n in range(n_dim):
                        acc += dy[y_off + n] * wd[w_off + n]
                    xg[x_off + k] += acc
        if w.requires_grad and w.grad is not None:
            wg = w.grad
            for m in range(rows):
                x_off = m * k_dim
                y_off = m * n_dim
                for k in range(k_dim):
                    x_val = xd[x_off + k]
                    w_off = k * n_dim
                    for n in range(n_dim):
                        wg[w_off + n] += x_val * dy[y_off + n]

    out._backward = _backward
    return out


def layernorm_3d(x: Tensor, gamma: Tensor, beta: Tensor, eps: float = 1e-5) -> Tensor:
    """LayerNorm over last dimension D for X of shape (B, S, D)."""
    bsz, seq, d_dim = x.shape
    if gamma.shape != (d_dim,) or beta.shape != (d_dim,):
        raise ValueError("LayerNorm parameter shape mismatch")
    rows = bsz * seq
    xd = x.data
    gd = gamma.data
    bd = beta.data

    x_hat = [0.0] * (rows * d_dim)
    inv_stds = [0.0] * rows
    out_data = [0.0] * (rows * d_dim)

    for m in range(rows):
        off = m * d_dim
        mean = sum(xd[off : off + d_dim]) / d_dim
        var = 0.0
        for j in range(d_dim):
            diff = xd[off + j] - mean
            var += diff * diff
        var /= d_dim
        inv_std = 1.0 / math.sqrt(var + eps)
        inv_stds[m] = inv_std
        for j in range(d_dim):
            xh = (xd[off + j] - mean) * inv_std
            x_hat[off + j] = xh
            out_data[off + j] = xh * gd[j] + bd[j]

    req = x.requires_grad or gamma.requires_grad or beta.requires_grad
    out = Tensor(
        out_data,
        x.shape,
        requires_grad=req,
        _children=(x, gamma, beta),
        _op="LayerNorm",
    )

    def _backward() -> None:
        if out.grad is None:
            return
        dy = out.grad
        if gamma.requires_grad and gamma.grad is not None:
            gg = gamma.grad
            for m in range(rows):
                off = m * d_dim
                for j in range(d_dim):
                    gg[j] += dy[off + j] * x_hat[off + j]
        if beta.requires_grad and beta.grad is not None:
            bg = beta.grad
            for m in range(rows):
                off = m * d_dim
                for j in range(d_dim):
                    bg[j] += dy[off + j]
        if x.requires_grad and x.grad is not None:
            xg = x.grad
            for m in range(rows):
                off = m * d_dim
                inv_std = inv_stds[m]
                dx_hat = [dy[off + j] * gd[j] for j in range(d_dim)]
                sum_dxh = sum(dx_hat)
                sum_dxh_xh = sum(dx_hat[j] * x_hat[off + j] for j in range(d_dim))
                scale = inv_std / d_dim
                for j in range(d_dim):
                    xg[off + j] += scale * (
                        d_dim * dx_hat[j] - sum_dxh - x_hat[off + j] * sum_dxh_xh
                    )

    out._backward = _backward
    return out


def swiglu_mlp(x: Tensor, w_gate: Tensor, w_up: Tensor, w_down: Tensor) -> Tensor:
    """SwiGLU feed-forward block: (SiLU(X @ W_gate) * (X @ W_up)) @ W_down."""
    gate = silu(linear_3d(x, w_gate))
    up = linear_3d(x, w_up)
    hidden = mul_tensors(gate, up)
    return linear_3d(hidden, w_down)


def reshape_tensor(x: Tensor, new_shape: Tuple[int, ...]) -> Tensor:
    if _prod(x.shape) != _prod(new_shape):
        raise ValueError(f"Cannot reshape {x.shape} -> {new_shape}")
    out = Tensor(
        x.data,
        new_shape,
        requires_grad=x.requires_grad,
        _children=(x,),
        _op="Reshape",
    )

    def _backward() -> None:
        if out.grad is None or not x.requires_grad or x.grad is None:
            return
        xg = x.grad
        og = out.grad
        for i in range(len(og)):
            xg[i] += og[i]

    out._backward = _backward
    return out


def scaled_dot_product_attention(
    q: Tensor, k: Tensor, v: Tensor, causal: bool = True
) -> Tensor:
    """Exact SDPA on shapes (B, S, H, Dh) supporting causal LLM and bidirectional DiT."""
    bsz, seq, n_heads, d_head = q.shape
    if k.shape != q.shape or v.shape != q.shape:
        raise ValueError("SDPA requires matching (B, S, H, Dh) shapes")
    scale = 1.0 / math.sqrt(d_head)
    qd, kd, vd = q.data, k.data, v.data

    def idx4(b: int, s: int, h: int, d: int) -> int:
        return (((b * seq + s) * n_heads + h) * d_head) + d

    probs_cache: List[List[List[List[float]]]] = [
        [[[0.0] * seq for _ in range(seq)] for _ in range(n_heads)]
        for _ in range(bsz)
    ]
    out_data = [0.0] * len(qd)

    for b in range(bsz):
        for h in range(n_heads):
            for i in range(seq):
                q_off = ((b * seq + i) * n_heads + h) * d_head
                scores = [0.0] * seq
                max_s = -1e30
                for j in range(seq):
                    if causal and j > i:
                        scores[j] = -1e30
                    else:
                        k_off = ((b * seq + j) * n_heads + h) * d_head
                        dot = 0.0
                        for d in range(d_head):
                            dot += qd[q_off + d] * kd[k_off + d]
                        s_val = dot * scale
                        scores[j] = s_val
                        if s_val > max_s:
                            max_s = s_val
                exps = [0.0] * seq
                denom = 0.0
                limit_j = i + 1 if causal else seq
                for j in range(limit_j):
                    ev = math.exp(scores[j] - max_s)
                    exps[j] = ev
                    denom += ev
                row_p = probs_cache[b][h][i]
                for j in range(limit_j):
                    row_p[j] = exps[j] / denom

                o_off = q_off
                for j in range(limit_j):
                    p_ij = row_p[j]
                    v_off = ((b * seq + j) * n_heads + h) * d_head
                    for d in range(d_head):
                        out_data[o_off + d] += p_ij * vd[v_off + d]

    req = q.requires_grad or k.requires_grad or v.requires_grad
    out = Tensor(
        out_data,
        q.shape,
        requires_grad=req,
        _children=(q, k, v),
        _op="ScaledDotProductAttention",
    )

    def _backward() -> None:
        if out.grad is None:
            return
        do = out.grad
        qg = q.grad if q.requires_grad else None
        kg = k.grad if k.requires_grad else None
        vg = v.grad if v.requires_grad else None

        for b in range(bsz):
            for h in range(n_heads):
                for i in range(seq):
                    limit_j = i + 1 if causal else seq
                    row_p = probs_cache[b][h][i]
                    o_off = ((b * seq + i) * n_heads + h) * d_head
                    dp = [0.0] * seq
                    for j in range(limit_j):
                        v_off = ((b * seq + j) * n_heads + h) * d_head
                        p_ij = row_p[j]
                        dot_dov = 0.0
                        for d in range(d_head):
                            do_val = do[o_off + d]
                            if vg is not None:
                                vg[v_off + d] += p_ij * do_val
                            dot_dov += do_val * vd[v_off + d]
                        dp[j] = dot_dov

                    sum_p_dp = 0.0
                    for j in range(limit_j):
                        sum_p_dp += row_p[j] * dp[j]

                    q_off = o_off
                    for j in range(limit_j):
                        ds_ij = row_p[j] * (dp[j] - sum_p_dp) * scale
                        k_off = ((b * seq + j) * n_heads + h) * d_head
                        if qg is not None:
                            for d in range(d_head):
                                qg[q_off + d] += ds_ij * kd[k_off + d]
                        if kg is not None:
                            for d in range(d_head):
                                kg[k_off + d] += ds_ij * qd[q_off + d]

    _ = idx4
    out._backward = _backward
    return out


def cross_entropy_loss(
    logits: Tensor, targets: Sequence[int], total_token_divisor: Optional[int] = None
) -> Tensor:
    """CrossEntropy over logits of shape (B, S, V) with exact log-softmax backward."""
    bsz, seq, vocab = logits.shape
    rows = bsz * seq
    if len(targets) != rows:
        raise ValueError(f"Expected {rows} target tokens, got {len(targets)}")
    denom_tokens = float(total_token_divisor if total_token_divisor is not None else rows)
    ld = logits.data
    probs = [0.0] * (rows * vocab)
    total_nll = 0.0

    for m in range(rows):
        off = m * vocab
        max_z = max(ld[off : off + vocab])
        sum_exp = 0.0
        for v in range(vocab):
            ev = math.exp(ld[off + v] - max_z)
            probs[off + v] = ev
            sum_exp += ev
        lse = max_z + math.log(sum_exp)
        tgt = targets[m]
        total_nll += lse - ld[off + tgt]
        for v in range(vocab):
            probs[off + v] /= sum_exp

    out = Tensor(
        [total_nll / denom_tokens],
        (1,),
        requires_grad=logits.requires_grad,
        _children=(logits,),
        _op="CrossEntropy",
    )

    def _backward() -> None:
        if out.grad is None or not logits.requires_grad or logits.grad is None:
            return
        scale = out.grad[0] / denom_tokens
        lg = logits.grad
        for m in range(rows):
            off = m * vocab
            tgt = targets[m]
            for v in range(vocab):
                ind = 1.0 if v == tgt else 0.0
                lg[off + v] += scale * (probs[off + v] - ind)

    out._backward = _backward
    return out


# ---------------------------------------------------------------------------
# Multi-Rank Collective Primitives with Exact Conjugate Autograd
# ---------------------------------------------------------------------------

class CollectiveStats:
    def __init__(self) -> None:
        self.all_reduce_calls = 0
        self.all_gather_calls = 0
        self.reduce_scatter_calls = 0
        self.all_to_all_calls = 0
        self.bytes_transferred = 0


def dist_all_gather_seq(
    shards: List[Tensor], stats: CollectiveStats
) -> List[Tensor]:
    """Forward: AllGather (B, S/R, D) -> replicated (B, S, D) on each rank.
    Backward: ReduceScatter(SUM) of dOut_r slices back to each rank's (B, S/R, D).
    """
    world = len(shards)
    bsz, s_local, d_dim = shards[0].shape
    s_full = s_local * world
    stats.all_gather_calls += 1
    stats.bytes_transferred += world * (world - 1) * len(shards[0].data) * 8

    gathered_data = [0.0] * (bsz * s_full * d_dim)
    for b in range(bsz):
        for r in range(world):
            sd = shards[r].data
            src_off = b * s_local * d_dim
            dst_off = (b * s_full + r * s_local) * d_dim
            gathered_data[dst_off : dst_off + s_local * d_dim] = sd[
                src_off : src_off + s_local * d_dim
            ]

    req = any(s.requires_grad for s in shards)
    outputs: List[Tensor] = [
        Tensor(
            gathered_data,
            (bsz, s_full, d_dim),
            requires_grad=req,
            _children=tuple(shards),
            _op="AllGatherSeq",
        )
        for _ in range(world)
    ]

    for r_out in range(world):
        out_r = outputs[r_out]

        def _make_bw(node: Tensor) -> Callable[[], None]:
            def _bw() -> None:
                if node.grad is None:
                    return
                stats.reduce_scatter_calls += 1
                ng = node.grad
                for b in range(bsz):
                    for r_dst in range(world):
                        dst_t = shards[r_dst]
                        if not dst_t.requires_grad or dst_t.grad is None:
                            continue
                        dg = dst_t.grad
                        src_off = (b * s_full + r_dst * s_local) * d_dim
                        dst_off = b * s_local * d_dim
                        for j in range(s_local * d_dim):
                            dg[dst_off + j] += ng[src_off + j]

            return _bw

        out_r._backward = _make_bw(out_r)

    return outputs


def dist_reduce_scatter_seq(
    partials: List[Tensor], stats: CollectiveStats
) -> List[Tensor]:
    """Forward: ReduceScatter(SUM) across R ranks from (B, S, D) -> (B, S/R, D) per rank.
    Backward: AllGather of (B, S/R, D) gradients back to (B, S, D) on every rank.
    """
    world = len(partials)
    bsz, s_full, d_dim = partials[0].shape
    s_local = s_full // world
    stats.reduce_scatter_calls += 1
    stats.bytes_transferred += (world - 1) * len(partials[0].data) * 8

    req = any(p.requires_grad for p in partials)
    outputs: List[Tensor] = []
    for r in range(world):
        out_r_data = [0.0] * (bsz * s_local * d_dim)
        for b in range(bsz):
            dst_off = b * s_local * d_dim
            src_off = (b * s_full + r * s_local) * d_dim
            for src_rank in range(world):
                pd = partials[src_rank].data
                for j in range(s_local * d_dim):
                    out_r_data[dst_off + j] += pd[src_off + j]
        out_r = Tensor(
            out_r_data,
            (bsz, s_local, d_dim),
            requires_grad=req,
            _children=tuple(partials),
            _op="ReduceScatterSeq",
        )

        def _make_bw(rank_idx: int, node: Tensor) -> Callable[[], None]:
            def _bw() -> None:
                if node.grad is None:
                    return
                stats.all_gather_calls += 1
                ng = node.grad
                for b in range(bsz):
                    src_off = b * s_local * d_dim
                    dst_off = (b * s_full + rank_idx * s_local) * d_dim
                    for p in partials:
                        if not p.requires_grad or p.grad is None:
                            continue
                        pg = p.grad
                        for j in range(s_local * d_dim):
                            pg[dst_off + j] += ng[src_off + j]

            return _bw

        out_r._backward = _make_bw(r, out_r)
        outputs.append(out_r)

    return outputs


def dist_ulysses_seq_to_head(
    shards: List[Tensor], stats: CollectiveStats
) -> List[Tensor]:
    """Ulysses AllToAll: (B, S/R, H, Dh) -> (B, S, H/R, Dh) on each rank r."""
    world = len(shards)
    bsz, s_local, n_heads, d_head = shards[0].shape
    if n_heads % world != 0:
        raise ValueError(f"Heads {n_heads} must be divisible by world size {world}")
    h_local = n_heads // world
    s_full = s_local * world
    stats.all_to_all_calls += 1
    stats.bytes_transferred += (world - 1) * len(shards[0].data) * 8

    req = any(s.requires_grad for s in shards)
    outputs: List[Tensor] = []

    for r_dst in range(world):
        out_data = [0.0] * (bsz * s_full * h_local * d_head)
        for b in range(bsz):
            for r_src in range(world):
                sd = shards[r_src].data
                for sl in range(s_local):
                    s_global = r_src * s_local + sl
                    for hl in range(h_local):
                        h_global = r_dst * h_local + hl
                        src_off = (((b * s_local + sl) * n_heads + h_global) * d_head)
                        dst_off = (((b * s_full + s_global) * h_local + hl) * d_head)
                        out_data[dst_off : dst_off + d_head] = sd[
                            src_off : src_off + d_head
                        ]
        out_r = Tensor(
            out_data,
            (bsz, s_full, h_local, d_head),
            requires_grad=req,
            _children=tuple(shards),
            _op="UlyssesAllToAllSeqToHead",
        )

        def _make_bw(dst_rank: int, node: Tensor) -> Callable[[], None]:
            def _bw() -> None:
                if node.grad is None:
                    return
                stats.all_to_all_calls += 1
                ng = node.grad
                for b in range(bsz):
                    for r_src in range(world):
                        st = shards[r_src]
                        if not st.requires_grad or st.grad is None:
                            continue
                        sg = st.grad
                        for sl in range(s_local):
                            s_global = r_src * s_local + sl
                            for hl in range(h_local):
                                h_global = dst_rank * h_local + hl
                                src_off = (
                                    ((b * s_local + sl) * n_heads + h_global) * d_head
                                )
                                dst_off = (
                                    ((b * s_full + s_global) * h_local + hl) * d_head
                                )
                                for d in range(d_head):
                                    sg[src_off + d] += ng[dst_off + d]

            return _bw

        out_r._backward = _make_bw(r_dst, out_r)
        outputs.append(out_r)

    return outputs


def dist_ulysses_head_to_seq(
    shards: List[Tensor], stats: CollectiveStats
) -> List[Tensor]:
    """Inverse Ulysses AllToAll: (B, S, H/R, Dh) -> (B, S/R, H, Dh) on each rank r."""
    world = len(shards)
    bsz, s_full, h_local, d_head = shards[0].shape
    s_local = s_full // world
    n_heads = h_local * world
    stats.all_to_all_calls += 1
    stats.bytes_transferred += (world - 1) * len(shards[0].data) * 8

    req = any(s.requires_grad for s in shards)
    outputs: List[Tensor] = []

    for r_dst in range(world):
        out_data = [0.0] * (bsz * s_local * n_heads * d_head)
        for b in range(bsz):
            for sl in range(s_local):
                s_global = r_dst * s_local + sl
                for r_src in range(world):
                    sd = shards[r_src].data
                    for hl in range(h_local):
                        h_global = r_src * h_local + hl
                        src_off = (((b * s_full + s_global) * h_local + hl) * d_head)
                        dst_off = (((b * s_local + sl) * n_heads + h_global) * d_head)
                        out_data[dst_off : dst_off + d_head] = sd[
                            src_off : src_off + d_head
                        ]
        out_r = Tensor(
            out_data,
            (bsz, s_local, n_heads, d_head),
            requires_grad=req,
            _children=tuple(shards),
            _op="UlyssesAllToAllHeadToSeq",
        )

        def _make_bw(dst_rank: int, node: Tensor) -> Callable[[], None]:
            def _bw() -> None:
                if node.grad is None:
                    return
                stats.all_to_all_calls += 1
                ng = node.grad
                for b in range(bsz):
                    for sl in range(s_local):
                        s_global = dst_rank * s_local + sl
                        for r_src in range(world):
                            st = shards[r_src]
                            if not st.requires_grad or st.grad is None:
                                continue
                            sg = st.grad
                            for hl in range(h_local):
                                h_global = r_src * h_local + hl
                                src_off = (
                                    ((b * s_full + s_global) * h_local + hl) * d_head
                                )
                                dst_off = (
                                    ((b * s_local + sl) * n_heads + h_global) * d_head
                                )
                                for d in range(d_head):
                                    sg[src_off + d] += ng[dst_off + d]

            return _bw

        out_r._backward = _make_bw(r_dst, out_r)
        outputs.append(out_r)

    return outputs


def dist_sum_losses(rank_losses: List[Tensor], stats: CollectiveStats) -> Tensor:
    """AllReduce sum of per-rank partial scalar losses."""
    stats.all_reduce_calls += 1
    stats.bytes_transferred += len(rank_losses) * 8
    total = sum(t.data[0] for t in rank_losses)
    out = Tensor(
        [total],
        (1,),
        requires_grad=any(t.requires_grad for t in rank_losses),
        _children=tuple(rank_losses),
        _op="AllReduceLossSum",
    )

    def _bw() -> None:
        if out.grad is None:
            return
        g0 = out.grad[0]
        for t in rank_losses:
            if t.requires_grad and t.grad is not None:
                t.grad[0] += g0

    out._backward = _bw
    return out


# ---------------------------------------------------------------------------
# Weight Sharding Helpers & ZeRO-1 Optimizer Verification
# ---------------------------------------------------------------------------

def _rand_list(n: int, rng: random.Random, scale: float = 0.12) -> List[float]:
    return [rng.uniform(-scale, scale) for _ in range(n)]


def _slice_cols_2d(data: List[float], rows: int, cols: int, c0: int, c1: int) -> List[float]:
    w = c1 - c0
    out = [0.0] * (rows * w)
    for r in range(rows):
        out[r * w : (r + 1) * w] = data[r * cols + c0 : r * cols + c1]
    return out


def _concat_cols_2d(shards: List[List[float]], rows: int, cols_per_rank: int) -> List[float]:
    world = len(shards)
    cols = cols_per_rank * world
    out = [0.0] * (rows * cols)
    for r_idx, shard in enumerate(shards):
        c0 = r_idx * cols_per_rank
        for row in range(rows):
            out[row * cols + c0 : row * cols + c0 + cols_per_rank] = shard[
                row * cols_per_rank : (row + 1) * cols_per_rank
            ]
    return out


def _slice_rows_2d(data: List[float], cols: int, r0: int, r1: int) -> List[float]:
    return data[r0 * cols : r1 * cols]


def _measure_unsharded_adamw_bytes(param_vectors: Dict[str, List[float]]) -> Tuple[int, Dict[str, List[float]]]:
    """Allocates full-model FP64 AdamW state (master_w, m, v) on a single rank under tracemalloc."""
    tracemalloc.start()
    snap_before = tracemalloc.take_snapshot()
    state_buffers: List[array] = []
    for vec in param_vectors.values():
        state_buffers.append(array("d", vec))            # master weight
        state_buffers.append(array("d", [0.0] * len(vec)))  # first moment m
        state_buffers.append(array("d", [0.0] * len(vec)))  # second moment v
    snap_after = tracemalloc.take_snapshot()
    tracemalloc.stop()
    diff_bytes = sum(
        stat.size_diff for stat in snap_after.compare_to(snap_before, "lineno") if stat.size_diff > 0
    )
    _ = len(state_buffers)
    return diff_bytes, {}


def _measure_zero1_rank0_adamw_bytes(
    replicated_param_vectors: Dict[str, List[float]],
    tp_rank0_param_vectors: Dict[str, List[float]],
    world_size: int,
) -> int:
    """Allocates Rank-0's ZeRO-1 partitioned AdamW state (1/R of replicated + TP shard) under tracemalloc."""
    tracemalloc.start()
    snap_before = tracemalloc.take_snapshot()
    rank0_buffers: List[array] = []
    for vec in replicated_param_vectors.values():
        shard_len = math.ceil(len(vec) / world_size)
        rank0_buffers.append(array("d", vec[:shard_len]))
        rank0_buffers.append(array("d", [0.0] * shard_len))
        rank0_buffers.append(array("d", [0.0] * shard_len))
    for vec in tp_rank0_param_vectors.values():
        rank0_buffers.append(array("d", vec))
        rank0_buffers.append(array("d", [0.0] * len(vec)))
        rank0_buffers.append(array("d", [0.0] * len(vec)))
    snap_after = tracemalloc.take_snapshot()
    tracemalloc.stop()
    diff_bytes = sum(
        stat.size_diff for stat in snap_after.compare_to(snap_before, "lineno") if stat.size_diff > 0
    )
    _ = len(rank0_buffers)
    return diff_bytes


def _adamw_update_in_place(
    w: List[float],
    g: List[float],
    m: List[float],
    v: List[float],
    lr: float = 1e-3,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    weight_decay: float = 0.01,
    step: int = 1,
) -> List[float]:
    bc1 = 1.0 - (beta1 ** step)
    bc2 = 1.0 - (beta2 ** step)
    updated = [0.0] * len(w)
    for i in range(len(w)):
        gi = g[i]
        m[i] = beta1 * m[i] + (1.0 - beta1) * gi
        v[i] = beta2 * v[i] + (1.0 - beta2) * (gi * gi)
        m_hat = m[i] / bc1
        v_hat = v[i] / bc2
        updated[i] = w[i] * (1.0 - lr * weight_decay) - lr * m_hat / (math.sqrt(v_hat) + eps)
    return updated


def verify_sharded_autograd(
    world_size: int = 4,
    batch_size: int = 2,
    seq_len: int = 16,
    d_model: int = 32,
    n_heads: int = 4,
    d_ff: int = 64,
    vocab_size: int = 32,
    architecture: str = "llama_causal",
    seed: int = 42,
) -> Dict[str, object]:
    """Runs single-rank reference vs. R-rank sharded (Ulysses SP + Megatron TP + ZeRO-1)
    forward, backward, and AdamW step, verifying exact L_inf gradient and parameter parity.
    """
    t0 = time.perf_counter()
    if world_size not in (2, 4, 8):
        raise ValueError("world_size must be 2, 4, or 8")
    if seq_len % world_size != 0:
        raise ValueError(f"seq_len ({seq_len}) must be divisible by world_size ({world_size})")
    if n_heads % world_size != 0:
        raise ValueError(f"n_heads ({n_heads}) must be divisible by world_size ({world_size})")
    if d_model % n_heads != 0:
        raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
    if d_ff % world_size != 0:
        raise ValueError(f"d_ff ({d_ff}) must be divisible by world_size ({world_size})")

    causal = architecture != "flux_mmdit_bidirectional"
    d_head = d_model // n_heads
    rng = random.Random(seed)

    # Initial input activations & target token IDs
    x_init = _rand_list(batch_size * seq_len * d_model, rng, scale=0.5)
    targets = [rng.randrange(vocab_size) for _ in range(batch_size * seq_len)]

    # Initial full model weights
    init_weights: Dict[str, Tuple[List[float], Tuple[int, ...]]] = {
        "ln1_gamma": ([1.0 + rng.uniform(-0.05, 0.05) for _ in range(d_model)], (d_model,)),
        "ln1_beta": (_rand_list(d_model, rng, 0.02), (d_model,)),
        "w_q": (_rand_list(d_model * d_model, rng, 0.14), (d_model, d_model)),
        "w_k": (_rand_list(d_model * d_model, rng, 0.14), (d_model, d_model)),
        "w_v": (_rand_list(d_model * d_model, rng, 0.14), (d_model, d_model)),
        "w_o": (_rand_list(d_model * d_model, rng, 0.14), (d_model, d_model)),
        "ln2_gamma": ([1.0 + rng.uniform(-0.05, 0.05) for _ in range(d_model)], (d_model,)),
        "ln2_beta": (_rand_list(d_model, rng, 0.02), (d_model,)),
        "w_gate": (_rand_list(d_model * d_ff, rng, 0.14), (d_model, d_ff)),
        "w_up": (_rand_list(d_model * d_ff, rng, 0.14), (d_model, d_ff)),
        "w_down": (_rand_list(d_ff * d_model, rng, 0.14), (d_ff, d_model)),
        "lm_head": (_rand_list(d_model * vocab_size, rng, 0.14), (d_model, vocab_size)),
    }

    # -----------------------------------------------------------------------
    # 1. Unsharded Single-Rank FP64 Reference Forward + Backward + AdamW
    # -----------------------------------------------------------------------
    t_single_start = time.perf_counter()
    x_ref = Tensor(x_init, (batch_size, seq_len, d_model), requires_grad=True, name="x_ref")
    ref_params: Dict[str, Tensor] = {
        name: Tensor(vec, shp, requires_grad=True, name=name)
        for name, (vec, shp) in init_weights.items()
    }

    h_norm1 = layernorm_3d(x_ref, ref_params["ln1_gamma"], ref_params["ln1_beta"])
    q_ref = reshape_tensor(
        linear_3d(h_norm1, ref_params["w_q"]), (batch_size, seq_len, n_heads, d_head)
    )
    k_ref = reshape_tensor(
        linear_3d(h_norm1, ref_params["w_k"]), (batch_size, seq_len, n_heads, d_head)
    )
    v_ref = reshape_tensor(
        linear_3d(h_norm1, ref_params["w_v"]), (batch_size, seq_len, n_heads, d_head)
    )
    attn_ref = reshape_tensor(
        scaled_dot_product_attention(q_ref, k_ref, v_ref, causal=causal),
        (batch_size, seq_len, d_model),
    )
    h1_ref = add_tensors(x_ref, linear_3d(attn_ref, ref_params["w_o"]))
    h_norm2 = layernorm_3d(h1_ref, ref_params["ln2_gamma"], ref_params["ln2_beta"])
    mlp_ref = swiglu_mlp(
        h_norm2, ref_params["w_gate"], ref_params["w_up"], ref_params["w_down"]
    )
    h2_ref = add_tensors(h1_ref, mlp_ref)
    logits_ref = linear_3d(h2_ref, ref_params["lm_head"])
    loss_ref = cross_entropy_loss(logits_ref, targets)
    loss_ref.backward()
    single_rank_ms = (time.perf_counter() - t_single_start) * 1000.0

    # Reference AdamW updated parameters
    ref_updated_weights: Dict[str, List[float]] = {}
    for name, param in ref_params.items():
        assert param.grad is not None
        ref_updated_weights[name] = _adamw_update_in_place(
            param.data,
            param.grad,
            [0.0] * len(param.data),
            [0.0] * len(param.data),
        )

    # -----------------------------------------------------------------------
    # 2. Multi-Rank Sharded Execution (R Ranks: Ulysses SP + TP + ZeRO-1)
    # -----------------------------------------------------------------------
    t_dist_start = time.perf_counter()
    cstats = CollectiveStats()
    s_local = seq_len // world_size
    dff_local = d_ff // world_size

    # Sequence-shard input X and target tokens across R ranks
    x_shards: List[Tensor] = []
    target_shards: List[List[int]] = []
    for r in range(world_size):
        xs_data = [0.0] * (batch_size * s_local * d_model)
        ts_data: List[int] = []
        for b in range(batch_size):
            src_off = (b * seq_len + r * s_local) * d_model
            dst_off = b * s_local * d_model
            xs_data[dst_off : dst_off + s_local * d_model] = x_init[
                src_off : src_off + s_local * d_model
            ]
            t_off = b * seq_len + r * s_local
            ts_data.extend(targets[t_off : t_off + s_local])
        x_shards.append(
            Tensor(xs_data, (batch_size, s_local, d_model), requires_grad=True, name=f"x_r{r}")
        )
        target_shards.append(ts_data)

    # Sequence-parallel replicated params (optimizer states partitioned by ZeRO-1)
    sp_param_names = [
        "ln1_gamma",
        "ln1_beta",
        "w_q",
        "w_k",
        "w_v",
        "w_o",
        "ln2_gamma",
        "ln2_beta",
        "lm_head",
    ]
    rank_sp_params: List[Dict[str, Tensor]] = []
    rank_tp_params: List[Dict[str, Tensor]] = []

    for r in range(world_size):
        sp_dict: Dict[str, Tensor] = {}
        for name in sp_param_names:
            vec, shp = init_weights[name]
            sp_dict[name] = Tensor(vec, shp, requires_grad=True, name=f"{name}_r{r}")
        rank_sp_params.append(sp_dict)

        # ColumnParallel shards for w_gate and w_up: (d_model, d_ff / R)
        c0, c1 = r * dff_local, (r + 1) * dff_local
        wg_shard = _slice_cols_2d(init_weights["w_gate"][0], d_model, d_ff, c0, c1)
        wu_shard = _slice_cols_2d(init_weights["w_up"][0], d_model, d_ff, c0, c1)
        # RowParallel shard for w_down: (d_ff / R, d_model)
        wd_shard = _slice_rows_2d(init_weights["w_down"][0], d_model, c0, c1)

        rank_tp_params.append(
            {
                "w_gate": Tensor(wg_shard, (d_model, dff_local), requires_grad=True),
                "w_up": Tensor(wu_shard, (d_model, dff_local), requires_grad=True),
                "w_down": Tensor(wd_shard, (dff_local, d_model), requires_grad=True),
            }
        )

    # Stage A: Ulysses Sequence-Parallel Attention across R ranks
    q_local_shards: List[Tensor] = []
    k_local_shards: List[Tensor] = []
    v_local_shards: List[Tensor] = []
    for r in range(world_size):
        sp = rank_sp_params[r]
        ln1_r = layernorm_3d(x_shards[r], sp["ln1_gamma"], sp["ln1_beta"])
        q_local_shards.append(
            reshape_tensor(linear_3d(ln1_r, sp["w_q"]), (batch_size, s_local, n_heads, d_head))
        )
        k_local_shards.append(
            reshape_tensor(linear_3d(ln1_r, sp["w_k"]), (batch_size, s_local, n_heads, d_head))
        )
        v_local_shards.append(
            reshape_tensor(linear_3d(ln1_r, sp["w_v"]), (batch_size, s_local, n_heads, d_head))
        )

    # AllToAll transpose: (B, S/R, H, Dh) -> (B, S, H/R, Dh)
    q_head_shards = dist_ulysses_seq_to_head(q_local_shards, cstats)
    k_head_shards = dist_ulysses_seq_to_head(k_local_shards, cstats)
    v_head_shards = dist_ulysses_seq_to_head(v_local_shards, cstats)

    attn_head_outs = [
        scaled_dot_product_attention(
            q_head_shards[r], k_head_shards[r], v_head_shards[r], causal=causal
        )
        for r in range(world_size)
    ]

    # Inverse AllToAll transpose: (B, S, H/R, Dh) -> (B, S/R, H, Dh) -> (B, S/R, D)
    attn_seq_shards = dist_ulysses_head_to_seq(attn_head_outs, cstats)

    h1_shards: List[Tensor] = []
    ln2_local_shards: List[Tensor] = []
    for r in range(world_size):
        sp = rank_sp_params[r]
        attn_flat_r = reshape_tensor(attn_seq_shards[r], (batch_size, s_local, d_model))
        h1_r = add_tensors(x_shards[r], linear_3d(attn_flat_r, sp["w_o"]))
        h1_shards.append(h1_r)
        ln2_local_shards.append(layernorm_3d(h1_r, sp["ln2_gamma"], sp["ln2_beta"]))

    # Stage B: Megatron SP AllGather -> ColumnParallel + RowParallel SwiGLU -> ReduceScatter
    ln2_gathered = dist_all_gather_seq(ln2_local_shards, cstats)
    mlp_row_partials: List[Tensor] = []
    for r in range(world_size):
        tp = rank_tp_params[r]
        mlp_row_partials.append(
            swiglu_mlp(ln2_gathered[r], tp["w_gate"], tp["w_up"], tp["w_down"])
        )
    mlp_scattered = dist_reduce_scatter_seq(mlp_row_partials, cstats)

    # Residual + Sequence-sharded CrossEntropy Loss -> AllReduce Loss
    partial_losses: List[Tensor] = []
    for r in range(world_size):
        h2_r = add_tensors(h1_shards[r], mlp_scattered[r])
        logits_r = linear_3d(h2_r, rank_sp_params[r]["lm_head"])
        partial_losses.append(
            cross_entropy_loss(
                logits_r,
                target_shards[r],
                total_token_divisor=batch_size * seq_len,
            )
        )

    dist_loss = dist_sum_losses(partial_losses, cstats)
    dist_loss.backward()

    # Stage C: ZeRO-1 ReduceScatter on SP gradients + Partitioned AdamW Update + AllGather
    sharded_full_grads: Dict[str, List[float]] = {}
    sharded_updated_weights: Dict[str, List[float]] = {}

    for name in sp_param_names:
        n_el = len(init_weights[name][0])
        shard_len = math.ceil(n_el / world_size)
        cstats.reduce_scatter_calls += 1
        cstats.bytes_transferred += (world_size - 1) * n_el * 8

        assembled_grad: List[float] = []
        assembled_updated: List[float] = []
        for r_owner in range(world_size):
            i0 = r_owner * shard_len
            i1 = min(n_el, (r_owner + 1) * shard_len)
            if i0 >= i1:
                continue
            # Owner rank reduces its 1/R slice across all R ranks
            reduced_slice = [0.0] * (i1 - i0)
            for r_src in range(world_size):
                rg = rank_sp_params[r_src][name].grad
                assert rg is not None
                for k in range(i1 - i0):
                    reduced_slice[k] += rg[i0 + k]
            assembled_grad.extend(reduced_slice)
            # Owner rank runs AdamW strictly on its 1/R optimizer state slice
            w_slice = init_weights[name][0][i0:i1]
            upd_slice = _adamw_update_in_place(
                w_slice,
                reduced_slice,
                [0.0] * len(w_slice),
                [0.0] * len(w_slice),
            )
            assembled_updated.extend(upd_slice)

        cstats.all_gather_calls += 1
        cstats.bytes_transferred += (world_size - 1) * n_el * 8
        sharded_full_grads[name] = assembled_grad
        sharded_updated_weights[name] = assembled_updated

    # Assemble TP weight gradients and TP rank-local AdamW updates
    for col_name in ("w_gate", "w_up"):
        grad_shards: List[List[float]] = []
        upd_shards: List[List[float]] = []
        for r in range(world_size):
            tp_t = rank_tp_params[r][col_name]
            assert tp_t.grad is not None
            grad_shards.append(tp_t.grad)
            upd_shards.append(
                _adamw_update_in_place(
                    tp_t.data,
                    tp_t.grad,
                    [0.0] * len(tp_t.data),
                    [0.0] * len(tp_t.data),
                )
            )
        sharded_full_grads[col_name] = _concat_cols_2d(grad_shards, d_model, dff_local)
        sharded_updated_weights[col_name] = _concat_cols_2d(upd_shards, d_model, dff_local)

    row_grad_concat: List[float] = []
    row_upd_concat: List[float] = []
    for r in range(world_size):
        tp_d = rank_tp_params[r]["w_down"]
        assert tp_d.grad is not None
        row_grad_concat.extend(tp_d.grad)
        row_upd_concat.extend(
            _adamw_update_in_place(
                tp_d.data,
                tp_d.grad,
                [0.0] * len(tp_d.data),
                [0.0] * len(tp_d.data),
            )
        )
    sharded_full_grads["w_down"] = row_grad_concat
    sharded_updated_weights["w_down"] = row_upd_concat

    # Verify input activation gradient parity as well
    assembled_dx = [0.0] * len(x_init)
    for b in range(batch_size):
        for r in range(world_size):
            xg = x_shards[r].grad
            assert xg is not None
            src_off = b * s_local * d_model
            dst_off = (b * seq_len + r * s_local) * d_model
            assembled_dx[dst_off : dst_off + s_local * d_model] = xg[
                src_off : src_off + s_local * d_model
            ]

    sharded_exec_ms = (time.perf_counter() - t_dist_start) * 1000.0

    # -----------------------------------------------------------------------
    # 3. Compute Exact L_inf Diffs & Real tracemalloc Optimizer Memory
    # -----------------------------------------------------------------------
    per_tensor_parity: List[Dict[str, object]] = []
    global_max_grad_diff = 0.0
    global_max_param_diff = 0.0
    total_params = 0

    for name, (_, shp) in init_weights.items():
        ref_g = ref_params[name].grad
        assert ref_g is not None
        dist_g = sharded_full_grads[name]
        max_g_diff = max(abs(a - b) for a, b in zip(ref_g, dist_g))
        max_w_diff = max(
            abs(a - b) for a, b in zip(ref_updated_weights[name], sharded_updated_weights[name])
        )
        global_max_grad_diff = max(global_max_grad_diff, max_g_diff)
        global_max_param_diff = max(global_max_param_diff, max_w_diff)
        total_params += len(ref_g)
        sharding_mode = (
            "ColumnParallel TP"
            if name in ("w_gate", "w_up")
            else ("RowParallel TP" if name == "w_down" else "Ulysses SP + ZeRO-1")
        )
        per_tensor_parity.append(
            {
                "tensor": name,
                "shape": list(shp),
                "sharding": sharding_mode,
                "max_abs_grad_diff": round(max_g_diff, 18),
                "max_abs_param_diff": round(max_w_diff, 18),
                "grad_l2_norm": round(math.sqrt(sum(v * v for v in ref_g)), 6),
            }
        )

    assert x_ref.grad is not None
    input_dx_max_diff = max(abs(a - b) for a, b in zip(x_ref.grad, assembled_dx))
    loss_abs_diff = abs(loss_ref.data[0] - dist_loss.data[0])

    unsharded_opt_bytes, _ = _measure_unsharded_adamw_bytes(
        {k: v[0] for k, v in init_weights.items()}
    )
    rank0_opt_bytes = _measure_zero1_rank0_adamw_bytes(
        {k: init_weights[k][0] for k in sp_param_names},
        {k: rank_tp_params[0][k].data for k in ("w_gate", "w_up", "w_down")},
        world_size,
    )
    opt_mem_reduction_pct = (
        (1.0 - (rank0_opt_bytes / max(1, unsharded_opt_bytes))) * 100.0
    )

    total_ms = (time.perf_counter() - t0) * 1000.0
    return {
        "architecture": architecture,
        "causal_mask": causal,
        "world_size": world_size,
        "shapes": {
            "batch_size": batch_size,
            "seq_len": seq_len,
            "seq_per_rank": s_local,
            "d_model": d_model,
            "n_heads": n_heads,
            "heads_per_rank_ulysses": n_heads // world_size,
            "d_ff": d_ff,
            "d_ff_per_rank_tp": dff_local,
            "vocab_size": vocab_size,
            "total_parameters": total_params,
        },
        "single_rank_loss": round(loss_ref.data[0], 10),
        "sharded_loss": round(dist_loss.data[0], 10),
        "loss_abs_diff": loss_abs_diff,
        "max_abs_grad_diff": global_max_grad_diff,
        "max_abs_param_diff": global_max_param_diff,
        "input_dx_max_diff": input_dx_max_diff,
        "parity_verified": bool(
            global_max_grad_diff < 1e-10 and global_max_param_diff < 1e-10
        ),
        "memory_tracemalloc": {
            "unsharded_adamw_bytes": unsharded_opt_bytes,
            "zero1_tp_rank0_adamw_bytes": rank0_opt_bytes,
            "per_rank_reduction_pct": round(opt_mem_reduction_pct, 2),
            "compression_ratio_x": round(
                unsharded_opt_bytes / max(1, rank0_opt_bytes), 2
            ),
        },
        "collectives": {
            "ulysses_all_to_all_calls": cstats.all_to_all_calls,
            "sp_all_gather_calls": cstats.all_gather_calls,
            "sp_and_zero1_reduce_scatter_calls": cstats.reduce_scatter_calls,
            "loss_all_reduce_calls": cstats.all_reduce_calls,
            "bytes_transferred": cstats.bytes_transferred,
        },
        "timing_ms": {
            "single_rank_ms": round(single_rank_ms, 3),
            "sharded_multi_rank_ms": round(sharded_exec_ms, 3),
            "total_verify_ms": round(total_ms, 3),
        },
        "per_tensor_parity": per_tensor_parity,
    }
