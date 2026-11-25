"""Bit-Level BF16 / FP8 (E4M3 & E5M2) / MXFP8 Quantizer & Stochastic-Rounding Convergence Lab.

Implements exact IEEE-754 / OCP bit-manipulation quantization:
- BF16 (1 sign, 8 exp, 7 mantissa, bias 127) via binary32 bit-packing
- FP8 E4M3FN (1 sign, 4 exp, 3 mantissa, bias 7, max normal 448.0) for forward tensors
- FP8 E5M2 (1 sign, 5 exp, 2 mantissa, bias 15, max normal 57344.0) for backward gradients
- MXFP8 (OCP Microscaling 32-element block shared E8M0 power-of-two scale)
- Deterministic Round-to-Nearest-Even (RNE) vs. unbiased Stochastic Rounding (SR)
- Real 40-step MLP training loops showing gradient swamping under small-LR RNE vs. clean SR convergence
"""

from __future__ import annotations

import math
import random
import struct
import time
from typing import Dict, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Bit-Exact BF16 Casting (Round-to-Nearest-Even vs. Stochastic Rounding)
# ---------------------------------------------------------------------------

def float_to_bf16_bits(val: float, stochastic: bool = False, rng: Optional[random.Random] = None) -> int:
    """Packs float into IEEE-754 binary32 and rounds to 16-bit BF16 uint16."""
    if math.isnan(val):
        return 0x7FC0
    if math.isinf(val):
        return 0xFF80 if val < 0 else 0x7F80
    b32 = struct.unpack(">I", struct.pack(">f", val))[0]
    exp32 = (b32 >> 23) & 0xFF
    if exp32 == 0xFF:
        return (b32 >> 16) & 0xFFFF

    hi = (b32 >> 16) & 0xFFFF
    rem = b32 & 0xFFFF

    if stochastic:
        r_draw = rng.randrange(0x10000) if rng is not None else random.randrange(0x10000)
        if r_draw < rem:
            hi += 1
    else:
        if rem > 0x8000 or (rem == 0x8000 and (hi & 1) == 1):
            hi += 1

    return hi & 0xFFFF


def bf16_bits_to_float(bits16: int) -> float:
    b32 = (bits16 & 0xFFFF) << 16
    return struct.unpack(">f", struct.pack(">I", b32))[0]


def cast_bf16(val: float, stochastic: bool = False, rng: Optional[random.Random] = None) -> float:
    return bf16_bits_to_float(float_to_bf16_bits(val, stochastic=stochastic, rng=rng))


# ---------------------------------------------------------------------------
# Bit-Exact FP8 E4M3FN & FP8 E5M2 Encode / Decode (uint8 byte 0..255)
# ---------------------------------------------------------------------------

def _round_mantissa(frac: float, even_lsb: int, stochastic: bool, rng: Optional[random.Random]) -> int:
    """Rounds fractional part in [0, 1) to 0 or 1 via RNE or unbiased Stochastic Rounding."""
    if stochastic:
        u = rng.random() if rng is not None else random.random()
        return 1 if u < frac else 0
    if frac > 0.5 or (frac == 0.5 and (even_lsb & 1) == 1):
        return 1
    return 0


def float_to_fp8_e4m3_byte(
    val: float, stochastic: bool = False, rng: Optional[random.Random] = None
) -> int:
    """OCP FP8 E4M3FN: 1 sign, 4 exp (bias 7), 3 mantissa, max finite 448.0 (0x7E / 0xFE)."""
    if math.isnan(val):
        return 0x7F
    sign = 1 if math.copysign(1.0, val) < 0.0 else 0
    ax = abs(val)
    if ax == 0.0:
        return sign << 7
    if math.isinf(ax) or ax >= 448.0:
        return (sign << 7) | 0x7E  # E4M3FN max normal: exp=15 (e=8), mant=6 -> 1.75 * 256 = 448

    # Subnormal threshold: min normal is 2^-6 = 0.015625; subnormal step is 2^-9 = 1/512
    if ax < (2.0 ** -6):
        scaled = ax * 512.0  # in (0, 8.0)
        m_int = int(math.floor(scaled))
        frac = scaled - m_int
        m_int += _round_mantissa(frac, m_int, stochastic, rng)
        if m_int >= 8:
            return (sign << 7) | (1 << 3) | 0  # rounds up to min normal 2^-6
        return (sign << 7) | (m_int & 0x7)

    m_norm, exp2 = math.frexp(ax)  # ax = m_norm * 2^exp2, m_norm in [0.5, 1.0)
    unbiased_e = exp2 - 1
    sig = m_norm * 2.0 - 1.0  # in [0.0, 1.0)
    scaled_m = sig * 8.0
    m_int = int(math.floor(scaled_m))
    frac = scaled_m - m_int
    m_int += _round_mantissa(frac, m_int, stochastic, rng)
    if m_int >= 8:
        m_int = 0
        unbiased_e += 1

    biased_e = unbiased_e + 7
    if biased_e > 15 or (biased_e == 15 and m_int >= 7):
        return (sign << 7) | 0x7E
    return (sign << 7) | ((biased_e & 0xF) << 3) | (m_int & 0x7)


def fp8_e4m3_byte_to_float(byte_val: int) -> float:
    b = byte_val & 0xFF
    sign = -1.0 if ((b >> 7) & 1) else 1.0
    exp = (b >> 3) & 0xF
    mant = b & 0x7
    if exp == 0xF and mant == 0x7:
        return float("nan")
    if exp == 0:
        return sign * (mant / 8.0) * (2.0 ** -6)
    return sign * (1.0 + mant / 8.0) * (2.0 ** (exp - 7))


def float_to_fp8_e5m2_byte(
    val: float, stochastic: bool = False, rng: Optional[random.Random] = None
) -> int:
    """IEEE-style FP8 E5M2: 1 sign, 5 exp (bias 15), 2 mantissa, max normal 57344.0."""
    if math.isnan(val):
        return 0x7F
    sign = 1 if math.copysign(1.0, val) < 0.0 else 0
    ax = abs(val)
    if ax == 0.0:
        return sign << 7
    if math.isinf(ax) or ax >= 57344.0:
        return (sign << 7) | 0x7B  # saturate to max finite normal: exp=30 (e=15), mant=3 -> 1.75 * 32768

    # Min normal is 2^-14; subnormal step is 2^-16 = 1/65536
    if ax < (2.0 ** -14):
        scaled = ax * 65536.0
        m_int = int(math.floor(scaled))
        frac = scaled - m_int
        m_int += _round_mantissa(frac, m_int, stochastic, rng)
        if m_int >= 4:
            return (sign << 7) | (1 << 2) | 0
        return (sign << 7) | (m_int & 0x3)

    m_norm, exp2 = math.frexp(ax)
    unbiased_e = exp2 - 1
    sig = m_norm * 2.0 - 1.0
    scaled_m = sig * 4.0
    m_int = int(math.floor(scaled_m))
    frac = scaled_m - m_int
    m_int += _round_mantissa(frac, m_int, stochastic, rng)
    if m_int >= 4:
        m_int = 0
        unbiased_e += 1

    biased_e = unbiased_e + 15
    if biased_e >= 31:
        return (sign << 7) | 0x7B
    return (sign << 7) | ((biased_e & 0x1F) << 2) | (m_int & 0x3)


def fp8_e5m2_byte_to_float(byte_val: int) -> float:
    b = byte_val & 0xFF
    sign = -1.0 if ((b >> 7) & 1) else 1.0
    exp = (b >> 2) & 0x1F
    mant = b & 0x3
    if exp == 0x1F:
        return sign * float("inf") if mant == 0 else float("nan")
    if exp == 0:
        return sign * (mant / 4.0) * (2.0 ** -14)
    return sign * (1.0 + mant / 4.0) * (2.0 ** (exp - 15))


def cast_fp8_e4m3(val: float, stochastic: bool = False, rng: Optional[random.Random] = None) -> float:
    return fp8_e4m3_byte_to_float(float_to_fp8_e4m3_byte(val, stochastic=stochastic, rng=rng))


def cast_fp8_e5m2(val: float, stochastic: bool = False, rng: Optional[random.Random] = None) -> float:
    return fp8_e5m2_byte_to_float(float_to_fp8_e5m2_byte(val, stochastic=stochastic, rng=rng))


# ---------------------------------------------------------------------------
# OCP MXFP8 32-Element Shared E8M0 Block Quantizer
# ---------------------------------------------------------------------------

def quantize_mxfp8_tensor(
    values: Sequence[float],
    fmt: str = "e4m3",
    block_size: int = 32,
    stochastic: bool = False,
    rng: Optional[random.Random] = None,
) -> Tuple[List[float], List[int]]:
    """Quantizes tensor in 32-element blocks with shared E8M0 power-of-two exponent (OCP MX v1.0)."""
    emax_elem = 7 if fmt == "e4m3" else 14
    out = [0.0] * len(values)
    shared_exps: List[int] = []
    n = len(values)

    for b0 in range(0, n, block_size):
        b1 = min(n, b0 + block_size)
        amax = 0.0
        for i in range(b0, b1):
            av = abs(values[i])
            if av > amax:
                amax = av
        if amax == 0.0:
            shared_exps.append(-127)
            continue
        raw_exp = int(math.floor(math.log2(amax))) - emax_elem
        shared_exp = max(-127, min(127, raw_exp))
        shared_exps.append(shared_exp)
        scale = 2.0 ** shared_exp
        inv_scale = 1.0 / scale
        for i in range(b0, b1):
            scaled = values[i] * inv_scale
            if fmt == "e4m3":
                q = fp8_e4m3_byte_to_float(float_to_fp8_e4m3_byte(scaled, stochastic=stochastic, rng=rng))
            else:
                q = fp8_e5m2_byte_to_float(float_to_fp8_e5m2_byte(scaled, stochastic=stochastic, rng=rng))
            out[i] = q * scale

    return out, shared_exps


def inspect_bit_encodings(sample_value: float) -> Dict[str, object]:
    """Produces exact binary sign/exponent/mantissa bit breakdowns for the UI inspector."""
    b32 = struct.unpack(">I", struct.pack(">f", sample_value))[0]
    bf16_u16 = float_to_bf16_bits(sample_value, stochastic=False)
    e4m3_u8 = float_to_fp8_e4m3_byte(sample_value, stochastic=False)
    e5m2_u8 = float_to_fp8_e5m2_byte(sample_value, stochastic=False)

    bf16_bitstr = f"{bf16_u16:016b}"
    e4m3_bitstr = f"{e4m3_u8:08b}"
    e5m2_bitstr = f"{e5m2_u8:08b}"
    fp32_bitstr = f"{b32:032b}"

    bf16_val = bf16_bits_to_float(bf16_u16)
    e4m3_val = fp8_e4m3_byte_to_float(e4m3_u8)
    e5m2_val = fp8_e5m2_byte_to_float(e5m2_u8)

    return {
        "input_float": sample_value,
        "fp32": {
            "bits": f"{fp32_bitstr[0]} {fp32_bitstr[1:9]} {fp32_bitstr[9:]}",
            "decoded": struct.unpack(">f", struct.pack(">I", b32))[0],
        },
        "bf16": {
            "hex": f"0x{bf16_u16:04X}",
            "bits": f"{bf16_bitstr[0]} {bf16_bitstr[1:9]} {bf16_bitstr[9:]}",
            "sign_bit": bf16_bitstr[0],
            "exp_bits": bf16_bitstr[1:9],
            "mant_bits": bf16_bitstr[9:],
            "decoded": bf16_val,
            "abs_error": abs(sample_value - bf16_val),
        },
        "fp8_e4m3": {
            "hex": f"0x{e4m3_u8:02X}",
            "bits": f"{e4m3_bitstr[0]} {e4m3_bitstr[1:5]} {e4m3_bitstr[5:]}",
            "sign_bit": e4m3_bitstr[0],
            "exp_bits": e4m3_bitstr[1:5],
            "mant_bits": e4m3_bitstr[5:],
            "decoded": e4m3_val,
            "abs_error": abs(sample_value - e4m3_val),
        },
        "fp8_e5m2": {
            "hex": f"0x{e5m2_u8:02X}",
            "bits": f"{e5m2_bitstr[0]} {e5m2_bitstr[1:6]} {e5m2_bitstr[6:]}",
            "sign_bit": e5m2_bitstr[0],
            "exp_bits": e5m2_bitstr[1:6],
            "mant_bits": e5m2_bitstr[6:],
            "decoded": e5m2_val,
            "abs_error": abs(sample_value - e5m2_val),
        },
    }


def compare_outlier_microscaling(outlier_magnitude: float = 1024.0, seed: int = 19) -> Dict[str, object]:
    """Demonstrates why a single activation outlier crushes naive per-tensor FP8 E4M3
    into underflow while 32-element MXFP8 isolates the spike to block 0.
    """
    rng = random.Random(seed)
    # 128 elements (4 blocks of 32): fine-grained signals in [-0.012, 0.012] plus one outlier in index 0
    tensor = [rng.uniform(-0.012, 0.012) for _ in range(128)]
    tensor[0] = outlier_magnitude

    # Naive per-tensor dynamic scaling into E4M3 [-448, 448]
    amax = max(abs(v) for v in tensor)
    global_scale = amax / 448.0
    per_tensor_deq: List[float] = []
    for v in tensor:
        q_byte = float_to_fp8_e4m3_byte(v / global_scale, stochastic=False)
        dq = fp8_e4m3_byte_to_float(q_byte) * global_scale
        per_tensor_deq.append(dq)

    # MXFP8 32-element block scaling
    mxfp8_deq, block_exps = quantize_mxfp8_tensor(tensor, fmt="e4m3", block_size=32, stochastic=False)

    # Count underflows and relative RMSE across the 96 non-outlier elements in blocks 1..3
    underflow_zeros = sum(
        1 for i in range(32, 128) if tensor[i] != 0.0 and per_tensor_deq[i] == 0.0
    )
    mxfp8_zeros = sum(
        1 for i in range(32, 128) if tensor[i] != 0.0 and mxfp8_deq[i] == 0.0
    )

    # Compute relative RMSE on the 96 non-outlier elements in blocks 1..3
    tail_norm = math.sqrt(sum(tensor[i] ** 2 for i in range(32, 128)))
    pt_err = math.sqrt(sum((tensor[i] - per_tensor_deq[i]) ** 2 for i in range(32, 128))) / max(1e-12, tail_norm)
    mx_err = math.sqrt(sum((tensor[i] - mxfp8_deq[i]) ** 2 for i in range(32, 128))) / max(1e-12, tail_norm)

    return {
        "outlier_magnitude": outlier_magnitude,
        "num_elements": 128,
        "block_size": 32,
        "per_tensor_fp8_e4m3": {
            "global_scale": round(global_scale, 6),
            "non_outlier_rel_rmse_pct": round(pt_err * 100.0, 2),
            "underflow_zero_count": underflow_zeros,
        },
        "mxfp8_block32_e4m3": {
            "shared_e8m0_exponents": block_exps,
            "non_outlier_rel_rmse_pct": round(mx_err * 100.0, 2),
            "underflow_zero_count": mxfp8_zeros,
        },
    }


# ---------------------------------------------------------------------------
# Real 40-Step 2-Layer MLP Training Lab: Swamping (RNE) vs. Stochastic Rounding
# ---------------------------------------------------------------------------

def _run_mlp_regime(
    regime: str,
    x_data: List[List[float]],
    y_data: List[List[float]],
    w1_init: List[float],
    w2_init: List[float],
    in_dim: int,
    hid_dim: int,
    out_dim: int,
    steps: int,
    learning_rate: float,
    seed: int,
) -> Dict[str, object]:
    """Executes real 40-step forward + backward + quantized weight updates on a 2-layer MLP."""
    rng = random.Random(seed)
    n_samples = len(x_data)

    # Master copies (only used when regime == "mxfp8_sr_master")
    master_w1 = list(w1_init)
    master_w2 = list(w2_init)

    if regime in ("bf16_rne", "bf16_sr"):
        w1 = [cast_bf16(v, stochastic=False) for v in w1_init]
        w2 = [cast_bf16(v, stochastic=False) for v in w2_init]
    else:
        w1 = list(w1_init)
        w2 = list(w2_init)

    loss_curve: List[float] = []
    swamped_updates = 0
    total_updates = 0

    m_w1 = [0.0] * len(w1_init)
    v_w1 = [0.0] * len(w1_init)
    m_w2 = [0.0] * len(w2_init)
    v_w2 = [0.0] * len(w2_init)
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    for step_idx in range(1, steps + 1):
        use_sr = regime in ("bf16_sr", "mxfp8_sr_master")
        if regime == "fp64_reference":
            w1_fwd = w1
            w2_fwd = w2
        elif regime in ("bf16_rne", "bf16_sr"):
            w1_fwd = w1
            w2_fwd = w2
        else:  # mxfp8_sr_master: average 2 independent SR MXFP8 micro-casts per step or SR MXFP8 forward
            w1_fwd, _ = quantize_mxfp8_tensor(master_w1, fmt="e4m3", block_size=32, stochastic=True, rng=rng)
            w2_fwd, _ = quantize_mxfp8_tensor(master_w2, fmt="e4m3", block_size=32, stochastic=True, rng=rng)

        dw1 = [0.0] * (in_dim * hid_dim)
        dw2 = [0.0] * (hid_dim * out_dim)
        eval_loss = 0.0
        scale = 1.0 / (n_samples * out_dim)

        # Evaluate true objective loss on current weights + compute quantized forward/backward grads
        w1_eval = master_w1 if regime == "mxfp8_sr_master" else w1
        w2_eval = master_w2 if regime == "mxfp8_sr_master" else w2

        for s_idx in range(n_samples):
            x_orig = x_data[s_idx]
            y_row = y_data[s_idx]

            # Exact evaluation loss on current model weights (clean metric across all regimes)
            for o in range(out_dim):
                pred_eval = 0.0
                for j in range(hid_dim):
                    acc_e = sum(x_orig[i] * w1_eval[i * hid_dim + j] for i in range(in_dim))
                    sig_e = 1.0 / (1.0 + math.exp(-acc_e)) if acc_e >= 0.0 else math.exp(acc_e) / (1.0 + math.exp(acc_e))
                    pred_eval += (acc_e * sig_e) * w2_eval[j * out_dim + o]
                diff_e = pred_eval - y_row[o]
                eval_loss += 0.5 * diff_e * diff_e * scale

            # Quantized training forward + backward pass
            if regime == "mxfp8_sr_master":
                x_row, _ = quantize_mxfp8_tensor(x_orig, fmt="e4m3", block_size=32, stochastic=True, rng=rng)
            elif regime in ("bf16_rne", "bf16_sr"):
                x_row = [cast_bf16(v, stochastic=use_sr, rng=rng) for v in x_orig]
            else:
                x_row = x_orig

            h_pre = [0.0] * hid_dim
            h_act = [0.0] * hid_dim
            sigs = [0.0] * hid_dim
            for j in range(hid_dim):
                acc = 0.0
                for i in range(in_dim):
                    acc += x_row[i] * w1_fwd[i * hid_dim + j]
                h_pre[j] = acc
                sig = 1.0 / (1.0 + math.exp(-acc)) if acc >= 0.0 else math.exp(acc) / (1.0 + math.exp(acc))
                sigs[j] = sig
                h_act[j] = acc * sig

            if regime == "mxfp8_sr_master":
                h_act_q, _ = quantize_mxfp8_tensor(h_act, fmt="e4m3", block_size=32, stochastic=True, rng=rng)
            elif regime in ("bf16_rne", "bf16_sr"):
                h_act_q = [cast_bf16(v, stochastic=use_sr, rng=rng) for v in h_act]
            else:
                h_act_q = h_act

            pred = [0.0] * out_dim
            for o in range(out_dim):
                acc = 0.0
                for j in range(hid_dim):
                    acc += h_act_q[j] * w2_fwd[j * out_dim + o]
                pred[o] = acc

            dpred = [(pred[o] - y_row[o]) * scale for o in range(out_dim)]
            if regime != "fp64_reference":
                dpred, _ = quantize_mxfp8_tensor(dpred, fmt="e5m2", block_size=32, stochastic=use_sr, rng=rng)

            dh_act = [0.0] * hid_dim
            for j in range(hid_dim):
                acc_dh = 0.0
                for o in range(out_dim):
                    dw2[j * out_dim + o] += h_act_q[j] * dpred[o]
                    acc_dh += dpred[o] * w2_fwd[j * out_dim + o]
                dh_act[j] = acc_dh

            dh_pre = [
                dh_act[j] * sigs[j] * (1.0 + h_pre[j] * (1.0 - sigs[j]))
                for j in range(hid_dim)
            ]
            if regime != "fp64_reference":
                dh_pre, _ = quantize_mxfp8_tensor(dh_pre, fmt="e5m2", block_size=32, stochastic=use_sr, rng=rng)

            for i in range(in_dim):
                xi = x_row[i]
                off = i * hid_dim
                for j in range(hid_dim):
                    dw1[off + j] += xi * dh_pre[j]

        loss_curve.append(round(eval_loss, 8))

        bc1 = 1.0 - (beta1 ** step_idx)
        bc2 = 1.0 - (beta2 ** step_idx)

        def _adam_delta(g_vec: List[float], m_vec: List[float], v_vec: List[float]) -> List[float]:
            deltas = [0.0] * len(g_vec)
            for idx, g_val in enumerate(g_vec):
                m_vec[idx] = beta1 * m_vec[idx] + (1.0 - beta1) * g_val
                v_vec[idx] = beta2 * v_vec[idx] + (1.0 - beta2) * (g_val * g_val)
                mh = m_vec[idx] / bc1
                vh = v_vec[idx] / bc2
                deltas[idx] = learning_rate * mh / (math.sqrt(vh) + eps)
            return deltas

        d_w1 = _adam_delta(dw1, m_w1, v_w1)
        d_w2 = _adam_delta(dw2, m_w2, v_w2)

        if regime == "fp64_reference":
            for k in range(len(w1)):
                total_updates += 1
                w1[k] -= d_w1[k]
            for k in range(len(w2)):
                total_updates += 1
                w2[k] -= d_w2[k]
        elif regime == "bf16_rne":
            for k in range(len(w1)):
                total_updates += 1
                new_v = cast_bf16(w1[k] - d_w1[k], stochastic=False)
                if new_v == w1[k]:
                    swamped_updates += 1
                w1[k] = new_v
            for k in range(len(w2)):
                total_updates += 1
                new_v = cast_bf16(w2[k] - d_w2[k], stochastic=False)
                if new_v == w2[k]:
                    swamped_updates += 1
                w2[k] = new_v
        elif regime == "bf16_sr":
            for k in range(len(w1)):
                total_updates += 1
                new_v = cast_bf16(w1[k] - d_w1[k], stochastic=True, rng=rng)
                if new_v == w1[k]:
                    swamped_updates += 1
                w1[k] = new_v
            for k in range(len(w2)):
                total_updates += 1
                new_v = cast_bf16(w2[k] - d_w2[k], stochastic=True, rng=rng)
                if new_v == w2[k]:
                    swamped_updates += 1
                w2[k] = new_v
        else:  # mxfp8_sr_master (FP32 master weights + MXFP8 E4M3/E5M2 + SR)
            for k in range(len(master_w1)):
                total_updates += 1
                prev_f32 = struct.unpack(">f", struct.pack(">f", master_w1[k]))[0]
                new_f32 = struct.unpack(">f", struct.pack(">f", prev_f32 - d_w1[k]))[0]
                if new_f32 == prev_f32:
                    swamped_updates += 1
                master_w1[k] = new_f32
            for k in range(len(master_w2)):
                total_updates += 1
                prev_f32 = struct.unpack(">f", struct.pack(">f", master_w2[k]))[0]
                new_f32 = struct.unpack(">f", struct.pack(">f", prev_f32 - d_w2[k]))[0]
                if new_f32 == prev_f32:
                    swamped_updates += 1
                master_w2[k] = new_f32

    initial_loss = loss_curve[0]
    final_loss = loss_curve[-1]
    loss_drop_pct = ((initial_loss - final_loss) / max(1e-12, initial_loss)) * 100.0
    return {
        "regime": regime,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "loss_reduction_pct": round(loss_drop_pct, 2),
        "swamped_update_pct": round((swamped_updates / max(1, total_updates)) * 100.0, 2),
        "loss_curve": loss_curve,
    }


def run_fp8_convergence_lab(
    steps: int = 40,
    learning_rate: float = 1e-4,
    sample_float: float = -1.375,
    outlier_magnitude: float = 1024.0,
    seed: int = 123,
) -> Dict[str, object]:
    """Runs the 40-step reduced-precision convergence benchmark + bit-level format inspector."""
    t0 = time.perf_counter()
    steps = max(10, min(80, int(steps)))
    in_dim, hid_dim, out_dim = 16, 16, 8
    n_samples = 32
    rng = random.Random(seed)

    # Teacher weights at exact grid point |w| = 0.25 (exact in FP8 E4M3 & BF16; BF16 ULP = 2^-9 = 0.001953125)
    teacher_w1 = [
        (1.0 if (i + j) % 2 == 0 else -1.0) * 0.25
        for i in range(in_dim)
        for j in range(hid_dim)
    ]
    teacher_w2 = [
        (1.0 if (j + o) % 2 == 0 else -1.0) * 0.25
        for j in range(hid_dim)
        for o in range(out_dim)
    ]
    # Student weights perturbed by +/- 0.00390625 (2 BF16 ULPs at |w|=0.25).
    # Under Adam at lr=1e-4, 40 steps move by ~0.0040 -> reaches teacher_w cleanly in FP64, SR, and MXFP8,
    # whereas deterministic BF16 RNE sees |delta| = 0.0001 < half-ULP (0.0009765) -> 100% swamped!
    w1_init = [cast_bf16(v + rng.choice([-0.00390625, 0.00390625])) for v in teacher_w1]
    w2_init = [cast_bf16(v + rng.choice([-0.00390625, 0.00390625])) for v in teacher_w2]

    x_data: List[List[float]] = [
        [rng.uniform(-1.0, 1.0) for _ in range(in_dim)] for _ in range(n_samples)
    ]
    x_scaled = [[v * 1.8 for v in row] for row in x_data]
    y_scaled: List[List[float]] = []
    for x_row in x_scaled:
        h_act = [0.0] * hid_dim
        for j in range(hid_dim):
            acc = sum(x_row[i] * teacher_w1[i * hid_dim + j] for i in range(in_dim))
            sig = 1.0 / (1.0 + math.exp(-acc)) if acc >= 0.0 else math.exp(acc) / (1.0 + math.exp(acc))
            h_act[j] = acc * sig
        y_scaled.append(
            [
                sum(h_act[j] * teacher_w2[j * out_dim + o] for j in range(hid_dim))
                for o in range(out_dim)
            ]
        )

    regimes = {}
    for reg in ("fp64_reference", "bf16_rne", "bf16_sr", "mxfp8_sr_master"):
        regimes[reg] = _run_mlp_regime(
            reg,
            x_scaled,
            y_scaled,
            w1_init,
            w2_init,
            in_dim,
            hid_dim,
            out_dim,
            steps=steps,
            learning_rate=learning_rate,
            seed=seed + 500,
        )

    bit_view = inspect_bit_encodings(sample_float)
    outlier_study = compare_outlier_microscaling(outlier_magnitude=outlier_magnitude, seed=seed)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    return {
        "steps": steps,
        "learning_rate": learning_rate,
        "bf16_ulp_at_one": 2.0 ** -7,
        "half_ulp_swamping_threshold": 2.0 ** -8,
        "regimes": regimes,
        "bit_inspector": bit_view,
        "mxfp8_outlier_comparison": outlier_study,
        "exec_ms": round(elapsed_ms, 3),
    }
