#!/usr/bin/env python3
# Copyright (c) 2026 vibeyolo
# SPDX-License-Identifier: Apache-2.0
#
# extract.py — Generate stimulus + reference for the layer_10_cv2 integration
# test. This is the exit conv of the second C3k2 block in YOLO26n:
#
#   /model.4/cv2/conv  Conv 1×1, Cin=96, Cout=128, stride=1, pad=0
#
# Slice (ONNX subgraph):
#   /model.4/Concat_output_0                  (fp32, 96 channels, 80x80)
#    → DynamicQuantizeLinear  (u8, s_a, zp_a)
#    → ConvInteger            (96 → 128, K=1, stride=1, pad=0)
#    → Cast i32 → fp32
#    → Mul (s_a * s_w)
#    → Add bias               (fp32)
#    → Sigmoid → Mul          (SiLU)
#    → /model.4/cv2/act/Mul_output_0          (fp32 reference)
#
# The 96-ch input is the external concat of:
#   - /model.4/Slice_output_0     (32 ch)   cv1 first half
#   - /model.4/Slice_1_output_0   (32 ch)   cv1 second half
#   - /model.4/m.0/Add_output_0   (32 ch)   bottleneck m.0 residual
# (verified via onnx.shape_inference; concat axis=1, 32+32+32 = 96.)
# We accept the ORT-quantised concatenated u8 tensor directly.
#
# Pipeline mirrors layer_7_cv1 (P_CIN-phased K=1 conv):
#   • NCH_OUT × dotN(N=P_CIN=8)
#   • N_PHASE=12 phase accumulator
#   • >>> ACC_SHIFT  (chosen empirically to keep |sum_q| inside fp16 max)
#   • requant (fp16 scale + bias)
#   • act_silu (LUT)

import os, json
import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper

HERE = os.path.dirname(os.path.abspath(__file__))
STIM = os.path.join(HERE, "stim")
os.makedirs(STIM, exist_ok=True)

MODEL_PATH = "/home/marty/Documents/aif/vibeyolo/integ/yolo26n/model_int8.onnx"

# ── Geometry ──────────────────────────────────────────────────────────────
# Layer 10: 96 → 128, K=1, stride=1, pad=0. Input/output 80×80.
NCH_IN  = 96
NCH_OUT = 128
K       = 1
STRIDE  = 1
ROI_H, ROI_W = 8, 8
OUT_H, OUT_W = ROI_H, ROI_W
R, C = 16, 16   # ROI top-left in 80x80

# ── ONNX initializers (L10 conv params) ────────────────────────────────────
model = onnx.load(MODEL_PATH)
init_by_name = {init.name: numpy_helper.to_array(init) for init in model.graph.initializer}

W_q  = init_by_name['onnx::Conv_1681_quantized'].astype(np.int32)   # (128,96,1,1)
s_w  = float(init_by_name['onnx::Conv_1681_scale'])
zp_w = int(init_by_name['onnx::Conv_1681_zero_point'])
bias = init_by_name['onnx::Conv_1682'].astype(np.float32)           # (128,)
assert zp_w == 0
assert W_q.shape == (NCH_OUT, NCH_IN, K, K), f"Got {W_q.shape}"

sum_w = W_q.reshape(NCH_OUT, -1).sum(axis=1).astype(np.int32)        # (128,)

# ── ORT session with intermediate outputs ──────────────────────────────────
TARGET_OUT = '/model.4/cv2/act/Mul_output_0'                          # fp32
IN_Q       = '/model.4/Concat_output_0_quantized'                     # u8 (1,96,80,80)
IN_S       = '/model.4/Concat_output_0_scale'                         # fp32 scalar
IN_ZP      = '/model.4/Concat_output_0_zero_point'                    # u8

EXTRA_OUTS = [
    (TARGET_OUT, onnx.TensorProto.FLOAT, None),
    (IN_Q,       onnx.TensorProto.UINT8, None),
    (IN_S,       onnx.TensorProto.FLOAT, None),
    (IN_ZP,      onnx.TensorProto.UINT8, None),
]

mod_model = onnx.load(MODEL_PATH)
existing = [o.name for o in mod_model.graph.output]
for nm, tp, _shape in EXTRA_OUTS:
    if nm not in existing:
        mod_model.graph.output.append(onnx.helper.make_tensor_value_info(nm, tp, None))
TMP_PATH = os.path.join(HERE, "_model_with_intermediate.onnx")
onnx.save(mod_model, TMP_PATH)

so = ort.SessionOptions()
so.log_severity_level = 3
sess = ort.InferenceSession(TMP_PATH, sess_options=so, providers=['CPUExecutionProvider'])
ipt_name = sess.get_inputs()[0].name
print(f"ORT input: {ipt_name}")

def run_ort(img_f32):
    outs = sess.run([TARGET_OUT, IN_Q, IN_S, IN_ZP], {ipt_name: img_f32})
    return outs[0], outs[1], float(outs[2]), int(outs[3])

# ── Stimuli ───────────────────────────────────────────────────────────────
def make_pixel_input(kind, seed=0):
    img = np.zeros((1, 3, 640, 640), dtype=np.float32)
    rng = np.random.RandomState(seed)
    if kind == 'rand':
        img[0] = rng.uniform(0.0, 1.0, size=(3, 640, 640)).astype(np.float32)
    elif kind == 'half':
        img[0] = 0.5
    elif kind == 'gradient':
        gx = np.tile(np.linspace(0.0, 1.0, 640, dtype=np.float32), (640, 1))
        img[0, 0] = gx; img[0, 1] = gx.T; img[0, 2] = 0.5 * (gx + gx.T)
    elif kind == 'rand_low':
        img[0] = rng.uniform(0.15, 0.80, size=(3, 640, 640)).astype(np.float32)
    elif kind == 'rand_high':
        img[0] = rng.uniform(0.5, 1.0, size=(3, 640, 640)).astype(np.float32)
    else:
        raise ValueError(kind)
    return img

# Pre-SiLU output range pick. L10 sits roughly in a similar regime to L5/L7;
# tighten/widen automatically below from the data, but start with the same
# default as L5 (12/127), then refine.
S_OUT_PRE  = 6.0 / 127.0
S_OUT_SILU = 6.0 / 127.0

def f32_to_fp16(x):  return np.float16(x).view(np.uint16)

# ── HW reference (fp32 model of the SV-equivalent pipeline) ───────────────
def hw_reference(qx_u8_full, s_a, zp_a):
    """qx_u8_full: (1,96,80,80) uint8 ORT-quantised L10 input."""
    s_acc = s_a * s_w                                                # per-tensor
    pad_u8 = qx_u8_full[0, :, R:R+ROI_H, C:C+ROI_W].astype(np.int32)
    x_i8 = np.clip(pad_u8 - 128, -128, 127).astype(np.int8)
    bias_eff = bias + s_acc * (128 - zp_a) * sum_w                   # (128,)

    out = np.zeros((NCH_OUT, OUT_H, OUT_W), dtype=np.float32)
    sum_q_per_pix = np.zeros((NCH_OUT, OUT_H, OUT_W), dtype=np.int64)
    for c in range(NCH_OUT):
        for oy in range(OUT_H):
            for ox in range(OUT_W):
                window = x_i8[:, oy, ox].astype(np.int32)            # (96,)
                acc = int((W_q[c, :, 0, 0] * window).sum())
                sum_q_per_pix[c, oy, ox] = acc
                pre = acc * s_acc + bias_eff[c]
                sig = 1.0 / (1.0 + np.exp(-pre))
                out[c, oy, ox] = pre * sig
    return out, bias_eff, x_i8, s_acc, sum_q_per_pix

# ── Weight packing ────────────────────────────────────────────────────────
# 1x1: full reduction is NCH_IN=96. P_CIN=N_LANE=8 → N_PHASE=12.
N_LANE_FULL = NCH_IN          # 96
N_LANE      = 8               # P_CIN from scale_pkg LAYER_10_P_CIN
N_PHASE     = N_LANE_FULL // N_LANE  # 12
assert N_LANE_FULL == N_LANE * N_PHASE

# Pack (NCH_OUT, N_PHASE, N_LANE) — the order the TB will stream.
W_phase = np.zeros((NCH_OUT, N_PHASE, N_LANE), dtype=np.int8)
for c in range(NCH_OUT):
    for ph in range(N_PHASE):
        for k in range(N_LANE):
            W_phase[c, ph, k] = W_q[c, ph*N_LANE + k, 0, 0]

with open(os.path.join(STIM, "weights.i8.hex"), "w") as f:
    for c in range(NCH_OUT):
        for ph in range(N_PHASE):
            for k in range(N_LANE):
                f.write(f"{int(W_phase[c, ph, k]) & 0xFF:02x}\n")

# ── Samples ───────────────────────────────────────────────────────────────
samples = [
    ("rand0",     'rand',      0),
    ("rand1",     'rand',      1),
    ("rand2",     'rand',      7),
    ("rand3",     'rand',      13),
    ("rand_low",  'rand_low',  2),
    ("rand_high", 'rand_high', 3),
    ("half",      'half',      0),
    ("gradient",  'gradient',  0),
]

# Worst-case |sum_q| theoretical bound, informational.
max_abs_sum_w = int(np.max(np.abs(W_q).reshape(NCH_OUT, -1).sum(axis=1)))
print(f"max per-chan |sum(|w|)| = {max_abs_sum_w} "
      f"(theoretical |sum_q| bound vs i8∈[-128,127] ≈ {128*max_abs_sum_w})")

# Empirically choose ACC_SHIFT so observed |sum_q| stays well under fp16 max
# (≈65504). Pick the smallest shift that keeps the worst sample under THRESH.
THRESH = 32768
ACC_SHIFT = 0

manifest = {
    "out_h": OUT_H, "out_w": OUT_W, "nch_out": NCH_OUT, "nch_in": NCH_IN,
    "K": K, "stride": STRIDE, "n_lane_full": N_LANE_FULL,
    "n_lane": N_LANE, "n_phase": N_PHASE,
    "roi_h": ROI_H, "roi_w": ROI_W, "R": R, "C": C,
    "s_w": s_w, "s_out_pre": S_OUT_PRE, "s_out_silu": S_OUT_SILU,
    "samples": [],
}

# First pass: collect observed |sum_q| max to size ACC_SHIFT.
worst_sum_q = 0
ort_cache = []
for sname, kind, seed in samples:
    img = make_pixel_input(kind, seed=seed)
    ort_out_full, qx_u8_full, s_a, zp_a = run_ort(img)
    if s_a == 0.0:
        s_a = 1.0
    _, _, _, _, sum_q = hw_reference(qx_u8_full, s_a, zp_a)
    worst_sum_q = max(worst_sum_q, int(np.max(np.abs(sum_q))))
    ort_cache.append((sname, ort_out_full, qx_u8_full, s_a, zp_a))

# Tiled conv_layer requant.sv autoscales; ACC_SHIFT pre-shift not needed.
ACC_SHIFT = 0
print(f"worst |sum_q| observed = {worst_sum_q} → ACC_SHIFT={ACC_SHIFT}")
manifest["acc_shift"] = ACC_SHIFT
manifest["worst_sum_q"] = worst_sum_q

# Second pass: emit stim with chosen ACC_SHIFT.
for (sname, ort_out_full, qx_u8_full, s_a, zp_a) in ort_cache:
    ort_roi = ort_out_full[0, :, R:R+OUT_H, C:C+OUT_W]               # (128,8,8)
    hw_out, bias_eff, x_i8, s_acc, _ = hw_reference(qx_u8_full, s_a, zp_a)

    # RTL arithmetic-shifts sum_q right by ACC_SHIFT before requant;
    # compensate by pre-multiplying scale_fp16 by 2^ACC_SHIFT.
    scale_fp16 = np.array(
        [f32_to_fp16((s_acc / S_OUT_PRE) * (1 << ACC_SHIFT))] * NCH_OUT,
        dtype=np.uint16)
    bias_fp16  = np.array([f32_to_fp16(bias_eff[c] / S_OUT_PRE)
                           for c in range(NCH_OUT)], dtype=np.uint16)

    # Input i8 (ROI_H × ROI_W × NCH_IN), order h,w,kc.
    with open(os.path.join(STIM, f"{sname}.input_i8.hex"), "w") as f:
        for h in range(ROI_H):
            for w in range(ROI_W):
                for kc in range(NCH_IN):
                    f.write(f"{int(x_i8[kc, h, w]) & 0xFF:02x}\n")
    with open(os.path.join(STIM, f"{sname}.scale_fp16.hex"), "w") as f:
        for c in range(NCH_OUT):
            f.write(f"{int(scale_fp16[c]):04x}\n")
    with open(os.path.join(STIM, f"{sname}.bias_fp16.hex"), "w") as f:
        for c in range(NCH_OUT):
            f.write(f"{int(bias_fp16[c]):04x}\n")
    with open(os.path.join(STIM, f"{sname}.ref_ort.f32.hex"), "w") as f:
        for oy in range(OUT_H):
            for ox in range(OUT_W):
                for c in range(NCH_OUT):
                    bits = np.float32(ort_roi[c, oy, ox]).view(np.uint32)
                    f.write(f"{int(bits):08x}\n")
    with open(os.path.join(STIM, f"{sname}.ref_hw.f32.hex"), "w") as f:
        for oy in range(OUT_H):
            for ox in range(OUT_W):
                for c in range(NCH_OUT):
                    bits = np.float32(hw_out[c, oy, ox]).view(np.uint32)
                    f.write(f"{int(bits):08x}\n")

    diff = ort_roi - hw_out
    mae = float(np.mean(np.abs(diff)))
    mxe = float(np.max(np.abs(diff)))
    out_range = float(ort_roi.max() - ort_roi.min())
    cos = float(np.sum(ort_roi * hw_out) /
                (np.linalg.norm(ort_roi) * np.linalg.norm(hw_out) + 1e-12))
    print(f"[{sname}] s_a={s_a:.5f} zp_a={zp_a} "
          f"pre-SiLU range~{ort_roi.min():.2f}..{ort_roi.max():.2f} "
          f"ORT vs HW-ref: max_abs={mxe:.4f} mae={mae:.4f} "
          f"out_range={out_range:.3f} cos={cos:.6f}")

    manifest["samples"].append({
        "name": sname, "s_a": s_a, "zp_a": zp_a, "s_acc": s_acc,
        "ort_vs_hw_max_abs": mxe, "ort_vs_hw_mae": mae,
        "ort_vs_hw_cos": cos, "out_range": out_range,
    })

with open(os.path.join(STIM, "manifest.json"), "w") as f:
    json.dump(manifest, f, indent=2)

# ── SiLU LUT (diagnostic) ─────────────────────────────────────────────────
def silu_lut_bytes(in_scale, out_scale):
    lut = np.zeros(256, dtype=np.int8)
    for idx in range(256):
        cp = idx if idx < 128 else idx - 256
        f = cp * in_scale
        if f >= 0.0:
            s = f / (1.0 + np.exp(-f))
        else:
            ef = np.exp(f); s = (f * ef) / (1.0 + ef)
        q = round(s / out_scale)
        q = max(-128, min(127, int(q)))
        lut[idx] = q
    return lut

lut = silu_lut_bytes(S_OUT_PRE, S_OUT_SILU)
with open(os.path.join(STIM, "silu_lut.i8.hex"), "w") as f:
    for v in lut:
        f.write(f"{int(v) & 0xFF:02x}\n")

print(f"\nWrote stimulus + reference to {STIM}")
