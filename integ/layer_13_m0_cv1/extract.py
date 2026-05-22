#!/usr/bin/env python3
# Copyright (c) 2026 vibeyolo
# SPDX-License-Identifier: Apache-2.0
#
# extract.py — Generate stimulus + reference for the layer_13_m0_cv1
# integration test. Slice: YOLO26n /model.6/m.0/cv1 Conv-BN-SiLU — the first
# conv inside the bottleneck of C3k2 block #3 (/model.6). Because YOLO26n
# uses c3k=False in this deeper stage, the bottleneck is a (1×1 + 1×1)
# pair instead of (3×3 + 3×3), so this conv is K=1.
#
#   /model.6/Slice_1_output_0_quantized  (u8, 64ch, 40x40, s_a, zp_a)
#    → ConvInteger (64→32, 1x1, stride 1, pad 0, zp_w=0)
#    → Cast i32→fp32 → Mul (s_a·s_w) → Add bias
#    → Sigmoid → Mul   (SiLU)
#
# The conv input is the second half of /model.6/cv1's 128-channel output
# (C3k2 split). Treat it as a 64-channel u8 tensor; pull both the
# quantized values and (s_a, zp_a) straight from ORT intermediate outputs.
#
# Structural pattern mirrors integ/layer_7_cv1 (1×1, phased reduction) and
# integ/layer_8_m0_cv1 (m.0/cv1 role / Slice_1 input).

import os, json
import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper

HERE = os.path.dirname(os.path.abspath(__file__))
STIM = os.path.join(HERE, "stim")
os.makedirs(STIM, exist_ok=True)

MODEL_PATH = "/home/marty/Documents/aif/vibeyolo/integ/yolo26n/model_int8.onnx"

# ── Geometry ──
# L13: 64→32, 1x1, stride 1, pad 0. Plane 40x40 → 40x40. Drive 8×8 ROI.
NCH_IN  = 64
NCH_OUT = 32
K       = 1
STRIDE  = 1
ROI_H, ROI_W = 8, 8
OUT_H, OUT_W = ROI_H, ROI_W
R, C = 12, 12   # top-left of crop in 40x40 layer-13-input space

# ── Load ONNX initializers ──
model = onnx.load(MODEL_PATH)
init_by_name = {init.name: numpy_helper.to_array(init) for init in model.graph.initializer}

W_q  = init_by_name['onnx::Conv_1690_quantized'].astype(np.int32)   # (32,64,1,1)
s_w  = float(init_by_name['onnx::Conv_1690_scale'])
zp_w = int(init_by_name['onnx::Conv_1690_zero_point'])
bias = init_by_name['onnx::Conv_1691'].astype(np.float32)           # (32,)
assert zp_w == 0
assert W_q.shape == (NCH_OUT, NCH_IN, K, K), f"got {W_q.shape}"

# Per-output-channel sum of weights, used to fold the u8→i8 zp shift.
sum_w = W_q.reshape(NCH_OUT, -1).sum(axis=1).astype(np.int32)  # (32,)

# ── ORT session with extra outputs ──
TARGET_OUT = '/model.6/m.0/cv1/act/Mul_output_0'           # layer-13 post-SiLU
L13_IN_Q   = '/model.6/Slice_1_output_0_quantized'         # u8 (1,64,40,40)
L13_IN_S   = '/model.6/Slice_1_output_0_scale'
L13_IN_ZP  = '/model.6/Slice_1_output_0_zero_point'

EXTRA_OUTS = [
    (TARGET_OUT, onnx.TensorProto.FLOAT, None),
    (L13_IN_Q,   onnx.TensorProto.UINT8, None),
    (L13_IN_S,   onnx.TensorProto.FLOAT, None),
    (L13_IN_ZP,  onnx.TensorProto.UINT8, None),
]

mod_model = onnx.load(MODEL_PATH)
existing = [o.name for o in mod_model.graph.output]
for nm, tp, shape in EXTRA_OUTS:
    if nm not in existing:
        mod_model.graph.output.append(onnx.helper.make_tensor_value_info(nm, tp, shape))
TMP_PATH = os.path.join(HERE, "_model_with_intermediate.onnx")
onnx.save(mod_model, TMP_PATH)

so = ort.SessionOptions()
so.log_severity_level = 3
sess = ort.InferenceSession(TMP_PATH, sess_options=so, providers=['CPUExecutionProvider'])
ipt_name = sess.get_inputs()[0].name
print(f"ORT input: {ipt_name}")

def run_ort(img_f32):
    outs = sess.run([TARGET_OUT, L13_IN_Q, L13_IN_S, L13_IN_ZP], {ipt_name: img_f32})
    return outs[0], outs[1], float(outs[2]), int(outs[3])

# ── Inputs ──
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

# Inner C3k2 conv ranges tend to be small; reuse ±2.0/127 like sibling layers.
S_OUT_PRE  = 4.0 / 127.0
S_OUT_SILU = 4.0 / 127.0

def f32_to_fp16(x):  return np.float16(x).view(np.uint16)

# ── HW reference (fp32 forward through the SV-equivalent pipeline) ──
def hw_reference(qx_u8_full, s_a, zp_a):
    """qx_u8_full: (1,64,40,40) uint8."""
    s_acc = s_a * s_w  # per-tensor accumulator scale (zp_w==0)

    x_u8 = qx_u8_full[0, :, R:R+ROI_H, C:C+ROI_W].astype(np.int32)  # (64,H,W)
    x_i8 = np.clip(x_u8 - 128, -128, 127).astype(np.int8)

    bias_eff = bias + s_acc * (128 - zp_a) * sum_w  # (32,)

    out = np.zeros((NCH_OUT, OUT_H, OUT_W), dtype=np.float32)
    sum_q_per_pix = np.zeros((NCH_OUT, OUT_H, OUT_W), dtype=np.int64)
    for c in range(NCH_OUT):
        for oy in range(OUT_H):
            for ox in range(OUT_W):
                acc = int((W_q[c, :, 0, 0] * x_i8[:, oy, ox].astype(np.int32)).sum())
                sum_q_per_pix[c, oy, ox] = acc
                pre = acc * s_acc + bias_eff[c]
                sig = 1.0 / (1.0 + np.exp(-pre))
                out[c, oy, ox] = pre * sig
    return out, bias_eff, x_i8, s_acc, sum_q_per_pix

# ── Weight packing ──
# 1x1: full reduction is NCH_IN=64. P_CIN=N_LANE=2 → N_PHASE=32.
N_LANE_FULL = NCH_IN          # 64
N_LANE      = 2               # P_CIN from scale_pkg LAYER_13_P_CIN
N_PHASE     = N_LANE_FULL // N_LANE  # 32
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

# ── Samples ──
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

# Worst-case |sum_q| theoretical bound (all 64 i8 values at ±128, all
# weights at ±max|w|): rarely realized but used to pick ACC_SHIFT safely.
max_abs_w_per_chan = np.max(np.abs(W_q).reshape(NCH_OUT, -1).sum(axis=1))
print(f"max |sum_w_abs| per channel = {max_abs_w_per_chan} "
      f"(theoretical |sum_q| bound vs i8∈[-128,127] ≈ {128*max_abs_w_per_chan})")

# Choose ACC_SHIFT to keep observed |sum_q| < ~32k (well inside fp16 max),
# determined empirically below from the actual samples.
ACC_SHIFT = 0  # will be raised if observed |sum_q| exceeds threshold

manifest = {
    "out_h": OUT_H, "out_w": OUT_W, "nch_out": NCH_OUT, "nch_in": NCH_IN,
    "K": K, "stride": STRIDE, "n_lane_full": N_LANE_FULL,
    "n_lane": N_LANE, "n_phase": N_PHASE,
    "roi_h": ROI_H, "roi_w": ROI_W, "R": R, "C": C,
    "s_w": s_w, "s_out_pre": S_OUT_PRE, "s_out_silu": S_OUT_SILU,
    "samples": [],
}

# First pass: collect observed |sum_q| max so we can decide ACC_SHIFT.
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

# fp16 max normal magnitude ≈ 65504; pick smallest ACC_SHIFT keeping a margin.
THRESH = 32768
while (worst_sum_q >> ACC_SHIFT) > THRESH:
    ACC_SHIFT += 1
print(f"worst |sum_q| observed = {worst_sum_q} → ACC_SHIFT={ACC_SHIFT}")
manifest["acc_shift"] = ACC_SHIFT
manifest["worst_sum_q"] = worst_sum_q

for (sname, ort_out_full, qx_u8_full, s_a, zp_a) in ort_cache:
    ort_roi = ort_out_full[0, :, R:R+OUT_H, C:C+OUT_W]  # (32,8,8)
    hw_out, bias_eff, x_i8, s_acc, sum_q = hw_reference(qx_u8_full, s_a, zp_a)

    # Per-channel fp16 scale and bias to feed requant.
    # RTL arithmetic-shifts sum_q right by ACC_SHIFT; pre-multiply scale to
    # restore the value.
    scale_fp16 = np.array(
        [f32_to_fp16((s_acc / S_OUT_PRE) * (1 << ACC_SHIFT))] * NCH_OUT,
        dtype=np.uint16)
    bias_fp16  = np.array([f32_to_fp16(bias_eff[c] / S_OUT_PRE) for c in range(NCH_OUT)],
                          dtype=np.uint16)

    # ── Input i8 (ROI_H × ROI_W × NCH_IN), order h,w,kc ──
    inp_path = os.path.join(STIM, f"{sname}.input_i8.hex")
    with open(inp_path, "w") as f:
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
    print(f"[{sname}] s_a={s_a:.5f} zp_a={zp_a} pre-SiLU range~{ort_roi.min():.2f}..{ort_roi.max():.2f} "
          f"max|sum_q|={int(np.max(np.abs(sum_q)))} "
          f"ORT vs HW-ref: max_abs={mxe:.4f} mae={mae:.4f} out_range={out_range:.3f} cos={cos:.6f}")

    manifest["samples"].append({
        "name": sname, "s_a": s_a, "zp_a": zp_a, "s_acc": s_acc,
        "max_abs_sum_q": int(np.max(np.abs(sum_q))),
        "ort_vs_hw_max_abs": mxe, "ort_vs_hw_mae": mae,
        "ort_vs_hw_cos": cos, "out_range": out_range,
    })

with open(os.path.join(STIM, "manifest.json"), "w") as f:
    json.dump(manifest, f, indent=2)

# ── SiLU LUT (diagnostic) ──
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

print(f"\nWrote stimulus + reference to {STIM} (ACC_SHIFT={ACC_SHIFT})")
