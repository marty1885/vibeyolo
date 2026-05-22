#!/usr/bin/env python3
# Copyright (c) 2026 vibeyolo
# SPDX-License-Identifier: Apache-2.0
#
# extract.py — Generate stimulus + reference for the layer_12_cv1 integration
# test (YOLO26n /model.6/cv1: Conv 1x1, 128→128, stride 1, pad 0, followed by
# SiLU; entry of the third C3k2 block, operating on 40×40 feature maps).
#
# ONNX slice:
#   /model.5/act/Mul_output_0  (post-L11 SiLU, fp32)
#    → DynamicQuantizeLinear   (u8, s_a, zp_a)
#    → ConvInteger             (1x1, stride 1, pad 0, 128→128, w_zp=0)
#    → Cast i32→fp32
#    → Mul (s_a · s_w)
#    → Add bias
#    → Sigmoid → Mul (SiLU)
#    → /model.6/cv1/act/Mul_output_0
#
# Mirrors integ/layer_7_cv1 (same op type, 64→64). C_IN doubled to 128 so we
# expect ~2x worst-case |acc| → ACC_SHIFT chosen empirically from probe.

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
# Layer 12 cv1: 128→128, 1x1, stride 1, pad 0. Plane 40x40, drive 8x8 ROI.
NCH_IN  = 128
NCH_OUT = 128
K       = 1
STRIDE  = 1
ROI_H, ROI_W = 8, 8
OUT_H, OUT_W = ROI_H, ROI_W
R, C = 8, 8   # top-left of crop in 40x40 layer-12-input space

# ── Load ONNX initializers ──
model = onnx.load(MODEL_PATH)
init_by_name = {init.name: numpy_helper.to_array(init) for init in model.graph.initializer}

W_q  = init_by_name['onnx::Conv_1687_quantized'].astype(np.int32)   # (128,128,1,1)
s_w  = float(init_by_name['onnx::Conv_1687_scale'])
zp_w = int(init_by_name['onnx::Conv_1687_zero_point'])
bias = init_by_name['onnx::Conv_1688'].astype(np.float32)           # (128,)
assert zp_w == 0
assert W_q.shape == (NCH_OUT, NCH_IN, K, K), f"got {W_q.shape}"

# Per-output-channel sum of weights, used to fold the u8→i8 zp shift.
sum_w = W_q.reshape(NCH_OUT, -1).sum(axis=1).astype(np.int32)  # (128,)
print(f"|sum_w| max = {np.max(np.abs(sum_w))}  (informational)")

# ── ORT session with extra outputs ──
TARGET_OUT = '/model.6/cv1/act/Mul_output_0'              # layer-12-cv1 SiLU output
IN_Q       = '/model.5/act/Mul_output_0_quantized'        # u8 (1,128,40,40)
IN_S       = '/model.5/act/Mul_output_0_scale'            # scalar fp32
IN_ZP      = '/model.5/act/Mul_output_0_zero_point'       # uint8

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

S_OUT_PRE  = 4.0 / 127.0
S_OUT_SILU = 4.0 / 127.0

def f32_to_fp16(x):  return np.float16(x).view(np.uint16)

# ── HW reference (fp32 forward through the SV-equivalent pipeline) ──
def hw_reference(qx_u8_full, s_a, zp_a):
    """qx_u8_full: (1,128,40,40) uint8."""
    s_acc = s_a * s_w  # per-tensor accumulator scale (zp_w==0)

    x_u8 = qx_u8_full[0, :, R:R+ROI_H, C:C+ROI_W].astype(np.int32)  # (128,H,W)
    x_i8 = np.clip(x_u8 - 128, -128, 127).astype(np.int8)

    bias_eff = bias + s_acc * (128 - zp_a) * sum_w  # (128,)

    # Returns sum_q (int) too, to probe ACC_SHIFT empirically.
    out = np.zeros((NCH_OUT, OUT_H, OUT_W), dtype=np.float32)
    accs = np.zeros((NCH_OUT, OUT_H, OUT_W), dtype=np.int64)
    for c in range(NCH_OUT):
        for oy in range(OUT_H):
            for ox in range(OUT_W):
                acc = int((W_q[c, :, 0, 0] * x_i8[:, oy, ox].astype(np.int32)).sum())
                accs[c, oy, ox] = acc
                pre = acc * s_acc + bias_eff[c]
                sig = 1.0 / (1.0 + np.exp(-pre))
                out[c, oy, ox] = pre * sig
    return out, bias_eff, x_i8, s_acc, accs

# ── Weight packing ──
# 1x1: full reduction is NCH_IN=128. P_CIN=N_LANE=4 → N_PHASE=32.
N_LANE_FULL = NCH_IN          # 128
N_LANE      = 4               # LAYER_12_P_CIN
N_PHASE     = N_LANE_FULL // N_LANE  # 32
assert N_LANE_FULL == N_LANE * N_PHASE

# Pack (NCH_OUT, N_PHASE, N_LANE).
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

# ── First pass: probe |acc| across all samples to choose ACC_SHIFT ──
FP16_MAX = 65504.0
probe_results = []
for sname, kind, seed in samples:
    img = make_pixel_input(kind, seed=seed)
    _, qx_u8_full, s_a, zp_a = run_ort(img)
    if s_a == 0.0:
        s_a = 1.0
    _, _, _, _, accs = hw_reference(qx_u8_full, s_a, zp_a)
    probe_results.append((sname, qx_u8_full, s_a, zp_a, np.max(np.abs(accs))))

worst_abs_acc = max(p[4] for p in probe_results)
print(f"\nWorst |acc| across probes = {worst_abs_acc}")
ACC_SHIFT = 0
while (worst_abs_acc >> ACC_SHIFT) > FP16_MAX:
    ACC_SHIFT += 1
# Add a touch of headroom (full frame may exceed ROI worst), bias to ACC_SHIFT≥3.
if ACC_SHIFT < 3:
    ACC_SHIFT = 3
print(f"Chosen ACC_SHIFT = {ACC_SHIFT}  (post-shift |acc| ≤ {worst_abs_acc >> ACC_SHIFT})")

manifest = {
    "out_h": OUT_H, "out_w": OUT_W, "nch_out": NCH_OUT, "nch_in": NCH_IN,
    "K": K, "stride": STRIDE, "n_lane_full": N_LANE_FULL,
    "n_lane": N_LANE, "n_phase": N_PHASE,
    "roi_h": ROI_H, "roi_w": ROI_W, "R": R, "C": C,
    "s_w": s_w, "s_out_pre": S_OUT_PRE, "s_out_silu": S_OUT_SILU,
    "acc_shift": ACC_SHIFT,
    "worst_abs_acc_probe": int(worst_abs_acc),
    "samples": [],
}

for (sname, qx_u8_full, s_a, zp_a, _wac), (_, kind, seed) in zip(probe_results, samples):
    img = make_pixel_input(kind, seed=seed)
    ort_out_full, _, _, _ = run_ort(img)
    ort_roi = ort_out_full[0, :, R:R+OUT_H, C:C+OUT_W]  # (128,8,8)
    hw_out, bias_eff, x_i8, s_acc, _accs = hw_reference(qx_u8_full, s_a, zp_a)

    # Per-channel fp16 scale and bias to feed requant.
    # Pre-multiply scale by 2^ACC_SHIFT (so the >>>ACC_SHIFT in RTL is undone).
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
          f"ORT vs HW-ref: max_abs={mxe:.4f} mae={mae:.4f} out_range={out_range:.3f} cos={cos:.6f}")

    manifest["samples"].append({
        "name": sname, "s_a": s_a, "zp_a": zp_a, "s_acc": s_acc,
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

print(f"\nWrote stimulus + reference to {STIM}")
print(f"ACC_SHIFT = {ACC_SHIFT} (must match RTL parameter)")
