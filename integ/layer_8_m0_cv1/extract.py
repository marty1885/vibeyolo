#!/usr/bin/env python3
# Copyright (c) 2026 vibeyolo
# SPDX-License-Identifier: Apache-2.0
#
# extract.py — Generate stimulus + reference for the layer_8_m0_cv1 integration
# test. Slice: YOLO26n /model.4/m.0/cv1 Conv-BN-SiLU (the first conv inside
# the C3k2 bottleneck of /model.4).
#
#   /model.4/Slice_1_output_0_quantized  (u8, 32ch, 80x80, s_a, zp_a)
#    → ConvInteger (32→16, 3x3, stride 1, pad 1, zp_w=0)
#    → Cast i32→fp32 → Mul (s_a·s_w) → Add bias
#    → Sigmoid → Mul   (SiLU)
#
# The conv input is the second half of /model.4/cv1's 64-channel output
# (C3k2's split). We treat it as a given 32-channel u8 tensor and pull both
# the quantized values and (s_a, zp_a) straight from ORT intermediate outputs.

import os, json
import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper

HERE = os.path.dirname(os.path.abspath(__file__))
STIM = os.path.join(HERE, "stim")
os.makedirs(STIM, exist_ok=True)

MODEL_PATH = "/home/marty/Documents/aif/vibeyolo/integ/yolo26n/model_int8.onnx"

# Geometry: L8 = 32→16, 3x3, stride 1, pad 1. Input 80x80 → output 80x80.
# Drive an 8x8 output ROI inside the 80x80 grid (input ROI is 10x10x32 with
# the 1-pixel pad ring drawn from the actual ORT-quantized neighbours).
NCH_IN  = 32
NCH_OUT = 16
K       = 3
STRIDE  = 1
ROI_H, ROI_W = 8, 8
OUT_H, OUT_W = ROI_H, ROI_W
PAD_H, PAD_W = ROI_H + 2, ROI_W + 2     # 10x10
# Position inside the 80x80 layer-8-input space.
R, C = 20, 20

# ── Load ONNX initializers (layer-8 conv params) ──
model = onnx.load(MODEL_PATH)
init_by_name = {init.name: numpy_helper.to_array(init) for init in model.graph.initializer}

W_q  = init_by_name['onnx::Conv_1675_quantized'].astype(np.int32)   # (16,32,3,3)
s_w  = float(init_by_name['onnx::Conv_1675_scale'])
zp_w = int(init_by_name['onnx::Conv_1675_zero_point'])
bias = init_by_name['onnx::Conv_1676'].astype(np.float32)            # (16,)
assert zp_w == 0
assert W_q.shape == (NCH_OUT, NCH_IN, K, K), f"Got {W_q.shape}"

sum_w = W_q.reshape(NCH_OUT, -1).sum(axis=1).astype(np.int32)        # (16,)

# ── ORT session with extra outputs ──
TARGET_OUT = '/model.4/m.0/cv1/act/Mul_output_0'              # layer-8 post-SiLU
L8_IN_Q    = '/model.4/Slice_1_output_0_quantized'            # u8 (1,32,80,80)
L8_IN_S    = '/model.4/Slice_1_output_0_scale'
L8_IN_ZP   = '/model.4/Slice_1_output_0_zero_point'

EXTRA_OUTS = [
    (TARGET_OUT, onnx.TensorProto.FLOAT, None),
    (L8_IN_Q,    onnx.TensorProto.UINT8, None),
    (L8_IN_S,    onnx.TensorProto.FLOAT, None),
    (L8_IN_ZP,   onnx.TensorProto.UINT8, None),
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
    outs = sess.run([TARGET_OUT, L8_IN_Q, L8_IN_S, L8_IN_ZP], {ipt_name: img_f32})
    return outs[0], outs[1], float(outs[2]), int(outs[3])

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

# Per-channel output scales (pre- and post-SiLU). C3k2 inner-conv activations
# are typically small. We use the same ±2.0/127 scale as layer_3.
S_OUT_PRE  = 4.0 / 127.0
S_OUT_SILU = 4.0 / 127.0

# RTL pre-shifts dot_acc by ACC_SHIFT bits before requant to keep |acc| within
# fp16 normal range. Compensate by multiplying scale_fp16 by 2**ACC_SHIFT.
ACC_SHIFT = 0  # tiled requant.sv autoscales

def f32_to_fp16(x):  return np.float16(x).view(np.uint16)

def hw_reference(qx_in_u8_full, s_a, zp_a):
    """qx_in_u8_full: (1,32,80,80) uint8 — ORT's L8-input quantization."""
    s_acc = s_a * s_w

    pad_u8 = np.full((NCH_IN, PAD_H, PAD_W), zp_a, dtype=np.int32)
    r0, r1 = R - 1, R - 1 + PAD_H
    c0, c1 = C - 1, C - 1 + PAD_W
    H_FULL = qx_in_u8_full.shape[2]
    W_FULL = qx_in_u8_full.shape[3]
    sr0, sr1 = max(0, r0), min(H_FULL, r1)
    sc0, sc1 = max(0, c0), min(W_FULL, c1)
    pad_u8[:, sr0-r0:sr1-r0, sc0-c0:sc1-c0] = \
        qx_in_u8_full[0, :, sr0:sr1, sc0:sc1].astype(np.int32)

    x_i8 = np.clip(pad_u8 - 128, -128, 127).astype(np.int8)
    bias_eff = bias + s_acc * (128 - zp_a) * sum_w       # (16,)

    out = np.zeros((NCH_OUT, OUT_H, OUT_W), dtype=np.float32)
    for c in range(NCH_OUT):
        for oy in range(OUT_H):
            for ox in range(OUT_W):
                iy = oy * STRIDE
                ix = ox * STRIDE
                window = x_i8[:, iy:iy+K, ix:ix+K].astype(np.int32)   # (32,3,3)
                acc = int((W_q[c] * window).sum())
                pre = acc * s_acc + bias_eff[c]
                sig = 1.0 / (1.0 + np.exp(-pre))
                out[c, oy, ox] = pre * sig
    return out, bias_eff, x_i8, s_acc

# Lane index = (kh*K + kw)*Cin + kc
N_LANE = K * K * NCH_IN     # 288

def pack_window_index(kh, kw, kc):
    return (kh * K + kw) * NCH_IN + kc

W_flat = np.zeros((NCH_OUT, N_LANE), dtype=np.int8)
for c in range(NCH_OUT):
    for kh in range(K):
        for kw in range(K):
            for kc in range(NCH_IN):
                W_flat[c, pack_window_index(kh, kw, kc)] = W_q[c, kc, kh, kw]

with open(os.path.join(STIM, "weights.i8.hex"), "w") as f:
    for c in range(NCH_OUT):
        for k in range(N_LANE):
            f.write(f"{int(W_flat[c, k]) & 0xFF:02x}\n")

samples = [
    ("rand0",     'rand',      0),
    ("rand1",     'rand',      1),
    ("half",      'half',      0),
    ("gradient",  'gradient',  0),
    ("rand_low",  'rand_low',  2),
    ("rand_high", 'rand_high', 3),
]

manifest = {
    "out_h": OUT_H, "out_w": OUT_W, "nch_out": NCH_OUT, "nch_in": NCH_IN,
    "K": K, "stride": STRIDE, "n_lane": N_LANE,
    "roi_h": ROI_H, "roi_w": ROI_W, "pad_h": PAD_H, "pad_w": PAD_W,
    "R": R, "C": C, "s_w": s_w,
    "s_out_pre": S_OUT_PRE, "s_out_silu": S_OUT_SILU,
    "samples": [],
}

for sname, kind, seed in samples:
    img = make_pixel_input(kind, seed=seed)
    ort_out_full, qx_in_u8_full, s_a, zp_a = run_ort(img)
    if s_a == 0.0:
        s_a = 1.0
    ort_roi = ort_out_full[0, :, R:R+OUT_H, C:C+OUT_W]      # (16,8,8)
    hw_out, bias_eff, x_i8, s_acc = hw_reference(qx_in_u8_full, s_a, zp_a)

    scale_fp16 = np.array([f32_to_fp16((s_acc * (1 << ACC_SHIFT)) / S_OUT_PRE)] * NCH_OUT, dtype=np.uint16)
    bias_fp16  = np.array([f32_to_fp16(bias_eff[c] / S_OUT_PRE) for c in range(NCH_OUT)],
                          dtype=np.uint16)

    inp_path = os.path.join(STIM, f"{sname}.input_i8.hex")
    with open(inp_path, "w") as f:
        for h in range(PAD_H):
            for w in range(PAD_W):
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
