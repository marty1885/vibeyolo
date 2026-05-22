#!/usr/bin/env python3
# Copyright (c) 2026 vibeyolo
# SPDX-License-Identifier: Apache-2.0
#
# extract.py — Generate stimulus + reference for the layer_11 integration test.
#
# Slice: YOLO26n /model.5/conv Conv 3x3, Cin=128, Cout=128, stride=2, pad=1.
# Input plane is 80x80 (from /model.4/cv2/act/Mul_output_0), output 40x40.
#
#   /model.4/cv2/act/Mul_output_0
#    -> DynamicQuantizeLinear   (u8, s_a, zp_a)
#    -> ConvInteger             (3x3, stride 2, pad 1, 128 -> 128)
#    -> Cast i32 -> fp32
#    -> Mul (s_a * s_w)
#    -> Add bias                (fp32)
#    -> Sigmoid -> Mul          (SiLU)
#    -> /model.5/act/Mul_output_0
#
# Strategy (mirrors integ/layer_6/extract.py):
#   * Take an 18x18x128 input ROI on the 80x80 plane (pad-1 ring from real
#     neighbour pixels). Output ROI is 8x8x128 on the 40x40 plane.
#   * u8 -> i8 fold: `i8 = u8 - 128`, push the zp into the per-channel bias as
#       bias_eff[c] = bias[c] + s_a*s_w*(128 - zp_a)*sum_w[c]

import os, json
import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper

HERE = os.path.dirname(os.path.abspath(__file__))
STIM = os.path.join(HERE, "stim")
os.makedirs(STIM, exist_ok=True)

MODEL_PATH = "/home/marty/Documents/aif/vibeyolo/integ/yolo26n/model_int8.onnx"

# Geometry
NCH_IN  = 128
NCH_OUT = 128
K       = 3
STRIDE  = 2
ROI_H, ROI_W = 16, 16        # output 8x8
OUT_H, OUT_W = ROI_H // STRIDE, ROI_W // STRIDE
PAD_H, PAD_W = ROI_H + 2, ROI_W + 2   # 18x18 (pad ring of 1)
# ROI top-left on the 80x80 layer-11 input plane. Even (R,C) so output ROI
# starts at integer (R//2, C//2) in 40x40 output space.
R, C = 16, 16

# ONNX initializers (L11 conv params)
model = onnx.load(MODEL_PATH)
init_by_name = {init.name: numpy_helper.to_array(init) for init in model.graph.initializer}

W_q  = init_by_name['onnx::Conv_1684_quantized'].astype(np.int32)   # (128,128,3,3)
s_w  = float(init_by_name['onnx::Conv_1684_scale'])
zp_w = int(init_by_name['onnx::Conv_1684_zero_point'])
bias = init_by_name['onnx::Conv_1685'].astype(np.float32)           # (128,)
assert zp_w == 0
assert W_q.shape == (NCH_OUT, NCH_IN, K, K), f"Got {W_q.shape}"

sum_w = W_q.reshape(NCH_OUT, -1).sum(axis=1).astype(np.int32)        # (128,)

# ORT session with intermediate outputs
TARGET_OUT = '/model.5/act/Mul_output_0'                       # fp32 reference
IN_Q       = '/model.4/cv2/act/Mul_output_0_quantized'         # u8 (1,128,80,80)
IN_S       = '/model.4/cv2/act/Mul_output_0_scale'             # fp32 scalar
IN_ZP      = '/model.4/cv2/act/Mul_output_0_zero_point'        # u8

EXTRA_OUTS = [
    (TARGET_OUT, onnx.TensorProto.FLOAT, None),
    (IN_Q,       onnx.TensorProto.UINT8, None),
    (IN_S,       onnx.TensorProto.FLOAT, None),
    (IN_ZP,      onnx.TensorProto.UINT8, None),
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
    outs = sess.run([TARGET_OUT, IN_Q, IN_S, IN_ZP], {ipt_name: img_f32})
    return outs[0], outs[1], float(outs[2]), int(outs[3])

# Stimuli
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

# Output scale for SiLU LUT.
# Empirical pre-SiLU range on normalised 0..1 pixel inputs is ~ -0.3..3.0 and
# post-SiLU range ~ -0.28..3.0. We pick S_OUT_PRE = S_OUT_SILU = 4/127 (~0.0315)
# to span +/-4 with LUT step ~0.0315 -- about 1% of the output dynamic range,
# i.e. LUT quantisation is the dominant DUT-vs-ORT error and stays small.
S_OUT_PRE  = 4.0 / 127.0
S_OUT_SILU = 4.0 / 127.0

def f32_to_fp16(x): return np.float16(x).view(np.uint16)

# HW reference (fp32 model of SV-equivalent pipeline)
def hw_reference(qx_u8_full, s_a, zp_a):
    """qx_u8_full: (1,128,80,80) uint8 ORT-quantised L11 input."""
    s_acc = s_a * s_w
    # Pull 18x18x128 padded ROI.
    pad_u8 = np.full((NCH_IN, PAD_H, PAD_W), zp_a, dtype=np.int32)
    r0, r1 = R - 1, R - 1 + PAD_H
    c0, c1 = C - 1, C - 1 + PAD_W
    H_FULL = qx_u8_full.shape[2]
    W_FULL = qx_u8_full.shape[3]
    sr0, sr1 = max(0, r0), min(H_FULL, r1)
    sc0, sc1 = max(0, c0), min(W_FULL, c1)
    pad_u8[:, sr0-r0:sr1-r0, sc0-c0:sc1-c0] = qx_u8_full[0, :, sr0:sr1, sc0:sc1].astype(np.int32)

    x_i8 = np.clip(pad_u8 - 128, -128, 127).astype(np.int8)
    bias_eff = bias + s_acc * (128 - zp_a) * sum_w   # (128,)

    out = np.zeros((NCH_OUT, OUT_H, OUT_W), dtype=np.float32)
    max_abs_acc = 0
    for c in range(NCH_OUT):
        for oy in range(OUT_H):
            for ox in range(OUT_W):
                iy = oy * STRIDE
                ix = ox * STRIDE
                window = x_i8[:, iy:iy+K, ix:ix+K].astype(np.int32)  # (128,3,3)
                acc = int((W_q[c] * window).sum())
                if abs(acc) > max_abs_acc: max_abs_acc = abs(acc)
                pre = acc * s_acc + bias_eff[c]
                sig = 1.0 / (1.0 + np.exp(-pre))
                out[c, oy, ox] = pre * sig
    return out, bias_eff, x_i8, s_acc, max_abs_acc

# Pack weights flat: (Cout, N_LANE), lane = (kh*K + kw)*Cin + kc
N_LANE = K * K * NCH_IN   # 1152

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

# ACC_SHIFT: legacy workaround for the old fully-unrolled layer (fp16 saturation
# on the i32->fp16 step). The new tiled requant.sv autoscales i32 inputs
# internally, so layers no longer need to pre-multiply scale_fp16. Kept at 0
# for forward compatibility / documentation.
ACC_SHIFT = 0

manifest = {
    "out_h": OUT_H, "out_w": OUT_W, "nch_out": NCH_OUT, "nch_in": NCH_IN,
    "K": K, "stride": STRIDE, "n_lane": N_LANE,
    "roi_h": ROI_H, "roi_w": ROI_W, "pad_h": PAD_H, "pad_w": PAD_W,
    "R": R, "C": C, "s_w": s_w,
    "s_out_pre": S_OUT_PRE, "s_out_silu": S_OUT_SILU,
    "acc_shift": ACC_SHIFT,
    "samples": [],
}

global_max_acc = 0
for sname, kind, seed in samples:
    img = make_pixel_input(kind, seed=seed)
    ort_out_full, qx_u8_full, s_a, zp_a = run_ort(img)
    if s_a == 0.0:
        s_a = 1.0
    ort_roi = ort_out_full[0, :, R//STRIDE:R//STRIDE+OUT_H, C//STRIDE:C//STRIDE+OUT_W]
    hw_out, bias_eff, x_i8, s_acc, max_acc = hw_reference(qx_u8_full, s_a, zp_a)
    global_max_acc = max(global_max_acc, max_acc)

    scale_fp16 = np.array([f32_to_fp16((s_acc / S_OUT_PRE) * (1 << ACC_SHIFT))] * NCH_OUT,
                          dtype=np.uint16)
    bias_fp16  = np.array([f32_to_fp16(bias_eff[c] / S_OUT_PRE)
                           for c in range(NCH_OUT)], dtype=np.uint16)

    # Input i8 dump (PAD_H * PAD_W * NCH_IN), order h,w,kc.
    with open(os.path.join(STIM, f"{sname}.input_i8.hex"), "w") as f:
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
    print(f"[{sname}] s_a={s_a:.5f} zp_a={zp_a} "
          f"pre-SiLU range~{ort_roi.min():.2f}..{ort_roi.max():.2f} "
          f"max|acc|={max_acc} (>>{ACC_SHIFT} = {max_acc >> ACC_SHIFT}) "
          f"ORT vs HW-ref: max_abs={mxe:.4f} mae={mae:.4f} "
          f"out_range={out_range:.3f} cos={cos:.6f}")

    manifest["samples"].append({
        "name": sname, "s_a": s_a, "zp_a": zp_a, "s_acc": s_acc,
        "max_abs_acc": int(max_acc),
        "ort_vs_hw_max_abs": mxe, "ort_vs_hw_mae": mae,
        "ort_vs_hw_cos": cos, "out_range": out_range,
    })

print(f"\nglobal max|acc| = {global_max_acc} ; after >>{ACC_SHIFT} = {global_max_acc >> ACC_SHIFT} (fp16 max = 65504)")

with open(os.path.join(STIM, "manifest.json"), "w") as f:
    json.dump(manifest, f, indent=2)

# SiLU LUT diagnostic dump
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
