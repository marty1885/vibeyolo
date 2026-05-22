#!/usr/bin/env python3
# Copyright (c) 2026 vibeyolo
# SPDX-License-Identifier: Apache-2.0
#
# golden.py — Stimulus + reference for the tiled layer_2 integration test
# (YOLO26n /model.2/cv1 Conv-BN-SiLU, 1x1, 32 -> 32, stride 1, pad 0).
#
# Mirrors the conv_layer-shim pattern used by integ/layer_1 + integ/layer_11.

import os, json
import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper

HERE = os.path.dirname(os.path.abspath(__file__))
STIM = os.path.join(HERE, "stim")
os.makedirs(STIM, exist_ok=True)

MODEL_PATH = "/home/marty/Documents/aif/vibeyolo/integ/yolo26n/model_int8.onnx"

NCH_IN  = 32
NCH_OUT = 32
K       = 1
STRIDE  = 1
ROI_H, ROI_W = 8, 8
OUT_H, OUT_W = ROI_H, ROI_W
PAD_H, PAD_W = ROI_H, ROI_W   # K=1 -> no pad ring
R, C = 32, 32

model = onnx.load(MODEL_PATH)
init_by_name = {init.name: numpy_helper.to_array(init) for init in model.graph.initializer}
W_q  = init_by_name['onnx::Conv_1657_quantized'].astype(np.int32)   # (32,32,1,1)
s_w  = float(init_by_name['onnx::Conv_1657_scale'])
zp_w = int(init_by_name['onnx::Conv_1657_zero_point'])
bias = init_by_name['onnx::Conv_1658'].astype(np.float32)
assert zp_w == 0
assert W_q.shape == (NCH_OUT, NCH_IN, K, K)
sum_w = W_q.reshape(NCH_OUT, -1).sum(axis=1).astype(np.int32)

TARGET_OUT = '/model.2/cv1/act/Mul_output_0'
L1_OUT_Q   = '/model.1/act/Mul_output_0_quantized'
L1_OUT_S   = '/model.1/act/Mul_output_0_scale'
L1_OUT_ZP  = '/model.1/act/Mul_output_0_zero_point'
EXTRA_OUTS = [
    (TARGET_OUT, onnx.TensorProto.FLOAT, None),
    (L1_OUT_Q,   onnx.TensorProto.UINT8, None),
    (L1_OUT_S,   onnx.TensorProto.FLOAT, None),
    (L1_OUT_ZP,  onnx.TensorProto.UINT8, None),
]
mod = onnx.load(MODEL_PATH)
existing = [o.name for o in mod.graph.output]
for nm, tp, _ in EXTRA_OUTS:
    if nm not in existing:
        mod.graph.output.append(onnx.helper.make_tensor_value_info(nm, tp, None))
TMP_PATH = os.path.join(HERE, "_model_with_intermediate.onnx")
onnx.save(mod, TMP_PATH)
so = ort.SessionOptions(); so.log_severity_level = 3
sess = ort.InferenceSession(TMP_PATH, sess_options=so, providers=['CPUExecutionProvider'])
ipt_name = sess.get_inputs()[0].name

def run_ort(img):
    o = sess.run([TARGET_OUT, L1_OUT_Q, L1_OUT_S, L1_OUT_ZP], {ipt_name: img})
    return o[0], o[1], float(o[2]), int(o[3])

def make_pixel_input(kind, seed=0):
    img = np.zeros((1, 3, 640, 640), dtype=np.float32)
    rng = np.random.RandomState(seed)
    if kind == 'rand':       img[0] = rng.uniform(0.0, 1.0, size=(3, 640, 640)).astype(np.float32)
    elif kind == 'half':     img[0] = 0.5
    elif kind == 'gradient':
        gx = np.tile(np.linspace(0.0, 1.0, 640, dtype=np.float32), (640, 1))
        img[0, 0] = gx; img[0, 1] = gx.T; img[0, 2] = 0.5 * (gx + gx.T)
    elif kind == 'rand_low': img[0] = rng.uniform(0.15, 0.80, size=(3, 640, 640)).astype(np.float32)
    elif kind == 'rand_high':img[0] = rng.uniform(0.5, 1.0, size=(3, 640, 640)).astype(np.float32)
    else: raise ValueError(kind)
    return img

# Output scales sized to fit the post-SiLU range across all tiles.
S_OUT_PRE  = 24.0 / 127.0
S_OUT_SILU = 24.0 / 127.0

def f32_to_fp16(x): return np.float16(x).view(np.uint16)

def hw_reference(qx_u8_full, s_a, zp_a):
    s_acc = s_a * s_w
    x_u8 = qx_u8_full[0, :, R:R+ROI_H, C:C+ROI_W].astype(np.int32)
    x_i8 = np.clip(x_u8 - 128, -128, 127).astype(np.int8)
    bias_eff = bias + s_acc * (128 - zp_a) * sum_w
    out = np.zeros((NCH_OUT, OUT_H, OUT_W), dtype=np.float32)
    for c in range(NCH_OUT):
        for oy in range(OUT_H):
            for ox in range(OUT_W):
                acc = int((W_q[c, :, 0, 0] * x_i8[:, oy, ox].astype(np.int32)).sum())
                pre = acc * s_acc + bias_eff[c]
                sig = 1.0 / (1.0 + np.exp(-pre))
                out[c, oy, ox] = pre * sig
    return out, bias_eff, x_i8, s_acc

# Pack weights flat: (Cout, N_LANE) with N_LANE = K*K*NCH_IN = NCH_IN
N_LANE = K * K * NCH_IN
W_flat = np.zeros((NCH_OUT, N_LANE), dtype=np.int8)
for c in range(NCH_OUT):
    for kc in range(NCH_IN):
        W_flat[c, kc] = W_q[c, kc, 0, 0]

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
    ort_full, qx_u8, s_a, zp_a = run_ort(img)
    if s_a == 0.0: s_a = 1.0
    ort_roi = ort_full[0, :, R:R+OUT_H, C:C+OUT_W]
    hw_out, bias_eff, x_i8, s_acc = hw_reference(qx_u8, s_a, zp_a)
    scale_fp16 = np.array([f32_to_fp16(s_acc / S_OUT_PRE)] * NCH_OUT, dtype=np.uint16)
    bias_fp16  = np.array([f32_to_fp16(bias_eff[c] / S_OUT_PRE) for c in range(NCH_OUT)], dtype=np.uint16)

    with open(os.path.join(STIM, f"{sname}.input_i8.hex"), "w") as f:
        for h in range(PAD_H):
            for w in range(PAD_W):
                for kc in range(NCH_IN):
                    f.write(f"{int(x_i8[kc, h, w]) & 0xFF:02x}\n")
    with open(os.path.join(STIM, f"{sname}.scale_fp16.hex"), "w") as f:
        for c in range(NCH_OUT): f.write(f"{int(scale_fp16[c]):04x}\n")
    with open(os.path.join(STIM, f"{sname}.bias_fp16.hex"), "w") as f:
        for c in range(NCH_OUT): f.write(f"{int(bias_fp16[c]):04x}\n")
    with open(os.path.join(STIM, f"{sname}.ref_ort.f32.hex"), "w") as f:
        for oy in range(OUT_H):
            for ox in range(OUT_W):
                for c in range(NCH_OUT):
                    f.write(f"{int(np.float32(ort_roi[c,oy,ox]).view(np.uint32)):08x}\n")
    with open(os.path.join(STIM, f"{sname}.ref_hw.f32.hex"), "w") as f:
        for oy in range(OUT_H):
            for ox in range(OUT_W):
                for c in range(NCH_OUT):
                    f.write(f"{int(np.float32(hw_out[c,oy,ox]).view(np.uint32)):08x}\n")
    diff = ort_roi - hw_out
    mae = float(np.mean(np.abs(diff))); mxe = float(np.max(np.abs(diff)))
    out_range = float(ort_roi.max() - ort_roi.min())
    cos = float(np.sum(ort_roi * hw_out) / (np.linalg.norm(ort_roi) * np.linalg.norm(hw_out) + 1e-12))
    print(f"[{sname}] s_a={s_a:.5f} zp_a={zp_a} range~{ort_roi.min():.2f}..{ort_roi.max():.2f} "
          f"hw vs ort: mae={mae:.4f} max={mxe:.4f} cos={cos:.6f}")
    manifest["samples"].append({"name": sname, "s_a": s_a, "zp_a": zp_a, "s_acc": s_acc,
                                "ort_vs_hw_max_abs": mxe, "ort_vs_hw_mae": mae,
                                "ort_vs_hw_cos": cos, "out_range": out_range})

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
        q = max(-128, min(127, int(round(s / out_scale))))
        lut[idx] = q
    return lut

lut = silu_lut_bytes(S_OUT_PRE, S_OUT_SILU)
with open(os.path.join(STIM, "silu_lut.i8.hex"), "w") as f:
    for v in lut: f.write(f"{int(v)&0xFF:02x}\n")

print(f"Wrote stim + ref to {STIM}")
