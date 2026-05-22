#!/usr/bin/env python3
# Copyright (c) 2026 vibeyolo
# SPDX-License-Identifier: Apache-2.0
#
# extract.py — Stimulus + reference for layer_18 (/model.6/m.0/m/m.1/cv2).
# 3x3, 32->32, stride 1, pad 1. Second conv of the second nested Bottleneck.
# Bottleneck.add=True ⇒ output = bottleneck_input + silu(cv2(...)).
#
# Conv input plane:   /model.6/m.0/m/m.0/cv1/act/Mul_output_0  (= L15 output)
# Residual plane:     /model.6/m.0/cv1/act/Mul_output_0        (= L13 output, the bottleneck input)
# Target output:      /model.6/m.0/m/m.0/Add_output_0          (post-Add, fp32)

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
K       = 3
STRIDE  = 1
ROI_H, ROI_W = 8, 8
OUT_H, OUT_W = ROI_H, ROI_W
PAD_H, PAD_W = ROI_H + 2, ROI_W + 2
R, C = 12, 12

model = onnx.load(MODEL_PATH)
inits = {i.name: numpy_helper.to_array(i) for i in model.graph.initializer}

W_q  = inits['onnx::Conv_1702_quantized'].astype(np.int32)
s_w  = float(inits['onnx::Conv_1702_scale'])
zp_w = int(inits['onnx::Conv_1702_zero_point'])
bias = inits['onnx::Conv_1703'].astype(np.float32)
assert zp_w == 0
assert W_q.shape == (NCH_OUT, NCH_IN, K, K), f"got {W_q.shape}"
sum_w = W_q.reshape(NCH_OUT, -1).sum(axis=1).astype(np.int32)

TARGET_OUT = '/model.6/m.0/m/m.1/Add_output_0'
IN_Q  = '/model.6/m.0/m/m.1/cv1/act/Mul_output_0_quantized'
IN_S  = '/model.6/m.0/m/m.1/cv1/act/Mul_output_0_scale'
IN_ZP = '/model.6/m.0/m/m.1/cv1/act/Mul_output_0_zero_point'
RES_Q  = '/model.6/m.0/m/m.0/Add_output_0_quantized'
RES_S  = '/model.6/m.0/m/m.0/Add_output_0_scale'
RES_ZP = '/model.6/m.0/m/m.0/Add_output_0_zero_point'

mod_model = onnx.load(MODEL_PATH)
existing = {o.name for o in mod_model.graph.output}
for nm, tp in [(TARGET_OUT, onnx.TensorProto.FLOAT),
               (IN_Q, onnx.TensorProto.UINT8),(IN_S, onnx.TensorProto.FLOAT),(IN_ZP, onnx.TensorProto.UINT8),
               (RES_Q, onnx.TensorProto.UINT8),(RES_S, onnx.TensorProto.FLOAT),(RES_ZP, onnx.TensorProto.UINT8)]:
    if nm not in existing:
        mod_model.graph.output.append(onnx.helper.make_tensor_value_info(nm, tp, None))
TMP = os.path.join(HERE, "_model_with_intermediate.onnx")
onnx.save(mod_model, TMP)

so = ort.SessionOptions(); so.log_severity_level = 3
sess = ort.InferenceSession(TMP, sess_options=so, providers=['CPUExecutionProvider'])
ipt = sess.get_inputs()[0].name

def run_ort(img):
    o = sess.run([TARGET_OUT, IN_Q, IN_S, IN_ZP, RES_Q, RES_S, RES_ZP], {ipt: img})
    return o[0], o[1], float(o[2]), int(o[3]), o[4], float(o[5]), int(o[6])

def make_input(kind, seed=0):
    img = np.zeros((1,3,640,640), dtype=np.float32)
    rng = np.random.RandomState(seed)
    if kind == 'rand':       img[0] = rng.uniform(0,1,(3,640,640)).astype(np.float32)
    elif kind == 'half':     img[0] = 0.5
    elif kind == 'gradient':
        gx = np.tile(np.linspace(0,1,640,dtype=np.float32),(640,1))
        img[0,0]=gx; img[0,1]=gx.T; img[0,2]=0.5*(gx+gx.T)
    elif kind == 'rand_low':  img[0] = rng.uniform(0.15,0.80,(3,640,640)).astype(np.float32)
    elif kind == 'rand_high': img[0] = rng.uniform(0.5,1.0,(3,640,640)).astype(np.float32)
    else: raise ValueError(kind)
    return img

def f32_to_fp16(x): return np.float16(x).view(np.uint16)

def hw_reference(qx_u8_full, s_a, zp_a, qr_u8_full, s_r, zp_r, S_OUT_SILU):
    """Mimics conv_layer w/ RESIDUAL: pre-add silu_y is on the same int8 grid
       as the post-add output (both quantised by S_OUT_SILU)."""
    s_acc = s_a * s_w
    pad_u8 = np.full((NCH_IN, PAD_H, PAD_W), zp_a, dtype=np.int32)
    r0, r1 = R - 1, R - 1 + PAD_H
    c0, c1 = C - 1, C - 1 + PAD_W
    H_FULL, W_FULL = qx_u8_full.shape[2], qx_u8_full.shape[3]
    sr0, sr1 = max(0,r0), min(H_FULL,r1); sc0, sc1 = max(0,c0), min(W_FULL,c1)
    pad_u8[:, sr0-r0:sr1-r0, sc0-c0:sc1-c0] = qx_u8_full[0,:,sr0:sr1,sc0:sc1].astype(np.int32)
    x_i8 = np.clip(pad_u8 - 128, -128, 127).astype(np.int8)
    bias_eff = bias + s_acc * (128 - zp_a) * sum_w

    # Residual ROI (no padding, K=1 effectively for skip)
    r_u8 = qr_u8_full[0, :, R:R+OUT_H, C:C+OUT_W].astype(np.int32)
    r_i8 = np.clip(r_u8 - 128, -128, 127).astype(np.int8)
    # Residual fp32 value (this is what we want to add)
    r_f32 = (r_u8 - zp_r).astype(np.float32) * s_r   # (NCH_OUT, OUT_H, OUT_W)

    pre_silu_min, pre_silu_max = 1e30, -1e30
    silu_out = np.zeros((NCH_OUT, OUT_H, OUT_W), dtype=np.float32)
    for c in range(NCH_OUT):
        for oy in range(OUT_H):
            for ox in range(OUT_W):
                iy = oy * STRIDE; ix = ox * STRIDE
                window = x_i8[:, iy:iy+K, ix:ix+K].astype(np.int32)
                acc = int((W_q[c] * window).sum())
                pre = acc * s_acc + bias_eff[c]
                pre_silu_min = min(pre_silu_min, pre); pre_silu_max = max(pre_silu_max, pre)
                sig = 1.0/(1.0 + np.exp(-pre))
                silu_out[c,oy,ox] = pre * sig

    # Pre-quantised SiLU (i8 grid), then add residual
    silu_i8 = np.clip(np.round(silu_out / S_OUT_SILU), -128, 127).astype(np.int8)
    add_f32 = silu_out + r_f32
    # Re-quantise to same S_OUT_SILU grid
    add_i8  = np.clip(np.round(add_f32 / S_OUT_SILU), -128, 127).astype(np.int8)
    add_dequant = add_i8.astype(np.float32) * S_OUT_SILU
    return add_dequant, silu_out, silu_i8, r_i8, r_f32, bias_eff, s_acc, (pre_silu_min, pre_silu_max)

samples = [
    ("rand0",'rand',0),("rand1",'rand',1),("rand2",'rand',7),("rand3",'rand',13),
    ("rand_low",'rand_low',2),("rand_high",'rand_high',3),
    ("half",'half',0),("gradient",'gradient',0),
]

# Pass 1: pick scales from observed ranges.
pre_lo, pre_hi, silu_hi, add_hi = 0.0, 0.0, 0.0, 0.0
cache = []
S_TMP = 4.0/127.0
for sname, kind, seed in samples:
    img = make_input(kind, seed=seed)
    o = run_ort(img)
    ort_out, qx_u8, s_a, zp_a, qr_u8, s_r, zp_r = o
    if s_a == 0.0: s_a = 1.0
    if s_r == 0.0: s_r = 1.0
    _, silu, _, _, rf, _, _, (pmin, pmax) = hw_reference(qx_u8, s_a, zp_a, qr_u8, s_r, zp_r, S_TMP)
    pre_lo = min(pre_lo, pmin); pre_hi = max(pre_hi, pmax)
    silu_hi = max(silu_hi, float(np.max(np.abs(silu))))
    roi = ort_out[0,:,R:R+OUT_H,C:C+OUT_W]
    add_hi = max(add_hi, float(np.max(np.abs(roi))))
    cache.append((sname, ort_out, qx_u8, s_a, zp_a, qr_u8, s_r, zp_r))

print(f"observed pre-SiLU [{pre_lo:.3f},{pre_hi:.3f}], |silu_max|~{silu_hi:.3f}, |add_max|~{add_hi:.3f}")
def pick(amp):
    raw = round(amp * 1.2 * 1000.0) / 1000.0
    return max(raw/127.0, 2.0/127.0)
S_OUT_PRE  = pick(max(abs(pre_lo), abs(pre_hi)))
# Use larger of silu/add range — both share the post-add grid.
S_OUT_SILU = pick(max(silu_hi, add_hi))
print(f"chose S_OUT_PRE={S_OUT_PRE:.6f} (={S_OUT_PRE*127:.3f}/127), "
      f"S_OUT_SILU={S_OUT_SILU:.6f} (={S_OUT_SILU*127:.3f}/127)")

N_LANE_FULL = K * K * NCH_IN
W_flat = np.zeros((NCH_OUT, N_LANE_FULL), dtype=np.int8)
for c in range(NCH_OUT):
    for kh in range(K):
        for kw in range(K):
            for kc in range(NCH_IN):
                W_flat[c, (kh*K+kw)*NCH_IN + kc] = W_q[c, kc, kh, kw]
with open(os.path.join(STIM, "weights.i8.hex"), "w") as f:
    for c in range(NCH_OUT):
        for k in range(N_LANE_FULL):
            f.write(f"{int(W_flat[c,k]) & 0xFF:02x}\n")

manifest = {
    "out_h": OUT_H, "out_w": OUT_W, "nch_out": NCH_OUT, "nch_in": NCH_IN,
    "K": K, "stride": STRIDE, "n_lane_full": N_LANE_FULL,
    "roi_h": ROI_H, "roi_w": ROI_W, "pad_h": PAD_H, "pad_w": PAD_W,
    "R": R, "C": C, "s_w": s_w,
    "s_out_pre": S_OUT_PRE, "s_out_silu": S_OUT_SILU,
    "samples": [],
}

for (sname, ort_out, qx_u8, s_a, zp_a, qr_u8, s_r, zp_r) in cache:
    ort_roi = ort_out[0,:,R:R+OUT_H, C:C+OUT_W]
    add_dq, silu, silu_i8, r_i8, r_f32, bias_eff, s_acc, _ = hw_reference(
        qx_u8, s_a, zp_a, qr_u8, s_r, zp_r, S_OUT_SILU)

    scale_fp16 = np.array([f32_to_fp16(s_acc / S_OUT_PRE)]*NCH_OUT, dtype=np.uint16)
    bias_fp16  = np.array([f32_to_fp16(bias_eff[c]/S_OUT_PRE) for c in range(NCH_OUT)], dtype=np.uint16)

    # Residual side-band (per-pixel, per-channel; constant per cit-sweep)
    # r_scale = s_r  (folds u8 i8 shift via bias)
    # r_bias_chan_const = (128 - zp_r) * s_r / S_OUT_SILU
    r_scale_fp16 = np.array([f32_to_fp16(s_r)]*NCH_OUT, dtype=np.uint16)
    r_bias_const = (128 - zp_r) * s_r / S_OUT_SILU
    r_bias_fp16  = np.array([f32_to_fp16(r_bias_const)]*NCH_OUT, dtype=np.uint16)

    # Dump input ROI (NCH_IN pad halo) and residual ROI (NCH_OUT, no halo)
    with open(os.path.join(STIM, f"{sname}.input_i8.hex"), "w") as f:
        # x_i8 was returned implicitly via hw_reference; recompute
        pad_u8 = np.full((NCH_IN, PAD_H, PAD_W), zp_a, dtype=np.int32)
        r0, r1 = R - 1, R - 1 + PAD_H; c0, c1 = C - 1, C - 1 + PAD_W
        H_FULL, W_FULL = qx_u8.shape[2], qx_u8.shape[3]
        sr0, sr1 = max(0,r0), min(H_FULL,r1); sc0, sc1 = max(0,c0), min(W_FULL,c1)
        pad_u8[:, sr0-r0:sr1-r0, sc0-c0:sc1-c0] = qx_u8[0,:,sr0:sr1,sc0:sc1].astype(np.int32)
        x_i8 = np.clip(pad_u8 - 128, -128, 127).astype(np.int8)
        for h in range(PAD_H):
            for w in range(PAD_W):
                for kc in range(NCH_IN):
                    f.write(f"{int(x_i8[kc,h,w]) & 0xFF:02x}\n")
    with open(os.path.join(STIM, f"{sname}.residual_i8.hex"), "w") as f:
        for oy in range(OUT_H):
            for ox in range(OUT_W):
                for c in range(NCH_OUT):
                    f.write(f"{int(r_i8[c,oy,ox]) & 0xFF:02x}\n")
    with open(os.path.join(STIM, f"{sname}.scale_fp16.hex"), "w") as f:
        for c in range(NCH_OUT): f.write(f"{int(scale_fp16[c]):04x}\n")
    with open(os.path.join(STIM, f"{sname}.bias_fp16.hex"), "w") as f:
        for c in range(NCH_OUT): f.write(f"{int(bias_fp16[c]):04x}\n")
    with open(os.path.join(STIM, f"{sname}.r_scale_fp16.hex"), "w") as f:
        for c in range(NCH_OUT): f.write(f"{int(r_scale_fp16[c]):04x}\n")
    with open(os.path.join(STIM, f"{sname}.r_bias_fp16.hex"), "w") as f:
        for c in range(NCH_OUT): f.write(f"{int(r_bias_fp16[c]):04x}\n")
    with open(os.path.join(STIM, f"{sname}.ref_ort.f32.hex"), "w") as f:
        for oy in range(OUT_H):
            for ox in range(OUT_W):
                for c in range(NCH_OUT):
                    bits = np.float32(ort_roi[c,oy,ox]).view(np.uint32)
                    f.write(f"{int(bits):08x}\n")
    with open(os.path.join(STIM, f"{sname}.ref_hw.f32.hex"), "w") as f:
        for oy in range(OUT_H):
            for ox in range(OUT_W):
                for c in range(NCH_OUT):
                    bits = np.float32(add_dq[c,oy,ox]).view(np.uint32)
                    f.write(f"{int(bits):08x}\n")

    diff = ort_roi - add_dq
    mxe = float(np.max(np.abs(diff))); mae = float(np.mean(np.abs(diff)))
    out_range = float(ort_roi.max() - ort_roi.min())
    cos = float(np.sum(ort_roi*add_dq)/(np.linalg.norm(ort_roi)*np.linalg.norm(add_dq)+1e-12))
    print(f"[{sname}] s_a={s_a:.5f} zp_a={zp_a} s_r={s_r:.5f} zp_r={zp_r} "
          f"ORT vs HW: max_abs={mxe:.4f} mae={mae:.4f} cos={cos:.6f}")
    manifest["samples"].append({"name":sname,"s_a":s_a,"zp_a":zp_a,"s_r":s_r,"zp_r":zp_r,
        "s_acc":s_acc,"ort_vs_hw_max_abs":mxe,"ort_vs_hw_mae":mae,"ort_vs_hw_cos":cos,
        "out_range":out_range})

with open(os.path.join(STIM, "manifest.json"), "w") as f:
    json.dump(manifest, f, indent=2)
print(f"\nWrote stim+ref to {STIM}")
