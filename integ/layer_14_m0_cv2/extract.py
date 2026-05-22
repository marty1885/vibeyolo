#!/usr/bin/env python3
# Copyright (c) 2026 vibeyolo
# SPDX-License-Identifier: Apache-2.0
#
# extract.py — Generate stimulus + reference for the layer_14_m0_cv2 integ test.
#
# Slice: YOLO26n /model.6/m.0/cv2 Conv-BN-SiLU — second conv of the C3k
# wrapper at /model.6/m.0. K=1, 64→32, stride 1, pad 0, 40×40 plane. Same
# input plane as L13 (cv1 sibling).
#
# Pipeline through the model:
#   /model.6/Slice_1_output_0_quantized (u8, s_a, zp_a, 1×64×40×40)
#    → ConvInteger (32×64×1×1, zp_w = 0)
#    → ×(s_a * s_w) → +bias → SiLU
#    → /model.6/m.0/cv2/act/Mul_output_0
#
# Non-residual layer (cv2 in c3k=True wrapper feeds Concat, no Add).

import os, json
import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper

HERE = os.path.dirname(os.path.abspath(__file__))
STIM = os.path.join(HERE, "stim")
os.makedirs(STIM, exist_ok=True)

MODEL_PATH = "/home/marty/Documents/aif/vibeyolo/integ/yolo26n/model_int8.onnx"

NCH_IN  = 64
NCH_OUT = 32
K       = 1
STRIDE  = 1
ROI_H, ROI_W = 8, 8
OUT_H, OUT_W = ROI_H, ROI_W
PAD_H, PAD_W = ROI_H, ROI_W  # K=1 → no halo
R, C = 12, 12

model = onnx.load(MODEL_PATH)
inits = {i.name: numpy_helper.to_array(i) for i in model.graph.initializer}

W_q  = inits['onnx::Conv_1705_quantized'].astype(np.int32)
s_w  = float(inits['onnx::Conv_1705_scale'])
zp_w = int(inits['onnx::Conv_1705_zero_point'])
bias = inits['onnx::Conv_1706'].astype(np.float32)
assert zp_w == 0
assert W_q.shape == (NCH_OUT, NCH_IN, K, K), f"got {W_q.shape}"

sum_w = W_q.reshape(NCH_OUT, -1).sum(axis=1).astype(np.int32)

TARGET_OUT = '/model.6/m.0/cv2/act/Mul_output_0'
IN_Q  = '/model.6/Slice_1_output_0_quantized'
IN_S  = '/model.6/Slice_1_output_0_scale'
IN_ZP = '/model.6/Slice_1_output_0_zero_point'

mod_model = onnx.load(MODEL_PATH)
existing = {o.name for o in mod_model.graph.output}
for nm, tp in [(TARGET_OUT, onnx.TensorProto.FLOAT),
               (IN_Q, onnx.TensorProto.UINT8),
               (IN_S, onnx.TensorProto.FLOAT),
               (IN_ZP, onnx.TensorProto.UINT8)]:
    if nm not in existing:
        mod_model.graph.output.append(onnx.helper.make_tensor_value_info(nm, tp, None))
TMP = os.path.join(HERE, "_model_with_intermediate.onnx")
onnx.save(mod_model, TMP)

so = ort.SessionOptions(); so.log_severity_level = 3
sess = ort.InferenceSession(TMP, sess_options=so, providers=['CPUExecutionProvider'])
ipt = sess.get_inputs()[0].name

def run_ort(img):
    o = sess.run([TARGET_OUT, IN_Q, IN_S, IN_ZP], {ipt: img})
    return o[0], o[1], float(o[2]), int(o[3])

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

def hw_reference(qx_u8_full, s_a, zp_a, S_OUT_PRE):
    s_acc = s_a * s_w
    x_u8 = qx_u8_full[0, :, R:R+ROI_H, C:C+ROI_W].astype(np.int32)
    x_i8 = np.clip(x_u8 - 128, -128, 127).astype(np.int8)
    bias_eff = bias + s_acc * (128 - zp_a) * sum_w
    out = np.zeros((NCH_OUT, OUT_H, OUT_W), dtype=np.float32)
    pre_min, pre_max = 1e30, -1e30
    for c in range(NCH_OUT):
        for oy in range(OUT_H):
            for ox in range(OUT_W):
                acc = int((W_q[c,:,0,0] * x_i8[:,oy,ox].astype(np.int32)).sum())
                pre = acc * s_acc + bias_eff[c]
                pre_min = min(pre_min, pre); pre_max = max(pre_max, pre)
                sig = 1.0/(1.0 + np.exp(-pre))
                out[c,oy,ox] = pre * sig
    return out, bias_eff, x_i8, s_acc, (pre_min, pre_max)

samples = [
    ("rand0",'rand',0),("rand1",'rand',1),("rand2",'rand',7),("rand3",'rand',13),
    ("rand_low",'rand_low',2),("rand_high",'rand_high',3),
    ("half",'half',0),("gradient",'gradient',0),
]

# First pass to pick S_OUT_PRE/S_OUT_SILU
pre_lo, pre_hi, post_hi = 0.0, 0.0, 0.0
cache = []
S_TMP = 4.0/127.0
for sname, kind, seed in samples:
    img = make_input(kind, seed=seed)
    ort_out, qx_u8, s_a, zp_a = run_ort(img)
    if s_a == 0.0: s_a = 1.0
    _, _, _, _, (pmin, pmax) = hw_reference(qx_u8, s_a, zp_a, S_TMP)
    pre_lo = min(pre_lo, pmin); pre_hi = max(pre_hi, pmax)
    roi = ort_out[0,:,R:R+OUT_H,C:C+OUT_W]
    post_hi = max(post_hi, float(np.max(np.abs(roi))))
    cache.append((sname, ort_out, qx_u8, s_a, zp_a))

print(f"observed pre-SiLU range [{pre_lo:.3f}, {pre_hi:.3f}], |post-SiLU| max ~{post_hi:.3f}")
PRE_AMP = max(abs(pre_lo), abs(pre_hi), 1e-3)
SILU_AMP = max(post_hi, 1e-3)
def pick_scale(amp):
    # Round up to nearest 0.001/127 with ~20% headroom; matches .sv literal.
    raw = round(amp * 1.2 * 1000.0) / 1000.0
    return raw / 127.0
S_OUT_PRE  = max(pick_scale(PRE_AMP),  2.0/127.0)
S_OUT_SILU = max(pick_scale(SILU_AMP), 2.0/127.0)
print(f"chose S_OUT_PRE={S_OUT_PRE:.6f} (={S_OUT_PRE*127:.3f}/127), "
      f"S_OUT_SILU={S_OUT_SILU:.6f} (={S_OUT_SILU*127:.3f}/127)")

N_LANE_FULL = NCH_IN
# pack weights flat (NCH_OUT, N_LANE_FULL): lane = kc (K=1)
W_flat = np.zeros((NCH_OUT, N_LANE_FULL), dtype=np.int8)
for c in range(NCH_OUT):
    for kc in range(NCH_IN):
        W_flat[c, kc] = W_q[c, kc, 0, 0]
with open(os.path.join(STIM, "weights.i8.hex"), "w") as f:
    for c in range(NCH_OUT):
        for k in range(N_LANE_FULL):
            f.write(f"{int(W_flat[c,k]) & 0xFF:02x}\n")

manifest = {
    "out_h": OUT_H, "out_w": OUT_W, "nch_out": NCH_OUT, "nch_in": NCH_IN,
    "K": K, "stride": STRIDE, "n_lane_full": N_LANE_FULL,
    "roi_h": ROI_H, "roi_w": ROI_W, "pad_h": PAD_H, "pad_w": PAD_W,
    "R": R, "C": C, "s_w": s_w, "s_out_pre": S_OUT_PRE, "s_out_silu": S_OUT_SILU,
    "acc_shift": 0, "samples": [],
}

for (sname, ort_out, qx_u8, s_a, zp_a) in cache:
    ort_roi = ort_out[0,:,R:R+OUT_H, C:C+OUT_W]
    hw_out, bias_eff, x_i8, s_acc, _ = hw_reference(qx_u8, s_a, zp_a, S_OUT_PRE)

    scale_fp16 = np.array([f32_to_fp16(s_acc / S_OUT_PRE)]*NCH_OUT, dtype=np.uint16)
    bias_fp16  = np.array([f32_to_fp16(bias_eff[c]/S_OUT_PRE) for c in range(NCH_OUT)], dtype=np.uint16)

    with open(os.path.join(STIM, f"{sname}.input_i8.hex"), "w") as f:
        for h in range(PAD_H):
            for w in range(PAD_W):
                for kc in range(NCH_IN):
                    f.write(f"{int(x_i8[kc,h,w]) & 0xFF:02x}\n")
    with open(os.path.join(STIM, f"{sname}.scale_fp16.hex"), "w") as f:
        for c in range(NCH_OUT): f.write(f"{int(scale_fp16[c]):04x}\n")
    with open(os.path.join(STIM, f"{sname}.bias_fp16.hex"), "w") as f:
        for c in range(NCH_OUT): f.write(f"{int(bias_fp16[c]):04x}\n")
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
                    bits = np.float32(hw_out[c,oy,ox]).view(np.uint32)
                    f.write(f"{int(bits):08x}\n")

    diff = ort_roi - hw_out
    mxe = float(np.max(np.abs(diff))); mae = float(np.mean(np.abs(diff)))
    out_range = float(ort_roi.max() - ort_roi.min())
    cos = float(np.sum(ort_roi*hw_out) / (np.linalg.norm(ort_roi)*np.linalg.norm(hw_out)+1e-12))
    print(f"[{sname}] s_a={s_a:.5f} zp_a={zp_a} pre~[{hw_out.min():.2f},{hw_out.max():.2f}] "
          f"ORT vs HW: max_abs={mxe:.4f} mae={mae:.4f} cos={cos:.6f}")
    manifest["samples"].append({"name":sname,"s_a":s_a,"zp_a":zp_a,"s_acc":s_acc,
        "ort_vs_hw_max_abs":mxe,"ort_vs_hw_mae":mae,"ort_vs_hw_cos":cos,
        "out_range":out_range})

with open(os.path.join(STIM, "manifest.json"), "w") as f:
    json.dump(manifest, f, indent=2)

print(f"\nWrote stim+ref to {STIM}")
