#!/usr/bin/env python3
# Copyright (c) 2026 vibeyolo
# SPDX-License-Identifier: Apache-2.0
#
# golden.py — Stimulus + reference for tiled layer_5 (YOLO26n /model.2/cv2
# Conv-BN-SiLU, 1x1, 48 -> 64, stride 1, pad 0). Input is the concatenated
# /model.2/Concat_output_0 (48ch).

import os, json
import numpy as np
import onnx, onnxruntime as ort
from onnx import numpy_helper

HERE = os.path.dirname(os.path.abspath(__file__))
STIM = os.path.join(HERE, "stim"); os.makedirs(STIM, exist_ok=True)
MODEL_PATH = "/home/marty/Documents/aif/vibeyolo/integ/yolo26n/model_int8.onnx"

NCH_IN, NCH_OUT, K, STRIDE = 48, 64, 1, 1
ROI_H, ROI_W = 8, 8
OUT_H, OUT_W = ROI_H, ROI_W
PAD_H, PAD_W = ROI_H, ROI_W
R, C = 32, 32

model = onnx.load(MODEL_PATH)
init = {i.name: numpy_helper.to_array(i) for i in model.graph.initializer}
W_q  = init['onnx::Conv_1666_quantized'].astype(np.int32)
s_w  = float(init['onnx::Conv_1666_scale'])
zp_w = int(init['onnx::Conv_1666_zero_point'])
bias = init['onnx::Conv_1667'].astype(np.float32)
assert zp_w == 0 and W_q.shape == (NCH_OUT, NCH_IN, K, K)
sum_w = W_q.reshape(NCH_OUT, -1).sum(axis=1).astype(np.int32)

TARGET_OUT = '/model.2/cv2/act/Mul_output_0'
IN_Q  = '/model.2/Concat_output_0_quantized'
IN_S  = '/model.2/Concat_output_0_scale'
IN_ZP = '/model.2/Concat_output_0_zero_point'

mod = onnx.load(MODEL_PATH)
ex = [o.name for o in mod.graph.output]
for nm, tp in [(TARGET_OUT, onnx.TensorProto.FLOAT), (IN_Q, onnx.TensorProto.UINT8),
               (IN_S, onnx.TensorProto.FLOAT), (IN_ZP, onnx.TensorProto.UINT8)]:
    if nm not in ex:
        mod.graph.output.append(onnx.helper.make_tensor_value_info(nm, tp, None))
TMP = os.path.join(HERE, "_model_with_intermediate.onnx")
onnx.save(mod, TMP)
so = ort.SessionOptions(); so.log_severity_level = 3
sess = ort.InferenceSession(TMP, sess_options=so, providers=['CPUExecutionProvider'])
ipt = sess.get_inputs()[0].name

def run_ort(img):
    o = sess.run([TARGET_OUT, IN_Q, IN_S, IN_ZP], {ipt: img})
    return o[0], o[1], float(o[2]), int(o[3])

def make_pixel_input(kind, seed=0):
    img = np.zeros((1, 3, 640, 640), dtype=np.float32)
    rng = np.random.RandomState(seed)
    if   kind == 'rand':      img[0] = rng.uniform(0,1,(3,640,640)).astype(np.float32)
    elif kind == 'half':      img[0] = 0.5
    elif kind == 'gradient':
        gx = np.tile(np.linspace(0,1,640,dtype=np.float32), (640,1))
        img[0,0]=gx; img[0,1]=gx.T; img[0,2]=0.5*(gx+gx.T)
    elif kind == 'rand_low':  img[0] = rng.uniform(0.15,0.80,(3,640,640)).astype(np.float32)
    elif kind == 'rand_high': img[0] = rng.uniform(0.5,1.0,(3,640,640)).astype(np.float32)
    else: raise ValueError(kind)
    return img

S_OUT_PRE  = 12.0 / 127.0
S_OUT_SILU = 12.0 / 127.0

def f32_to_fp16(x): return np.float16(x).view(np.uint16)

def hw_reference(qx_full, s_a, zp_a):
    s_acc = s_a * s_w
    x_u8 = qx_full[0, :, R:R+ROI_H, C:C+ROI_W].astype(np.int32)
    x_i8 = np.clip(x_u8 - 128, -128, 127).astype(np.int8)
    bias_eff = bias + s_acc * (128 - zp_a) * sum_w
    out = np.zeros((NCH_OUT, OUT_H, OUT_W), dtype=np.float32)
    for c in range(NCH_OUT):
        for oy in range(OUT_H):
            for ox in range(OUT_W):
                acc = int((W_q[c,:,0,0] * x_i8[:,oy,ox].astype(np.int32)).sum())
                pre = acc * s_acc + bias_eff[c]
                sig = 1.0/(1.0+np.exp(-pre))
                out[c,oy,ox] = pre * sig
    return out, bias_eff, x_i8, s_acc

N_LANE = K * K * NCH_IN
W_flat = np.zeros((NCH_OUT, N_LANE), dtype=np.int8)
for c in range(NCH_OUT):
    for kc in range(NCH_IN):
        W_flat[c, kc] = W_q[c, kc, 0, 0]
with open(os.path.join(STIM, "weights.i8.hex"), "w") as f:
    for c in range(NCH_OUT):
        for k in range(N_LANE):
            f.write(f"{int(W_flat[c,k])&0xFF:02x}\n")

samples = [("rand0",'rand',0), ("rand1",'rand',1), ("half",'half',0),
           ("gradient",'gradient',0), ("rand_low",'rand_low',2), ("rand_high",'rand_high',3)]

manifest = {"out_h":OUT_H,"out_w":OUT_W,"nch_out":NCH_OUT,"nch_in":NCH_IN,"K":K,"stride":STRIDE,
            "n_lane":N_LANE,"roi_h":ROI_H,"roi_w":ROI_W,"pad_h":PAD_H,"pad_w":PAD_W,"R":R,"C":C,
            "s_w":s_w,"s_out_pre":S_OUT_PRE,"s_out_silu":S_OUT_SILU,"samples":[]}

for sname, kind, seed in samples:
    img = make_pixel_input(kind, seed=seed)
    ort_full, qx, s_a, zp_a = run_ort(img)
    if s_a == 0.0: s_a = 1.0
    ort_roi = ort_full[0, :, R:R+OUT_H, C:C+OUT_W]
    hw_out, bias_eff, x_i8, s_acc = hw_reference(qx, s_a, zp_a)
    scale_fp16 = np.array([f32_to_fp16(s_acc/S_OUT_PRE)]*NCH_OUT, dtype=np.uint16)
    bias_fp16 = np.array([f32_to_fp16(bias_eff[c]/S_OUT_PRE) for c in range(NCH_OUT)], dtype=np.uint16)

    with open(os.path.join(STIM, f"{sname}.input_i8.hex"), "w") as f:
        for h in range(PAD_H):
            for w in range(PAD_W):
                for kc in range(NCH_IN):
                    f.write(f"{int(x_i8[kc,h,w])&0xFF:02x}\n")
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
    mae = float(np.mean(np.abs(ort_roi-hw_out))); mxe = float(np.max(np.abs(ort_roi-hw_out)))
    rng_o = float(ort_roi.max() - ort_roi.min())
    cos = float(np.sum(ort_roi*hw_out)/(np.linalg.norm(ort_roi)*np.linalg.norm(hw_out)+1e-12))
    print(f"[{sname}] s_a={s_a:.5f} zp_a={zp_a} range~{ort_roi.min():.3f}..{ort_roi.max():.3f} "
          f"mae={mae:.4f} max={mxe:.4f} cos={cos:.6f}")
    manifest["samples"].append({"name":sname,"s_a":s_a,"zp_a":zp_a,"s_acc":s_acc,
                                "ort_vs_hw_max_abs":mxe,"ort_vs_hw_mae":mae,
                                "ort_vs_hw_cos":cos,"out_range":rng_o})

with open(os.path.join(STIM, "manifest.json"), "w") as f:
    json.dump(manifest, f, indent=2)

def silu_lut(in_s, out_s):
    lut = np.zeros(256, dtype=np.int8)
    for idx in range(256):
        cp = idx if idx<128 else idx-256
        fv = cp * in_s
        s = fv/(1+np.exp(-fv)) if fv>=0 else (fv*np.exp(fv))/(1+np.exp(fv))
        lut[idx] = max(-128, min(127, int(round(s/out_s))))
    return lut
with open(os.path.join(STIM, "silu_lut.i8.hex"), "w") as f:
    for v in silu_lut(S_OUT_PRE, S_OUT_SILU): f.write(f"{int(v)&0xFF:02x}\n")

print(f"Wrote {STIM}")
