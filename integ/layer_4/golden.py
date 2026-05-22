#!/usr/bin/env python3
# Copyright (c) 2026 vibeyolo
# SPDX-License-Identifier: Apache-2.0
#
# golden.py — Stimulus + reference for tiled layer_4 (YOLO26n
# /model.2/m.0/cv2 Conv-BN-SiLU + residual add, 3x3, 8 -> 16, stride 1, pad 1).
# Residual is the 16-channel /model.2/Slice_1_output_0 added after the conv's
# SiLU via add_rq (conv_layer RESIDUAL=1).
#
# add_rq convention (matches conv_layer):
#   y = sat_i8( (a_i8 * S_OUT_SILU + b_i8 * r_scale) * (1/S_OUT_SILU) + r_bias )
# where r_bias absorbs the per-channel DC term s_r*(128 - zp_r)/S_OUT_SILU.

import os, json
import numpy as np
import onnx, onnxruntime as ort
from onnx import numpy_helper

HERE = os.path.dirname(os.path.abspath(__file__))
STIM = os.path.join(HERE, "stim"); os.makedirs(STIM, exist_ok=True)
MODEL_PATH = "/home/marty/Documents/aif/vibeyolo/integ/yolo26n/model_int8.onnx"

NCH_IN, NCH_OUT, K, STRIDE = 8, 16, 3, 1
ROI_H, ROI_W = 8, 8
OUT_H, OUT_W = ROI_H, ROI_W
PAD_H, PAD_W = ROI_H + 2, ROI_W + 2
R, C = 32, 32

model = onnx.load(MODEL_PATH)
init = {i.name: numpy_helper.to_array(i) for i in model.graph.initializer}
W_q  = init['onnx::Conv_1663_quantized'].astype(np.int32)
s_w  = float(init['onnx::Conv_1663_scale'])
zp_w = int(init['onnx::Conv_1663_zero_point'])
bias = init['onnx::Conv_1664'].astype(np.float32)
assert zp_w == 0 and W_q.shape == (NCH_OUT, NCH_IN, K, K)
sum_w = W_q.reshape(NCH_OUT, -1).sum(axis=1).astype(np.int32)

# ORT outputs
CV1_OUT_Q  = '/model.2/m.0/cv1/act/Mul_output_0_quantized'
CV1_OUT_S  = '/model.2/m.0/cv1/act/Mul_output_0_scale'
CV1_OUT_ZP = '/model.2/m.0/cv1/act/Mul_output_0_zero_point'
SL1_Q      = '/model.2/Slice_1_output_0_quantized'
SL1_S      = '/model.2/Slice_1_output_0_scale'
SL1_ZP     = '/model.2/Slice_1_output_0_zero_point'
ADD_OUT    = '/model.2/m.0/Add_output_0'

mod = onnx.load(MODEL_PATH)
ex = [o.name for o in mod.graph.output]
for nm, tp in [(ADD_OUT, onnx.TensorProto.FLOAT),
               (CV1_OUT_Q, onnx.TensorProto.UINT8), (CV1_OUT_S, onnx.TensorProto.FLOAT),
               (CV1_OUT_ZP, onnx.TensorProto.UINT8),
               (SL1_Q, onnx.TensorProto.UINT8), (SL1_S, onnx.TensorProto.FLOAT),
               (SL1_ZP, onnx.TensorProto.UINT8)]:
    if nm not in ex:
        mod.graph.output.append(onnx.helper.make_tensor_value_info(nm, tp, None))
TMP = os.path.join(HERE, "_model_with_intermediate.onnx")
onnx.save(mod, TMP)
so = ort.SessionOptions(); so.log_severity_level = 3
sess = ort.InferenceSession(TMP, sess_options=so, providers=['CPUExecutionProvider'])
ipt = sess.get_inputs()[0].name

ORDER = [ADD_OUT, CV1_OUT_Q, CV1_OUT_S, CV1_OUT_ZP, SL1_Q, SL1_S, SL1_ZP]
def run_ort(img):
    o = sess.run(ORDER, {ipt: img})
    return {'add_out': o[0], 'cv1_q': o[1], 'cv1_s': float(o[2]), 'cv1_zp': int(o[3]),
            'sl1_q': o[4], 'sl1_s': float(o[5]), 'sl1_zp': int(o[6])}

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

# conv_layer fixes inv_out_scale = 1/S_OUT_SILU, so S_OUT_ADD == S_OUT_SILU.
S_OUT_PRE  = 32.0 / 127.0
S_OUT_SILU = 32.0 / 127.0

def f32_to_fp16(x): return np.float16(x).view(np.uint16)

def hw_reference(qx_full, s_a, zp_a, sl1_full, s_r, zp_r):
    s_acc = s_a * s_w
    pad_u8 = np.full((NCH_IN, PAD_H, PAD_W), zp_a, dtype=np.int32)
    r0, r1 = R-1, R-1+PAD_H; c0, c1 = C-1, C-1+PAD_W
    HF = qx_full.shape[2]; WF = qx_full.shape[3]
    sr0, sr1 = max(0,r0), min(HF,r1); sc0, sc1 = max(0,c0), min(WF,c1)
    pad_u8[:, sr0-r0:sr1-r0, sc0-c0:sc1-c0] = qx_full[0, :, sr0:sr1, sc0:sc1].astype(np.int32)
    x_i8 = np.clip(pad_u8 - 128, -128, 127).astype(np.int8)
    bias_eff = bias + s_acc * (128 - zp_a) * sum_w

    # Conv + SiLU (fp32)
    cv2_silu_f = np.zeros((NCH_OUT, OUT_H, OUT_W), dtype=np.float32)
    for c in range(NCH_OUT):
        for oy in range(OUT_H):
            for ox in range(OUT_W):
                w = x_i8[:, oy:oy+K, ox:ox+K].astype(np.int32)
                acc = int((W_q[c] * w).sum())
                pre = acc * s_acc + bias_eff[c]
                sig = 1.0/(1.0+np.exp(-pre))
                cv2_silu_f[c,oy,ox] = pre * sig
    # Requant to int8 with S_OUT_SILU
    cv2_silu_i8 = np.clip(np.round(cv2_silu_f / S_OUT_SILU), -128, 127).astype(np.int8)

    # Residual 16ch from Slice_1
    sl1_pad = sl1_full[0, :, R:R+OUT_H, C:C+OUT_W].astype(np.int32)
    sl1_i8  = np.clip(sl1_pad - 128, -128, 127).astype(np.int8)

    # Final add per add_rq math (S_OUT_ADD == S_OUT_SILU here):
    # value_f = S_OUT_SILU * cv2_silu_i8 + s_r * (sl1_i8 + 128 - zp_r)
    out = np.zeros((NCH_OUT, OUT_H, OUT_W), dtype=np.float32)
    for c in range(NCH_OUT):
        for oy in range(OUT_H):
            for ox in range(OUT_W):
                a_f = S_OUT_SILU * float(cv2_silu_i8[c, oy, ox])
                b_f = s_r * float(sl1_i8[c, oy, ox]) + s_r * (128 - zp_r)
                out[c, oy, ox] = a_f + b_f
    return out, cv2_silu_f, cv2_silu_i8, sl1_i8, bias_eff, x_i8, s_acc

N_LANE = K * K * NCH_IN  # 72
W_flat = np.zeros((NCH_OUT, N_LANE), dtype=np.int8)
for c in range(NCH_OUT):
    for kh in range(K):
        for kw in range(K):
            for kc in range(NCH_IN):
                W_flat[c, (kh*K+kw)*NCH_IN + kc] = W_q[c, kc, kh, kw]
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
    o = run_ort(img)
    s_a = o['cv1_s'] if o['cv1_s'] != 0 else 1.0; zp_a = o['cv1_zp']
    s_r = o['sl1_s'] if o['sl1_s'] != 0 else 1.0; zp_r = o['sl1_zp']
    ort_add_roi = o['add_out'][0, :, R:R+OUT_H, C:C+OUT_W]
    hw_out, cv2_silu_f, cv2_silu_i8, sl1_i8, bias_eff, x_i8, s_acc = \
        hw_reference(o['cv1_q'], s_a, zp_a, o['sl1_q'], s_r, zp_r)

    # Conv requant coefficients (pre-SiLU)
    scale_fp16 = np.array([f32_to_fp16(s_acc / S_OUT_PRE)]*NCH_OUT, dtype=np.uint16)
    bias_fp16  = np.array([f32_to_fp16(bias_eff[c] / S_OUT_PRE) for c in range(NCH_OUT)], dtype=np.uint16)

    # Residual per-channel scale and bias (passed as r_scale_i / r_bias_i)
    # r_scale_i[c] = s_r (residual fp scale -> fp16)
    # r_bias_i [c] = s_r * (128 - zp_r) / S_OUT_SILU
    r_scale_fp16 = np.array([f32_to_fp16(s_r)]*NCH_OUT, dtype=np.uint16)
    r_bias_fp16  = np.array([f32_to_fp16(s_r * (128 - zp_r) / S_OUT_SILU)]*NCH_OUT, dtype=np.uint16)

    with open(os.path.join(STIM, f"{sname}.input_i8.hex"), "w") as f:
        for h in range(PAD_H):
            for w in range(PAD_W):
                for kc in range(NCH_IN):
                    f.write(f"{int(x_i8[kc,h,w])&0xFF:02x}\n")
    with open(os.path.join(STIM, f"{sname}.resid_i8.hex"), "w") as f:
        for oy in range(OUT_H):
            for ox in range(OUT_W):
                for c in range(NCH_OUT):
                    f.write(f"{int(sl1_i8[c,oy,ox])&0xFF:02x}\n")
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
                    f.write(f"{int(np.float32(ort_add_roi[c,oy,ox]).view(np.uint32)):08x}\n")
    with open(os.path.join(STIM, f"{sname}.ref_hw.f32.hex"), "w") as f:
        for oy in range(OUT_H):
            for ox in range(OUT_W):
                for c in range(NCH_OUT):
                    f.write(f"{int(np.float32(hw_out[c,oy,ox]).view(np.uint32)):08x}\n")
    mae = float(np.mean(np.abs(ort_add_roi-hw_out))); mxe = float(np.max(np.abs(ort_add_roi-hw_out)))
    rng_o = float(ort_add_roi.max() - ort_add_roi.min())
    cos = float(np.sum(ort_add_roi*hw_out)/(np.linalg.norm(ort_add_roi)*np.linalg.norm(hw_out)+1e-12))
    print(f"[{sname}] s_a={s_a:.5f} zp_a={zp_a} s_r={s_r:.5f} zp_r={zp_r} "
          f"add range~{ort_add_roi.min():.3f}..{ort_add_roi.max():.3f} "
          f"mae={mae:.4f} max={mxe:.4f} cos={cos:.6f}")
    manifest["samples"].append({"name":sname,"s_a":s_a,"zp_a":zp_a,"s_r":s_r,"zp_r":zp_r,
                                "s_acc":s_acc,"ort_vs_hw_max_abs":mxe,"ort_vs_hw_mae":mae,
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
