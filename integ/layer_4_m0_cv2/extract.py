#!/usr/bin/env python3
# Copyright (c) 2026 vibeyolo
# SPDX-License-Identifier: Apache-2.0
#
# extract.py — Generate stimulus + reference for the layer_4_m0_cv2 micro-
# integration test.
#
# Slice: ONNX nodes 39..47 — the SECOND conv inside the C3k2 bottleneck
# /model.2/m.0, plus the bottleneck's residual add.
#
#   /model.2/m.0/cv1/act/Mul_output_0  (post-SiLU of cv1, 8ch)
#    → DynamicQuantizeLinear  (u8, s_a, zp_a)                      [node 38]
#    → ConvInteger            (8 → 16, 3x3, stride 1, pad 1)       [node 39]
#    → Cast / Mul / Add bias                                       [nodes 40..44]
#    → Sigmoid → Mul (SiLU)   → /model.2/m.0/cv2/act/Mul_output_0  [nodes 45,46]
#    → Add /model.2/Slice_1_output_0  (residual — 16ch, post-SiLU) [node 47]
#
# The residual second operand `/model.2/Slice_1_output_0` is the 16-channel
# second half of `/model.2/cv1/act/Mul_output_0`. It is independently
# DynamicQuantizeLinear'd elsewhere (we use ORT's own dyn-quant for it on
# the HW path so add_rq can combine two int8 streams with their own scales).
#
# Geometry: input 160×160. We pick a 16×16 inside ROI (pad-1 ring → 18×18
# input window), output 16×16×16. Three+ tiles are generated.
#
# Math summary (HW path):
#   cv2 path (per output pixel, c=0..15):
#       acc = sum_{kh,kw,kc} w_i8[c,kc,kh,kw] * x_i8[kc, iy+kh, ix+kw]
#       bias_eff[c] = bias[c] + s_a * s_w * (128 - zp_a) * sum_w[c]
#       pre_silu = acc * (s_a*s_w) + bias_eff[c]
#       post_silu = pre_silu * sigmoid(pre_silu)
#   requant of post_silu to int8 with S_OUT_SILU.
#   Residual is the int8 stream Slice_1_q - 128 (with its own scale s_r).
#   add_rq combines: y_fp = post_silu + s_r*(slice1_i8) , then requant to
#     S_OUT_ADD (post-add int8 scale).
#
# Pattern follows integ/layer_1/golden.py.

import os, json
import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper

HERE = os.path.dirname(os.path.abspath(__file__))
STIM = os.path.join(HERE, "stim")
os.makedirs(STIM, exist_ok=True)

MODEL_PATH = "/home/marty/Documents/aif/vibeyolo/integ/yolo26n/model_int8.onnx"

# ── Geometry: layer 4, /model.2/m.0/cv2 conv ──
# 8 → 16, K=3, stride=1, pad=1. Input/output H=W=160.
NCH_IN  = 8
NCH_OUT = 16
K       = 3
STRIDE  = 1
ROI_H, ROI_W = 16, 16            # inside region (output is 16×16)
OUT_H, OUT_W = ROI_H, ROI_W       # stride 1 ⇒ same as ROI
PAD_H, PAD_W = ROI_H + 2, ROI_W + 2   # 18×18 (1-pixel pad ring)
# Place ROI on the 160×160 plane.
R, C = 32, 32

# ── ONNX initializers (cv2 conv params) ──
model = onnx.load(MODEL_PATH)
init_by_name = {init.name: numpy_helper.to_array(init) for init in model.graph.initializer}

W_q  = init_by_name['onnx::Conv_1663_quantized'].astype(np.int32)   # (16,8,3,3)
s_w  = float(init_by_name['onnx::Conv_1663_scale'])
zp_w = int(init_by_name['onnx::Conv_1663_zero_point'])
bias = init_by_name['onnx::Conv_1664'].astype(np.float32)           # (16,)
assert zp_w == 0
assert W_q.shape == (NCH_OUT, NCH_IN, K, K), f"Got {W_q.shape}"

sum_w = W_q.reshape(NCH_OUT, -1).sum(axis=1).astype(np.int32)  # (16,)

# ── Extra ORT outputs ──
# cv1 post-SiLU dyn-quantized tensor (= cv2 input).
CV1_OUT_Q  = '/model.2/m.0/cv1/act/Mul_output_0_quantized'
CV1_OUT_S  = '/model.2/m.0/cv1/act/Mul_output_0_scale'
CV1_OUT_ZP = '/model.2/m.0/cv1/act/Mul_output_0_zero_point'
# cv2 post-SiLU (pre-residual) — fp32 reference for the convolution stage.
CV2_POST_SILU = '/model.2/m.0/cv2/act/Mul_output_0'
# Slice_1 dyn-quantized — residual second operand (16ch).
SL1_Q  = '/model.2/Slice_1_output_0_quantized'
SL1_S  = '/model.2/Slice_1_output_0_scale'
SL1_ZP = '/model.2/Slice_1_output_0_zero_point'
# Slice_1 fp32 (just for cross-check / oracle)
SL1_F  = '/model.2/Slice_1_output_0'
# Final residual-add output (= layer_4_m0_cv2 final output).
ADD_OUT = '/model.2/m.0/Add_output_0'

EXTRA_OUTS = [
    (ADD_OUT,       onnx.TensorProto.FLOAT, None),
    (CV2_POST_SILU, onnx.TensorProto.FLOAT, None),
    (CV1_OUT_Q,     onnx.TensorProto.UINT8, None),
    (CV1_OUT_S,     onnx.TensorProto.FLOAT, None),
    (CV1_OUT_ZP,    onnx.TensorProto.UINT8, None),
    (SL1_Q,         onnx.TensorProto.UINT8, None),
    (SL1_S,         onnx.TensorProto.FLOAT, None),
    (SL1_ZP,        onnx.TensorProto.UINT8, None),
    (SL1_F,         onnx.TensorProto.FLOAT, None),
]

mod_model = onnx.load(MODEL_PATH)
existing = {o.name for o in mod_model.graph.output}
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

ORDER = [ADD_OUT, CV2_POST_SILU, CV1_OUT_Q, CV1_OUT_S, CV1_OUT_ZP,
         SL1_Q, SL1_S, SL1_ZP, SL1_F]
def run_ort(img_f32):
    o = sess.run(ORDER, {ipt_name: img_f32})
    return {
        'add_out':       o[0],
        'cv2_post_silu': o[1],
        'cv1_q':         o[2],
        'cv1_s':         float(o[3]),
        'cv1_zp':        int(o[4]),
        'sl1_q':         o[5],
        'sl1_s':         float(o[6]),
        'sl1_zp':        int(o[7]),
        'sl1_f':         o[8],
    }

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

# ── Scale choices ──
# Empirical pre-SiLU and post-SiLU ranges of cv2 stay modest; the residual
# add can blow up a bit because Slice_1 itself spans the cv1 output. We
# size output scales to comfortably cover ±80.
S_OUT_PRE   = 80.0 / 127.0   # cv2 pre-SiLU (requant from int32 acc → int8)
S_OUT_SILU  = 80.0 / 127.0   # cv2 post-SiLU int8 stream feeding add_rq
S_OUT_ADD   = 80.0 / 127.0   # final post-residual int8 scale

def f32_to_fp16(x):
    return np.float16(x).view(np.uint16)

# ── HW reference (fp32 forward through the SV-equivalent pipeline) ──
def hw_reference(qx_cv1_full, s_a, zp_a, qsl1_full, s_r, zp_r):
    """qx_cv1_full: (1,8,160,160) uint8. qsl1_full: (1,16,160,160) uint8."""
    s_acc = s_a * s_w

    # 18×18×8 padded ROI from cv1 quantized tensor.
    pad_u8 = np.full((NCH_IN, PAD_H, PAD_W), zp_a, dtype=np.int32)
    r0, r1 = R - 1, R - 1 + PAD_H
    c0, c1 = C - 1, C - 1 + PAD_W
    H_FULL = qx_cv1_full.shape[2]; W_FULL = qx_cv1_full.shape[3]
    sr0, sr1 = max(0, r0), min(H_FULL, r1)
    sc0, sc1 = max(0, c0), min(W_FULL, c1)
    pad_u8[:, sr0-r0:sr1-r0, sc0-c0:sc1-c0] = \
        qx_cv1_full[0, :, sr0:sr1, sc0:sc1].astype(np.int32)

    x_i8 = np.clip(pad_u8 - 128, -128, 127).astype(np.int8)
    bias_eff = bias + s_acc * (128 - zp_a) * sum_w   # (16,)

    # cv2 conv + SiLU (fp32, exactly what HW will do up to fp16 rounding).
    cv2_silu_f = np.zeros((NCH_OUT, OUT_H, OUT_W), dtype=np.float32)
    for c in range(NCH_OUT):
        for oy in range(OUT_H):
            for ox in range(OUT_W):
                iy = oy + 0  # pad-1 already in pad_u8
                ix = ox + 0
                window = x_i8[:, iy:iy+K, ix:ix+K].astype(np.int32)  # (8,3,3)
                acc = int((W_q[c] * window).sum())
                pre = acc * s_acc + bias_eff[c]
                sig = 1.0 / (1.0 + np.exp(-pre))
                cv2_silu_f[c, oy, ox] = pre * sig

    # cv2 post-SiLU requantized to int8 with S_OUT_SILU.
    cv2_silu_i8 = np.clip(np.round(cv2_silu_f / S_OUT_SILU), -128, 127).astype(np.int8)

    # Residual second operand: Slice_1 quantized ROI (16ch), folded to i8.
    sl1_pad = qsl1_full[0, :, R:R+OUT_H, C:C+OUT_W].astype(np.int32)  # (16,16,16)
    sl1_i8  = np.clip(sl1_pad - 128, -128, 127).astype(np.int8)
    # Slice_1 effective scale (fp32) once folded with the u8→i8 shift:
    # value_f = s_r * (u8 - zp_r) = s_r*(i8+128 - zp_r) = s_r*i8 + s_r*(128-zp_r)
    # The (128-zp_r) DC term per-channel we absorb into add_rq's bias.

    # Final residual add (fp32 emulation of add_rq math).
    out = np.zeros((NCH_OUT, OUT_H, OUT_W), dtype=np.float32)
    for c in range(NCH_OUT):
        for oy in range(OUT_H):
            for ox in range(OUT_W):
                a_f = S_OUT_SILU * float(cv2_silu_i8[c, oy, ox])
                b_f = s_r * float(sl1_i8[c, oy, ox]) + s_r * (128 - zp_r)
                out[c, oy, ox] = a_f + b_f
    return out, cv2_silu_f, cv2_silu_i8, sl1_i8, bias_eff, x_i8, s_acc

# ── Pack window: lane index = (kh*K + kw)*Cin + kc ──
N_LANE = K * K * NCH_IN   # 72

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

# ── Samples ──
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
    "s_out_pre": S_OUT_PRE, "s_out_silu": S_OUT_SILU, "s_out_add": S_OUT_ADD,
    "samples": [],
}

for sname, kind, seed in samples:
    img = make_pixel_input(kind, seed=seed)
    o = run_ort(img)
    s_a, zp_a = (o['cv1_s'] if o['cv1_s'] != 0.0 else 1.0), o['cv1_zp']
    s_r, zp_r = (o['sl1_s'] if o['sl1_s'] != 0.0 else 1.0), o['sl1_zp']

    ort_add_roi = o['add_out'][0, :, R:R+OUT_H, C:C+OUT_W]   # (16,16,16)
    hw_out, cv2_silu_f, cv2_silu_i8, sl1_i8, bias_eff, x_i8, s_acc = \
        hw_reference(o['cv1_q'], s_a, zp_a, o['sl1_q'], s_r, zp_r)

    # Per-channel fp16 scale/bias to feed cv2 requant (pre-SiLU stage).
    # We use S_OUT_PRE here (post-SiLU still int8 with S_OUT_SILU after
    # act_silu LUT). scale_fp16[c] = s_acc / S_OUT_PRE.
    scale_fp16 = np.array([f32_to_fp16(s_acc / S_OUT_PRE)] * NCH_OUT, dtype=np.uint16)
    bias_fp16  = np.array([f32_to_fp16(bias_eff[c] / S_OUT_PRE) for c in range(NCH_OUT)],
                          dtype=np.uint16)

    # Residual add_rq coefficients (per-output add):
    #   scale_a_fp16 = S_OUT_SILU   (cv2-stream weight)
    #   scale_b_fp16 = s_r          (residual weight per i8 unit)
    #   inv_out_scale_fp16 = 1.0 / S_OUT_ADD
    #   bias_fp16 = s_r * (128 - zp_r)   (DC term from u8→i8 fold)
    #
    # add_rq's bias is added *after* inv_out_scale, so we must scale the
    # DC term by inv_out_scale ourselves: bias_eff_add = s_r*(128-zp_r)/S_OUT_ADD.
    inv_out_add = f32_to_fp16(1.0 / S_OUT_ADD)
    sa_fp16     = f32_to_fp16(S_OUT_SILU)
    sb_fp16     = f32_to_fp16(s_r)
    bias_add_fp16 = f32_to_fp16(s_r * (128 - zp_r) / S_OUT_ADD)

    # ── Dump input i8 (PAD_H * PAD_W * NCH_IN), order h,w,kc ──
    with open(os.path.join(STIM, f"{sname}.input_i8.hex"), "w") as f:
        for h in range(PAD_H):
            for w in range(PAD_W):
                for kc in range(NCH_IN):
                    f.write(f"{int(x_i8[kc, h, w]) & 0xFF:02x}\n")

    # ── Dump residual i8 (OUT_H * OUT_W * NCH_OUT), order oy,ox,c ──
    with open(os.path.join(STIM, f"{sname}.resid_i8.hex"), "w") as f:
        for oy in range(OUT_H):
            for ox in range(OUT_W):
                for c in range(NCH_OUT):
                    f.write(f"{int(sl1_i8[c, oy, ox]) & 0xFF:02x}\n")

    # Scale + bias for cv2 path
    with open(os.path.join(STIM, f"{sname}.scale_fp16.hex"), "w") as f:
        for c in range(NCH_OUT):
            f.write(f"{int(scale_fp16[c]):04x}\n")
    with open(os.path.join(STIM, f"{sname}.bias_fp16.hex"), "w") as f:
        for c in range(NCH_OUT):
            f.write(f"{int(bias_fp16[c]):04x}\n")

    # add_rq scalars (shared across all output channels).
    with open(os.path.join(STIM, f"{sname}.add_rq_scalars_fp16.hex"), "w") as f:
        # order: scale_a, scale_b, inv_out_scale, bias
        f.write(f"{int(sa_fp16):04x}\n")
        f.write(f"{int(sb_fp16):04x}\n")
        f.write(f"{int(inv_out_add):04x}\n")
        f.write(f"{int(bias_add_fp16):04x}\n")

    # ORT + HW reference (fp32, order oy,ox,c) for the FINAL post-residual output.
    with open(os.path.join(STIM, f"{sname}.ref_ort.f32.hex"), "w") as f:
        for oy in range(OUT_H):
            for ox in range(OUT_W):
                for c in range(NCH_OUT):
                    bits = np.float32(ort_add_roi[c, oy, ox]).view(np.uint32)
                    f.write(f"{int(bits):08x}\n")
    with open(os.path.join(STIM, f"{sname}.ref_hw.f32.hex"), "w") as f:
        for oy in range(OUT_H):
            for ox in range(OUT_W):
                for c in range(NCH_OUT):
                    bits = np.float32(hw_out[c, oy, ox]).view(np.uint32)
                    f.write(f"{int(bits):08x}\n")

    diff = ort_add_roi - hw_out
    mae = float(np.mean(np.abs(diff)))
    mxe = float(np.max(np.abs(diff)))
    out_range = float(ort_add_roi.max() - ort_add_roi.min())
    cos = float(np.sum(ort_add_roi * hw_out) /
                (np.linalg.norm(ort_add_roi) * np.linalg.norm(hw_out) + 1e-12))
    print(f"[{sname}] s_a={s_a:.5f} zp_a={zp_a} s_r={s_r:.5f} zp_r={zp_r} "
          f"pre-SiLU range~{cv2_silu_f.min():.2f}..{cv2_silu_f.max():.2f} "
          f"add range~{ort_add_roi.min():.2f}..{ort_add_roi.max():.2f} "
          f"ORT vs HW-ref: max_abs={mxe:.4f} mae={mae:.4f} range={out_range:.3f} cos={cos:.6f}")

    manifest["samples"].append({
        "name": sname, "s_a": s_a, "zp_a": zp_a,
        "s_r": s_r, "zp_r": zp_r, "s_acc": s_acc,
        "ort_vs_hw_max_abs": mxe, "ort_vs_hw_mae": mae,
        "ort_vs_hw_cos": cos, "out_range": out_range,
    })

with open(os.path.join(STIM, "manifest.json"), "w") as f:
    json.dump(manifest, f, indent=2)

# ── SiLU LUT (same shape as stem_l0 / layer_1) ──
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
