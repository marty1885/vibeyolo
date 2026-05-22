#!/usr/bin/env python3
# Copyright (c) 2026 vibeyolo
# SPDX-License-Identifier: Apache-2.0
#
# golden.py — Generate stimulus + reference for the stem_l0 integration test.
#
# Steps:
#   1. Load /model.0 stem params from yolo26n int8 ONNX.
#   2. Build random + directed uint8 inputs (a 16x16x3 ROI inside 640x640).
#   3. Run ORT on the FULL model with the prepared inputs and crop the 8x8x16
#      stem-output ROI as the "ORT fp32 reference".
#   4. Compute the "HW-format reference" by:
#         a) input_i8 = clip(input_u8 - 128, -128, 127)   (we choose zp=128;
#            see note below — we measure the actual ORT zp on each sample and
#            adjust the per-channel bias accordingly so the HW path is exact).
#         b) For each output channel c:
#               acc_i32 = sum_{kh,kw,kc} w_i8[c,kc,kh,kw] * x_i8[kc, y*2+kh-1, x*2+kw-1]
#                       + sum_w[c] * (zp_used - 128)     (zp-fold correction)
#               s_acc[c] = s_a * s_w[c]   (here s_w is shared scalar -> s_a*s_w)
#               pre_silu = acc_i32 * s_acc + bias[c]
#               post_silu = pre_silu * sigmoid(pre_silu)
#         c) Quantize: choose s_out_pre (pre-SiLU int8 scale) and s_out_silu
#            (post-SiLU int8 scale) so the int8 LUT-based pipeline can run.
#   5. Dump stimulus + references to hex files in stim/.

import os, sys, json, struct
import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper

HERE = os.path.dirname(os.path.abspath(__file__))
STIM = os.path.join(HERE, "stim")
os.makedirs(STIM, exist_ok=True)

MODEL_PATH = "/home/marty/Documents/aif/vibeyolo/integ/yolo26n/model_int8.onnx"

# ---------- ROI geometry ----------
# Use a 16x16 patch of input (stride 2, pad 1, k=3) → output 8x8 per channel.
# We place the ROI inside the full 640x640 image and run ORT on the full
# image; then crop the conv-output to the matching 8x8 ROI.
#
# Output (y,x) takes input window (y*2-1..y*2+1, x*2-1..x*2+1).
# To make the ROI self-contained (no neighbors outside leak in), we let the
# input ROI be 16x16 at image position (R, C) and pad with zeros in the
# image OUTSIDE the ROI. Output ROI is then 8x8 at (R//2, C//2). For the
# ROI edges, the conv reaches one pixel outside; we let the surrounding
# input image be 0 so the "outside" contribution is 0, matching the
# "pad 1 with zeros" semantics that the conv itself implements. Combined,
# this means: for input rows R-1 and R+16, columns C-1 and C+16 we keep
# zero. So we effectively get the exact same result as running just the
# 16x16 patch with pad 1.
R, C = 16, 16          # top-left placement of ROI
ROI_H, ROI_W = 16, 16
OUT_H, OUT_W = ROI_H // 2, ROI_W // 2
NCH_IN = 3
NCH_OUT = 16

# ---------- Load ONNX and extract initializers ----------
model = onnx.load(MODEL_PATH)
init_by_name = {init.name: numpy_helper.to_array(init) for init in model.graph.initializer}

W_q   = init_by_name['onnx::Conv_1651_quantized'].astype(np.int32)   # (16,3,3,3) int8
s_w   = float(init_by_name['onnx::Conv_1651_scale'])                 # scalar fp32
zp_w  = int(init_by_name['onnx::Conv_1651_zero_point'])              # 0
bias  = init_by_name['onnx::Conv_1652'].astype(np.float32)           # (16,)
assert zp_w == 0
print(f"W shape={W_q.shape} dtype={W_q.dtype} s_w={s_w} bias.shape={bias.shape}")

# Per-channel sum of weights (for zp folding).
# sum over (kc,kh,kw) of w[c,kc,kh,kw]
sum_w = W_q.reshape(NCH_OUT, -1).sum(axis=1).astype(np.int32)  # (16,)

# ---------- Build sample inputs ----------
#
# YOLO26n receives float pixel_values that the model dyn-quantises to uint8.
# Real upstream pipelines feed pixel/255.0 (range 0..1). At that scale, the
# pre-SiLU activations of /model.0 sit in roughly ±60 and post-SiLU in
# [-0.28, ~60] — well within fp16 precision (~10 mantissa bits → ULP at 60
# is ~0.06, totally fine for our epsilon comparison).
#
# We therefore generate inputs as normalized floats in [0,1].
def make_input(kind, seed=0):
    """Returns a float32 (1,3,640,640) image (values in [0,1]) with the ROI populated.
    Outside the ROI, the image is zero (so the conv's "pad 1" zeros from
    outside the ROI match the surrounding-image-zero condition exactly)."""
    img = np.zeros((1, NCH_IN, 640, 640), dtype=np.float32)
    rng = np.random.RandomState(seed)
    if kind == 'rand':
        patch = rng.uniform(0.0, 1.0, size=(NCH_IN, ROI_H, ROI_W)).astype(np.float32)
    elif kind == 'half':
        patch = np.full((NCH_IN, ROI_H, ROI_W), 0.5, dtype=np.float32)
    elif kind == 'all0':
        patch = np.zeros((NCH_IN, ROI_H, ROI_W), dtype=np.float32)
    elif kind == 'all1':
        patch = np.full((NCH_IN, ROI_H, ROI_W), 1.0, dtype=np.float32)
    elif kind == 'gradient':
        gx = np.tile(np.linspace(0.0, 1.0, ROI_W, dtype=np.float32), (ROI_H, 1))
        patch = np.stack([gx, gx.T, (gx + gx.T) * 0.5], axis=0)
    elif kind == 'rand_low':
        patch = rng.uniform(0.15, 0.80, size=(NCH_IN, ROI_H, ROI_W)).astype(np.float32)
    else:
        raise ValueError(kind)
    img[0, :, R:R+ROI_H, C:C+ROI_W] = patch
    return img, patch

# ---------- ORT session — run the full model up to node 8 ----------
# We add explicit graph outputs for the intermediate tensors we need.
TARGET = '/model.0/act/Mul_output_0'
EXTRA_OUTS = [
    (TARGET,                       onnx.TensorProto.FLOAT, None),
    ('pixel_values_scale',         onnx.TensorProto.FLOAT, None),
    ('pixel_values_zero_point',    onnx.TensorProto.UINT8, None),
    ('pixel_values_quantized',     onnx.TensorProto.UINT8, None),
]

mod_model = onnx.load(MODEL_PATH)
existing = [o.name for o in mod_model.graph.output]
for nm, tp, shape in EXTRA_OUTS:
    if nm not in existing:
        mod_model.graph.output.append(onnx.helper.make_tensor_value_info(nm, tp, shape))
TMP_PATH = os.path.join(HERE, "_model_with_intermediate.onnx")
onnx.save(mod_model, TMP_PATH)

# pixel_values is float32 in the ONNX schema (dyn-quant happens inside).
# We give float values matching our uint8 patch.
so = ort.SessionOptions()
so.log_severity_level = 3
sess = ort.InferenceSession(TMP_PATH, sess_options=so, providers=['CPUExecutionProvider'])
ipt_name = sess.get_inputs()[0].name
ipt_type = sess.get_inputs()[0].type
print(f"ORT input: {ipt_name} type={ipt_type}")

def run_ort(img_f32):
    x = img_f32.astype(np.float32)
    outs = sess.run([TARGET, 'pixel_values_scale', 'pixel_values_zero_point',
                     'pixel_values_quantized'],
                    {ipt_name: x})
    return outs[0], float(outs[1]), int(outs[2]), outs[3]   # post-SiLU, s_a, zp_a, qx (u8)

# ---------- HW reference (what our SV pipeline should produce, in fp32) ----------
# The HW does int8 symmetric, zp=0. To emulate ORT's u8-dyn-quant we:
#   - measure ORT's zp_a (uint8 zero-point) and s_a per sample,
#   - convert input to i8 via x_i8 = clip(x_u8 - 128, -128, 127),
#   - per-output-channel bias adjustment:
#         bias_eff[c] = bias[c] + s_a * s_w * (128 - zp_a) * sum_w[c]
#     so that:
#         (acc_i32 over i8) * s_a * s_w + bias_eff
#       == (acc_i32 over u8) * s_a * s_w
#          - s_a*s_w*zp_a*sum_w + bias                    (ORT formula)
#       == ORT pre-SiLU exactly (modulo fp16 precision).
#
# Then we apply SiLU. This is the "expected output of the HW pipeline run on
# the same image, using ORT's s_a/zp_a folded into the bias" — i.e. the
# floating-point reference we want our SV to match within epsilon.

def hw_reference(qx_full_u8, s_a, zp_a):
    """Compute the HW-format reference output (fp32) at the 8x8 ROI.

    The HW pipeline uses int8 symmetric activations (zp_hw = 0). ORT delivers
    uint8 quantised inputs with (s_a, zp_a). We adopt the standard rebase:
        x_i8 = clip(qx_u8 - 128, -128, 127)
    so the int8 lane carries (qx_u8 - 128). The HW accumulator therefore
    differs from the ORT accumulator by `-128 * sum_w[c]`. We absorb this
    into the per-channel bias plus the ORT zp_a folding:
        bias_eff[c] = bias[c] + s_a * s_w * (128 - zp_a) * sum_w[c]
    yielding:
        pre[c] = (sum_i8 * s_a * s_w) + bias_eff[c]
              == ORT's ConvInteger result * s_a*s_w + bias - s_a*s_w*zp_a*sum_w
              == ORT's pre-SiLU at the same pixel (modulo per-op rounding).
    """
    s_acc = s_a * s_w  # scalar — s_w is per-tensor

    # Extract the (18x18) ROI window (including 1-pixel pad from neighbours,
    # which is 0 outside the ROI since we placed the patch on a zero canvas).
    pad_u8 = np.zeros((NCH_IN, ROI_H + 2, ROI_W + 2), dtype=np.int32)
    pad_u8[:, 1:1+ROI_H, 1:1+ROI_W] = qx_full_u8[0, :, R:R+ROI_H, C:C+ROI_W].astype(np.int32)
    # Surrounding image is zero ⇒ ORT quantises to u8 = zp_a (rounded).
    # For zp_a=0 (the common case here), pad of zeros in u8 is correct.
    if zp_a != 0:
        # Fill the 1-pixel ring with zp_a so the "outside" u8 matches ORT.
        pad_u8[:, 0, :] = zp_a
        pad_u8[:, -1, :] = zp_a
        pad_u8[:, :, 0] = zp_a
        pad_u8[:, :, -1] = zp_a
        pad_u8[:, 1:1+ROI_H, 1:1+ROI_W] = qx_full_u8[0, :, R:R+ROI_H, C:C+ROI_W].astype(np.int32)

    # i8 = u8 - 128, saturating clamp (in normal u8 range it's safe).
    x_i8 = np.clip(pad_u8 - 128, -128, 127).astype(np.int8)

    # Per-channel adjusted bias
    bias_eff = bias + s_acc * (128 - zp_a) * sum_w  # (16,)

    out = np.zeros((NCH_OUT, OUT_H, OUT_W), dtype=np.float32)
    acc_arr = np.zeros((NCH_OUT, OUT_H, OUT_W), dtype=np.int32)
    for c in range(NCH_OUT):
        for oy in range(OUT_H):
            for ox in range(OUT_W):
                iy = oy * 2
                ix = ox * 2
                window = x_i8[:, iy:iy+3, ix:ix+3].astype(np.int32)  # (3,3,3)
                acc = int((W_q[c] * window).sum())
                acc_arr[c, oy, ox] = acc
                pre = acc * s_acc + bias_eff[c]
                sig = 1.0 / (1.0 + np.exp(-pre))
                out[c, oy, ox] = pre * sig

    return out, acc_arr, s_acc, bias_eff, x_i8

# ---------- HW SV scaling choices ----------
# We need to pick:
#   - For each output channel c, an fp16 "scale" and "bias" feeding requant.
#     The requant does: y_i8 = sat(round( fp16(acc_i32) * scale + bias ))
#     We want y_i8 * s_out_pre[c] ≈ pre_silu, so scale[c] = s_acc / s_out_pre[c]
#     and bias_fed[c] = bias_eff[c] / s_out_pre[c].
#   - SiLU LUT: InScale = s_out_pre[c], OutScale = s_out_silu[c].
#
# Per-channel scales make the SiLU LUT per-channel parametric — which means
# 16 different act_silu instances (each with its own LUT). That's fine and
# matches how the HW would actually pipeline per output channel.
#
# We choose s_out_pre[c] = pre_max / 127 where pre_max is the max |pre_silu|
# expected for that channel — but we want the choice to be data-independent,
# so we use 6.0 / 127 as a fixed "stem activation scale" (most YOLO stems
# range ±5 pre-SiLU). The post-SiLU range for SiLU(x) when x∈[-6,6] is
# roughly [-0.28, 6], so s_out_silu = 6.0/127 also works.

# Pre-SiLU range for /model.0 on normalized 0..1 inputs is empirically ±64,
# so s_out_pre = 64/127 ≈ 0.504 lets the int8 mid-stage cover [-64, 64].
# Post-SiLU has the same range (SiLU saturates to f for large +f), so same.
S_OUT_PRE  = 64.0 / 127.0
S_OUT_SILU = 64.0 / 127.0

# ---------- fp16 helpers ----------
def f32_to_fp16(x):
    return np.float16(x).view(np.uint16)

def fp16_to_f32(h):
    return float(np.uint16(h).view(np.float16))

# ---------- Generate stimulus per sample ----------
samples = [
    ("rand0",    'rand',    0),
    ("rand1",    'rand',    1),
    ("half",     'half',    0),
    ("all0",     'all0',    0),
    ("all1",     'all1',    0),
    ("gradient", 'gradient',0),
    ("rand_low", 'rand_low',2),
]

# ---------- Write per-sample weight & scale files (one-time, shared across samples) ----------
# Weights as i8 hex, per output-channel-major: 16 channels × 27 elements each.
# Order chosen to match SV indexing: for c in 0..15, for k in 0..26, where
# k indexes (kc, kh, kw) with kc fastest in kw? We'll use natural ONNX order:
#   k = (kh * 3 + kw) * 3 + kc    (so kc innermost ↔ 3 channels packed at each pixel)
# Actually, more useful for SV: match how we read the input window:
#   For each output pixel we have 27 (a,b) lanes; lane index = (kh*3 + kw)*3 + kc.
def pack_window_index(kh, kw, kc):
    return (kh * 3 + kw) * 3 + kc

W_flat = np.zeros((NCH_OUT, 27), dtype=np.int8)
for c in range(NCH_OUT):
    for kh in range(3):
        for kw in range(3):
            for kc in range(NCH_IN):
                W_flat[c, pack_window_index(kh, kw, kc)] = W_q[c, kc, kh, kw]

with open(os.path.join(STIM, "weights.i8.hex"), "w") as f:
    for c in range(NCH_OUT):
        for k in range(27):
            v = int(W_flat[c, k]) & 0xFF
            f.write(f"{v:02x}\n")

# ---------- Run ORT + HW for each sample ----------
manifest = {
    "samples": [],
    "out_h": OUT_H, "out_w": OUT_W,
    "nch_out": NCH_OUT, "nch_in": NCH_IN,
    "roi_h": ROI_H, "roi_w": ROI_W,
    "R": R, "C": C,
    "s_w": s_w,
    "s_out_pre":  S_OUT_PRE,
    "s_out_silu": S_OUT_SILU,
    "bias_fp32": bias.tolist(),
    "sum_w": sum_w.tolist(),
}

for sname, kind, seed in samples:
    img, patch = make_input(kind, seed=seed)
    ort_out_full, s_a, zp_a, qx_full = run_ort(img)
    ort_roi = ort_out_full[0, :, R//2:R//2+OUT_H, C//2:C//2+OUT_W]  # (16,8,8)

    if s_a == 0.0:
        # Degenerate: all-zero image. Skip the divide.
        s_a = 1.0
    hw_out, acc_arr, s_acc, bias_eff, x_i8 = hw_reference(qx_full, s_a, zp_a)

    # Per-channel fp16 scale and bias fed to requant:
    scale_fp16 = np.array([f32_to_fp16(s_acc / S_OUT_PRE)] * NCH_OUT, dtype=np.uint16)
    bias_fp16  = np.array([f32_to_fp16(bias_eff[c] / S_OUT_PRE) for c in range(NCH_OUT)],
                          dtype=np.uint16)

    # Write input i8 (padded 18x18x3 → flatten in row-major, kc fastest)
    # We pass 18x18 padded so that for each (oy,ox) we extract the 3x3 window
    # directly. Layout: for h in 0..17, for w in 0..17, for kc in 0..2  -> uint8
    in_pad = x_i8  # shape (3, 18, 18)
    inp_path = os.path.join(STIM, f"{sname}.input_i8.hex")
    with open(inp_path, "w") as f:
        for h in range(ROI_H + 2):
            for w in range(ROI_W + 2):
                for kc in range(NCH_IN):
                    v = int(in_pad[kc, h, w]) & 0xFF
                    f.write(f"{v:02x}\n")

    # Write per-channel fp16 scale + bias (16 channels)
    with open(os.path.join(STIM, f"{sname}.scale_fp16.hex"), "w") as f:
        for c in range(NCH_OUT):
            f.write(f"{int(scale_fp16[c]):04x}\n")
    with open(os.path.join(STIM, f"{sname}.bias_fp16.hex"), "w") as f:
        for c in range(NCH_OUT):
            f.write(f"{int(bias_fp16[c]):04x}\n")

    # ORT reference (fp32) for output ROI — 8x8x16, channel fastest? Use
    # (oy, ox, c) order.
    with open(os.path.join(STIM, f"{sname}.ref_ort.f32.hex"), "w") as f:
        for oy in range(OUT_H):
            for ox in range(OUT_W):
                for c in range(NCH_OUT):
                    v = float(ort_roi[c, oy, ox])
                    bits = np.float32(v).view(np.uint32)
                    f.write(f"{int(bits):08x}\n")

    # HW reference (fp32) for output ROI
    with open(os.path.join(STIM, f"{sname}.ref_hw.f32.hex"), "w") as f:
        for oy in range(OUT_H):
            for ox in range(OUT_W):
                for c in range(NCH_OUT):
                    v = float(hw_out[c, oy, ox])
                    bits = np.float32(v).view(np.uint32)
                    f.write(f"{int(bits):08x}\n")

    # Quick stats
    diff = ort_roi - hw_out
    mae = float(np.mean(np.abs(diff)))
    mxe = float(np.max(np.abs(diff)))
    out_range = float(ort_roi.max() - ort_roi.min())
    cos = float(np.sum(ort_roi * hw_out) /
                (np.linalg.norm(ort_roi) * np.linalg.norm(hw_out) + 1e-12))
    print(f"[{sname}] s_a={s_a:.5f} zp_a={zp_a} "
          f"ORT vs HW-ref: max_abs={mxe:.4f} mae={mae:.4f} "
          f"out_range={out_range:.3f} cos={cos:.6f}")

    manifest["samples"].append({
        "name": sname,
        "s_a": s_a, "zp_a": zp_a, "s_acc": s_acc,
        "bias_eff": bias_eff.tolist(),
        "ort_vs_hw_max_abs": mxe,
        "ort_vs_hw_mae": mae,
        "ort_vs_hw_cos": cos,
        "out_range": out_range,
    })

with open(os.path.join(STIM, "manifest.json"), "w") as f:
    json.dump(manifest, f, indent=2)

# ---------- Build the SiLU LUT (one per output channel — but here all 16 channels
# share s_out_pre and s_out_silu so the LUT is the same for all). We bake it
# in Python and dump to a hex file so the TB can preload the SV LUT or check
# against it. (The act_silu RTL bakes its own LUT in `initial`, so we don't
# strictly need to dump — but we dump for traceability.)
def silu_lut_bytes(in_scale, out_scale):
    lut = np.zeros(256, dtype=np.int8)
    for idx in range(256):
        cp = idx if idx < 128 else idx - 256
        f = cp * in_scale
        # robust SiLU
        if f >= 0.0:
            s = f / (1.0 + np.exp(-f))
        else:
            ef = np.exp(f)
            s = (f * ef) / (1.0 + ef)
        q = round(s / out_scale)   # banker's rounding to nearest int
        q = max(-128, min(127, int(q)))
        lut[idx] = q
    return lut

lut = silu_lut_bytes(S_OUT_PRE, S_OUT_SILU)
with open(os.path.join(STIM, "silu_lut.i8.hex"), "w") as f:
    for v in lut:
        f.write(f"{int(v) & 0xFF:02x}\n")

print("\nWrote stimulus + reference to", STIM)
print(f"Per-sample manifest: {os.path.join(STIM, 'manifest.json')}")
