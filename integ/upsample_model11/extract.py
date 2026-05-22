#!/usr/bin/env python3
# Copyright (c) 2026 vibeyolo
# SPDX-License-Identifier: Apache-2.0
#
# Upsample integration extractor for the P4->P3 neck join in YOLO26n.
#
# Models the ONNX subgraph:
#
#     /model.10/cv2/act/Mul_output_0   (B x 256 x 20 x 20  float)
#                                       │
#                                       ▼
#                            /model.11/Resize  (nearest, scale=2)
#                                       │
#                                       ▼
#                                /model.11/Resize_output_0
#                                       │      (B x 256 x 40 x 40)
#                                       │
#                                       │            /model.6/cv2/act/Mul_output_0
#                                       │            (B x 128 x 40 x 40  float)
#                                       │                  │
#                                       ▼                  ▼
#                              /model.12/Concat (axis=1)
#                                       │
#                                       ▼
#                            /model.12/Concat_output_0  (B x 384 x 40 x 40)
#
# Quantization plan (mirrors SPPF integration block):
#   - Pick a single output-tensor scale S_OUT that covers BOTH input streams'
#     range (so the downstream layer can consume one int8 tensor at one
#     scale, which is what ORT's DynamicQuantizeLinear effectively does
#     after the float Concat).
#   - Quantize the 20x20x256 float A input at S_OUT  ->  A_i8
#   - Quantize the 40x40x128 float B input at S_OUT  ->  B_i8
#   - The DUT receives A_i8 + B_i8 already on the same int8 grid, performs
#     nearest-neighbour 2x upsample on A (scale-preserving) and
#     channel-concat with B (also scale-preserving), and emits the
#     40x40x384 i8 tensor.
#   - Compare DUT output against the ORT float Concat, dequantized.
#     Nearest-neighbour Resize + Concat is mathematically lossless on the
#     int8 grid; the cosine drop vs ORT is purely from the single S_OUT
#     quantization step.
#
# There is no padding hazard here (no spatial kernel), so the whole
# 40x40 frame is valid for the cosine comparison.

import json
import os
import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, TensorProto

HERE = os.path.dirname(os.path.abspath(__file__))
STIM = os.path.join(HERE, "stim")
os.makedirs(STIM, exist_ok=True)
MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    "/home/marty/Documents/aif/vibeyolo/integ/yolo26n/model_int8.onnx",
)

H_A, W_A, C_A = 20, 20, 256   # A: /model.10/cv2 output
H_B, W_B, C_B = 40, 40, 128   # B: /model.6/cv2 output (P3 skip)
H_O, W_O, C_O = 40, 40, 384   # Output: concat(upsample(A), B)
assert C_O == C_A + C_B
assert H_O == 2 * H_A and W_O == 2 * W_A and H_O == H_B and W_O == W_B

A_NAME = "/model.10/cv2/act/Mul_output_0"
B_NAME = "/model.6/cv2/act/Mul_output_0"
RESIZE_OUT = "/model.11/Resize_output_0"
CONCAT_OUT = "/model.12/Concat_output_0"

# Expose intermediate tensors as model outputs so ORT will yield them.
mod_model = onnx.load(MODEL_PATH)
existing = {o.name for o in mod_model.graph.output}
for nm in [A_NAME, B_NAME, RESIZE_OUT, CONCAT_OUT]:
    if nm not in existing:
        mod_model.graph.output.append(
            helper.make_tensor_value_info(nm, TensorProto.FLOAT, None)
        )
tmp_path = os.path.join(HERE, "_model_with_intermediate.onnx")
onnx.save(mod_model, tmp_path)

so = ort.SessionOptions()
so.log_severity_level = 3
sess = ort.InferenceSession(tmp_path, sess_options=so,
                            providers=["CPUExecutionProvider"])
ipt_name = sess.get_inputs()[0].name


def make_input(kind, seed=0):
    img = np.zeros((1, 3, 640, 640), dtype=np.float32)
    rng = np.random.RandomState(seed)
    if kind == "rand":
        img[0] = rng.uniform(0.0, 1.0, (3, 640, 640)).astype(np.float32)
    elif kind == "rand_low":
        img[0] = rng.uniform(0.15, 0.80, (3, 640, 640)).astype(np.float32)
    elif kind == "rand_high":
        img[0] = rng.uniform(0.5, 1.0, (3, 640, 640)).astype(np.float32)
    elif kind == "half":
        img[0] = 0.5
    elif kind == "gradient":
        gx = np.tile(np.linspace(0, 1, 640, dtype=np.float32), (640, 1))
        img[0, 0] = gx
        img[0, 1] = gx.T
        img[0, 2] = 0.5 * (gx + gx.T)
    else:
        raise ValueError(kind)
    return img


def run_ort(img):
    a, b, rsz, cc = sess.run(
        [A_NAME, B_NAME, RESIZE_OUT, CONCAT_OUT], {ipt_name: img}
    )
    return np.asarray(a), np.asarray(b), np.asarray(rsz), np.asarray(cc)


samples = [
    ("rand0",    "rand",     0),
    ("rand1",    "rand",     1),
    ("rand2",    "rand",     2),
    ("half",     "half",     0),
    ("gradient", "gradient", 0),
]

# ── First pass: observe per-sample joint amplitude (A + B). The DUT
# treats both streams as already-on-the-same-int8-grid, so we pick a
# per-sample S_OUT that covers both. In a real chip this scale would
# come from a calibrated upstream fp16 rescale; here we re-derive it
# per sample to stay faithful to the rule-of-thumb max*1.1/127.
ort_cache = []
sample_s_out = {}
for sname, kind, seed in samples:
    img = make_input(kind, seed)
    a, b, rsz, cc = run_ort(img)
    joint_max = max(float(np.abs(a).max()), float(np.abs(b).max()))
    s_this = max(joint_max * 1.1 / 127.0, 1e-3 / 127.0)
    sample_s_out[sname] = s_this
    ort_cache.append((sname, a, b, rsz, cc))
    print(f"  {sname}: joint max-abs = {joint_max:.4f}  ->  s_out = {s_this:.6f}")


def quantize_i8(x_f32, scale):
    q = np.round(x_f32 / scale).astype(np.int32)
    return np.clip(q, -128, 127).astype(np.int8)


def sw_upsample_concat_i8(a_i8, b_i8):
    """Software int8 'HW reference' computed by the same operations the
    DUT performs: nearest-neighbour 2x upsample on a_i8, then channel-
    concat with b_i8."""
    Ca_, Ha_, Wa_ = a_i8.shape
    Cb_, Hb_, Wb_ = b_i8.shape
    assert Hb_ == 2 * Ha_ and Wb_ == 2 * Wa_
    # NN 2x via index broadcasting (ho = h*2 + dh, wo = w*2 + dw).
    a_up = np.repeat(np.repeat(a_i8, 2, axis=1), 2, axis=2)  # (Ca, 2*Ha, 2*Wa)
    out = np.concatenate([a_up, b_i8], axis=0)               # (Ca+Cb, 2*Ha, 2*Wa)
    return out


manifest = {
    "H_A": H_A, "W_A": W_A, "C_A": C_A,
    "H_B": H_B, "W_B": W_B, "C_B": C_B,
    "H_O": H_O, "W_O": W_O, "C_O": C_O,
    "samples": [],
}

for (sname, a_f, b_f, rsz_f, cc_f) in ort_cache:
    a_f = a_f[0]   # (256, 20, 20)
    b_f = b_f[0]   # (128, 40, 40)
    cc_f = cc_f[0] # (384, 40, 40)

    S_OUT = sample_s_out[sname]
    a_i8 = quantize_i8(a_f, S_OUT)
    b_i8 = quantize_i8(b_f, S_OUT)

    out_i8 = sw_upsample_concat_i8(a_i8, b_i8)  # (384, 40, 40)

    # Cosine vs ORT (full frame — no padding/ROI restriction here).
    out_f_hw  = out_i8.astype(np.float32) * S_OUT
    out_f_ort = cc_f.astype(np.float32)
    a = out_f_hw.reshape(-1)
    b = out_f_ort.reshape(-1)
    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    mxe = float(np.max(np.abs(a - b)))
    print(f"[{sname}]  cos(HW_sw vs ORT) = {cos:.6f}   max_abs_err = {mxe:.4f}")

    # ── Emit hex files ────────────────────────────────────────────
    # A input frame: pixel-major, channel-minor (h,w,c). H_A*W_A*C_A bytes.
    with open(os.path.join(STIM, f"{sname}.a_i8.hex"), "w") as f:
        for h in range(H_A):
            for w in range(W_A):
                for c in range(C_A):
                    f.write(f"{int(a_i8[c, h, w]) & 0xFF:02x}\n")

    # B input frame: pixel-major (h,w,c). H_B*W_B*C_B bytes.
    with open(os.path.join(STIM, f"{sname}.b_i8.hex"), "w") as f:
        for h in range(H_B):
            for w in range(W_B):
                for c in range(C_B):
                    f.write(f"{int(b_i8[c, h, w]) & 0xFF:02x}\n")

    # Golden output frame: pixel-major (h,w,c) over C_O = 384 chans, where
    # the lower C_A bytes are the upsampled-A pixel and the upper C_B
    # bytes are the B pixel (this matches the DUT's odata layout).
    with open(os.path.join(STIM, f"{sname}.golden_i8.hex"), "w") as f:
        for h in range(H_O):
            for w in range(W_O):
                for c in range(C_O):
                    f.write(f"{int(out_i8[c, h, w]) & 0xFF:02x}\n")

    # ORT reference (float32) for the full output frame.
    with open(os.path.join(STIM, f"{sname}.ref_ort.f32.hex"), "w") as f:
        for h in range(H_O):
            for w in range(W_O):
                for c in range(C_O):
                    bits = np.float32(cc_f[c, h, w]).view(np.uint32)
                    f.write(f"{int(bits):08x}\n")

    manifest["samples"].append({
        "name": sname,
        "s_out": S_OUT,
        "cos_sw_vs_ort": cos,
        "max_abs_err":   mxe,
    })

with open(os.path.join(STIM, "manifest.json"), "w") as f:
    json.dump(manifest, f, indent=2)
print(f"Wrote stim+ref to {STIM}")
