#!/usr/bin/env python3
# Copyright (c) 2026 vibeyolo
# SPDX-License-Identifier: Apache-2.0
#
# Upsample integration extractor for the P3->detect neck join in YOLO26n
# (second of the two neck upsamples; near-clone of upsample_model11).
#
# Models the ONNX subgraph:
#
#     /model.13/cv2/act/Mul_output_0   (B x 128 x 40 x 40  float)
#                                       │
#                                       ▼
#                            /model.14/Resize  (nearest, scale=2)
#                                       │
#                                       ▼
#                                /model.14/Resize_output_0
#                                       │      (B x 128 x 80 x 80)
#                                       │
#                                       │            /model.4/cv2/act/Mul_output_0
#                                       │            (B x 128 x 80 x 80  float)
#                                       │                  │
#                                       ▼                  ▼
#                              /model.15/Concat (axis=1)
#                                       │
#                                       ▼
#                            /model.15/Concat_output_0  (B x 256 x 80 x 80)
#
# Concat order verified via onnx.shape_inference: [Resize_out, /model.4/cv2].
# This output then feeds DynamicQuantizeLinear and /model.16/cv1 Conv.
#
# Quantization plan (identical to upsample_model11):
#   - Pick a single output-tensor scale S_OUT that covers BOTH input streams'
#     range. In a real chip an upstream per-stream fp16 rescale would put
#     both producers on this common grid; here we re-derive per-sample to
#     follow the max*1.1/127 rule of thumb.
#   - Quantize A (40x40x128, the upsample source) and B (80x80x128, the
#     P3 skip) at S_OUT.
#   - DUT does NN-2x upsample of A + channel-concat with B and emits
#     80x80x256 i8.
#   - Compare against ORT float Concat dequantized.

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

H_A, W_A, C_A = 40, 40, 128   # A: /model.13/cv2 output (upsample src)
H_B, W_B, C_B = 80, 80, 128   # B: /model.4/cv2  output (P3 skip)
H_O, W_O, C_O = 80, 80, 256   # Output: concat(upsample(A), B)
assert C_O == C_A + C_B
assert H_O == 2 * H_A and W_O == 2 * W_A and H_O == H_B and W_O == W_B

A_NAME = "/model.13/cv2/act/Mul_output_0"
B_NAME = "/model.4/cv2/act/Mul_output_0"
RESIZE_OUT = "/model.14/Resize_output_0"
CONCAT_OUT = "/model.15/Concat_output_0"

# Expose intermediate tensors as model outputs.
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
    Ca_, Ha_, Wa_ = a_i8.shape
    Cb_, Hb_, Wb_ = b_i8.shape
    assert Hb_ == 2 * Ha_ and Wb_ == 2 * Wa_
    a_up = np.repeat(np.repeat(a_i8, 2, axis=1), 2, axis=2)
    out = np.concatenate([a_up, b_i8], axis=0)
    return out


manifest = {
    "H_A": H_A, "W_A": W_A, "C_A": C_A,
    "H_B": H_B, "W_B": W_B, "C_B": C_B,
    "H_O": H_O, "W_O": W_O, "C_O": C_O,
    "samples": [],
}

for (sname, a_f, b_f, rsz_f, cc_f) in ort_cache:
    a_f = a_f[0]
    b_f = b_f[0]
    cc_f = cc_f[0]

    S_OUT = sample_s_out[sname]
    a_i8 = quantize_i8(a_f, S_OUT)
    b_i8 = quantize_i8(b_f, S_OUT)

    out_i8 = sw_upsample_concat_i8(a_i8, b_i8)

    out_f_hw  = out_i8.astype(np.float32) * S_OUT
    out_f_ort = cc_f.astype(np.float32)
    a = out_f_hw.reshape(-1)
    b = out_f_ort.reshape(-1)
    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    mxe = float(np.max(np.abs(a - b)))
    print(f"[{sname}]  cos(HW_sw vs ORT) = {cos:.6f}   max_abs_err = {mxe:.4f}")

    with open(os.path.join(STIM, f"{sname}.a_i8.hex"), "w") as f:
        for h in range(H_A):
            for w in range(W_A):
                for c in range(C_A):
                    f.write(f"{int(a_i8[c, h, w]) & 0xFF:02x}\n")

    with open(os.path.join(STIM, f"{sname}.b_i8.hex"), "w") as f:
        for h in range(H_B):
            for w in range(W_B):
                for c in range(C_B):
                    f.write(f"{int(b_i8[c, h, w]) & 0xFF:02x}\n")

    with open(os.path.join(STIM, f"{sname}.golden_i8.hex"), "w") as f:
        for h in range(H_O):
            for w in range(W_O):
                for c in range(C_O):
                    f.write(f"{int(out_i8[c, h, w]) & 0xFF:02x}\n")

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
