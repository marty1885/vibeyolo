#!/usr/bin/env python3
# Copyright (c) 2026 vibeyolo
# SPDX-License-Identifier: Apache-2.0
#
# SPPF extractor: drives the real model.9 SPPF subgraph in ORT, then
# computes an int8-quantized version of the same subgraph in software
# to use as the golden the SystemVerilog DUT must match.
#
# Pipeline being modeled:
#     /model.9/cv1/conv/Conv_output_0  (float32, B×128×20×20)
#       │
#       ├── (cv1 stream)
#       │
#       └── MaxPool 5×5 s=1 p=2  → mp1
#                │
#                └── MaxPool ... → mp2
#                         │
#                         └── MaxPool ... → mp3
#       Concat axis=1 → B×512×20×20
#
# For HW we quantize the float cv1 output to int8 with a chosen scale
# S_IN (covers the observed range). MaxPool is exact on int8 (max
# preserves the scale). Concat is just channel-wise packing.
#
# Padding note: linebuf_kxk uses ZERO padding; ONNX MaxPool uses -INF
# (ignored cells). For a 13×13 effective receptive field on a 20×20
# frame, only the inner 8×8 ROI (positions 6..13 × 6..13) is fully
# valid for both schemes. We compare DUT vs ORT/golden on that ROI.

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

H, W, C, K = 20, 20, 128, 5
P_RF = 3 * (K // 2)            # total padding distance per side = 6
ROI_LO = P_RF                  # 6
ROI_HI = H - P_RF              # 14 (exclusive)
ROI_H = ROI_HI - ROI_LO        # 8
ROI_W = ROI_HI - ROI_LO        # 8

CV1_OUT  = "/model.9/cv1/conv/Conv_output_0"
MP1_OUT  = "/model.9/m/MaxPool_output_0"
MP2_OUT  = "/model.9/m_1/MaxPool_output_0"
MP3_OUT  = "/model.9/m_2/MaxPool_output_0"
CONCAT   = "/model.9/Concat_output_0"

# Expose the intermediate tensors as outputs of the model so we can
# capture them directly from ORT.
mod_model = onnx.load(MODEL_PATH)
existing = {o.name for o in mod_model.graph.output}
for nm in [CV1_OUT, MP1_OUT, MP2_OUT, MP3_OUT, CONCAT]:
    if nm not in existing:
        mod_model.graph.output.append(
            helper.make_tensor_value_info(nm, TensorProto.FLOAT, None)
        )
tmp_path = os.path.join(HERE, "_model_with_intermediate.onnx")
onnx.save(mod_model, tmp_path)

so = ort.SessionOptions()
so.log_severity_level = 3
sess = ort.InferenceSession(tmp_path, sess_options=so, providers=["CPUExecutionProvider"])
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
    o = sess.run([CV1_OUT, MP1_OUT, MP2_OUT, MP3_OUT, CONCAT], {ipt_name: img})
    return tuple(np.asarray(t) for t in o)


# ── First pass: observe cv1 amplitude across samples to size S_IN ────
samples = [
    ("rand0",     "rand",      0),
    ("rand1",     "rand",      1),
    ("rand2",     "rand",      2),
    ("half",      "half",      0),
    ("gradient",  "gradient",  0),
]

cv1_max_abs = 0.0
ort_cache = []
for sname, kind, seed in samples:
    img = make_input(kind, seed)
    cv1, mp1, mp2, mp3, cc = run_ort(img)
    cv1_max_abs = max(cv1_max_abs, float(np.abs(cv1).max()))
    ort_cache.append((sname, cv1, mp1, mp2, mp3, cc))

# Pick S_IN = max(|cv1|) * 1.1 / 127  (rule of thumb from the codebase).
S_IN = max(cv1_max_abs * 1.1 / 127.0, 1e-3 / 127.0)
print(f"observed cv1 max-abs = {cv1_max_abs:.4f} -> S_IN = {S_IN:.6f}")


def quantize_i8(x_f32, scale):
    q = np.round(x_f32 / scale).astype(np.int32)
    return np.clip(q, -128, 127).astype(np.int8)


def sw_sppf_i8(cv1_i8):
    """Software 'HW reference': compute the int8 SPPF chain with the
    same zero-padding semantics that the RTL (via linebuf) implements."""
    C_, H_, W_ = cv1_i8.shape
    P = K // 2

    def maxpool_zeropad(x_i8):
        # Pad with zeros (matches linebuf), then K×K max.
        padded = np.zeros((C_, H_ + 2 * P, W_ + 2 * P), dtype=np.int16)
        padded[:, P:P + H_, P:P + W_] = x_i8.astype(np.int16)
        out = np.full((C_, H_, W_), -32768, dtype=np.int16)
        for ky in range(K):
            for kx in range(K):
                window = padded[:, ky:ky + H_, kx:kx + W_]
                out = np.maximum(out, window)
        return out.astype(np.int8)

    mp1 = maxpool_zeropad(cv1_i8)
    mp2 = maxpool_zeropad(mp1)
    mp3 = maxpool_zeropad(mp2)
    return mp1, mp2, mp3


manifest = {
    "H": H, "W": W, "C": C, "K": K,
    "roi_lo": ROI_LO, "roi_hi": ROI_HI, "roi_h": ROI_H, "roi_w": ROI_W,
    "s_in": S_IN,
    "samples": [],
}

for (sname, cv1, mp1_ort, mp2_ort, mp3_ort, concat_ort) in ort_cache:
    # cv1 frame: B×128×20×20 (B=1). Take the entire frame.
    cv1_f = cv1[0]  # shape (128, 20, 20)
    cv1_i8 = quantize_i8(cv1_f, S_IN)

    # Software int8 SPPF for golden.
    mp1_i8_sw, mp2_i8_sw, mp3_i8_sw = sw_sppf_i8(cv1_i8)

    # Build a 512-channel int8 'concat' frame in {cv1, mp1, mp2, mp3} order
    # — same order the RTL emits.
    concat_i8_sw = np.concatenate(
        [cv1_i8, mp1_i8_sw, mp2_i8_sw, mp3_i8_sw], axis=0
    ).astype(np.int8)  # (512, 20, 20)

    # Compute cosine vs ORT on the inner ROI (after dequantizing the
    # HW golden back to float).
    concat_f_hw  = concat_i8_sw.astype(np.float32) * S_IN
    concat_f_ort = concat_ort[0].astype(np.float32)

    sl = (slice(None), slice(ROI_LO, ROI_HI), slice(ROI_LO, ROI_HI))
    a = concat_f_hw[sl].reshape(-1)
    b = concat_f_ort[sl].reshape(-1)
    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    mxe = float(np.max(np.abs(a - b)))
    print(f"[{sname}] inner-ROI cos(HW_sw vs ORT) = {cos:.6f}  max_abs={mxe:.4f}")

    # ── Emit hex files ────────────────────────────────────────
    # Input frame: row-major (h,w,c) channel-minor, H*W*C bytes
    with open(os.path.join(STIM, f"{sname}.input_i8.hex"), "w") as f:
        for h in range(H):
            for w in range(W):
                for c in range(C):
                    f.write(f"{int(cv1_i8[c, h, w]) & 0xFF:02x}\n")

    # Golden output frame: (h,w,4*c) channel-minor, H*W*4*C bytes
    with open(os.path.join(STIM, f"{sname}.golden_i8.hex"), "w") as f:
        for h in range(H):
            for w in range(W):
                for ch in range(4 * C):
                    f.write(f"{int(concat_i8_sw[ch, h, w]) & 0xFF:02x}\n")

    # ORT reference (float32) for the ROI, channel-minor (h,w,4*c)
    with open(os.path.join(STIM, f"{sname}.ref_ort.f32.hex"), "w") as f:
        for h in range(ROI_LO, ROI_HI):
            for w in range(ROI_LO, ROI_HI):
                for ch in range(4 * C):
                    bits = np.float32(concat_f_ort[ch, h, w]).view(np.uint32)
                    f.write(f"{int(bits):08x}\n")

    manifest["samples"].append({
        "name": sname,
        "inner_roi_cos_sw_vs_ort": cos,
        "inner_roi_max_abs_err":  mxe,
    })

with open(os.path.join(STIM, "manifest.json"), "w") as f:
    json.dump(manifest, f, indent=2)
print(f"Wrote stim+ref to {STIM}")
