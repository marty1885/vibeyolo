#!/usr/bin/env python3
# Copyright (c) 2026 vibeyolo
# SPDX-License-Identifier: Apache-2.0
#
# Attention block extractor for YOLO26n /model.22 A2C2f attention sub-block.
#
# Parametric clone of integ/attn_model10/extract.py. ONNX inspection
# confirms the inner attention shape sequence under
#   /model.22/m.0/m.0.1/...
# is byte-identical to /model.10/m/m.0/... (HEADS=2, DIM_Q=32, DIM_K=32,
# DIM_V=64, N=400, 20x20 spatial, C_FE=128, C_QKV=256). Same flash_attn
# IP and same structural shim apply; only the tensor names and per-tensor
# scales change.
#
# Topology (mirrors /model.10 attention):
#   Inputs (with per-tensor int8 scales):
#     QKV    : /model.22/m.0/m.0.1/attn/qkv/conv/Conv_output_0   256ch x 20x20
#     PE     : /model.22/m.0/m.0.1/attn/pe/conv/Conv_output_0    128ch x 20x20
#     PROJ   : /model.22/m.0/m.0.1/attn/proj/conv/Conv_output_0  128ch x 20x20
#     SPLIT1 : /model.22/m.0/m.0.0/Add_output_0                  128ch x 20x20
#              (residual src for /model.22/m.0/m.0.1/Add, i.e. the
#              output of the m.0.0 conv-pair upstream of the attention
#              sub-block. In /model.10 this corresponded to the second
#              split of /model.10/Split; in A2C2f the corresponding role
#              is filled by the m.0.0 output that flows into the m.0.1
#              residual stream.)
#     FFN1   : /model.22/m.0/m.0.1/ffn/ffn.1/conv/Conv_output_0  128ch x 20x20
#
#   Block boundaries:
#     ATTN_ADD : /model.22/m.0/m.0.1/attn/Add_output_0           (post pe-add)
#     PROJ_RES : /model.22/m.0/m.0.1/Add_output_0                (post proj-residual)
#     FINAL    : /model.22/m.0/m.0.1/Add_1_output_0              (final block output)
#
# Output: ATTN_FINAL int8 at S_OUT (per-sample fp16 scale chosen to cover
# the joint range across all internal residual streams).

import json
import os
import math
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

H, W = 20, 20
N = H * W                  # 400 tokens
HEADS = 2
DIM_Q = 32
DIM_K = 32
DIM_V = 64
C_QKV = HEADS * (DIM_Q + DIM_K + DIM_V)   # 256
C_FE  = HEADS * DIM_V                      # 128
SOFTMAX_SCALE = 1.0 / math.sqrt(DIM_Q)     # ~0.17677669

# Tensor names from the ONNX graph (verified via shape_inference).
QKV_OUT    = "/model.22/m.0/m.0.1/attn/qkv/conv/Conv_output_0"
PE_OUT     = "/model.22/m.0/m.0.1/attn/pe/conv/Conv_output_0"
PROJ_OUT   = "/model.22/m.0/m.0.1/attn/proj/conv/Conv_output_0"
SPLIT1_OUT = "/model.22/m.0/m.0.0/Add_output_0"
FFN1_OUT   = "/model.22/m.0/m.0.1/ffn/ffn.1/conv/Conv_output_0"
ATTN_ADD   = "/model.22/m.0/m.0.1/attn/Add_output_0"
PROJ_RES   = "/model.22/m.0/m.0.1/Add_output_0"
FINAL_ADD  = "/model.22/m.0/m.0.1/Add_1_output_0"

# Expose intermediate tensors as model outputs.
mod_model = onnx.load(MODEL_PATH)
existing = {o.name for o in mod_model.graph.output}
for nm in [QKV_OUT, PE_OUT, PROJ_OUT, SPLIT1_OUT, FFN1_OUT,
           ATTN_ADD, PROJ_RES, FINAL_ADD]:
    if nm not in existing:
        mod_model.graph.output.append(
            helper.make_tensor_value_info(nm, TensorProto.FLOAT, None)
        )
tmp_path = os.path.join(HERE, "_model_with_intermediate.onnx")
onnx.save(mod_model, tmp_path)

so = ort.SessionOptions(); so.log_severity_level = 3
sess = ort.InferenceSession(tmp_path, sess_options=so,
                            providers=["CPUExecutionProvider"])
ipt_name = sess.get_inputs()[0].name

OUTS = [QKV_OUT, PE_OUT, PROJ_OUT, SPLIT1_OUT, FFN1_OUT,
        ATTN_ADD, PROJ_RES, FINAL_ADD]


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
        img[0, 0] = gx; img[0, 1] = gx.T; img[0, 2] = 0.5 * (gx + gx.T)
    else:
        raise ValueError(kind)
    return img


def run_ort(img):
    res = sess.run(OUTS, {ipt_name: img})
    return {nm: np.asarray(t) for nm, t in zip(OUTS, res)}


def quantize_i8(x_f32, scale):
    q = np.round(x_f32 / scale).astype(np.int32)
    return np.clip(q, -128, 127).astype(np.int8)


def real_to_fp16(x):
    return np.float16(x)


def sw_attn(qkv_f32, pe_f32):
    """qkv_f32: (256, 20, 20)  pe_f32: (128, 20, 20)
    Returns float32 ATTN_ADD output (128, 20, 20)."""
    qkv = qkv_f32.reshape(HEADS, DIM_Q + DIM_K + DIM_V, N)
    Q = qkv[:, 0:DIM_Q, :]
    K = qkv[:, DIM_Q:DIM_Q + DIM_K, :]
    V = qkv[:, DIM_Q + DIM_K:, :]

    scores = np.einsum("hdn,hdm->hnm", Q, K).astype(np.float32)
    scores *= np.float32(SOFTMAX_SCALE)
    m = scores.max(axis=-1, keepdims=True)
    e = np.exp((scores - m).astype(np.float64))
    p = (e / e.sum(axis=-1, keepdims=True)).astype(np.float32)
    pT = np.transpose(p, (0, 2, 1))
    out = np.einsum("hdn,hnm->hdm", V, pT).astype(np.float32)
    out = out.reshape(HEADS * DIM_V, H, W)
    return out + pe_f32


samples = [
    ("rand0",    "rand",     0),
    ("rand1",    "rand",     1),
    ("rand2",    "rand",     2),
    ("half",     "half",     0),
    ("gradient", "gradient", 0),
]

print("Running ORT to gather conv outputs + golden boundaries...")
ort_cache = []
max_abs = {nm: 0.0 for nm in OUTS}
for sname, kind, seed in samples:
    img = make_input(kind, seed)
    res = run_ort(img)
    ort_cache.append((sname, res))
    for nm in OUTS:
        max_abs[nm] = max(max_abs[nm], float(np.abs(res[nm]).max()))
    print(f"  {sname} done")

print("Observed max-abs:")
for nm in OUTS:
    print(f"  {nm:60s}  {max_abs[nm]:.4f}")

def sc(nm):
    return max(max_abs[nm] * 1.1 / 127.0, 1e-3 / 127.0)


S_QKV   = sc(QKV_OUT)
S_PE    = sc(PE_OUT)
S_PROJ  = sc(PROJ_OUT)
S_SPL1  = sc(SPLIT1_OUT)
S_FFN1  = sc(FFN1_OUT)
S_AOUT  = sc(ATTN_ADD)
S_RES   = sc(PROJ_RES)
S_OUT   = sc(FINAL_ADD)

print(f"Scales chosen:  S_QKV={S_QKV:.6f}  S_PE={S_PE:.6f}  S_PROJ={S_PROJ:.6f}")
print(f"                S_SPL1={S_SPL1:.6f}  S_FFN1={S_FFN1:.6f}")
print(f"                S_AOUT={S_AOUT:.6f}  S_RES={S_RES:.6f}  S_OUT={S_OUT:.6f}")

manifest = {
    "H": H, "W": W,
    "C_QKV": C_QKV, "C_FE": C_FE,
    "HEADS": HEADS, "DIM_Q": DIM_Q, "DIM_K": DIM_K, "DIM_V": DIM_V,
    "N": N,
    "softmax_scale": SOFTMAX_SCALE,
    "S_QKV": S_QKV, "S_PE": S_PE, "S_PROJ": S_PROJ,
    "S_SPL1": S_SPL1, "S_FFN1": S_FFN1,
    "S_AOUT": S_AOUT, "S_RES": S_RES, "S_OUT": S_OUT,
    "samples": [],
}


def sw_block_end_to_end(qkv_i8, pe_i8, proj_i8, spl1_i8, ffn1_i8):
    qkv  = qkv_i8.astype(np.float32)  * np.float32(S_QKV)
    pe   = pe_i8.astype(np.float32)   * np.float32(S_PE)
    proj = proj_i8.astype(np.float32) * np.float32(S_PROJ)
    spl1 = spl1_i8.astype(np.float32) * np.float32(S_SPL1)
    ffn1 = ffn1_i8.astype(np.float32) * np.float32(S_FFN1)

    attn_out_f = sw_attn(qkv, pe)
    attn_out_i8 = quantize_i8(attn_out_f, S_AOUT)

    res_f = proj + spl1
    res_i8 = quantize_i8(res_f, S_RES)

    final_f = res_f + ffn1
    final_i8 = quantize_i8(final_f, S_OUT)

    return attn_out_i8, res_i8, final_i8, attn_out_f, res_f, final_f


for (sname, res) in ort_cache:
    qkv_f  = res[QKV_OUT][0]
    pe_f   = res[PE_OUT][0]
    proj_f = res[PROJ_OUT][0]
    spl1_f = res[SPLIT1_OUT][0]
    ffn1_f = res[FFN1_OUT][0]

    attn_ort  = res[ATTN_ADD][0]
    proj_ort  = res[PROJ_RES][0]
    final_ort = res[FINAL_ADD][0]

    qkv_i8  = quantize_i8(qkv_f,  S_QKV)
    pe_i8   = quantize_i8(pe_f,   S_PE)
    proj_i8 = quantize_i8(proj_f, S_PROJ)
    spl1_i8 = quantize_i8(spl1_f, S_SPL1)
    ffn1_i8 = quantize_i8(ffn1_f, S_FFN1)

    attn_sw_i8, res_sw_i8, final_sw_i8, attn_sw_f, res_sw_f, final_sw_f = \
        sw_block_end_to_end(qkv_i8, pe_i8, proj_i8, spl1_i8, ffn1_i8)

    def cos(a, b):
        a = a.reshape(-1).astype(np.float64); b = b.reshape(-1).astype(np.float64)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))

    cos_attn  = cos(attn_sw_i8.astype(np.float32) * S_AOUT, attn_ort)
    cos_res   = cos(res_sw_i8.astype(np.float32) * S_RES,  proj_ort)
    cos_final = cos(final_sw_i8.astype(np.float32) * S_OUT, final_ort)
    print(f"[{sname}]  cos(SW vs ORT):  attn={cos_attn:.6f}  res={cos_res:.6f}  final={cos_final:.6f}")

    def write_i8_pixmajor(path, tensor_chw):
        Ch, Hh, Ww = tensor_chw.shape
        with open(path, "w") as f:
            for h in range(Hh):
                for w in range(Ww):
                    for c in range(Ch):
                        f.write(f"{int(tensor_chw[c, h, w]) & 0xFF:02x}\n")

    def write_f32_pixmajor(path, tensor_chw):
        Ch, Hh, Ww = tensor_chw.shape
        with open(path, "w") as f:
            for h in range(Hh):
                for w in range(Ww):
                    for c in range(Ch):
                        bits = np.float32(tensor_chw[c, h, w]).view(np.uint32)
                        f.write(f"{int(bits):08x}\n")

    write_i8_pixmajor(os.path.join(STIM, f"{sname}.qkv_i8.hex"),  qkv_i8)
    write_i8_pixmajor(os.path.join(STIM, f"{sname}.pe_i8.hex"),   pe_i8)
    write_i8_pixmajor(os.path.join(STIM, f"{sname}.proj_i8.hex"), proj_i8)
    write_i8_pixmajor(os.path.join(STIM, f"{sname}.spl1_i8.hex"), spl1_i8)
    write_i8_pixmajor(os.path.join(STIM, f"{sname}.ffn1_i8.hex"), ffn1_i8)

    write_i8_pixmajor(os.path.join(STIM, f"{sname}.golden_final_i8.hex"), final_sw_i8)
    write_f32_pixmajor(os.path.join(STIM, f"{sname}.ref_final_f32.hex"),  final_ort)
    write_i8_pixmajor(os.path.join(STIM, f"{sname}.golden_attn_i8.hex"),  attn_sw_i8)
    write_f32_pixmajor(os.path.join(STIM, f"{sname}.ref_attn_f32.hex"),   attn_ort)

    manifest["samples"].append({
        "name": sname,
        "cos_sw_vs_ort_attn":  cos_attn,
        "cos_sw_vs_ort_res":   cos_res,
        "cos_sw_vs_ort_final": cos_final,
    })

with open(os.path.join(STIM, "manifest.json"), "w") as f:
    json.dump(manifest, f, indent=2)

def fp16_bits(x):
    return int(np.float16(x).view(np.uint16))

pkg_path = os.path.join(STIM, "attn_scales_pkg.sv")
with open(pkg_path, "w") as f:
    f.write("// AUTO-GENERATED by integ/attn_model22/extract.py - do not edit.\n")
    f.write("package attn_scales_pkg;\n")
    f.write(f"  localparam logic [15:0] S_QKV_FP16    = 16'h{fp16_bits(S_QKV):04x};\n")
    f.write(f"  localparam logic [15:0] S_PE_FP16     = 16'h{fp16_bits(S_PE):04x};\n")
    f.write(f"  localparam logic [15:0] S_PROJ_FP16   = 16'h{fp16_bits(S_PROJ):04x};\n")
    f.write(f"  localparam logic [15:0] S_SPL1_FP16   = 16'h{fp16_bits(S_SPL1):04x};\n")
    f.write(f"  localparam logic [15:0] S_FFN1_FP16   = 16'h{fp16_bits(S_FFN1):04x};\n")
    f.write(f"  localparam logic [15:0] INV_S_AOUT_FP16 = 16'h{fp16_bits(1.0 / S_AOUT):04x};\n")
    f.write(f"  localparam logic [15:0] INV_S_OUT_FP16  = 16'h{fp16_bits(1.0 / S_OUT):04x};\n")
    f.write(f"  localparam real PARAM_S_QKV   = {S_QKV:.10e};\n")
    f.write(f"  localparam real PARAM_S_PE    = {S_PE:.10e};\n")
    f.write(f"  localparam real PARAM_S_PROJ  = {S_PROJ:.10e};\n")
    f.write(f"  localparam real PARAM_S_SPL1  = {S_SPL1:.10e};\n")
    f.write(f"  localparam real PARAM_S_FFN1  = {S_FFN1:.10e};\n")
    f.write(f"  localparam real PARAM_S_AOUT  = {S_AOUT:.10e};\n")
    f.write(f"  localparam real PARAM_S_OUT   = {S_OUT:.10e};\n")
    f.write("endpackage\n")

print(f"Wrote stim+ref to {STIM}")
