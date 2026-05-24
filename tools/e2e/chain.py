#!/usr/bin/env python3
# Copyright (c) 2026 vibeyolo
# SPDX-License-Identifier: Apache-2.0
#
# chain.py — chained end-to-end "chip" run on a real image.
#
# Interprets the real ONNX graph node-by-node in numpy, BUT every conv is
# executed as the chip executes it (symmetric int8 in, int8 weights, fp16-ish
# requant + SiLU LUT, symmetric int8 out — the exact arithmetic Phase 1 proved
# equals the conv_stage RTL bit-for-bit). All non-conv ops (concat / slice /
# residual add / upsample / maxpool / attention matmul+softmax / detect
# decode+topk) run in fp32. Activations are carried as the chip's *reconstructed*
# values (int8_code * scale) so quantization error accumulates exactly where the
# chip quantizes. Final logits/pred_boxes are compared to ORT.
#
# Conv backend is pluggable:  CONV=numpy (default, fast, == RTL per Phase 1) or
#                             CONV=rtl   (drive the real conv_stage Verilator).
#
#   python3 tools/e2e/chain.py [--stop-after IDX] [--conv numpy|rtl]

import argparse
import json
import os
import re
import subprocess
import sys

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ONNX = os.path.join(ROOT, "integ/yolo26n/model_int8.onnx")
E2E = os.path.join(ROOT, "integ/generated/e2e")
COSIM = os.path.join(ROOT, "tools/e2e/cosim")
CALIB_JSON = os.path.join(E2E, "calib_scales.json")


def load_calib():
    """corpus-calibrated per-conv fixed scales (calibrate.py); name -> dict.
    Covers ALL 102 convs incl. the attn-internal ones meta.json misses."""
    if os.path.exists(CALIB_JSON):
        return json.load(open(CALIB_JSON)).get("scales", {})
    return {}


def cos(a, b):
    a = np.asarray(a).reshape(-1).astype(np.float64)
    b = np.asarray(b).reshape(-1).astype(np.float64)
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / d) if d else 0.0


# ───────────────────────── ONNX graph + helpers ─────────────────────────
M = onnx.load(ONNX)
G = M.graph
INIT = {i.name: numpy_helper.to_array(i) for i in G.initializer}
PROD = {o: n for n in G.node for o in n.output}
PROD_BY_NAME = {n.name: n for n in G.node}


def attr(node, name, default=None):
    for a in node.attribute:
        if a.name == name:
            if a.type == onnx.AttributeProto.INT: return a.i
            if a.type == onnx.AttributeProto.INTS: return list(a.ints)
            if a.type == onnx.AttributeProto.FLOAT: return a.f
            if a.type == onnx.AttributeProto.STRING: return a.s.decode()
            if a.type == onnx.AttributeProto.TENSOR: return numpy_helper.to_array(a.t)
    return default


# ── conv cluster discovery: ConvInteger -> (fp_in tensor, out tensor, params) ──
def conv_params(conv):
    qname = conv.output[0]
    pre_t = qname[:-len("_output_quantized")]
    bias_add = PROD[pre_t]
    bias = None
    for inp in bias_add.input:
        rs = PROD.get(inp)
        if rs is not None and rs.op_type == "Reshape" and rs.input[0] in INIT:
            c = INIT[rs.input[0]].astype(np.float32)
            if c.ndim == 1: bias = c
    base = conv.name.rsplit("/conv/Conv_quant", 1)[0].rsplit("/Conv_quant", 1)[0]
    act_t = base + "/act/Mul_output_0"
    silu = act_t in PROD
    out_t = act_t if silu else pre_t
    dql = PROD[conv.input[0]]          # DynamicQuantizeLinear
    fp_in_t = dql.input[0]             # fp32 activation feeding the conv
    W_raw = INIT[conv.input[1]].astype(np.int32)
    s_w = float(INIT[conv.input[1].replace("_quantized", "_scale")])
    dw = (W_raw.ndim == 4 and W_raw.shape[1] == 1 and W_raw.shape[0] > 1)
    K = W_raw.shape[2]
    strides = attr(conv, "strides", [1, 1]); pads = attr(conv, "pads", [0, 0, 0, 0])
    return dict(fp_in=fp_in_t, out=out_t, bias=bias, silu=silu, dw=dw,
                W_raw=W_raw, s_w=s_w, K=K, S=strides[0], PAD=pads[0],
                conv=conv, dql=dql, pre=pre_t)


CONV_BY_DQL = {}      # conv node name -> params (one DQL may feed several convs)
for n in G.node:
    if n.op_type == "ConvInteger":
        CONV_BY_DQL[n.name] = conv_params(n)

# canonical idx for logging / per-layer S_OUT lookup
BAL = os.path.join(ROOT, "integ/generated/scale/scale_report_balanced.md")
NAME2IDX = {}
for line in open(BAL):
    mm = re.match(r"\|\s*(\d+)\s*\|\s*(/model[^|]+?)\s*\|", line)
    if mm: NAME2IDX[mm.group(2).strip()] = int(mm.group(1))


def layer_sout(conv_name):
    idx = NAME2IDX.get(conv_name)
    if idx is None: return None
    mp = os.path.join(E2E, f"layer_{idx:03d}", "meta.json")
    if os.path.exists(mp):
        m = json.load(open(mp))
        return idx, m["s_out_pre"], m["s_out_silu"]
    return idx, None, None


# ───────────────────────── chip-faithful conv ─────────────────────────
def conv_numpy(fp_in, s_in, p, s_pre, s_silu, out_bits=8):
    W = p["W_raw"]; K = p["K"]; S = p["S"]; PAD = p["PAD"]
    COUT = W.shape[0]
    if p["dw"]:
        CIN = fp_in.shape[0]
        Wd = np.zeros((COUT, CIN, K, K), dtype=np.float32)
        for co in range(COUT): Wd[co, co] = W[co, 0]
        W = Wd
    else:
        W = W.astype(np.float32)
    CIN, H_IN, W_IN = fp_in.shape
    H_OUT = (H_IN + 2 * PAD - K) // S + 1
    W_OUT = (W_IN + 2 * PAD - K) // S + 1
    codes = np.clip(np.round(fp_in / s_in), -128, 127).astype(np.float32)
    xp = np.zeros((CIN, H_IN + 2 * PAD, W_IN + 2 * PAD), dtype=np.float32)
    xp[:, PAD:PAD + H_IN, PAD:PAD + W_IN] = codes
    cols = np.empty((CIN, K, K, H_OUT, W_OUT), dtype=np.float32)
    for kh in range(K):
        for kw in range(K):
            cols[:, kh, kw] = xp[:, kh:kh + H_OUT * S:S, kw:kw + W_OUT * S:S]
    acc = np.tensordot(W, cols, axes=([1, 2, 3], [0, 1, 2]))   # (COUT,H,W)
    pre = acc * (s_in * p["s_w"]) + p["bias"][:, None, None]
    pre_code = np.clip(np.round(pre / s_pre), -128, 127) * s_pre
    y = pre_code / (1.0 + np.exp(-pre_code)) if p["silu"] else pre_code
    # out_bits>8: finer output quantization over the SAME range (s_eff=s*127/qmax),
    # i.e. a wider output code out of the requant — MAC array unchanged. Used to
    # study whether a wider cls-logit output breaks the int8 detection score ties.
    qmax = 2 ** (out_bits - 1) - 1
    s_eff = s_silu * 127.0 / qmax
    out_code = np.clip(np.round(y / s_eff), -qmax - 1, qmax)
    return (out_code * s_eff).astype(np.float32)


def conv_rtl(fp_in, s_in, p, s_pre, s_silu, idx):
    """Drive the real conv_stage Verilator (reuses the Phase-1 build dir)."""
    W = p["W_raw"]; K = p["K"]; S = p["S"]; PAD = p["PAD"]; COUT = W.shape[0]
    CIN, H_IN, W_IN = fp_in.shape
    codes = np.clip(np.round(fp_in / s_in), -128, 127).astype(np.int32)
    if p["dw"]:
        Wd = np.zeros((COUT, CIN, K, K), dtype=np.int32)
        for co in range(COUT): Wd[co, co] = W[co, 0]
        W = Wd
    # Write to the SAME dir the Phase-1 binary baked its WINIT/SINIT/BINIT -G
    # paths to (compile-time strings), so the cached binary reads the chain's
    # ROMs at $readmemh time. (Overwrites the dump hex; re-dump to restore.)
    cdir = os.path.join(E2E, f"layer_{idx:03d}")
    os.makedirs(cdir, exist_ok=True)
    # input frame raster channel-fastest
    raster = np.transpose(codes, (1, 2, 0)).reshape(-1)
    def wr(path, vals, w):
        with open(path, "w") as f:
            for v in vals: f.write(f"%0{w}x\n" % (int(v) & ((1 << (4 * w)) - 1)))
    wr(os.path.join(cdir, "input_i8.hex"), raster, 2)
    N_LANE = K * K * CIN
    Wf = np.zeros((COUT, N_LANE), dtype=np.int32)
    for kh in range(K):
        for kw in range(K):
            Wf[:, (kh * K + kw) * CIN:(kh * K + kw) * CIN + CIN] = W[:, :, kh, kw]
    wr(os.path.join(cdir, "weights.i8.hex"), Wf.reshape(-1), 2)
    def f16(x): return int(np.float16(x).view(np.uint16))
    wr(os.path.join(cdir, "scale_fp16.hex"), [f16(s_in * p["s_w"] / s_pre)] * COUT, 4)
    wr(os.path.join(cdir, "bias_fp16.hex"), [f16(b / s_pre) for b in p["bias"]], 4)
    mp = os.path.join(E2E, f"layer_{idx:03d}", "meta.json")
    if os.path.exists(mp):
        m = json.load(open(mp)); pcin = m["P_CIN"]; pcout = m["P_COUT"]
    else:                                  # attn/ffn convs (not in the dump)
        pcin = min(CIN, 8); pcout = min(COUT, 16)
    if K * K * pcin < 2: pcin = min(CIN, 2)
    pcout = max(pcout, min(COUT, 2))
    env = dict(os.environ, CIN=str(CIN), COUT=str(COUT), K=str(K), STRIDE=str(S),
               PAD=str(PAD), H_IN=str(H_IN), W_IN=str(W_IN), P_COUT=str(pcout),
               P_CIN=str(pcin), SILU=str(int(p["silu"])), S_OUT_PRE=repr(s_pre),
               S_OUT_SILU=repr(s_silu), STIM=cdir, LAYER=f"layer_{idx:03d}",
               VERILATOR_JOBS="3")
    r = subprocess.run(["make", "-C", COSIM, "test"], env=env,
                       capture_output=True, text=True)
    if r.returncode != 0:
        sys.stderr.write(r.stdout[-1500:] + r.stderr[-1500:])
        raise RuntimeError(f"RTL conv L{idx} failed")
    H_OUT = (H_IN + 2 * PAD - K) // S + 1; W_OUT = (W_IN + 2 * PAD - K) // S + 1
    o = np.array([int(l, 16) for l in open(os.path.join(cdir, "out_i8.hex"))], dtype=np.int32)
    o = ((o + 128) % 256 - 128).reshape(H_OUT, W_OUT, COUT).transpose(2, 0, 1)
    return (o.astype(np.float32) * s_silu)


# ───────────────────────── cluster ownership (skip internals) ─────────────
CONSUMERS = {}
for n in G.node:
    for i in n.input:
        CONSUMERS.setdefault(i, []).append(n)

TRIGGER = CONV_BY_DQL    # conv node name -> params (trigger on the ConvInteger)
# OWNED = every node strictly inside a conv cluster (dql, ConvInteger, cast, the
# two scale Muls, bias add, sigmoid, act Mul). Walk forward from each DQL,
# bounded by the set of all cluster-output tensors so a DQL shared by several
# convs doesn't leak past any cluster boundary. ConvInteger nodes are also added
# but the dispatch checks TRIGGER first, so they still execute.
STOP = {p["out"] for p in CONV_BY_DQL.values()}
OWNED = set()
DQLS = {p["dql"].name for p in CONV_BY_DQL.values()}
for dqln in DQLS:
    q = [PROD_BY_NAME[dqln]]
    seen = set()
    while q:
        nd = q.pop()
        if nd.name in seen: continue
        seen.add(nd.name); OWNED.add(nd.name)
        for o in nd.output:
            if o in STOP: continue
            for c in CONSUMERS.get(o, []):
                if c.name not in seen: q.append(c)


# ───────────────────────── numpy op handlers ─────────────────────────
def get(vals, t):
    if t in vals: return vals[t]
    if t in INIT: return INIT[t]
    raise KeyError(f"missing tensor {t}")


def run_op(node, vals):
    op = node.op_type
    I = [get(vals, t) if t else None for t in node.input]
    if op == "Cast":
        to = attr(node, "to")
        m = {1: np.float32, 6: np.int32, 7: np.int64, 9: np.bool_, 11: np.float64}
        return [I[0].astype(m.get(to, np.float32))]
    if op == "Mul":  return [I[0] * I[1]]
    if op == "Add":  return [I[0] + I[1]]
    if op == "Sub":  return [I[0] - I[1]]
    if op == "Div":  return [I[0] / I[1]]
    if op == "Sigmoid": return [1.0 / (1.0 + np.exp(-I[0]))]
    if op == "Concat":  return [np.concatenate(I, axis=attr(node, "axis"))]
    if op == "Transpose":
        perm = attr(node, "perm"); return [np.transpose(I[0], perm)]
    if op == "Reshape":
        shp = list(I[1].astype(np.int64)); return [I[0].reshape(shp)]
    if op == "Unsqueeze":
        axes = I[1] if len(I) > 1 else attr(node, "axes")
        x = I[0]
        for ax in sorted(int(a) for a in np.atleast_1d(axes)): x = np.expand_dims(x, ax)
        return [x]
    if op == "Squeeze":
        axes = I[1] if len(I) > 1 else attr(node, "axes")
        if axes is None: return [np.squeeze(I[0])]
        return [np.squeeze(I[0], axis=tuple(int(a) for a in np.atleast_1d(axes)))]
    if op == "Softmax":
        ax = attr(node, "axis", -1); x = I[0]
        e = np.exp(x - x.max(axis=ax, keepdims=True))
        return [e / e.sum(axis=ax, keepdims=True)]
    if op == "MatMul": return [np.matmul(I[0], I[1])]
    if op == "Tile":   return [np.tile(I[0], I[1].astype(np.int64))]
    if op == "ReduceMax":
        axes = I[1] if len(I) > 1 else attr(node, "axes")
        kd = attr(node, "keepdims", 1)
        return [np.max(I[0], axis=tuple(int(a) for a in np.atleast_1d(axes)), keepdims=bool(kd))]
    if op == "Split":
        ax = attr(node, "axis", 0)
        split = I[1] if len(I) > 1 else attr(node, "split")
        if split is None:
            n = len(node.output); return list(np.array_split(I[0], n, axis=ax))
        idx = np.cumsum([int(s) for s in np.atleast_1d(split)])[:-1]
        return list(np.split(I[0], idx, axis=ax))
    if op == "Slice":
        data = I[0]; starts = I[1].astype(np.int64); ends = I[2].astype(np.int64)
        axes = I[3].astype(np.int64) if len(I) > 3 and I[3] is not None else np.arange(len(starts))
        steps = I[4].astype(np.int64) if len(I) > 4 and I[4] is not None else np.ones_like(starts)
        sl = [slice(None)] * data.ndim
        for s, e, a, st in zip(starts, ends, axes, steps):
            a = int(a) % data.ndim
            sl[a] = slice(int(s), int(e), int(st))
        return [data[tuple(sl)]]
    if op == "GatherElements":
        ax = attr(node, "axis", 0)
        return [np.take_along_axis(I[0], I[1].astype(np.int64), axis=ax)]
    if op == "TopK":
        k = int(np.atleast_1d(I[1])[0]); ax = attr(node, "axis", -1)
        largest = attr(node, "largest", 1)
        idx = np.argsort(I[0], axis=ax)
        if largest: idx = np.flip(idx, axis=ax)
        idx = np.take(idx, np.arange(k), axis=ax)
        val = np.take_along_axis(I[0], idx, axis=ax)
        return [val, idx.astype(np.int64)]
    if op == "MaxPool":
        x = I[0]; ks = attr(node, "kernel_shape"); st = attr(node, "strides", ks)
        pads = attr(node, "pads", [0, 0, 0, 0])
        N, C, H, W = x.shape
        xp = np.full((N, C, H + pads[0] + pads[2], W + pads[1] + pads[3]), -np.inf, np.float32)
        xp[:, :, pads[0]:pads[0] + H, pads[1]:pads[1] + W] = x
        HO = (xp.shape[2] - ks[0]) // st[0] + 1; WO = (xp.shape[3] - ks[1]) // st[1] + 1
        out = np.full((N, C, HO, WO), -np.inf, np.float32)
        for i in range(ks[0]):
            for j in range(ks[1]):
                out = np.maximum(out, xp[:, :, i:i + HO * st[0]:st[0], j:j + WO * st[1]:st[1]])
        return [out]
    if op == "Resize":
        x = I[0]
        scales = I[2] if len(I) > 2 and I[2] is not None and I[2].size else None
        sizes = I[3] if len(I) > 3 and I[3] is not None and I[3].size else None
        if sizes is not None:
            HO, WO = int(sizes[2]), int(sizes[3])
        else:
            HO, WO = int(round(x.shape[2] * scales[2])), int(round(x.shape[3] * scales[3]))
        ri = (np.arange(HO) * x.shape[2] // HO); ci = (np.arange(WO) * x.shape[3] // WO)
        return [x[:, :, ri][:, :, :, ci]]
    raise NotImplementedError(op)


# ───────────────────────── block RTL backends ─────────────────────────
def _wr_i8_raster(path, codes_chw):
    """write (C,H,W) int8 codes as raster channel-fastest hex: (h*W+w)*C+c."""
    r = np.transpose(codes_chw, (1, 2, 0)).reshape(-1)
    with open(path, "w") as f:
        for v in r: f.write("%02x\n" % (int(v) & 0xFF))


def _run_block(block_dir, env_extra):
    env = dict(os.environ, **env_extra)
    r = subprocess.run(["make", "-C", os.path.join(ROOT, block_dir), "test"],
                       env=env, capture_output=True, text=True)
    if r.returncode != 0:
        sys.stderr.write(r.stdout[-2000:] + r.stderr[-2000:])
        raise RuntimeError(f"block RTL {block_dir} failed")


def sppf_rtl(fp_in, s_in):
    """Real SPPF IP: C-ch frame -> 4C-ch frame (maxpool x3 + concat). Scale
    preserved (maxpool/concat don't rescale)."""
    C, H, W = fp_in.shape
    codes = np.clip(np.round(fp_in / s_in), -128, 127).astype(np.int32)
    bdir = os.path.join(E2E, "_blocks", "sppf"); os.makedirs(bdir, exist_ok=True)
    cin = os.path.join(bdir, "in.hex"); cout = os.path.join(bdir, "out.hex")
    _wr_i8_raster(cin, codes)
    _run_block("integ/sppf_model9/dv", dict(CHAIN_IN=cin, CHAIN_OUT=cout))
    o = np.array([int(l, 16) for l in open(cout)], dtype=np.int32)
    o = ((o + 128) % 256 - 128).reshape(H, W, 4 * C).transpose(2, 0, 1)
    return (o.astype(np.float32) * s_in), s_in


# block output tensor -> (handler, [input tensors]). Handler(*input_fp_arrays,
# *input_scales) -> (out_fp[C,H,W], out_scale).
def _upsample_rtl(block_dir, A_fp, B_fp, sA, sB):
    """Real upsample_concat IP: NN-2x upsample A then channel-concat with B.
    Both streams quantized at a shared (joint) symmetric scale, matching the
    block's single-S_OUT design; output carries that scale."""
    CA, HA, WA = A_fp.shape; CB, HB, WB = B_fp.shape
    s = pick_s_out(max(float(np.max(np.abs(A_fp))), float(np.max(np.abs(B_fp)))))
    Ac = np.clip(np.round(A_fp / s), -128, 127).astype(np.int32)
    Bc = np.clip(np.round(B_fp / s), -128, 127).astype(np.int32)
    bdir = os.path.join(E2E, "_blocks", os.path.basename(block_dir))
    os.makedirs(bdir, exist_ok=True)
    ca = os.path.join(bdir, "a.hex"); cb = os.path.join(bdir, "b.hex")
    co = os.path.join(bdir, "out.hex")
    _wr_i8_raster(ca, Ac); _wr_i8_raster(cb, Bc)
    _run_block(block_dir, dict(CHAIN_IN_A=ca, CHAIN_IN_B=cb, CHAIN_OUT=co))
    o = np.array([int(l, 16) for l in open(co)], dtype=np.int32)
    o = ((o + 128) % 256 - 128).reshape(HB, WB, CA + CB).transpose(2, 0, 1)
    return (o.astype(np.float32) * s), s


# detect head: the six int8 conv tails (box=cv2.x.2 4ch, cls=cv3.x.2 80ch) feed
# the real /model.23 IP, which does decode + per-scale dequant + cross-scale
# topk + gather internally and emits the final logits/pred_boxes directly.
DET_BOX = [f"/model.23/one2one_cv2.{i}/one2one_cv2.{i}.2/Conv_output_0" for i in range(3)]
DET_CLS = [f"/model.23/one2one_cv3.{i}/one2one_cv3.{i}.2/Conv_output_0" for i in range(3)]
DET_SCNT = [6400, 1600, 400]


def detect_rtl(vals, scales):
    """Run the real detect_head IP on the chain's six conv-tail outputs.
    Feeds each conv's int8 codes with that conv's own output scale as the
    block's runtime s_box/s_cls (chip-faithful — scales are DUT ports, not
    baked). Returns (logits[1,300,80], pred_boxes[1,300,4]), sorted by score
    descending to align with ORT's TopK order for comparison."""
    bdir = os.path.join(E2E, "_blocks", "detect"); os.makedirs(bdir, exist_ok=True)
    s_box, s_cls = [], []
    for si in range(3):
        bt, ct = DET_BOX[si], DET_CLS[si]
        bf = vals[bt][0] if vals[bt].ndim == 4 else vals[bt]      # (4,H,W)
        cf = vals[ct][0] if vals[ct].ndim == 4 else vals[ct]      # (80,H,W)
        sb = scales.get(bt, pick_s_out(float(np.max(np.abs(bf)))))
        scl = scales.get(ct, pick_s_out(float(np.max(np.abs(cf)))))
        s_box.append(sb); s_cls.append(scl)
        np.clip(np.round(bf / sb), -128, 127).astype(np.int8).tofile(
            os.path.join(bdir, f"box_s{si}.bin"))
        np.clip(np.round(cf / scl), -128, 127).astype(np.int8).tofile(
            os.path.join(bdir, f"cls_s{si}.bin"))
    np.array(s_box + s_cls, np.float32).tofile(os.path.join(bdir, "scales.f32"))
    cout = os.path.join(bdir, "out.bin")
    env = dict(CHAIN_OUT=cout, CHAIN_SCALES=os.path.join(bdir, "scales.f32"))
    for si in range(3):
        env[f"CHAIN_BOX_S{si}"] = os.path.join(bdir, f"box_s{si}.bin")
        env[f"CHAIN_CLS_S{si}"] = os.path.join(bdir, f"cls_s{si}.bin")
    _run_block("integ/detect_model23/dv", env)
    rec = np.fromfile(cout, dtype=np.uint16).reshape(-1, 1 + 4 + 80)
    box = rec[:, 1:5].view(np.float16).astype(np.float32)        # (K,4)
    log = rec[:, 5:].view(np.float16).astype(np.float32)         # (K,80)
    order = np.argsort(-log.max(axis=1))                          # ORT TopK order
    return log[order][None], box[order][None]


def attn_rtl(block_dir, qkv_fp, pe_fp):
    """Real attn IP: qkv(256ch)+pe(128ch) float -> ATTN_OUT(128ch).
    Re-quantizes inputs at the block's BAKED S_QKV/S_PE (the chip's fixed
    calibrated scales, read from the same manifest that generated
    attn_scales_pkg), drives one frame, reads ATTN_OUT x S_AOUT."""
    man = json.load(open(os.path.join(ROOT, block_dir, "stim", "manifest.json")))
    sq, sp, sa = man["S_QKV"], man["S_PE"], man["S_AOUT"]
    C, H, W = pe_fp.shape                                # (128,20,20)
    bdir = os.path.join(E2E, "_blocks", os.path.basename(block_dir))
    os.makedirs(bdir, exist_ok=True)
    qf = os.path.join(bdir, "qkv.hex"); pf = os.path.join(bdir, "pe.hex")
    co = os.path.join(bdir, "out.hex")
    _wr_i8_raster(qf, np.clip(np.round(qkv_fp / sq), -128, 127).astype(np.int32))
    _wr_i8_raster(pf, np.clip(np.round(pe_fp / sp), -128, 127).astype(np.int32))
    _run_block(os.path.join(block_dir, "dv"),
               dict(CHAIN_QKV=qf, CHAIN_PE=pf, CHAIN_OUT=co))
    o = np.array([int(l, 16) for l in open(co)], dtype=np.int32)
    o = ((o + 128) % 256 - 128).reshape(H, W, C).transpose(2, 0, 1)
    return (o.astype(np.float32) * sa), sa


BLOCK_RTL = {
    "/model.9/Concat_output_0":
        ("sppf", ["/model.9/cv1/conv/Conv_output_0"]),
    "/model.10/m/m.0/attn/Add_output_0":
        ("attn", ["/model.10/m/m.0/attn/qkv/conv/Conv_output_0",
                  "/model.10/m/m.0/attn/pe/conv/Conv_output_0"]),
    "/model.22/m.0/m.0.1/attn/Add_output_0":
        ("attn", ["/model.22/m.0/m.0.1/attn/qkv/conv/Conv_output_0",
                  "/model.22/m.0/m.0.1/attn/pe/conv/Conv_output_0"]),
    "/model.12/Concat_output_0":
        ("upsample", ["/model.10/cv2/act/Mul_output_0", "/model.6/cv2/act/Mul_output_0"]),
    "/model.15/Concat_output_0":
        ("upsample", ["/model.13/cv2/act/Mul_output_0", "/model.4/cv2/act/Mul_output_0"]),
}
BLOCK_FN = {
    "sppf": sppf_rtl,
    # upsample dir chosen by output tensor (model.11 vs model.14 shapes)
    "/model.12/Concat_output_0":
        lambda A, B, sA, sB: _upsample_rtl("integ/upsample_model11/dv", A, B, sA, sB),
    "/model.15/Concat_output_0":
        lambda A, B, sA, sB: _upsample_rtl("integ/upsample_model14/dv", A, B, sA, sB),
    # attn: real flash_attn block; ignores chain input scales (re-quantizes at
    # the block's baked S_QKV/S_PE). qkv,pe -> ATTN_OUT.
    "/model.10/m/m.0/attn/Add_output_0":
        lambda qkv, pe, sq, sp: attn_rtl("integ/attn_model10", qkv, pe),
    "/model.22/m.0/m.0.1/attn/Add_output_0":
        lambda qkv, pe, sq, sp: attn_rtl("integ/attn_model22", qkv, pe),
}


# ───────────────────────── chained interpreter ─────────────────────────
def pick_s_out(amp):
    return max(max(amp, 1e-3) * 1.1 / 127.0, 2.0 / 127.0)


def conv_pre(fp_in, s_in, p):
    """fp32 pre-activation the chip computes (codes_in quantized, acc in fp32)."""
    W = p["W_raw"]; K = p["K"]; S = p["S"]; PAD = p["PAD"]; COUT = W.shape[0]
    CIN, H_IN, W_IN = fp_in.shape
    if p["dw"]:
        Wd = np.zeros((COUT, CIN, K, K), dtype=np.float32)
        for co in range(COUT): Wd[co, co] = W[co, 0]
        W = Wd
    else:
        W = W.astype(np.float32)
    HO = (H_IN + 2 * PAD - K) // S + 1; WO = (W_IN + 2 * PAD - K) // S + 1
    codes = np.clip(np.round(fp_in / s_in), -128, 127).astype(np.float32)
    xp = np.zeros((CIN, H_IN + 2 * PAD, W_IN + 2 * PAD), dtype=np.float32)
    xp[:, PAD:PAD + H_IN, PAD:PAD + W_IN] = codes
    cols = np.empty((CIN, K, K, HO, WO), dtype=np.float32)
    for kh in range(K):
        for kw in range(K):
            cols[:, kh, kw] = xp[:, kh:kh + HO * S:S, kw:kw + WO * S:S]
    acc = np.tensordot(W, cols, axes=([1, 2, 3], [0, 1, 2]))
    return acc * (s_in * p["s_w"]) + p["bias"][:, None, None]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stop-after", type=int, default=10**9)
    ap.add_argument("--conv", choices=["numpy", "rtl"], default="numpy")
    ap.add_argument("--cls-bits", type=int, default=8,
                    help="output bit-width of the 3 cls-logit tail convs "
                         "(cv3.x.2); >8 = finer logit resolution to break detection "
                         "score ties (numpy model only; MAC array unchanged).")
    ap.add_argument("--scales", choices=["dynamic", "fixed"], default="dynamic",
                    help="dynamic: per-image pick_s_out (oracle, RTL can't do this). "
                         "fixed: the chip's baked per-layer calibrated S_OUT_PRE/"
                         "S_OUT_SILU from meta.json — faithful to the synthesized IP "
                         "and keeps the RTL per-layer build cache valid across images.")
    ap.add_argument("--rtl-blocks", default="",
                    help="comma list of blocks to run via real RTL IP "
                         "(sppf,upsample,attn,detect) or 'all'")
    ap.add_argument("--pixels", default="assets/pixel_values.npy")
    ap.add_argument("--image", default="assets/bus.jpg",
                    help="source image used as the letterbox base for rendering")
    ap.add_argument("--metrics-json", default="",
                    help="if set, write logits/boxes cos + detection counts as JSON")
    ap.add_argument("--render", nargs="?", const="integ/generated/e2e/chain_result.png",
                    default="integ/generated/e2e/chain_result.png",
                    help="output PNG path (side-by-side ORT vs chip); '' to disable")
    args = ap.parse_args()

    x = np.load(os.path.join(ROOT, args.pixels)).astype(np.float32)
    in_name = [i.name for i in G.input if i.name not in INIT][0]

    # ORT golden: final outputs + every conv cluster output (for per-layer cos)
    conv_out_tensors = [p["out"] for p in CONV_BY_DQL.values()]
    # Also expose RTL block outputs (e.g. attn Add) so we can print their cos.
    block_out_tensors = [ot for ot, (kind, _) in BLOCK_RTL.items()]
    g2 = onnx.load(ONNX); ex = {o.name for o in g2.graph.output}
    for t in conv_out_tensors + block_out_tensors:
        if t not in ex:
            g2.graph.output.append(onnx.helper.make_tensor_value_info(t, 1, None))
    tmp = os.path.join(E2E, "_chain_ort.onnx"); onnx.save(g2, tmp)
    so = ort.SessionOptions(); so.log_severity_level = 3
    sess = ort.InferenceSession(tmp, sess_options=so, providers=["CPUExecutionProvider"])
    want = ["logits", "pred_boxes"] + conv_out_tensors + block_out_tensors
    ortv = dict(zip(want, sess.run(want, {in_name: x})))

    rtl_blocks = set()
    if args.rtl_blocks:
        rtl_blocks = ({"sppf", "upsample", "attn", "detect"}
                      if args.rtl_blocks == "all"
                      else set(args.rtl_blocks.split(",")))

    CALIB = load_calib() if args.scales == "fixed" else {}
    if args.scales == "fixed":
        print(f"scales=fixed: {len(CALIB)} convs from calib_scales.json"
              if CALIB else "scales=fixed: no calib_scales.json, using meta.json")

    vals = {in_name: x[0]}      # carry as CHW (drop batch); ops use numpy broadcasting
    scales = {}
    n_conv = 0
    worst = (1.1, None)
    for node in G.node:
        if node.name in TRIGGER:
            p = TRIGGER[node.name]
            idx = NAME2IDX.get(p["conv"].name, -1)
            fp_in = vals[p["fp_in"]]
            if fp_in.ndim == 4: fp_in = fp_in[0]
            # fixed-scale mode: use the chip's baked per-conv calibrated scales,
            # exactly what conv_stage synthesizes as S_OUT_PRE/S_OUT_SILU.
            # Priority: corpus calib_scales.json (all 102 convs, percentile-based)
            # > per-layer meta.json (92 dumped, bus-only max-based) > dynamic.
            fixed = None
            if args.scales == "fixed":
                fixed = CALIB.get(p["conv"].name)
                if fixed is None and idx >= 0:
                    mp = os.path.join(E2E, f"layer_{idx:03d}", "meta.json")
                    if os.path.exists(mp):
                        fixed = json.load(open(mp))
            s_in = scales.get(p["fp_in"])
            if s_in is None:
                s_in = (fixed["s_in"] if fixed else
                        pick_s_out(float(np.max(np.abs(fp_in)))))
            pre = conv_pre(fp_in, s_in, p)
            s_pre = (fixed["s_out_pre"] if fixed else
                     pick_s_out(float(np.max(np.abs(pre)))))
            pre_code = np.clip(np.round(pre / s_pre), -128, 127) * s_pre
            y = pre_code / (1.0 + np.exp(-pre_code)) if p["silu"] else pre_code
            s_silu = (fixed["s_out_silu"] if fixed else
                      pick_s_out(float(np.max(np.abs(y)))))
            is_cls_tail = re.search(r"/one2one_cv3\.\d+\.2/Conv_quant$", p["conv"].name)
            ob = args.cls_bits if is_cls_tail else 8
            if args.conv == "rtl" and idx >= 0:
                out = conv_rtl(fp_in, s_in, p, s_pre, s_silu, idx)
            else:
                out = conv_numpy(fp_in, s_in, p, s_pre, s_silu, out_bits=ob)
            vals[p["out"]] = out[None]      # restore batch dim for downstream ops
            scales[p["out"]] = s_silu
            n_conv += 1
            if p["out"] in ortv:
                c = cos(out, ortv[p["out"]][0])
                if c < worst[0]: worst = (c, idx)
                print(f"  conv#{idx:3d} {p['conv'].name:46s} cos={c:.4f} "
                      f"shape={tuple(out.shape)}")
            if idx >= args.stop_after:
                print(f"-- stopped after conv#{idx} --"); return
        elif node.name in OWNED or node.op_type == "DynamicQuantizeLinear":
            continue   # post-conv cluster internals / conv input quant (no-op here)
        elif node.output and node.output[0] in BLOCK_RTL and \
                BLOCK_RTL[node.output[0]][0] in rtl_blocks:
            ot = node.output[0]; kind, ins = BLOCK_RTL[ot]
            args_fp = [vals[t][0] if vals[t].ndim == 4 else vals[t] for t in ins]
            scls = [scales.get(t, pick_s_out(float(np.max(np.abs(vals[t]))))) for t in ins]
            fn = BLOCK_FN.get(ot) or BLOCK_FN[kind]
            out, s_out = fn(*args_fp, *scls)
            vals[ot] = out[None]; scales[ot] = s_out
            cmsg = f" cos={cos(out, ortv[ot][0]):.5f}" if ot in ortv else ""
            print(f"  block[{kind}] RTL -> {ot}  shape={tuple(out.shape)}{cmsg}")
        else:
            try:
                outs = run_op(node, vals)
            except Exception as e:
                print(f"!! op {node.op_type} ({node.name}) failed: {e}")
                raise
            for nm, val in zip(node.output, outs):
                vals[nm] = val

    if "detect" in rtl_blocks:
        # Real detect-head IP replaces the numpy decode+topk+gather tail.
        vals["logits"], vals["pred_boxes"] = detect_rtl(vals, scales)
        print(f"  block[detect] RTL -> logits/pred_boxes  "
              f"shape={tuple(vals['logits'].shape)}")

    print(f"\nran {n_conv} convs; worst conv cos {worst[0]:.4f} at #{worst[1]}")
    for t in ["logits", "pred_boxes"]:
        if t in vals:
            print(f"{t}: cos={cos(vals[t], ortv[t]):.5f} shape={tuple(np.shape(vals[t]))}")
        else:
            print(f"{t}: NOT PRODUCED")

    if "logits" in vals and "pred_boxes" in vals:
        cd, od = compare_detections(vals["logits"], vals["pred_boxes"],
                                    ortv["logits"], ortv["pred_boxes"])
        if args.render:
            render_result(vals["logits"], vals["pred_boxes"],
                          ortv["logits"], ortv["pred_boxes"], args.render,
                          args.conv, args.image)
        if args.metrics_json:
            # per-detection IoU of each chip det to best same-class ORT det
            ious = []
            for c, cf, b in cd:
                best = max([_iou(b, ob_) for c_, cf_, ob_ in od if c_ == c],
                           default=0.0)
                ious.append(best)
            metrics = dict(
                image=os.path.basename(args.image),
                logits_cos=cos(vals["logits"], ortv["logits"]),
                boxes_cos=cos(vals["pred_boxes"], ortv["pred_boxes"]),
                worst_conv_cos=worst[0], worst_conv_idx=worst[1],
                chip_dets=len(cd), ort_dets=len(od),
                matched=int(sum(1 for i in ious if i >= 0.5)),
                median_iou=float(np.median(ious)) if ious else 0.0,
                render=args.render, conv_mode=args.conv,
                chip=[[int(c), float(cf), [float(v) for v in b]] for c, cf, b in cd],
                ort=[[int(c), float(cf), [float(v) for v in b]] for c, cf, b in od],
            )
            mp = os.path.join(ROOT, args.metrics_json)
            os.makedirs(os.path.dirname(mp), exist_ok=True)
            json.dump(metrics, open(mp, "w"), indent=2)
            print(f"wrote metrics -> {args.metrics_json}")


COCO = {i: n for i, n in enumerate(
    "person bicycle car motorcycle airplane bus train truck boat traffic_light "
    "fire_hydrant stop_sign parking_meter bench bird cat dog horse sheep cow "
    "elephant bear zebra giraffe backpack umbrella handbag tie suitcase frisbee "
    "skis snowboard sports_ball kite baseball_bat baseball_glove skateboard "
    "surfboard tennis_racket bottle wine_glass cup fork knife spoon bowl banana "
    "apple sandwich orange broccoli carrot hot_dog pizza donut cake chair couch "
    "potted_plant bed dining_table toilet tv laptop mouse remote keyboard "
    "cell_phone microwave oven toaster sink refrigerator book clock vase "
    "scissors teddy_bear hair_drier toothbrush".split())}


def _dets(logits, boxes, conf_thr=0.25):
    L = np.asarray(logits)[0]; B = np.asarray(boxes)[0]
    sc = 1.0 / (1.0 + np.exp(-L))
    cls = sc.argmax(1); conf = sc.max(1)
    keep = np.where(conf >= conf_thr)[0]
    keep = keep[np.argsort(-conf[keep])]
    return [(int(cls[i]), float(conf[i]), B[i]) for i in keep]


def _iou(a, b):
    # boxes are (cx,cy,w,h) normalized
    def corners(x): return (x[0]-x[2]/2, x[1]-x[3]/2, x[0]+x[2]/2, x[1]+x[3]/2)
    ax0, ay0, ax1, ay1 = corners(a); bx0, by0, bx1, by1 = corners(b)
    ix0, iy0 = max(ax0, bx0), max(ay0, by0); ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0, ix1-ix0), max(0, iy1-iy0); inter = iw*ih
    ua = (ax1-ax0)*(ay1-ay0) + (bx1-bx0)*(by1-by0) - inter
    return inter/ua if ua > 0 else 0.0


def compare_detections(cl, cb, ol, ob):
    cd, od = _dets(cl, cb), _dets(ol, ob)
    print(f"\n── detections (conf>=0.25) ──  chip={len(cd)}  ORT={len(od)}")
    print("  ORT (reference):")
    for c, cf, b in od:
        print(f"    {COCO.get(c, str(c)):8s} conf={cf:.3f} box={np.round(b,3)}")
    print("  chip (chained):")
    for c, cf, b in cd:
        # best IoU match to an ORT det of same class
        best = max([( _iou(b, ob_) , cf_) for c_, cf_, ob_ in od if c_ == c],
                   default=(0.0, 0.0))
        print(f"    {COCO.get(c, str(c)):8s} conf={cf:.3f} box={np.round(b,3)}  "
              f"bestIoU={best[0]:.2f}")
    return cd, od


def render_result(cl, cb, ol, ob, out_path, conv_mode, image="assets/bus.jpg"):
    """Draw ORT (reference) vs chained-chip detections side by side on the
    letterboxed 640 image and save a PNG."""
    from PIL import Image, ImageDraw
    import sys as _sys
    _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from preprocess import letterbox
    ipath = image if os.path.isabs(image) else os.path.join(ROOT, image)
    src = np.asarray(Image.open(ipath).convert("RGB"), dtype=np.uint8)
    base = Image.fromarray(letterbox(src))            # 640x640 RGB
    S = base.size[0]

    def draw(img, dets, color):
        d = ImageDraw.Draw(img)
        for c, cf, b in dets:
            cx, cy, w, h = (b * S).tolist()
            x0, y0, x1, y1 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
            d.rectangle([x0, y0, x1, y1], outline=color, width=3)
            lbl = f"{COCO.get(c, str(c))} {cf:.2f}"
            d.rectangle([x0, y0 - 12, x0 + 8 * len(lbl), y0], fill=color)
            d.text((x0 + 1, y0 - 11), lbl, fill=(0, 0, 0))
        return img

    left = draw(base.copy(), _dets(ol, ob), (0, 220, 0))      # ORT  = green
    right = draw(base.copy(), _dets(cl, cb), (255, 90, 0))    # chip = orange
    canvas = Image.new("RGB", (S * 2 + 12, S + 24), (20, 20, 20))
    canvas.paste(left, (0, 24)); canvas.paste(right, (S + 12, 24))
    d = ImageDraw.Draw(canvas)
    d.text((6, 6), f"ORT reference (fp32)  [{os.path.basename(image)}]",
           fill=(0, 220, 0))
    d.text((S + 18, 6), f"chip chained ({conv_mode} convs, symmetric int8)",
           fill=(255, 90, 0))
    op = os.path.join(ROOT, out_path)
    os.makedirs(os.path.dirname(op), exist_ok=True)
    canvas.save(op)
    print(f"\nrendered result -> {out_path}  ({S}x{S} each, ORT left / chip right)")


if __name__ == "__main__":
    main()
