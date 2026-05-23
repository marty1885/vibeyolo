#!/usr/bin/env python3
# Copyright (c) 2026 vibeyolo
# SPDX-License-Identifier: Apache-2.0
#
# dump_layers.py — real-image, full-frame per-layer stimulus for the
# layer-by-layer chip cosim.
#
# One ORT inference on a real preprocessed image (pixel_values.npy), capturing
# every conv layer's quantized int8 input plane + pre/post-activation reference,
# then emitting per-layer stim that drives `hw/ip/conv_stage` over the FULL
# frame (not the small ROIs the per-layer DV used). Reuses the proven
# u8->i8 + zero-point-into-bias fold and the S_OUT sizing rule (footgun #1/#5).
#
# Layer set + canonical index come from the balanced scale report (same source
# as gen_core.py). Per-conv dims come from scale_pkg_dv.sv. Tensor names are
# discovered structurally from the ONNX graph.
#
#   python3 tools/e2e/dump_layers.py [pixel_values.npy] [--layers 0,1,11]
#
# Output: integ/generated/e2e/layer_<NNN>/{meta.json, input_i8.hex,
#   weights.i8.hex, scale_fp16.hex, bias_fp16.hex, ref_ort.npy, input_i8.npy}

import argparse
import json
import os
import re
import sys

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ONNX = os.path.join(ROOT, "integ/yolo26n/model_int8.onnx")
BAL = os.path.join(ROOT, "integ/generated/scale/scale_report_balanced.md")
DVPKG = os.path.join(ROOT, "integ/scale/scale_pkg_dv.sv")
OUTROOT = os.path.join(ROOT, "integ/generated/e2e")


def parse_layer_map():
    """idx -> conv node name, from the balanced report (canonical order)."""
    idx2name = {}
    for line in open(BAL):
        m = re.match(r"\|\s*(\d+)\s*\|\s*(/model[^|]+?)\s*\|", line)
        if m:
            idx2name[int(m.group(1))] = m.group(2).strip()
    assert len(idx2name) == 102, len(idx2name)
    return idx2name


def parse_dv_dims():
    dv = open(DVPKG).read()
    def P(i, f):
        m = re.search(rf"LAYER_{i}_{f}\s*=\s*(\d+)", dv)
        return int(m.group(1)) if m else 0
    dims = {}
    for i in range(102):
        dims[i] = dict(K=P(i, "K"), STRIDE=P(i, "STRIDE") or 1, PAD=P(i, "PAD"),
                       P_COUT=P(i, "P_COUT"), P_CIN=P(i, "P_CIN"),
                       CIN=P(i, "CIN"), COUT=P(i, "COUT"))
    return dims


def f32_to_fp16_bits(x):
    return np.float16(x).view(np.uint16)


def discover(model, conv_name):
    """Return (conv_node, bias, pre_tensor, post_tensor, silu, depthwise)."""
    nodes = list(model.graph.node)
    init = {i.name: i for i in model.graph.initializer}
    producer = {o: n for n in nodes for o in n.output}
    conv = next(n for n in nodes if n.op_type == "ConvInteger" and n.name == conv_name)

    # Deterministic chain anchored to THIS conv's output (no fragile window
    # search): ConvInteger -> ... -> Add(bias) whose output is the conv output
    # name with the _output_quantized suffix stripped.
    qname = conv.output[0]
    assert qname.endswith("_output_quantized"), qname
    pre_tensor = qname[:-len("_output_quantized")]      # e.g. .../Conv_output_0
    bias_add = producer.get(pre_tensor)
    assert bias_add is not None and bias_add.op_type == "Add", \
        f"{conv_name}: no bias_add for {pre_tensor}"

    # bias initializer feeds the Reshape that feeds bias_add.input[1]
    bias = None
    for inp in bias_add.input:
        rs = producer.get(inp)
        if rs is not None and rs.op_type == "Reshape" and rs.input[0] in init:
            cand = numpy_helper.to_array(init[rs.input[0]]).astype(np.float32)
            if cand.ndim == 1:
                bias = cand
    assert bias is not None, f"{conv_name}: no bias initializer"

    # SiLU iff a "<base>/act/Mul_output_0" tensor exists fed by this conv's pre.
    base = conv_name.rsplit("/conv/Conv_quant", 1)[0].rsplit("/Conv_quant", 1)[0]
    act_t = base + "/act/Mul_output_0"
    silu = act_t in producer
    post_tensor = act_t if silu else pre_tensor

    W_raw = numpy_helper.to_array(init[conv.input[1]]).astype(np.int32)
    depthwise = (W_raw.ndim == 4 and W_raw.shape[1] == 1 and W_raw.shape[0] > 1)
    return conv, bias, pre_tensor, post_tensor, silu, depthwise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pixels", nargs="?", default="assets/pixel_values.npy")
    ap.add_argument("--layers", default=None, help="comma list of idx; default all")
    args = ap.parse_args()

    x = np.load(os.path.join(ROOT, args.pixels)).astype(np.float32)
    idx2name = parse_layer_map()
    dims = parse_dv_dims()
    sel = (sorted(int(i) for i in args.layers.split(",")) if args.layers
           else sorted(idx2name))

    model = onnx.load(ONNX)
    init_by_name = {i.name: numpy_helper.to_array(i) for i in model.graph.initializer}

    # Discover per-layer tensors and gather the set of intermediate outputs.
    info = {}
    want = set()
    for idx in sel:
        cname = idx2name[idx]
        conv, bias, pre_t, post_t, silu, dw = discover(model, cname)
        s_w = float(init_by_name[conv.input[1].replace("_quantized", "_scale")])
        zp_w = int(init_by_name[conv.input[3]])
        assert zp_w == 0, f"{cname}: weight zp {zp_w}"
        in_q = conv.input[0]
        in_s = in_q.replace("_quantized", "_scale")
        in_zp = conv.input[2]
        info[idx] = dict(cname=cname, conv=conv, bias=bias, pre=pre_t, post=post_t,
                         silu=silu, dw=dw, s_w=s_w, in_q=in_q, in_s=in_s, in_zp=in_zp)
        want.update([in_q, in_s, in_zp, pre_t, post_t])

    # Register intermediates as graph outputs and run ORT once.
    mod = onnx.load(ONNX)
    existing = {o.name for o in mod.graph.output}
    tp = {}
    for idx in sel:
        d = info[idx]
        tp[d["in_q"]] = onnx.TensorProto.UINT8
        tp[d["in_s"]] = onnx.TensorProto.FLOAT
        tp[d["in_zp"]] = onnx.TensorProto.UINT8
        tp[d["pre"]] = onnx.TensorProto.FLOAT
        tp[d["post"]] = onnx.TensorProto.FLOAT
    for nm in want:
        if nm not in existing:
            mod.graph.output.append(onnx.helper.make_tensor_value_info(nm, tp[nm], None))
    tmp = os.path.join(OUTROOT, "_model_e2e.onnx")
    os.makedirs(OUTROOT, exist_ok=True)
    onnx.save(mod, tmp)

    so = ort.SessionOptions(); so.log_severity_level = 3
    sess = ort.InferenceSession(tmp, sess_options=so, providers=["CPUExecutionProvider"])
    ipt = sess.get_inputs()[0].name
    out_names = sorted(want)
    print(f"running ORT once with {len(out_names)} intermediate outputs ...")
    res = dict(zip(out_names, sess.run(out_names, {ipt: x})))

    def pick_s_out(amp):
        return max(max(amp, 1e-3) * 1.1 / 127.0, 2.0 / 127.0)

    summary = []
    skipped = []
    for idx in sel:
        d = info[idx]
        K = dims[idx]["K"]; STRIDE = dims[idx]["STRIDE"]; PAD = dims[idx]["PAD"]
        qx = res[d["in_q"]]                       # (1,CIN,H,W) u8
        s_a = float(res[d["in_s"]]); zp_a = int(res[d["in_zp"]].reshape(-1)[0])
        if s_a == 0.0:
            s_a = 1.0
        pre = res[d["pre"]][0]                    # (COUT,Hout,Wout)
        post = res[d["post"]][0]
        _, CIN, H_IN, W_IN = qx.shape
        COUT, H_OUT, W_OUT = post.shape

        W_raw = init_by_name[d["conv"].input[1]].astype(np.int32)
        # The conv weight is authoritative for COUT/CIN/K. Attention/ffn convs
        # interface with flash_attn (fp16) — they have no clean int8 act output
        # (their bias_add feeds a residual Add / qkv split), so post-channels
        # won't match COUT. Those are covered by the attn integration block;
        # skip them in the per-conv conv_stage sweep.
        w_cout = W_raw.shape[0]
        block_internal = ("/attn/" in d["cname"]) or ("/ffn/" in d["cname"])
        if block_internal or post.shape[0] != w_cout:
            print(f"  L{idx:3d} {d['cname']:42s} SKIP (block-internal; "
                  f"post C={post.shape[0]} vs conv COUT={w_cout})")
            skipped.append((idx, d["cname"]))
            continue
        COUT = w_cout
        if d["dw"]:
            assert W_raw.shape == (COUT, 1, K, K)
            assert CIN == COUT, f"L{idx} dw CIN {CIN} != COUT {COUT}"
            W_q = np.zeros((COUT, CIN, K, K), dtype=np.int32)
            for co in range(COUT):
                W_q[co, co] = W_raw[co, 0]
        else:
            assert W_raw.shape == (COUT, CIN, K, K), f"L{idx} {W_raw.shape} vs {(COUT,CIN,K,K)}"
            W_q = W_raw

        # SYMMETRIC int8 injection (footgun: the chip is symmetric int8 zp=0
        # throughout — conv_layer/act_silu carry no zero-point. The model's
        # asymmetric u8-128 activation + zero-padding corrupts frame borders
        # because the model's activation-zero sits at u8=zp_a, not u8=128.
        # We requantize the true fp32 input activation symmetrically so int8 0
        # == activation 0 and the linebuf's zero-padding is exact. This is also
        # exactly what the chained chip feeds (act_silu's signed int8 output).
        fp_in = (qx[0].astype(np.float64) - zp_a) * s_a          # (CIN,H,W) fp32 act
        in_amp = float(np.max(np.abs(fp_in)))
        s_in = max(in_amp, 1e-3) / 127.0
        s_acc = s_in * d["s_w"]
        bias_eff = d["bias"].astype(np.float64)                  # no zp fold

        # int8 input plane, raster channel-fastest: (h*W+w)*CIN+c
        x_i8 = np.clip(np.round(fp_in / s_in), -128, 127).astype(np.int8)  # (CIN,H,W)
        x_i8_raster = np.transpose(x_i8, (1, 2, 0)).reshape(-1)            # H*W*CIN

        # S_OUT sizing from full-frame ORT amplitudes
        pre_amp = float(np.max(np.abs(pre)))
        post_amp = float(np.max(np.abs(post)))
        if d["silu"]:
            s_out_pre = pick_s_out(pre_amp)
            s_out_silu = pick_s_out(post_amp)
        else:
            s_out_pre = pick_s_out(pre_amp)
            s_out_silu = s_out_pre

        # weights: WROM[COUT][K*K*CIN], lane=(kh*K+kw)*CIN+kc
        N_LANE = K * K * CIN
        W_flat = np.zeros((COUT, N_LANE), dtype=np.int8)
        for kh in range(K):
            for kw in range(K):
                lane0 = (kh * K + kw) * CIN
                W_flat[:, lane0:lane0 + CIN] = W_q[:, :, kh, kw].astype(np.int8)

        odir = os.path.join(OUTROOT, f"layer_{idx:03d}")
        os.makedirs(odir, exist_ok=True)

        def wr_hex(path, vals, width):
            with open(path, "w") as f:
                fmt = f"%0{width}x\n"
                for v in vals:
                    f.write(fmt % (int(v) & ((1 << (width * 4)) - 1)))

        # requant ROM convention (matches layer_11 / generated extractors):
        # the requant emits an int8 in S_OUT_PRE-LSB units that act_silu
        # (InScale=S_OUT_PRE) consumes, so the on-die scale/bias are pre-divided
        # by S_OUT_PRE (ACC_SHIFT=0 in the current requant.sv).
        wr_hex(os.path.join(odir, "input_i8.hex"), x_i8_raster, 2)
        wr_hex(os.path.join(odir, "weights.i8.hex"), W_flat.reshape(-1), 2)
        wr_hex(os.path.join(odir, "scale_fp16.hex"),
               [f32_to_fp16_bits(np.float32(s_acc / s_out_pre))] * COUT, 4)
        wr_hex(os.path.join(odir, "bias_fp16.hex"),
               [f32_to_fp16_bits(np.float32(b / s_out_pre)) for b in bias_eff], 4)
        np.save(os.path.join(odir, "ref_ort.npy"), post.astype(np.float32))
        np.save(os.path.join(odir, "input_i8.npy"), x_i8)

        meta = dict(idx=idx, node=d["cname"], silu=int(d["silu"]),
                    depthwise=int(d["dw"]),
                    CIN=CIN, COUT=COUT, K=K, STRIDE=STRIDE, PAD=PAD,
                    H_IN=H_IN, W_IN=W_IN, H_OUT=H_OUT, W_OUT=W_OUT,
                    P_COUT=dims[idx]["P_COUT"], P_CIN=dims[idx]["P_CIN"],
                    s_a=s_a, zp_a=zp_a, s_w=d["s_w"], s_acc=s_acc, s_in=s_in,
                    s_out_pre=s_out_pre, s_out_silu=s_out_silu,
                    in_amp=in_amp, pre_amp=pre_amp, post_amp=post_amp)
        json.dump(meta, open(os.path.join(odir, "meta.json"), "w"), indent=2)
        summary.append((idx, d["cname"], CIN, COUT, K, STRIDE, H_IN, W_IN,
                        "silu" if d["silu"] else "no-act", "dw" if d["dw"] else "",
                        s_out_pre, s_out_silu))
        print(f"  L{idx:3d} {d['cname']:42s} {CIN:3d}->{COUT:3d} k{K}s{STRIDE} "
              f"{H_IN}x{W_IN} {'silu' if d['silu'] else 'noact':5s} "
              f"{'dw' if d['dw'] else '  '} "
              f"S_PRE={s_out_pre:.5f} S_SILU={s_out_silu:.5f}")

    print(f"\nwrote {len(summary)} conv_stage layers to {OUTROOT}; "
          f"skipped {len(skipped)} block-internal: {[s[0] for s in skipped]}")
    json.dump({"written": [s[0] for s in summary], "skipped": skipped},
              open(os.path.join(OUTROOT, "manifest.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
