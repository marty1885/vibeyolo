#!/usr/bin/env python3
# Copyright (c) 2026 vibeyolo
# SPDX-License-Identifier: Apache-2.0
#
# Non-destructive layer/scaling generator for YOLO26n integration work.
#
# This intentionally writes under integ/generated/ by default. The existing
# hand-built integ/layer_* directories remain the reference implementation.

import argparse
import math
import os
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "integ" / "scale"))

from gen_scale_pkg import infer_layer_specs  # noqa: E402

import onnx  # noqa: E402


def analyze_topology(model_path):
    """Return per-ConvInteger-index feature dict.

    For each conv layer (indexed in topological order, matching scale_pkg's
    LAYER_<i>_*) returns:
        {
          'has_silu':       bool,   # is the conv followed by a SiLU Mul(act/Mul)?
          'has_residual':   bool,   # does an Add consume this conv's chain output
                                    # AND the residual operand is produced earlier?
          'residual_src':   str | None,   # the residual operand tensor name
          'silu_out':       str | None,   # the SiLU Mul output (or post-bias add)
          'add_out':        str | None,   # the Add output (final output of layer)
        }
    """
    m = onnx.load(model_path)
    nodes = list(m.graph.node)
    out_to_pos = {}  # tensor -> producing node position
    for i, n in enumerate(nodes):
        for o in n.output:
            out_to_pos[o] = i

    conv_pos_list = []  # positions of ConvInteger nodes, in order
    for i, n in enumerate(nodes):
        if n.op_type == "ConvInteger":
            conv_pos_list.append(i)

    features = {}
    for idx, cpos in enumerate(conv_pos_list):
        conv = nodes[cpos]
        layer_base = conv.name.rsplit("/conv/Conv_quant", 1)[0]
        silu_out = None
        # Search forward (limited window) for the matching SiLU Mul.
        for n in nodes[cpos + 1:cpos + 15]:
            if n.op_type == "Mul" and n.name == layer_base + "/act/Mul":
                silu_out = n.output[0]
                break
        has_silu = silu_out is not None
        # Chain output that may feed Add residual.
        chain_out = silu_out
        if chain_out is None:
            for n in nodes[cpos + 1:cpos + 12]:
                if n.op_type == "Add" and n.name.endswith("_bias_add"):
                    chain_out = n.output[0]
                    break
        # Find Add that consumes the chain output and is NOT the bias_add.
        residual_src = None
        add_out = None
        if chain_out is not None:
            for n in nodes:
                if (n.op_type == "Add" and chain_out in n.input
                        and not n.name.endswith("_bias_add")):
                    others = [x for x in n.input if x != chain_out]
                    if len(others) == 1:
                        other = others[0]
                        # Only treat as residual if the other operand is produced
                        # earlier in topological order than this conv.
                        other_pos = out_to_pos.get(other, None)
                        if other_pos is not None and other_pos < cpos:
                            residual_src = other
                            add_out = n.output[0]
                            break
        features[idx] = {
            "has_silu": has_silu,
            "has_residual": residual_src is not None,
            "residual_src": residual_src,
            "silu_out": silu_out,
            "add_out": add_out,
        }
    return features



def divisors(n):
    out = []
    for i in range(1, int(math.sqrt(n)) + 1):
        if n % i == 0:
            out.append(i)
            if i * i != n:
                out.append(n // i)
    return sorted(out)


def short_name(layer_name):
    path = layer_name
    path = re.sub(r"/conv/Conv_quant$", "", path)
    path = re.sub(r"/Conv_quant$", "", path)
    path = path.strip("/")
    parts = path.split("/")
    if len(parts) >= 2 and parts[0] == "model":
        parts = parts[1:]
    return "_".join(p.replace(".", "") for p in parts if p not in ("conv",))


def macs_for(lay):
    return (lay["cout"] * (lay["cin"] // lay["group"]) * lay["kh"] * lay["kw"] *
            lay["H_out"] * lay["W_out"])


def choose_balanced(lay, target_cycles, max_p_pix=1, cout_cap=None, cin_cap=None):
    """Pick minimum-area legal factors that meet target_cycles.

    Area proxy is P_PIX * P_COUT * P_CIN. K*K lanes are folded inside dotN for
    every P_CIN, so K*K affects throughput but not the factor search legality.
    For grouped/depthwise convs, each output only sees Cin/group input channels.
    Ties prefer cycles closer to target, then less pixel parallelism.
    """
    k2 = lay["kh"] * lay["kw"]
    macs = macs_for(lay)
    cin_per_group = lay["cin"] // lay["group"]
    pix_limit = min(lay["H_out"] * lay["W_out"], max_p_pix)
    pix_choices = [d for d in divisors(lay["H_out"] * lay["W_out"]) if d <= pix_limit]
    cout_choices = divisors(lay["cout"])
    cin_choices = divisors(cin_per_group)
    if cout_cap is not None:
        cout_choices = [d for d in cout_choices if d <= cout_cap]
    if cin_cap is not None:
        cin_choices = [d for d in cin_choices if d <= cin_cap]

    best = None
    for p_pix in pix_choices:
        for p_cout in cout_choices:
            for p_cin in cin_choices:
                par = p_pix * p_cout * p_cin * k2
                cycles = math.ceil(macs / par)
                if cycles > target_cycles:
                    continue
                area = p_pix * p_cout * p_cin
                # Primary: area. Secondary: avoid too-fast stages. Tertiary:
                # avoid P_PIX unless it is needed, because it complicates linebufs.
                score = (area, target_cycles - cycles, p_pix, p_cin, p_cout)
                cand = (score, p_pix, p_cout, p_cin, cycles, macs)
                if best is None or cand[0] < best[0]:
                    best = cand
    if best is None:
        # Return the largest permitted point and let the report flag the miss.
        p_pix = pix_choices[-1]
        p_cout = cout_choices[-1]
        p_cin = cin_choices[-1]
        cycles = math.ceil(macs / (p_pix * p_cout * p_cin * k2))
        return p_pix, p_cout, p_cin, cycles, macs, False
    _, p_pix, p_cout, p_cin, cycles, macs = best
    return p_pix, p_cout, p_cin, cycles, macs, True


def emit_scale_pkg(layers, out_sv, out_report, target, max_p_pix, cout_cap, cin_cap, dv=False):
    out_sv.parent.mkdir(parents=True, exist_ok=True)
    if out_report:
        out_report.parent.mkdir(parents=True, exist_ok=True)

    header = "scale_pkg_dv.sv" if dv else "scale_pkg.sv"
    lines = [
        "// Copyright (c) 2026 vibeyolo",
        "// SPDX-License-Identifier: Apache-2.0",
        "//",
        f"// {header} -- generated by tools/layergen/layergen.py.",
        "// Balanced mode: minimize area while keeping each conv layer under T_FRAME.",
        "// Package name intentionally remains scale_pkg.",
        "",
        "// verilator lint_off UNUSEDPARAM",
        "// verilator lint_off DECLFILENAME",
        "package scale_pkg;",
        "",
        f"  localparam int T_FRAME    = {target};",
        f"  localparam int NUM_LAYERS = {len(layers)};",
        "",
    ]
    report = [
        f"# Balanced scale report -- YOLO26n (T_FRAME = {target})",
        "",
        "The optimizer minimizes `P_PIX * P_COUT * P_CIN` subject to `cycles <= T_FRAME`.",
        "Grouped/depthwise conv MACs use `Cin/group`. Ties prefer stages closer to the target.",
        "",
        f"- `max_p_pix`: {max_p_pix}",
        f"- `cout_cap`: {cout_cap if cout_cap is not None else 'none'}",
        f"- `cin_cap`: {cin_cap if cin_cap is not None else 'none'}",
        "",
        "| idx | name | Cin | Cout | K | HxW | P_PIX | P_COUT | P_CIN | area | cycles | meets |",
        "|-----|------|----:|-----:|--:|-----|------:|-------:|------:|-----:|-------:|:------|",
    ]

    total_area = 0
    worst = 0
    misses = 0
    for i, lay in enumerate(layers):
        assert lay["kh"] == lay["kw"], f"L{i} non-square kernel"
        assert lay["sh"] == lay["sw"], f"L{i} non-square stride"
        assert lay["ph"] == lay["pw"], f"L{i} asymmetric pad"
        p_pix, p_cout, p_cin, cycles, macs, meets = choose_balanced(
            lay, target, max_p_pix=max_p_pix, cout_cap=cout_cap, cin_cap=cin_cap)
        area = p_pix * p_cout * p_cin
        total_area += area
        worst = max(worst, cycles)
        misses += 0 if meets else 1
        lines += [
            f"  // L{i}: {lay['name']}",
            f"  //   shape Cin={lay['cin']} CinPerGroup={lay['cin'] // lay['group']} Cout={lay['cout']} K={lay['kh']} stride={lay['sh']} pad={lay['ph']} group={lay['group']}",
            f"  //   H_in={lay['H_in']} W_in={lay['W_in']} H_out={lay['H_out']} W_out={lay['W_out']}",
            f"  //   M={macs} cycles={cycles} target={target}{'' if meets else ' MISSES_TARGET'}",
            f"  localparam int LAYER_{i}_P_PIX  = {p_pix};",
            f"  localparam int LAYER_{i}_P_COUT = {p_cout};",
            f"  localparam int LAYER_{i}_P_CIN  = {p_cin};",
            f"  localparam int LAYER_{i}_H      = {lay['H_out']};",
            f"  localparam int LAYER_{i}_W      = {lay['W_out']};",
            f"  localparam int LAYER_{i}_K      = {lay['kh']};",
            f"  localparam int LAYER_{i}_CIN    = {lay['cin']};",
            f"  localparam int LAYER_{i}_COUT   = {lay['cout']};",
            f"  localparam int LAYER_{i}_STRIDE = {lay['sh']};",
            f"  localparam int LAYER_{i}_PAD    = {lay['ph']};",
            f"  localparam int LAYER_{i}_GROUP  = {lay['group']};",
            f"  localparam int LAYER_{i}_CYCLES = {cycles};",
            "",
        ]
        report.append(
            f"| {i} | {lay['name']} | {lay['cin']} | {lay['cout']} | {lay['kh']} | "
            f"{lay['H_out']}x{lay['W_out']} | {p_pix} | {p_cout} | {p_cin} | "
            f"{area} | {cycles} | {'yes' if meets else 'no'} |")

    lines += ["endpackage", "// verilator lint_on DECLFILENAME", "// verilator lint_on UNUSEDPARAM", ""]
    out_sv.write_text("\n".join(lines))

    if out_report:
        report += [
            "",
            f"**Total area proxy:** {total_area:,}",
            f"**Worst cycles:** {worst:,}",
            f"**Target misses:** {misses}",
            "",
        ]
        out_report.write_text("\n".join(report))


def sv_width(expr):
    return max(1, int(math.ceil(math.log2(max(2, expr)))))


def make_ref(path):
    path = Path(path).resolve()
    try:
        return "$(REPO_ROOT)/" + str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def write_s_out_params_placeholder(ldir):
    """Write a default stim/s_out_params.sv so that Verilator can find the file
    before the first `make stim` populates real values. Both `extract.py` and
    the RTL shim/TB consume this package to keep S_OUT_PRE/S_OUT_SILU in lockstep.

    The package name is intentionally globally unique per repo so multiple
    layers can coexist in one verilator command if ever needed. But because
    every layer's DV is built in isolation, a single name works fine.
    """
    sv = (
        "// Copyright (c) 2026 vibeyolo\n"
        "// SPDX-License-Identifier: Apache-2.0\n"
        "//\n"
        "// Placeholder generated by tools/layergen/layergen.py.\n"
        "// `make stim` (extract.py) overwrites this file with values derived\n"
        "// from observed activation amplitudes (S_OUT = max|silu_out|*1.1/127).\n"
        "package s_out_params;\n"
        "  localparam real S_OUT_PRE_VAL  = 8.0 / 127.0;\n"
        "  localparam real S_OUT_SILU_VAL = 8.0 / 127.0;\n"
        "endpackage\n"
    )
    (ldir / "stim").mkdir(parents=True, exist_ok=True)
    (ldir / "stim" / "s_out_params.sv").write_text(sv)




def emit_conv_silu_extract(lay, idx, ldir):
    # Allow depthwise (group == cin == cout): the extractor expands W_q from
    # (cout, 1, k, k) -> dense (cout, cin, k, k) with zeros so the driver / RTL
    # can stay group-1-agnostic.
    is_depthwise = lay["group"] != 1
    if is_depthwise:
        assert lay["group"] == lay["cin"] == lay["cout"], \
            "only pure depthwise (group==cin==cout) supported"
    roi_out_h = min(8, lay["H_out"])
    roi_out_w = min(8, lay["W_out"])
    roi_h = roi_out_h * lay["sh"]
    roi_w = roi_out_w * lay["sw"]
    pad_h = roi_h + 2 * lay["ph"]
    pad_w = roi_w + 2 * lay["pw"]
    # Keep the pilot ROI away from global boundaries when possible, aligned so
    # output ROI coordinates are integral for stride-2 layers.
    r = min(max(2 * lay["sh"], lay["ph"]), max(0, lay["H_in"] - roi_h - lay["ph"]))
    c = min(max(2 * lay["sw"], lay["pw"]), max(0, lay["W_in"] - roi_w - lay["pw"]))
    r -= r % lay["sh"]
    c -= c % lay["sw"]

    text = f'''#!/usr/bin/env python3
# Copyright (c) 2026 vibeyolo
# SPDX-License-Identifier: Apache-2.0
#
# Generated by tools/layergen/layergen.py.
# Ordinary Conv+SiLU extractor: group=1 only, no residual path.

import json
import os

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper

HERE = os.path.dirname(os.path.abspath(__file__))
STIM = os.path.join(HERE, "stim")
os.makedirs(STIM, exist_ok=True)

MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    "/home/marty/Documents/aif/vibeyolo/integ/yolo26n/model_int8.onnx",
)

LAYER_IDX = {idx}
LAYER_NAME = "{lay['name']}"
NCH_IN = {lay['cin']}
NCH_OUT = {lay['cout']}
K = {lay['kh']}
STRIDE = {lay['sh']}
PAD = {lay['ph']}
GROUP = {lay['group']}
IS_DEPTHWISE = {1 if is_depthwise else 0}
ROI_H, ROI_W = {roi_h}, {roi_w}
OUT_H, OUT_W = {roi_out_h}, {roi_out_w}
PAD_H, PAD_W = {pad_h}, {pad_w}
R, C = {r}, {c}

model = onnx.load(MODEL_PATH)
init_by_name = {{i.name: numpy_helper.to_array(i) for i in model.graph.initializer}}
conv = next(n for n in model.graph.node if n.op_type == "ConvInteger" and n.name == LAYER_NAME)

W_q_raw = init_by_name[conv.input[1]].astype(np.int32)
s_w = float(init_by_name[conv.input[1].replace("_quantized", "_scale")])
zp_w = int(init_by_name[conv.input[3]])
assert zp_w == 0
if IS_DEPTHWISE:
    assert W_q_raw.shape == (NCH_OUT, 1, K, K), f"got {{W_q_raw.shape}}"
    # Expand to dense: W_q[co, ci, kh, kw] = orig[co, 0, kh, kw] if co==ci else 0.
    W_q = np.zeros((NCH_OUT, NCH_IN, K, K), dtype=np.int32)
    for co in range(NCH_OUT):
        W_q[co, co] = W_q_raw[co, 0]
else:
    assert W_q_raw.shape == (NCH_OUT, NCH_IN, K, K), f"got {{W_q_raw.shape}}"
    W_q = W_q_raw

conv_pos = list(model.graph.node).index(conv)
layer_base = LAYER_NAME.rsplit("/conv/Conv_quant", 1)[0]
bias = None
target_out = None
for n in list(model.graph.node)[conv_pos + 1:]:
    if bias is None and n.op_type == "Reshape" and n.input and n.input[0] in init_by_name:
        cand = init_by_name[n.input[0]].astype(np.float32)
        if cand.shape == (NCH_OUT,):
            bias = cand
    if n.op_type == "Mul" and n.name == layer_base + "/act/Mul":
        target_out = n.output[0]
        break
assert bias is not None, "could not find following bias initializer"
assert target_out is not None, "could not find following SiLU Mul output"

IN_Q = conv.input[0]
IN_ZP = conv.input[2]
IN_S = IN_Q.replace("_quantized", "_scale")
sum_w = W_q.reshape(NCH_OUT, -1).sum(axis=1).astype(np.int32)

mod_model = onnx.load(MODEL_PATH)
existing = {{o.name for o in mod_model.graph.output}}
for nm, tp in [
    (target_out, onnx.TensorProto.FLOAT),
    (IN_Q, onnx.TensorProto.UINT8),
    (IN_S, onnx.TensorProto.FLOAT),
    (IN_ZP, onnx.TensorProto.UINT8),
]:
    if nm not in existing:
        mod_model.graph.output.append(onnx.helper.make_tensor_value_info(nm, tp, None))
tmp_path = os.path.join(HERE, "_model_with_intermediate.onnx")
onnx.save(mod_model, tmp_path)

so = ort.SessionOptions()
so.log_severity_level = 3
sess = ort.InferenceSession(tmp_path, sess_options=so, providers=["CPUExecutionProvider"])
ipt_name = sess.get_inputs()[0].name

def run_ort(img_f32):
    outs = sess.run([target_out, IN_Q, IN_S, IN_ZP], {{ipt_name: img_f32}})
    return outs[0], outs[1], float(outs[2]), int(outs[3])

def make_input(kind, seed=0):
    img = np.zeros((1, 3, 640, 640), dtype=np.float32)
    rng = np.random.RandomState(seed)
    if kind == "rand":
        img[0] = rng.uniform(0.0, 1.0, size=(3, 640, 640)).astype(np.float32)
    elif kind == "half":
        img[0] = 0.5
    elif kind == "gradient":
        gx = np.tile(np.linspace(0.0, 1.0, 640, dtype=np.float32), (640, 1))
        img[0, 0] = gx
        img[0, 1] = gx.T
        img[0, 2] = 0.5 * (gx + gx.T)
    elif kind == "rand_low":
        img[0] = rng.uniform(0.15, 0.80, size=(3, 640, 640)).astype(np.float32)
    elif kind == "rand_high":
        img[0] = rng.uniform(0.5, 1.0, size=(3, 640, 640)).astype(np.float32)
    else:
        raise ValueError(kind)
    return img

def f32_to_fp16(x):
    return np.float16(x).view(np.uint16)

def hw_reference(qx_u8_full, s_a, zp_a):
    s_acc = s_a * s_w
    pad_u8 = np.full((NCH_IN, PAD_H, PAD_W), zp_a, dtype=np.int32)
    r0, r1 = R - PAD, R - PAD + PAD_H
    c0, c1 = C - PAD, C - PAD + PAD_W
    h_full, w_full = qx_u8_full.shape[2], qx_u8_full.shape[3]
    sr0, sr1 = max(0, r0), min(h_full, r1)
    sc0, sc1 = max(0, c0), min(w_full, c1)
    pad_u8[:, sr0-r0:sr1-r0, sc0-c0:sc1-c0] = qx_u8_full[0, :, sr0:sr1, sc0:sc1].astype(np.int32)
    x_i8 = np.clip(pad_u8 - 128, -128, 127).astype(np.int8)
    bias_eff = bias + s_acc * (128 - zp_a) * sum_w

    out = np.zeros((NCH_OUT, OUT_H, OUT_W), dtype=np.float32)
    pre_min, pre_max = 1e30, -1e30
    max_abs_acc = 0
    for co in range(NCH_OUT):
        for oy in range(OUT_H):
            for ox in range(OUT_W):
                iy = oy * STRIDE
                ix = ox * STRIDE
                window = x_i8[:, iy:iy+K, ix:ix+K].astype(np.int32)
                acc = int((W_q[co] * window).sum())
                max_abs_acc = max(max_abs_acc, abs(acc))
                pre = acc * s_acc + bias_eff[co]
                pre_min = min(pre_min, pre)
                pre_max = max(pre_max, pre)
                out[co, oy, ox] = pre / (1.0 + np.exp(-pre))
    return out, bias_eff, x_i8, s_acc, max_abs_acc, (pre_min, pre_max)

N_LANE_FULL = K * K * NCH_IN
W_flat = np.zeros((NCH_OUT, N_LANE_FULL), dtype=np.int8)
for co in range(NCH_OUT):
    for kh in range(K):
        for kw in range(K):
            for kc in range(NCH_IN):
                lane = (kh * K + kw) * NCH_IN + kc
                W_flat[co, lane] = W_q[co, kc, kh, kw]

with open(os.path.join(STIM, "weights.i8.hex"), "w") as f:
    for co in range(NCH_OUT):
        for lane in range(N_LANE_FULL):
            f.write(f"{{int(W_flat[co, lane]) & 0xFF:02x}}\\n")

samples = [
    ("rand0", "rand", 0),
    ("rand1", "rand", 1),
    ("half", "half", 0),
    ("gradient", "gradient", 0),
    ("rand_low", "rand_low", 2),
    ("rand_high", "rand_high", 3),
]

cache = []
pre_lo, pre_hi, post_hi = 0.0, 0.0, 0.0
for sname, kind, seed in samples:
    img = make_input(kind, seed)
    ort_out, qx_u8, s_a, zp_a = run_ort(img)
    if s_a == 0.0:
        s_a = 1.0
    hw_out, _, _, _, _, (pmin, pmax) = hw_reference(qx_u8, s_a, zp_a)
    oy0, ox0 = R // STRIDE, C // STRIDE
    ort_roi = ort_out[0, :, oy0:oy0+OUT_H, ox0:ox0+OUT_W]
    pre_lo = min(pre_lo, pmin)
    pre_hi = max(pre_hi, pmax)
    post_hi = max(post_hi, float(np.max(np.abs(ort_roi))))
    cache.append((sname, ort_roi, qx_u8, s_a, zp_a))

def pick_s_out(amp):
    # HANDOFF footgun #5 empirical rule: S_OUT = max * 1.1 / 127. Sized so that
    # the LUT covers the dynamic range with ~10% headroom. Floor at 2/127 so
    # nearly-zero tensors still have a reasonable grid.
    raw = max(amp, 1e-3) * 1.1 / 127.0
    return max(raw, 2.0 / 127.0)

# S_OUT_PRE must cover the *pre-SiLU* dynamic range (LUT input grid). Otherwise
# negative pre-activations saturate to silu(-128*S_OUT_PRE) which is far from 0
# and corrupts the output. S_OUT_SILU sizes the *post-SiLU* output grid for
# LUT precision and the act_silu output quantization.
pre_amp  = max(abs(pre_lo), abs(pre_hi))
S_OUT_PRE  = pick_s_out(pre_amp)
S_OUT_SILU = pick_s_out(post_hi)
print(f"{{LAYER_NAME}} pre-SiLU=[{{pre_lo:.3f}}, {{pre_hi:.3f}}] post|max|={{post_hi:.3f}}")
print(f"S_OUT_PRE={{S_OUT_PRE:.8f}} S_OUT_SILU={{S_OUT_SILU:.8f}} "
      f"({{'dynamic' if S_OUT_PRE < 8.0/127.0 else 'default 8.0/127'}})")

# Emit SV package so the RTL shim/TB use the same constants. Lockstep critical
# (HANDOFF footgun #5: if extract.py's S_OUT diverges from RTL's, cos drops).
def _write_s_out_params_sv():
    txt = (
        "// Copyright (c) 2026 vibeyolo\\n"
        "// Auto-generated by extract.py (do not edit).\\n"
        "package s_out_params;\\n"
        f"  localparam real S_OUT_PRE_VAL  = {{S_OUT_PRE:.10e}};\\n"
        f"  localparam real S_OUT_SILU_VAL = {{S_OUT_SILU:.10e}};\\n"
        "endpackage\\n"
    )
    path = os.path.join(STIM, "s_out_params.sv")
    with open(path + ".tmp", "w") as f:
        f.write(txt)
    os.replace(path + ".tmp", path)
_write_s_out_params_sv()

manifest = {{
    "layer_idx": LAYER_IDX, "layer_name": LAYER_NAME,
    "out_h": OUT_H, "out_w": OUT_W, "nch_out": NCH_OUT, "nch_in": NCH_IN,
    "K": K, "stride": STRIDE, "pad": PAD, "n_lane_full": N_LANE_FULL,
    "roi_h": ROI_H, "roi_w": ROI_W, "pad_h": PAD_H, "pad_w": PAD_W,
    "R": R, "C": C, "s_w": s_w,
    "s_out_pre": S_OUT_PRE, "s_out_silu": S_OUT_SILU,
    "samples": [],
    "limitations": "generated ordinary group=1 Conv+SiLU DV; residual/grouped paths are not covered",
}}

global_max_acc = 0
for sname, ort_roi, qx_u8, s_a, zp_a in cache:
    hw_out, bias_eff, x_i8, s_acc, max_acc, _ = hw_reference(qx_u8, s_a, zp_a)
    global_max_acc = max(global_max_acc, max_acc)
    scale_fp16 = np.array([f32_to_fp16(s_acc / S_OUT_PRE)] * NCH_OUT, dtype=np.uint16)
    bias_fp16 = np.array([f32_to_fp16(bias_eff[co] / S_OUT_PRE) for co in range(NCH_OUT)], dtype=np.uint16)

    with open(os.path.join(STIM, f"{{sname}}.input_i8.hex"), "w") as f:
        for h in range(PAD_H):
            for w in range(PAD_W):
                for kc in range(NCH_IN):
                    f.write(f"{{int(x_i8[kc, h, w]) & 0xFF:02x}}\\n")
    with open(os.path.join(STIM, f"{{sname}}.scale_fp16.hex"), "w") as f:
        for co in range(NCH_OUT):
            f.write(f"{{int(scale_fp16[co]):04x}}\\n")
    with open(os.path.join(STIM, f"{{sname}}.bias_fp16.hex"), "w") as f:
        for co in range(NCH_OUT):
            f.write(f"{{int(bias_fp16[co]):04x}}\\n")
    with open(os.path.join(STIM, f"{{sname}}.ref_ort.f32.hex"), "w") as f:
        for oy in range(OUT_H):
            for ox in range(OUT_W):
                for co in range(NCH_OUT):
                    bits = np.float32(ort_roi[co, oy, ox]).view(np.uint32)
                    f.write(f"{{int(bits):08x}}\\n")
    with open(os.path.join(STIM, f"{{sname}}.ref_hw.f32.hex"), "w") as f:
        for oy in range(OUT_H):
            for ox in range(OUT_W):
                for co in range(NCH_OUT):
                    bits = np.float32(hw_out[co, oy, ox]).view(np.uint32)
                    f.write(f"{{int(bits):08x}}\\n")

    diff = ort_roi - hw_out
    mxe = float(np.max(np.abs(diff)))
    mae = float(np.mean(np.abs(diff)))
    out_range = float(ort_roi.max() - ort_roi.min())
    cos = float(np.sum(ort_roi * hw_out) / (np.linalg.norm(ort_roi) * np.linalg.norm(hw_out) + 1e-12))
    print(f"[{{sname}}] s_a={{s_a:.5f}} zp_a={{zp_a}} max|acc|={{max_acc}} "
          f"ORT vs HW-ref: max_abs={{mxe:.4f}} mae={{mae:.4f}} cos={{cos:.6f}}")
    manifest["samples"].append({{
        "name": sname, "s_a": s_a, "zp_a": zp_a, "s_acc": s_acc,
        "max_abs_acc": int(max_acc),
        "ort_vs_hw_max_abs": mxe, "ort_vs_hw_mae": mae,
        "ort_vs_hw_cos": cos, "out_range": out_range,
    }})

print(f"global max|acc| = {{global_max_acc}}")
with open(os.path.join(STIM, "manifest.json"), "w") as f:
    json.dump(manifest, f, indent=2)
print(f"Wrote stim+ref to {{STIM}}")
'''
    (ldir / "extract.py").write_text(text)
    os.chmod(ldir / "extract.py", 0o755)


def emit_conv_silu_test(lay, idx, ldir, p_pix, p_cout, p_cin):
    # Depthwise: emitter is identical; extract.py expands W to dense form.
    if lay["group"] != 1:
        assert lay["group"] == lay["cin"] == lay["cout"]
    roi_out_h = min(8, lay["H_out"])
    roi_out_w = min(8, lay["W_out"])
    roi_h = roi_out_h * lay["sh"]
    roi_w = roi_out_w * lay["sw"]
    pad_h = roi_h + 2 * lay["ph"]
    pad_w = roi_w + 2 * lay["pw"]
    n_cout_tile = math.ceil(lay["cout"] / p_cout)
    n_cin_tile = math.ceil(lay["cin"] / p_cin)
    n_lane_tile = lay["kh"] * lay["kw"] * p_cin
    n_lane_full = lay["kh"] * lay["kw"] * lay["cin"]
    dot_lat = 1 + math.ceil(math.log2(max(2, n_lane_tile)))

    text = f'''// Copyright (c) 2026 vibeyolo
// SPDX-License-Identifier: Apache-2.0
//
// Generated by tools/layergen/layergen.py.
// Ordinary Conv+SiLU DV: group=1 only, no residual path.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "Vlayer_{idx}_tb.h"
#include "sim_ctrl.h"

using DUT = Vlayer_{idx}_tb;

static constexpr int NCH_IN = {lay['cin']};
static constexpr int NCH_OUT = {lay['cout']};
static constexpr int K = {lay['kh']};
static constexpr int STRIDE = {lay['sh']};
static constexpr int PAD = {lay['ph']};
static constexpr int P_PIX = {p_pix};
static constexpr int P_COUT = {p_cout};
static constexpr int P_CIN = {p_cin};
static constexpr int N_COUT_TILE = {n_cout_tile};
static constexpr int N_CIN_TILE = {n_cin_tile};
static constexpr int N_LANE_TILE = {n_lane_tile};
static constexpr int N_LANE_FULL = {n_lane_full};
static constexpr int ROI_H = {roi_h};
static constexpr int ROI_W = {roi_w};
static constexpr int PAD_H = {pad_h};
static constexpr int PAD_W = {pad_w};
static constexpr int OUT_H = {roi_out_h};
static constexpr int OUT_W = {roi_out_w};
static constexpr int DOT_LAT = {dot_lat};
static constexpr int ACC_LAT = 1;
static constexpr int RQ_IN_LAT = 1;
static constexpr int REQUANT_LAT = 7;
static constexpr int SILU_LAT = 1;
static constexpr int TOTAL_LAT = DOT_LAT + ACC_LAT + RQ_IN_LAT + REQUANT_LAT + SILU_LAT;

static std::string stim_dir() {{ return std::string("../stim"); }}

static std::vector<uint32_t> load_hex(const std::string& path) {{
    std::ifstream f(path);
    if (!f) {{ fprintf(stderr, "could not open %s\\n", path.c_str()); std::exit(2); }}
    std::vector<uint32_t> v;
    std::string line;
    while (std::getline(f, line)) {{
        if (!line.empty()) v.push_back((uint32_t)std::stoul(line, nullptr, 16));
    }}
    return v;
}}

static float u32_to_f32(uint32_t bits) {{ float f; std::memcpy(&f, &bits, 4); return f; }}

static double load_manifest_number(const std::string& key) {{
    std::ifstream f(stim_dir() + "/manifest.json");
    if (!f) {{ fprintf(stderr, "could not open manifest.json\\n"); std::exit(2); }}
    std::string text((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    std::string needle = "\\"" + key + "\\"";
    size_t p = text.find(needle);
    if (p == std::string::npos) {{ fprintf(stderr, "manifest key missing: %s\\n", key.c_str()); std::exit(2); }}
    p = text.find(':', p);
    if (p == std::string::npos) {{ fprintf(stderr, "manifest key malformed: %s\\n", key.c_str()); std::exit(2); }}
    char* end = nullptr;
    double v = std::strtod(text.c_str() + p + 1, &end);
    if (end == text.c_str() + p + 1) {{ fprintf(stderr, "manifest value malformed: %s\\n", key.c_str()); std::exit(2); }}
    return v;
}}

struct Sample {{
    std::string name;
    std::vector<int8_t> input_i8;
    std::vector<uint16_t> scale_fp16, bias_fp16;
    std::vector<float> ref_ort, ref_hw;
}};

static Sample load_sample(const std::string& name) {{
    Sample s; s.name = name;
    auto inp = load_hex(stim_dir() + "/" + name + ".input_i8.hex");
    s.input_i8.resize(inp.size());
    for (size_t i = 0; i < inp.size(); i++) s.input_i8[i] = (int8_t)(uint8_t)(inp[i] & 0xFF);
    for (auto v : load_hex(stim_dir() + "/" + name + ".scale_fp16.hex")) s.scale_fp16.push_back((uint16_t)v);
    for (auto v : load_hex(stim_dir() + "/" + name + ".bias_fp16.hex")) s.bias_fp16.push_back((uint16_t)v);
    for (auto v : load_hex(stim_dir() + "/" + name + ".ref_ort.f32.hex")) s.ref_ort.push_back(u32_to_f32(v));
    for (auto v : load_hex(stim_dir() + "/" + name + ".ref_hw.f32.hex")) s.ref_hw.push_back(u32_to_f32(v));
    return s;
}}

template <typename T>
static void pack_bytes(T& dst, const int8_t* src, int nbytes) {{
    auto* p = reinterpret_cast<uint32_t*>(&dst);
    int nwords = (nbytes + 3) / 4;
    for (int i = 0; i < nwords; i++) p[i] = 0;
    for (int i = 0; i < nbytes; i++) p[i / 4] |= ((uint32_t)(uint8_t)src[i] << ((i % 4) * 8));
}}

template <typename T>
static void pack_u16(T& dst, const uint16_t* src, int nshorts) {{
    auto* p = reinterpret_cast<uint32_t*>(&dst);
    int nwords = (nshorts + 1) / 2;
    for (int i = 0; i < nwords; i++) p[i] = 0;
    for (int i = 0; i < nshorts; i++) p[i / 2] |= ((uint32_t)src[i] << ((i % 2) * 16));
}}

template <typename T>
static void unpack_bytes(const T& src, int8_t* dst, int nbytes) {{
    auto* p = reinterpret_cast<const uint32_t*>(&src);
    for (int i = 0; i < nbytes; i++) dst[i] = (int8_t)((p[i / 4] >> ((i % 4) * 8)) & 0xFFu);
}}

static void build_window_tile(const Sample& s, int pix_base, int cit,
                              int8_t out[P_PIX * N_LANE_TILE]) {{
    int kc_base = cit * P_CIN;
    for (int pp = 0; pp < P_PIX; pp++) {{
        int pix = pix_base + pp;
        int oy = pix / OUT_W;
        int ox = pix % OUT_W;
        for (int kh = 0; kh < K; kh++) {{
            for (int kw = 0; kw < K; kw++) {{
                int h_idx = oy * STRIDE + kh;
                int w_idx = ox * STRIDE + kw;
                for (int kc_local = 0; kc_local < P_CIN; kc_local++) {{
                    int kc = kc_base + kc_local;
                    int tile_lane = (kh * K + kw) * P_CIN + kc_local;
                    int off = pp * N_LANE_TILE + tile_lane;
                    out[off] = 0;
                    if (pix < OUT_H * OUT_W && kc < NCH_IN) {{
                        int in_off = (h_idx * PAD_W + w_idx) * NCH_IN + kc;
                        out[off] = s.input_i8[in_off];
                    }}
                }}
            }}
        }}
    }}
}}

static void build_weight_tile(const std::vector<int8_t>& w_full, int ct, int cit,
                              int8_t out[P_COUT * N_LANE_TILE]) {{
    int kc_base = cit * P_CIN;
    int co_base = ct * P_COUT;
    for (int co_local = 0; co_local < P_COUT; co_local++) {{
        int co_global = co_base + co_local;
        for (int kh = 0; kh < K; kh++) {{
            for (int kw = 0; kw < K; kw++) {{
                for (int kc_local = 0; kc_local < P_CIN; kc_local++) {{
                    int kc = kc_base + kc_local;
                    int full_lane = (kh * K + kw) * NCH_IN + kc;
                    int tile_lane = (kh * K + kw) * P_CIN + kc_local;
                    int off = co_local * N_LANE_TILE + tile_lane;
                    out[off] = 0;
                    if (co_global < NCH_OUT && kc < NCH_IN)
                        out[off] = w_full[co_global * N_LANE_FULL + full_lane];
                }}
            }}
        }}
    }}
}}

struct Stats {{ double max_abs=0, mae=0, cos=0, out_range=0; int n=0, worst_idx=-1; float worst_dut=0, worst_ref=0; }};

static Stats compute_stats(const std::vector<float>& dut, const std::vector<float>& ref) {{
    Stats st; st.n = (int)dut.size();
    double sum_abs=0, dot=0, na=0, nb=0;
    float mn=1e30f, mx=-1e30f;
    for (int i = 0; i < st.n; i++) {{
        double e = std::abs((double)dut[i] - (double)ref[i]);
        sum_abs += e;
        if (e > st.max_abs) {{ st.max_abs=e; st.worst_idx=i; st.worst_dut=dut[i]; st.worst_ref=ref[i]; }}
        dot += (double)dut[i] * ref[i]; na += (double)dut[i] * dut[i]; nb += (double)ref[i] * ref[i];
        mn = std::min(mn, ref[i]); mx = std::max(mx, ref[i]);
    }}
    st.mae = sum_abs / std::max(1, st.n);
    st.cos = dot / (std::sqrt(na) * std::sqrt(nb) + 1e-30);
    st.out_range = mx - mn;
    return st;
}}

int main(int argc, char** argv) {{
    SimCtrl<DUT> sim(argc, argv);
    sim.max_time = 80000000;

    auto w_raw = load_hex(stim_dir() + "/weights.i8.hex");
    if ((int)w_raw.size() != NCH_OUT * N_LANE_FULL) {{
        fprintf(stderr, "weights size mismatch: got %zu expected %d\\n", w_raw.size(), NCH_OUT * N_LANE_FULL);
        return 2;
    }}
    std::vector<int8_t> w_full(NCH_OUT * N_LANE_FULL);
    for (int i = 0; i < NCH_OUT * N_LANE_FULL; i++) w_full[i] = (int8_t)(uint8_t)(w_raw[i] & 0xFF);

    sim.dut->valid_i = 0; sim.dut->first_cin_i = 0; sim.dut->last_cin_i = 0; sim.dut->cout_tile_idx_i = 0;
    std::vector<int8_t> zero_x(P_PIX * N_LANE_TILE, 0), zero_w(P_COUT * N_LANE_TILE, 0);
    std::vector<uint16_t> zero_q(P_COUT, 0);
    pack_bytes(sim.dut->x_flat_i, zero_x.data(), P_PIX * N_LANE_TILE);
    pack_bytes(sim.dut->w_flat_i, zero_w.data(), P_COUT * N_LANE_TILE);
    pack_u16(sim.dut->scale_flat_i, zero_q.data(), P_COUT);
    pack_u16(sim.dut->bias_flat_i, zero_q.data(), P_COUT);
    sim.reset();

    const std::vector<std::string> sample_names = {{"rand0", "rand1", "half", "gradient", "rand_low", "rand_high"}};
    int n_pass = 0;
    double agg_cos = 0.0, agg_mae = 0.0, agg_max = 0.0;

    for (const auto& sname : sample_names) {{
        Sample s = load_sample(sname);
        std::vector<int8_t> y_i8(OUT_H * OUT_W * NCH_OUT, 0);
        const int N_OUT = OUT_H * OUT_W;
        const int N_PIX_GROUP = (N_OUT + P_PIX - 1) / P_PIX;
        const int BEATS_PER_GROUP = N_COUT_TILE * N_CIN_TILE;
        const int TOTAL_BEATS = N_PIX_GROUP * BEATS_PER_GROUP;
        struct Commit {{ int pix_base; int ct; }};
        std::vector<Commit> inflight;
        inflight.reserve(N_PIX_GROUP * N_COUT_TILE + TOTAL_LAT);
        int produced = 0;

        for (int step = 0; step < TOTAL_BEATS + TOTAL_LAT + 8; step++) {{
            if (step < TOTAL_BEATS) {{
                int pg = step / BEATS_PER_GROUP;
                int rem = step % BEATS_PER_GROUP;
                int ct = rem / N_CIN_TILE;
                int cit = rem % N_CIN_TILE;
                int pix_base = pg * P_PIX;

                int8_t win[P_PIX * N_LANE_TILE];
                build_window_tile(s, pix_base, cit, win);
                pack_bytes(sim.dut->x_flat_i, win, P_PIX * N_LANE_TILE);

                int8_t wtile[P_COUT * N_LANE_TILE];
                build_weight_tile(w_full, ct, cit, wtile);
                pack_bytes(sim.dut->w_flat_i, wtile, P_COUT * N_LANE_TILE);

                uint16_t sbuf[P_COUT], bbuf[P_COUT];
                int co_base = ct * P_COUT;
                for (int co = 0; co < P_COUT; co++) {{
                    sbuf[co] = (co_base + co < NCH_OUT) ? s.scale_fp16[co_base + co] : 0;
                    bbuf[co] = (co_base + co < NCH_OUT) ? s.bias_fp16[co_base + co] : 0;
                }}
                pack_u16(sim.dut->scale_flat_i, sbuf, P_COUT);
                pack_u16(sim.dut->bias_flat_i, bbuf, P_COUT);

                sim.dut->valid_i = 1;
                sim.dut->first_cin_i = (cit == 0);
                sim.dut->last_cin_i = (cit == N_CIN_TILE - 1);
                sim.dut->cout_tile_idx_i = (uint8_t)ct;
                if (cit == N_CIN_TILE - 1) inflight.push_back({{pix_base, ct}});
            }} else {{
                sim.dut->valid_i = 0; sim.dut->first_cin_i = 0; sim.dut->last_cin_i = 0;
            }}

            sim.tick();

            if (sim.dut->valid_o && produced < (int)inflight.size()) {{
                int pix_base = inflight[produced].pix_base;
                int ct = inflight[produced].ct;
                int got_ct = sim.dut->cout_tile_idx_o;
                if (got_ct != ct) fprintf(stderr, "cout_tile_idx mismatch at produced=%d: expected %d got %d\\n", produced, ct, got_ct);
                int8_t row[P_PIX * P_COUT];
                unpack_bytes(sim.dut->y_flat_o, row, P_PIX * P_COUT);
                int co_base = ct * P_COUT;
                for (int pp = 0; pp < P_PIX; pp++) {{
                    int pix = pix_base + pp;
                    if (pix >= N_OUT) continue;
                    for (int co = 0; co < P_COUT; co++) {{
                        int co_global = co_base + co;
                        if (co_global < NCH_OUT)
                            y_i8[pix * NCH_OUT + co_global] = row[pp * P_COUT + co];
                    }}
                }}
                produced++;
            }}
        }}

        if (produced != N_PIX_GROUP * N_COUT_TILE)
            printf("  WARN: produced=%d expected=%d\\n", produced, N_PIX_GROUP * N_COUT_TILE);

        double s_out_silu = load_manifest_number("s_out_silu");

        std::vector<float> dut_f(N_OUT * NCH_OUT);
        for (int i = 0; i < N_OUT * NCH_OUT; i++) dut_f[i] = (float)((double)y_i8[i] * s_out_silu);

        Stats st_ort = compute_stats(dut_f, s.ref_ort);
        Stats st_hw = compute_stats(dut_f, s.ref_hw);
        double pass_mae_thresh = 0.08 * std::max(1e-6, st_ort.out_range);
        bool pass = (st_ort.cos > 0.997) && (st_ort.mae < pass_mae_thresh);

        printf("\\n--- sample %s ---\\n", sname.c_str());
        printf("  DUT vs ORT   : max_abs=%.4f mae=%.4f cos=%.6f out_range=%.3f thresh=%.4f\\n",
               st_ort.max_abs, st_ort.mae, st_ort.cos, st_ort.out_range, pass_mae_thresh);
        printf("  DUT vs HW-ref: max_abs=%.4f mae=%.4f cos=%.6f\\n", st_hw.max_abs, st_hw.mae, st_hw.cos);
        if (st_ort.worst_idx >= 0) {{
            int i = st_ort.worst_idx;
            int co = i % NCH_OUT, ox = (i / NCH_OUT) % OUT_W, oy = (i / NCH_OUT) / OUT_W;
            printf("  worst-px: (oy=%d ox=%d c=%d) dut=%.4f ort=%.4f hw=%.4f\\n",
                   oy, ox, co, st_ort.worst_dut, st_ort.worst_ref, s.ref_hw[i]);
        }}
        printf("  => %s\\n", pass ? "PASS" : "FAIL");
        sim.check(pass, std::string("sample ") + sname + " pass");
        if (pass) n_pass++;
        agg_cos += st_ort.cos;
        agg_mae += st_ort.mae;
        agg_max = std::max(agg_max, st_ort.max_abs);
    }}

    printf("\\n========================================\\n");
    printf("Aggregate: %d/%zu samples passed\\n", n_pass, sample_names.size());
    printf("  avg cos   = %.6f\\n", agg_cos / sample_names.size());
    printf("  avg mae   = %.6f\\n", agg_mae / sample_names.size());
    printf("  worst max = %.6f\\n", agg_max);
    printf("Layer {idx} P_PIX=%d P_COUT=%d P_CIN=%d K=%d stride=%d pad=%d\\n", P_PIX, P_COUT, P_CIN, K, STRIDE, PAD);
    printf("========================================\\n");
    return sim.finish();
}}
'''
    (ldir / "dv" / f"layer_{idx}_test.cc").write_text(text)


# ====================================================================
# Residual-aware extractor + driver for layers where RESIDUAL=1.
# Mirrors integ/layer_16_m0_m_m0_cv2 but driven by the topology features.
# ====================================================================
def emit_conv_resid_extract(lay, idx, ldir, features):
    assert lay["group"] == 1
    has_silu = bool(features["has_silu"])
    # No-SiLU residual: pre-add stream is the post-requant value (no SiLU).
    # The IP uses scale_a_fp16 = real_to_fp16(S_OUT_SILU) for the add_rq "a"
    # input, while post_silu_y is just rq_y (scaled at S_OUT_PRE) when SILU=0.
    # We pin S_OUT_SILU = S_OUT_PRE in the no-silu case so the math lines up.
    roi_out_h = min(8, lay["H_out"])
    roi_out_w = min(8, lay["W_out"])
    roi_h = roi_out_h * lay["sh"]
    roi_w = roi_out_w * lay["sw"]
    pad_h = roi_h + 2 * lay["ph"]
    pad_w = roi_w + 2 * lay["pw"]
    r = min(max(2 * lay["sh"], lay["ph"]), max(0, lay["H_in"] - roi_h - lay["ph"]))
    c = min(max(2 * lay["sw"], lay["pw"]), max(0, lay["W_in"] - roi_w - lay["pw"]))
    r -= r % lay["sh"]
    c -= c % lay["sw"]
    residual_src = features["residual_src"]
    add_out = features["add_out"]

    text = f'''#!/usr/bin/env python3
# Copyright (c) 2026 vibeyolo
# SPDX-License-Identifier: Apache-2.0
#
# Generated by tools/layergen/layergen.py.
# Residual Conv+SiLU+Add extractor (group=1).

import json
import os
import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper

HERE = os.path.dirname(os.path.abspath(__file__))
STIM = os.path.join(HERE, "stim")
os.makedirs(STIM, exist_ok=True)

MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    "/home/marty/Documents/aif/vibeyolo/integ/yolo26n/model_int8.onnx",
)

LAYER_IDX = {idx}
LAYER_NAME = "{lay['name']}"
NCH_IN = {lay['cin']}
NCH_OUT = {lay['cout']}
K = {lay['kh']}
STRIDE = {lay['sh']}
PAD = {lay['ph']}
ROI_H, ROI_W = {roi_h}, {roi_w}
OUT_H, OUT_W = {roi_out_h}, {roi_out_w}
PAD_H, PAD_W = {pad_h}, {pad_w}
R, C = {r}, {c}
RESID_SRC = "{residual_src}"
ADD_OUT = "{add_out}"
HAS_SILU = {1 if has_silu else 0}

model = onnx.load(MODEL_PATH)
init_by_name = {{i.name: numpy_helper.to_array(i) for i in model.graph.initializer}}
conv = next(n for n in model.graph.node if n.op_type == "ConvInteger" and n.name == LAYER_NAME)

W_q = init_by_name[conv.input[1]].astype(np.int32)
s_w = float(init_by_name[conv.input[1].replace("_quantized", "_scale")])
zp_w = int(init_by_name[conv.input[3]])
assert zp_w == 0
assert W_q.shape == (NCH_OUT, NCH_IN, K, K), f"got {{W_q.shape}}"

conv_pos = list(model.graph.node).index(conv)
layer_base = LAYER_NAME.rsplit("/conv/Conv_quant", 1)[0]
bias = None
for n in list(model.graph.node)[conv_pos + 1:]:
    if bias is None and n.op_type == "Reshape" and n.input and n.input[0] in init_by_name:
        cand = init_by_name[n.input[0]].astype(np.float32)
        if cand.shape == (NCH_OUT,):
            bias = cand
            break
assert bias is not None, "could not find bias initializer"

IN_Q = conv.input[0]
IN_ZP = conv.input[2]
IN_S = IN_Q.replace("_quantized", "_scale")
RES_Q = RESID_SRC + "_quantized"
RES_S = RESID_SRC + "_scale"
RES_ZP = RESID_SRC + "_zero_point"
sum_w = W_q.reshape(NCH_OUT, -1).sum(axis=1).astype(np.int32)

mod_model = onnx.load(MODEL_PATH)
existing = {{o.name for o in mod_model.graph.output}}
for nm, tp in [
    (ADD_OUT, onnx.TensorProto.FLOAT),
    (IN_Q, onnx.TensorProto.UINT8),
    (IN_S, onnx.TensorProto.FLOAT),
    (IN_ZP, onnx.TensorProto.UINT8),
    (RES_Q, onnx.TensorProto.UINT8),
    (RES_S, onnx.TensorProto.FLOAT),
    (RES_ZP, onnx.TensorProto.UINT8),
]:
    if nm not in existing:
        mod_model.graph.output.append(onnx.helper.make_tensor_value_info(nm, tp, None))
tmp_path = os.path.join(HERE, "_model_with_intermediate.onnx")
onnx.save(mod_model, tmp_path)

so = ort.SessionOptions()
so.log_severity_level = 3
sess = ort.InferenceSession(tmp_path, sess_options=so, providers=["CPUExecutionProvider"])
ipt_name = sess.get_inputs()[0].name

def run_ort(img_f32):
    outs = sess.run([ADD_OUT, IN_Q, IN_S, IN_ZP, RES_Q, RES_S, RES_ZP], {{ipt_name: img_f32}})
    return outs[0], outs[1], float(outs[2]), int(outs[3]), outs[4], float(outs[5]), int(outs[6])

def make_input(kind, seed=0):
    img = np.zeros((1, 3, 640, 640), dtype=np.float32)
    rng = np.random.RandomState(seed)
    if kind == "rand":
        img[0] = rng.uniform(0.0, 1.0, size=(3, 640, 640)).astype(np.float32)
    elif kind == "half":
        img[0] = 0.5
    elif kind == "gradient":
        gx = np.tile(np.linspace(0.0, 1.0, 640, dtype=np.float32), (640, 1))
        img[0, 0] = gx; img[0, 1] = gx.T; img[0, 2] = 0.5 * (gx + gx.T)
    elif kind == "rand_low":
        img[0] = rng.uniform(0.15, 0.80, size=(3, 640, 640)).astype(np.float32)
    elif kind == "rand_high":
        img[0] = rng.uniform(0.5, 1.0, size=(3, 640, 640)).astype(np.float32)
    else:
        raise ValueError(kind)
    return img

def f32_to_fp16(x):
    return np.float16(x).view(np.uint16)

def hw_reference(qx_u8_full, s_a, zp_a, qr_u8_full, s_r, zp_r, S_OUT_SILU):
    s_acc = s_a * s_w
    pad_u8 = np.full((NCH_IN, PAD_H, PAD_W), zp_a, dtype=np.int32)
    r0, r1 = R - PAD, R - PAD + PAD_H
    c0, c1 = C - PAD, C - PAD + PAD_W
    h_full, w_full = qx_u8_full.shape[2], qx_u8_full.shape[3]
    sr0, sr1 = max(0, r0), min(h_full, r1)
    sc0, sc1 = max(0, c0), min(w_full, c1)
    pad_u8[:, sr0-r0:sr1-r0, sc0-c0:sc1-c0] = qx_u8_full[0, :, sr0:sr1, sc0:sc1].astype(np.int32)
    x_i8 = np.clip(pad_u8 - 128, -128, 127).astype(np.int8)
    bias_eff = bias + s_acc * (128 - zp_a) * sum_w

    # Residual ROI (no padding, K=1 effectively).
    r_u8 = qr_u8_full[0, :, R:R+OUT_H, C:C+OUT_W].astype(np.int32)
    r_i8 = np.clip(r_u8 - 128, -128, 127).astype(np.int8)

    silu_out = np.zeros((NCH_OUT, OUT_H, OUT_W), dtype=np.float32)
    pre_min, pre_max = 1e30, -1e30
    for co in range(NCH_OUT):
        for oy in range(OUT_H):
            for ox in range(OUT_W):
                iy = oy * STRIDE
                ix = ox * STRIDE
                window = x_i8[:, iy:iy+K, ix:ix+K].astype(np.int32)
                acc = int((W_q[co] * window).sum())
                pre = acc * s_acc + bias_eff[co]
                pre_min = min(pre_min, pre); pre_max = max(pre_max, pre)
                if HAS_SILU:
                    sig = 1.0 / (1.0 + np.exp(-pre))
                    silu_out[co, oy, ox] = pre * sig
                else:
                    # No SiLU: stream into add_rq is the post-requant value
                    # (i.e. the pre activation itself, but quantized to the
                    # S_OUT_PRE grid). We model that quantization below by
                    # passing through requant; here we keep float pre and
                    # let the i8 grid clamp on the silu_i8 line.
                    silu_out[co, oy, ox] = pre

    silu_i8 = np.clip(np.round(silu_out / S_OUT_SILU), -128, 127).astype(np.int8)
    r_f32 = (r_u8 - zp_r).astype(np.float32) * s_r
    add_f32 = silu_out + r_f32
    add_i8 = np.clip(np.round(add_f32 / S_OUT_SILU), -128, 127).astype(np.int8)
    add_dq = add_i8.astype(np.float32) * S_OUT_SILU
    return add_dq, silu_out, silu_i8, r_i8, bias_eff, x_i8, s_acc, (pre_min, pre_max)

samples = [
    ("rand0", "rand", 0), ("rand1", "rand", 1),
    ("half", "half", 0), ("gradient", "gradient", 0),
    ("rand_low", "rand_low", 2), ("rand_high", "rand_high", 3),
]

# Pass 1: pick scales empirically.
pre_lo, pre_hi, silu_hi, add_hi = 0.0, 0.0, 0.0, 0.0
cache = []
S_TMP = 4.0/127.0
for sname, kind, seed in samples:
    img = make_input(kind, seed)
    o = run_ort(img)
    ort_out, qx_u8, s_a, zp_a, qr_u8, s_r, zp_r = o
    if s_a == 0.0: s_a = 1.0
    if s_r == 0.0: s_r = 1.0
    _, silu, _, _, _, _, _, (pmin, pmax) = hw_reference(qx_u8, s_a, zp_a, qr_u8, s_r, zp_r, S_TMP)
    pre_lo = min(pre_lo, pmin); pre_hi = max(pre_hi, pmax)
    silu_hi = max(silu_hi, float(np.max(np.abs(silu))))
    roi = ort_out[0, :, R:R+OUT_H, C:C+OUT_W]
    add_hi = max(add_hi, float(np.max(np.abs(roi))))
    cache.append((sname, ort_out, qx_u8, s_a, zp_a, qr_u8, s_r, zp_r))

print(f"observed pre-SiLU=[{{pre_lo:.3f}},{{pre_hi:.3f}}] |silu|~{{silu_hi:.3f}} |add|~{{add_hi:.3f}}")

# Pick S_OUT_PRE/SILU from observed post-SiLU amplitude (HANDOFF footgun #5).
# Emit matching values into stim/s_out_params.sv so the RTL shim picks them up
# via parameter override.
def pick_s_out(amp):
    raw = max(amp, 1e-3) * 1.1 / 127.0
    return max(raw, 2.0 / 127.0)

pre_amp = max(abs(pre_lo), abs(pre_hi))
if HAS_SILU:
    S_OUT_PRE  = pick_s_out(pre_amp)
    # For residual layers, S_OUT_SILU also governs the residual+add output grid.
    # Use the wider of silu and add to be safe.
    S_OUT_SILU = pick_s_out(max(silu_hi, add_hi))
else:
    # No-SiLU: post-requant value flows straight into add_rq. The IP uses
    # FP16(S_OUT_SILU) as the scale_a for add_rq, so we MUST pin
    # S_OUT_SILU == S_OUT_PRE. Size to cover both the pre-activation and
    # the post-add amplitudes (whichever is wider).
    common = pick_s_out(max(pre_amp, add_hi))
    S_OUT_PRE  = common
    S_OUT_SILU = common
print(f"picked S_OUT_PRE={{S_OUT_PRE:.6f}} S_OUT_SILU={{S_OUT_SILU:.6f}} "
      f"({{'dynamic' if S_OUT_PRE < 8.0/127.0 else 'default 8.0/127'}})")

def _write_s_out_params_sv():
    txt = (
        "// Copyright (c) 2026 vibeyolo\\n"
        "// Auto-generated by extract.py (do not edit).\\n"
        "package s_out_params;\\n"
        f"  localparam real S_OUT_PRE_VAL  = {{S_OUT_PRE:.10e}};\\n"
        f"  localparam real S_OUT_SILU_VAL = {{S_OUT_SILU:.10e}};\\n"
        "endpackage\\n"
    )
    path = os.path.join(STIM, "s_out_params.sv")
    with open(path + ".tmp", "w") as f:
        f.write(txt)
    os.replace(path + ".tmp", path)
_write_s_out_params_sv()

N_LANE_FULL = K * K * NCH_IN
W_flat = np.zeros((NCH_OUT, N_LANE_FULL), dtype=np.int8)
for co in range(NCH_OUT):
    for kh in range(K):
        for kw in range(K):
            for kc in range(NCH_IN):
                W_flat[co, (kh*K+kw)*NCH_IN + kc] = W_q[co, kc, kh, kw]
with open(os.path.join(STIM, "weights.i8.hex"), "w") as f:
    for co in range(NCH_OUT):
        for lane in range(N_LANE_FULL):
            f.write(f"{{int(W_flat[co, lane]) & 0xFF:02x}}\\n")

manifest = {{
    "layer_idx": LAYER_IDX, "layer_name": LAYER_NAME,
    "out_h": OUT_H, "out_w": OUT_W, "nch_out": NCH_OUT, "nch_in": NCH_IN,
    "K": K, "stride": STRIDE, "pad": PAD, "n_lane_full": N_LANE_FULL,
    "roi_h": ROI_H, "roi_w": ROI_W, "pad_h": PAD_H, "pad_w": PAD_W,
    "R": R, "C": C, "s_w": s_w,
    "s_out_pre": S_OUT_PRE, "s_out_silu": S_OUT_SILU,
    "residual": True, "residual_src": RESID_SRC, "add_out": ADD_OUT,
    "samples": [],
}}

for sname, ort_out, qx_u8, s_a, zp_a, qr_u8, s_r, zp_r in cache:
    add_dq, silu_out, silu_i8, r_i8, bias_eff, x_i8, s_acc, _ = hw_reference(
        qx_u8, s_a, zp_a, qr_u8, s_r, zp_r, S_OUT_SILU)
    ort_roi = ort_out[0, :, R:R+OUT_H, C:C+OUT_W]

    scale_fp16 = np.array([f32_to_fp16(s_acc / S_OUT_PRE)] * NCH_OUT, dtype=np.uint16)
    bias_fp16  = np.array([f32_to_fp16(bias_eff[co]/S_OUT_PRE) for co in range(NCH_OUT)], dtype=np.uint16)

    # Residual side-band fold: r_scale=s_r, r_bias=(128-zp_r)*s_r/S_OUT_SILU
    r_scale_fp16 = np.array([f32_to_fp16(s_r)]*NCH_OUT, dtype=np.uint16)
    r_bias_const = (128 - zp_r) * s_r / S_OUT_SILU
    r_bias_fp16  = np.array([f32_to_fp16(r_bias_const)]*NCH_OUT, dtype=np.uint16)

    with open(os.path.join(STIM, f"{{sname}}.input_i8.hex"), "w") as f:
        for h in range(PAD_H):
            for w in range(PAD_W):
                for kc in range(NCH_IN):
                    f.write(f"{{int(x_i8[kc, h, w]) & 0xFF:02x}}\\n")
    with open(os.path.join(STIM, f"{{sname}}.residual_i8.hex"), "w") as f:
        for oy in range(OUT_H):
            for ox in range(OUT_W):
                for co in range(NCH_OUT):
                    f.write(f"{{int(r_i8[co, oy, ox]) & 0xFF:02x}}\\n")
    with open(os.path.join(STIM, f"{{sname}}.scale_fp16.hex"), "w") as f:
        for co in range(NCH_OUT):
            f.write(f"{{int(scale_fp16[co]):04x}}\\n")
    with open(os.path.join(STIM, f"{{sname}}.bias_fp16.hex"), "w") as f:
        for co in range(NCH_OUT):
            f.write(f"{{int(bias_fp16[co]):04x}}\\n")
    with open(os.path.join(STIM, f"{{sname}}.r_scale_fp16.hex"), "w") as f:
        for co in range(NCH_OUT):
            f.write(f"{{int(r_scale_fp16[co]):04x}}\\n")
    with open(os.path.join(STIM, f"{{sname}}.r_bias_fp16.hex"), "w") as f:
        for co in range(NCH_OUT):
            f.write(f"{{int(r_bias_fp16[co]):04x}}\\n")
    with open(os.path.join(STIM, f"{{sname}}.ref_ort.f32.hex"), "w") as f:
        for oy in range(OUT_H):
            for ox in range(OUT_W):
                for co in range(NCH_OUT):
                    bits = np.float32(ort_roi[co, oy, ox]).view(np.uint32)
                    f.write(f"{{int(bits):08x}}\\n")
    with open(os.path.join(STIM, f"{{sname}}.ref_hw.f32.hex"), "w") as f:
        for oy in range(OUT_H):
            for ox in range(OUT_W):
                for co in range(NCH_OUT):
                    bits = np.float32(add_dq[co, oy, ox]).view(np.uint32)
                    f.write(f"{{int(bits):08x}}\\n")

    diff = ort_roi - add_dq
    mxe = float(np.max(np.abs(diff))); mae = float(np.mean(np.abs(diff)))
    out_range = float(ort_roi.max() - ort_roi.min())
    cos = float(np.sum(ort_roi*add_dq)/(np.linalg.norm(ort_roi)*np.linalg.norm(add_dq)+1e-12))
    print(f"[{{sname}}] s_a={{s_a:.5f}} zp_a={{zp_a}} s_r={{s_r:.5f}} zp_r={{zp_r}} "
          f"ORT vs HW: max_abs={{mxe:.4f}} mae={{mae:.4f}} cos={{cos:.6f}}")
    manifest["samples"].append({{
        "name": sname, "s_a": s_a, "zp_a": zp_a,
        "s_r": s_r, "zp_r": zp_r, "s_acc": s_acc,
        "ort_vs_hw_max_abs": mxe, "ort_vs_hw_mae": mae,
        "ort_vs_hw_cos": cos, "out_range": out_range,
    }})

with open(os.path.join(STIM, "manifest.json"), "w") as f:
    json.dump(manifest, f, indent=2)
print(f"Wrote stim+ref to {{STIM}}")
'''
    (ldir / "extract.py").write_text(text)
    os.chmod(ldir / "extract.py", 0o755)


def emit_conv_resid_test(lay, idx, ldir, p_pix, p_cout, p_cin, features=None):
    """Residual conv test driver. Drives r_i side-band per-pixel."""
    assert lay["group"] == 1
    has_silu = True if (features is None) else bool(features["has_silu"])
    silu_lat_val = 1 if has_silu else 0
    roi_out_h = min(8, lay["H_out"])
    roi_out_w = min(8, lay["W_out"])
    roi_h = roi_out_h * lay["sh"]
    roi_w = roi_out_w * lay["sw"]
    pad_h = roi_h + 2 * lay["ph"]
    pad_w = roi_w + 2 * lay["pw"]
    n_cout_tile = math.ceil(lay["cout"] / p_cout)
    n_cin_tile = math.ceil(lay["cin"] / p_cin)
    n_lane_tile = lay["kh"] * lay["kw"] * p_cin
    n_lane_full = lay["kh"] * lay["kw"] * lay["cin"]
    dot_lat = 1 + math.ceil(math.log2(max(2, n_lane_tile)))

    text = f'''// Copyright (c) 2026 vibeyolo
// SPDX-License-Identifier: Apache-2.0
//
// Generated residual conv+silu+add DV (group=1).

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "Vlayer_{idx}_tb.h"
#include "sim_ctrl.h"

using DUT = Vlayer_{idx}_tb;

static constexpr int NCH_IN = {lay['cin']};
static constexpr int NCH_OUT = {lay['cout']};
static constexpr int K = {lay['kh']};
static constexpr int STRIDE = {lay['sh']};
static constexpr int PAD = {lay['ph']};
static constexpr int P_PIX = {p_pix};
static constexpr int P_COUT = {p_cout};
static constexpr int P_CIN = {p_cin};
static constexpr int N_COUT_TILE = {n_cout_tile};
static constexpr int N_CIN_TILE = {n_cin_tile};
static constexpr int N_LANE_TILE = {n_lane_tile};
static constexpr int N_LANE_FULL = {n_lane_full};
static constexpr int ROI_H = {roi_h};
static constexpr int ROI_W = {roi_w};
static constexpr int PAD_H = {pad_h};
static constexpr int PAD_W = {pad_w};
static constexpr int OUT_H = {roi_out_h};
static constexpr int OUT_W = {roi_out_w};
static constexpr int DOT_LAT = {dot_lat};
static constexpr int ACC_LAT = 1;
static constexpr int RQ_IN_LAT = 1;
static constexpr int REQUANT_LAT = 7;
static constexpr int SILU_LAT = {silu_lat_val};
static constexpr int ADDRQ_LAT = 12;
static constexpr int TOTAL_LAT = DOT_LAT + ACC_LAT + RQ_IN_LAT + REQUANT_LAT + SILU_LAT + ADDRQ_LAT;

static std::string stim_dir() {{ return std::string("../stim"); }}

static std::vector<uint32_t> load_hex(const std::string& path) {{
    std::ifstream f(path);
    if (!f) {{ fprintf(stderr, "could not open %s\\n", path.c_str()); std::exit(2); }}
    std::vector<uint32_t> v; std::string line;
    while (std::getline(f, line)) {{
        if (!line.empty()) v.push_back((uint32_t)std::stoul(line, nullptr, 16));
    }}
    return v;
}}
static float u32_to_f32(uint32_t bits) {{ float f; std::memcpy(&f, &bits, 4); return f; }}

static double load_manifest_number(const std::string& key) {{
    std::ifstream f(stim_dir() + "/manifest.json");
    if (!f) {{ fprintf(stderr, "could not open manifest.json\\n"); std::exit(2); }}
    std::string text((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    std::string needle = "\\"" + key + "\\"";
    size_t p = text.find(needle);
    if (p == std::string::npos) std::exit(2);
    p = text.find(':', p);
    if (p == std::string::npos) std::exit(2);
    char* end = nullptr;
    return std::strtod(text.c_str() + p + 1, &end);
}}

struct Sample {{
    std::string name;
    std::vector<int8_t> input_i8, residual_i8;
    std::vector<uint16_t> scale_fp16, bias_fp16, r_scale_fp16, r_bias_fp16;
    std::vector<float> ref_ort, ref_hw;
}};

static Sample load_sample(const std::string& name) {{
    Sample s; s.name = name;
    auto inp = load_hex(stim_dir() + "/" + name + ".input_i8.hex");
    s.input_i8.resize(inp.size());
    for (size_t i = 0; i < inp.size(); i++) s.input_i8[i] = (int8_t)(uint8_t)(inp[i] & 0xFF);
    auto res = load_hex(stim_dir() + "/" + name + ".residual_i8.hex");
    s.residual_i8.resize(res.size());
    for (size_t i = 0; i < res.size(); i++) s.residual_i8[i] = (int8_t)(uint8_t)(res[i] & 0xFF);
    for (auto v : load_hex(stim_dir() + "/" + name + ".scale_fp16.hex")) s.scale_fp16.push_back((uint16_t)v);
    for (auto v : load_hex(stim_dir() + "/" + name + ".bias_fp16.hex")) s.bias_fp16.push_back((uint16_t)v);
    for (auto v : load_hex(stim_dir() + "/" + name + ".r_scale_fp16.hex")) s.r_scale_fp16.push_back((uint16_t)v);
    for (auto v : load_hex(stim_dir() + "/" + name + ".r_bias_fp16.hex")) s.r_bias_fp16.push_back((uint16_t)v);
    for (auto v : load_hex(stim_dir() + "/" + name + ".ref_ort.f32.hex")) s.ref_ort.push_back(u32_to_f32(v));
    for (auto v : load_hex(stim_dir() + "/" + name + ".ref_hw.f32.hex")) s.ref_hw.push_back(u32_to_f32(v));
    return s;
}}

template <typename T> static void pack_bytes(T& dst, const int8_t* src, int n) {{
    auto* p = reinterpret_cast<uint32_t*>(&dst);
    int nw = (n + 3) / 4;
    for (int i = 0; i < nw; i++) p[i] = 0;
    for (int i = 0; i < n; i++) p[i / 4] |= ((uint32_t)(uint8_t)src[i] << ((i % 4) * 8));
}}
template <typename T> static void pack_u16(T& dst, const uint16_t* src, int n) {{
    auto* p = reinterpret_cast<uint32_t*>(&dst);
    int nw = (n + 1) / 2;
    for (int i = 0; i < nw; i++) p[i] = 0;
    for (int i = 0; i < n; i++) p[i / 2] |= ((uint32_t)src[i] << ((i % 2) * 16));
}}
template <typename T> static void unpack_bytes(const T& src, int8_t* dst, int n) {{
    auto* p = reinterpret_cast<const uint32_t*>(&src);
    for (int i = 0; i < n; i++) dst[i] = (int8_t)((p[i / 4] >> ((i % 4) * 8)) & 0xFFu);
}}

static void build_window_tile(const Sample& s, int pix_base, int cit,
                              int8_t out[P_PIX * N_LANE_TILE]) {{
    int kc_base = cit * P_CIN;
    for (int pp = 0; pp < P_PIX; pp++) {{
        int pix = pix_base + pp;
        int oy = pix / OUT_W, ox = pix % OUT_W;
        for (int kh = 0; kh < K; kh++) {{
            for (int kw = 0; kw < K; kw++) {{
                int h_idx = oy * STRIDE + kh, w_idx = ox * STRIDE + kw;
                for (int kc_local = 0; kc_local < P_CIN; kc_local++) {{
                    int kc = kc_base + kc_local;
                    int tile_lane = (kh * K + kw) * P_CIN + kc_local;
                    int off = pp * N_LANE_TILE + tile_lane;
                    out[off] = 0;
                    if (pix < OUT_H * OUT_W && kc < NCH_IN) {{
                        int in_off = (h_idx * PAD_W + w_idx) * NCH_IN + kc;
                        out[off] = s.input_i8[in_off];
                    }}
                }}
            }}
        }}
    }}
}}

static void build_weight_tile(const std::vector<int8_t>& w_full, int ct, int cit,
                              int8_t out[P_COUT * N_LANE_TILE]) {{
    int kc_base = cit * P_CIN;
    int co_base = ct * P_COUT;
    for (int co_local = 0; co_local < P_COUT; co_local++) {{
        int co_global = co_base + co_local;
        for (int kh = 0; kh < K; kh++) {{
            for (int kw = 0; kw < K; kw++) {{
                for (int kc_local = 0; kc_local < P_CIN; kc_local++) {{
                    int kc = kc_base + kc_local;
                    int full_lane = (kh * K + kw) * NCH_IN + kc;
                    int tile_lane = (kh * K + kw) * P_CIN + kc_local;
                    int off = co_local * N_LANE_TILE + tile_lane;
                    out[off] = 0;
                    if (co_global < NCH_OUT && kc < NCH_IN)
                        out[off] = w_full[co_global * N_LANE_FULL + full_lane];
                }}
            }}
        }}
    }}
}}

struct Stats {{ double max_abs=0, mae=0, cos=0, out_range=0; int n=0, worst_idx=-1;
                float worst_dut=0, worst_ref=0; }};
static Stats compute_stats(const std::vector<float>& dut, const std::vector<float>& ref) {{
    Stats st; st.n = (int)dut.size();
    double sum_abs=0, dot=0, na=0, nb=0; float mn=1e30f, mx=-1e30f;
    for (int i = 0; i < st.n; i++) {{
        double e = std::abs((double)dut[i] - (double)ref[i]);
        sum_abs += e;
        if (e > st.max_abs) {{ st.max_abs=e; st.worst_idx=i; st.worst_dut=dut[i]; st.worst_ref=ref[i]; }}
        dot += (double)dut[i]*ref[i]; na += (double)dut[i]*dut[i]; nb += (double)ref[i]*ref[i];
        if (ref[i] < mn) mn = ref[i]; if (ref[i] > mx) mx = ref[i];
    }}
    st.mae = sum_abs / std::max(1, st.n);
    st.cos = dot / (std::sqrt(na)*std::sqrt(nb) + 1e-30);
    st.out_range = mx - mn;
    return st;
}}

int main(int argc, char** argv) {{
    SimCtrl<DUT> sim(argc, argv);
    sim.max_time = 80000000;

    auto w_raw = load_hex(stim_dir() + "/weights.i8.hex");
    std::vector<int8_t> w_full(NCH_OUT * N_LANE_FULL);
    for (int i = 0; i < NCH_OUT * N_LANE_FULL; i++) w_full[i] = (int8_t)(uint8_t)(w_raw[i] & 0xFF);

    sim.dut->valid_i = 0; sim.dut->first_cin_i = 0; sim.dut->last_cin_i = 0;
    sim.dut->cout_tile_idx_i = 0;
    std::vector<int8_t> zero_x(P_PIX * N_LANE_TILE, 0), zero_w(P_COUT * N_LANE_TILE, 0);
    std::vector<uint16_t> zero_q(P_COUT, 0);
    std::vector<int8_t> zero_r(P_COUT, 0);
    pack_bytes(sim.dut->x_flat_i, zero_x.data(), P_PIX * N_LANE_TILE);
    pack_bytes(sim.dut->w_flat_i, zero_w.data(), P_COUT * N_LANE_TILE);
    pack_u16(sim.dut->scale_flat_i, zero_q.data(), P_COUT);
    pack_u16(sim.dut->bias_flat_i, zero_q.data(), P_COUT);
    pack_bytes(sim.dut->r_flat_i, zero_r.data(), P_PIX * P_COUT);
    pack_u16(sim.dut->r_scale_flat_i, zero_q.data(), P_COUT);
    pack_u16(sim.dut->r_bias_flat_i, zero_q.data(), P_COUT);
    sim.reset();

    const std::vector<std::string> sample_names = {{"rand0", "rand1", "half", "gradient", "rand_low", "rand_high"}};
    int n_pass = 0;
    double agg_cos = 0.0, agg_mae = 0.0, agg_max = 0.0;

    for (const auto& sname : sample_names) {{
        Sample s = load_sample(sname);
        std::vector<int8_t> y_i8(OUT_H * OUT_W * NCH_OUT, 0);
        const int N_OUT = OUT_H * OUT_W;
        const int N_PIX_GROUP = (N_OUT + P_PIX - 1) / P_PIX;
        const int BEATS_PER_GROUP = N_COUT_TILE * N_CIN_TILE;
        const int TOTAL_BEATS = N_PIX_GROUP * BEATS_PER_GROUP;
        struct Commit {{ int pix_base; int ct; }};
        std::vector<Commit> inflight;
        inflight.reserve(N_PIX_GROUP * N_COUT_TILE + TOTAL_LAT);
        int produced = 0;

        for (int step = 0; step < TOTAL_BEATS + TOTAL_LAT + 16; step++) {{
            if (step < TOTAL_BEATS) {{
                int pg = step / BEATS_PER_GROUP;
                int rem = step % BEATS_PER_GROUP;
                int ct = rem / N_CIN_TILE;
                int cit = rem % N_CIN_TILE;
                int pix_base = pg * P_PIX;

                int8_t win[P_PIX * N_LANE_TILE];
                build_window_tile(s, pix_base, cit, win);
                pack_bytes(sim.dut->x_flat_i, win, P_PIX * N_LANE_TILE);

                int8_t wtile[P_COUT * N_LANE_TILE];
                build_weight_tile(w_full, ct, cit, wtile);
                pack_bytes(sim.dut->w_flat_i, wtile, P_COUT * N_LANE_TILE);

                uint16_t sbuf[P_COUT], bbuf[P_COUT];
                int co_base = ct * P_COUT;
                for (int co = 0; co < P_COUT; co++) {{
                    sbuf[co] = (co_base + co < NCH_OUT) ? s.scale_fp16[co_base + co] : 0;
                    bbuf[co] = (co_base + co < NCH_OUT) ? s.bias_fp16[co_base + co] : 0;
                }}
                pack_u16(sim.dut->scale_flat_i, sbuf, P_COUT);
                pack_u16(sim.dut->bias_flat_i, bbuf, P_COUT);

                // Residual side-band: per-(pix,cout-tile), driver holds per cin-sweep.
                int8_t r_row[P_PIX * P_COUT];
                uint16_t rs_row[P_COUT], rb_row[P_COUT];
                for (int pp = 0; pp < P_PIX; pp++) {{
                    int pix = pix_base + pp;
                    for (int co = 0; co < P_COUT; co++) {{
                        int co_global = co_base + co;
                        r_row[pp * P_COUT + co] = 0;
                        if (pix < N_OUT && co_global < NCH_OUT) {{
                            r_row[pp * P_COUT + co] =
                                s.residual_i8[pix * NCH_OUT + co_global];
                        }}
                    }}
                }}
                for (int co = 0; co < P_COUT; co++) {{
                    int co_global = co_base + co;
                    rs_row[co] = (co_global < NCH_OUT) ? s.r_scale_fp16[co_global] : 0;
                    rb_row[co] = (co_global < NCH_OUT) ? s.r_bias_fp16[co_global] : 0;
                }}
                pack_bytes(sim.dut->r_flat_i, r_row, P_PIX * P_COUT);
                pack_u16(sim.dut->r_scale_flat_i, rs_row, P_COUT);
                pack_u16(sim.dut->r_bias_flat_i, rb_row, P_COUT);

                sim.dut->valid_i = 1;
                sim.dut->first_cin_i = (cit == 0);
                sim.dut->last_cin_i = (cit == N_CIN_TILE - 1);
                sim.dut->cout_tile_idx_i = (uint8_t)ct;
                if (cit == N_CIN_TILE - 1) inflight.push_back({{pix_base, ct}});
            }} else {{
                sim.dut->valid_i = 0; sim.dut->first_cin_i = 0; sim.dut->last_cin_i = 0;
            }}
            sim.tick();
            if (sim.dut->valid_o && produced < (int)inflight.size()) {{
                int pix_base = inflight[produced].pix_base;
                int ct = inflight[produced].ct;
                int got_ct = sim.dut->cout_tile_idx_o;
                if (got_ct != ct) fprintf(stderr, "cout_tile_idx mismatch at produced=%d: expected %d got %d\\n", produced, ct, got_ct);
                int8_t row[P_PIX * P_COUT];
                unpack_bytes(sim.dut->y_flat_o, row, P_PIX * P_COUT);
                int co_base = ct * P_COUT;
                for (int pp = 0; pp < P_PIX; pp++) {{
                    int pix = pix_base + pp;
                    if (pix >= N_OUT) continue;
                    for (int co = 0; co < P_COUT; co++) {{
                        int co_global = co_base + co;
                        if (co_global < NCH_OUT)
                            y_i8[pix * NCH_OUT + co_global] = row[pp * P_COUT + co];
                    }}
                }}
                produced++;
            }}
        }}

        double s_out_silu = load_manifest_number("s_out_silu");
        std::vector<float> dut_f(N_OUT * NCH_OUT);
        for (int i = 0; i < N_OUT * NCH_OUT; i++) dut_f[i] = (float)((double)y_i8[i] * s_out_silu);

        Stats st_ort = compute_stats(dut_f, s.ref_ort);
        Stats st_hw = compute_stats(dut_f, s.ref_hw);
        double pass_mae_thresh = 0.08 * std::max(1e-6, st_ort.out_range);
        bool pass = (st_ort.cos > 0.997) && (st_ort.mae < pass_mae_thresh);

        printf("\\n--- sample %s ---\\n", sname.c_str());
        printf("  DUT vs ORT   : max_abs=%.4f mae=%.4f cos=%.6f out_range=%.3f thresh=%.4f\\n",
               st_ort.max_abs, st_ort.mae, st_ort.cos, st_ort.out_range, pass_mae_thresh);
        printf("  DUT vs HW-ref: max_abs=%.4f mae=%.4f cos=%.6f\\n", st_hw.max_abs, st_hw.mae, st_hw.cos);
        printf("  => %s\\n", pass ? "PASS" : "FAIL");
        sim.check(pass, std::string("sample ") + sname + " pass");
        if (pass) n_pass++;
        agg_cos += st_ort.cos; agg_mae += st_ort.mae; agg_max = std::max(agg_max, st_ort.max_abs);
    }}

    printf("\\n========================================\\n");
    printf("Aggregate: %d/%zu samples passed\\n", n_pass, sample_names.size());
    printf("  avg cos = %.6f  avg mae = %.6f  worst max = %.6f\\n",
           agg_cos/sample_names.size(), agg_mae/sample_names.size(), agg_max);
    printf("Layer {idx} RESIDUAL P_PIX=%d P_COUT=%d P_CIN=%d\\n", P_PIX, P_COUT, P_CIN);
    printf("========================================\\n");
    return sim.finish();
}}
'''
    (ldir / "dv" / f"layer_{idx}_test.cc").write_text(text)


# ====================================================================
# No-SiLU tail variant: detect-head final convs.
# ====================================================================
def emit_conv_nosilu_extract(lay, idx, ldir, features):
    """Extract for SILU=0 layers (conv+bias only, output = post-bias fp32)."""
    is_depthwise = lay["group"] != 1
    if is_depthwise:
        assert lay["group"] == lay["cin"] == lay["cout"], \
            "only pure depthwise (group==cin==cout) supported"
    assert not features["has_silu"]
    assert not features["has_residual"]
    roi_out_h = min(8, lay["H_out"])
    roi_out_w = min(8, lay["W_out"])
    roi_h = roi_out_h * lay["sh"]
    roi_w = roi_out_w * lay["sw"]
    pad_h = roi_h + 2 * lay["ph"]
    pad_w = roi_w + 2 * lay["pw"]
    r = min(max(2 * lay["sh"], lay["ph"]), max(0, lay["H_in"] - roi_h - lay["ph"]))
    c = min(max(2 * lay["sw"], lay["pw"]), max(0, lay["W_in"] - roi_w - lay["pw"]))
    r -= r % lay["sh"]
    c -= c % lay["sw"]

    text = f'''#!/usr/bin/env python3
# Generated by tools/layergen/layergen.py — conv (no SiLU tail).
import json, os
import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper

HERE = os.path.dirname(os.path.abspath(__file__))
STIM = os.path.join(HERE, "stim"); os.makedirs(STIM, exist_ok=True)
MODEL_PATH = os.environ.get("MODEL_PATH", "/home/marty/Documents/aif/vibeyolo/integ/yolo26n/model_int8.onnx")

LAYER_IDX = {idx}
LAYER_NAME = "{lay['name']}"
NCH_IN = {lay['cin']}
NCH_OUT = {lay['cout']}
K = {lay['kh']}
STRIDE = {lay['sh']}
PAD = {lay['ph']}
GROUP = {lay['group']}
IS_DEPTHWISE = {1 if is_depthwise else 0}
ROI_H, ROI_W = {roi_h}, {roi_w}
OUT_H, OUT_W = {roi_out_h}, {roi_out_w}
PAD_H, PAD_W = {pad_h}, {pad_w}
R, C = {r}, {c}

model = onnx.load(MODEL_PATH)
init_by_name = {{i.name: numpy_helper.to_array(i) for i in model.graph.initializer}}
conv = next(n for n in model.graph.node if n.op_type == "ConvInteger" and n.name == LAYER_NAME)
W_q_raw = init_by_name[conv.input[1]].astype(np.int32)
s_w = float(init_by_name[conv.input[1].replace("_quantized", "_scale")])
zp_w = int(init_by_name[conv.input[3]]); assert zp_w == 0
if IS_DEPTHWISE:
    assert W_q_raw.shape == (NCH_OUT, 1, K, K), f"got {{W_q_raw.shape}}"
    W_q = np.zeros((NCH_OUT, NCH_IN, K, K), dtype=np.int32)
    for co in range(NCH_OUT):
        W_q[co, co] = W_q_raw[co, 0]
else:
    assert W_q_raw.shape == (NCH_OUT, NCH_IN, K, K)
    W_q = W_q_raw

conv_pos = list(model.graph.node).index(conv)
bias = None
target_out = None
# Target is the post-bias add output (no SiLU).
for n in list(model.graph.node)[conv_pos + 1:conv_pos + 12]:
    if bias is None and n.op_type == "Reshape" and n.input and n.input[0] in init_by_name:
        cand = init_by_name[n.input[0]].astype(np.float32)
        if cand.shape == (NCH_OUT,):
            bias = cand
    if n.op_type == "Add" and n.name.endswith("_bias_add"):
        target_out = n.output[0]
        break
assert bias is not None and target_out is not None

IN_Q = conv.input[0]; IN_ZP = conv.input[2]; IN_S = IN_Q.replace("_quantized", "_scale")
sum_w = W_q.reshape(NCH_OUT, -1).sum(axis=1).astype(np.int32)

mod_model = onnx.load(MODEL_PATH)
existing = {{o.name for o in mod_model.graph.output}}
for nm, tp in [(target_out, onnx.TensorProto.FLOAT),
               (IN_Q, onnx.TensorProto.UINT8),
               (IN_S, onnx.TensorProto.FLOAT),
               (IN_ZP, onnx.TensorProto.UINT8)]:
    if nm not in existing:
        mod_model.graph.output.append(onnx.helper.make_tensor_value_info(nm, tp, None))
tmp_path = os.path.join(HERE, "_model_with_intermediate.onnx"); onnx.save(mod_model, tmp_path)

so = ort.SessionOptions(); so.log_severity_level = 3
sess = ort.InferenceSession(tmp_path, sess_options=so, providers=["CPUExecutionProvider"])
ipt_name = sess.get_inputs()[0].name

def run_ort(img):
    o = sess.run([target_out, IN_Q, IN_S, IN_ZP], {{ipt_name: img}})
    return o[0], o[1], float(o[2]), int(o[3])

def make_input(kind, seed=0):
    img = np.zeros((1,3,640,640), dtype=np.float32)
    rng = np.random.RandomState(seed)
    if kind == "rand":      img[0] = rng.uniform(0,1,(3,640,640)).astype(np.float32)
    elif kind == "half":    img[0] = 0.5
    elif kind == "gradient":
        gx = np.tile(np.linspace(0,1,640,dtype=np.float32),(640,1))
        img[0,0]=gx; img[0,1]=gx.T; img[0,2]=0.5*(gx+gx.T)
    elif kind == "rand_low": img[0] = rng.uniform(0.15,0.80,(3,640,640)).astype(np.float32)
    elif kind == "rand_high":img[0] = rng.uniform(0.5,1.0,(3,640,640)).astype(np.float32)
    else: raise ValueError(kind)
    return img

def f32_to_fp16(x): return np.float16(x).view(np.uint16)

def hw_reference(qx_u8_full, s_a, zp_a):
    s_acc = s_a * s_w
    pad_u8 = np.full((NCH_IN, PAD_H, PAD_W), zp_a, dtype=np.int32)
    r0, r1 = R - PAD, R - PAD + PAD_H; c0, c1 = C - PAD, C - PAD + PAD_W
    h_full, w_full = qx_u8_full.shape[2], qx_u8_full.shape[3]
    sr0, sr1 = max(0,r0), min(h_full,r1); sc0, sc1 = max(0,c0), min(w_full,c1)
    pad_u8[:, sr0-r0:sr1-r0, sc0-c0:sc1-c0] = qx_u8_full[0,:,sr0:sr1,sc0:sc1].astype(np.int32)
    x_i8 = np.clip(pad_u8 - 128, -128, 127).astype(np.int8)
    bias_eff = bias + s_acc * (128 - zp_a) * sum_w
    out = np.zeros((NCH_OUT, OUT_H, OUT_W), dtype=np.float32)
    pre_min, pre_max = 1e30, -1e30
    for co in range(NCH_OUT):
        for oy in range(OUT_H):
            for ox in range(OUT_W):
                iy = oy * STRIDE; ix = ox * STRIDE
                window = x_i8[:, iy:iy+K, ix:ix+K].astype(np.int32)
                acc = int((W_q[co]*window).sum())
                pre = acc * s_acc + bias_eff[co]
                pre_min = min(pre_min, pre); pre_max = max(pre_max, pre)
                out[co, oy, ox] = pre
    return out, bias_eff, x_i8, s_acc, (pre_min, pre_max)

N_LANE_FULL = K * K * NCH_IN
W_flat = np.zeros((NCH_OUT, N_LANE_FULL), dtype=np.int8)
for co in range(NCH_OUT):
    for kh in range(K):
        for kw in range(K):
            for kc in range(NCH_IN):
                W_flat[co, (kh*K+kw)*NCH_IN+kc] = W_q[co, kc, kh, kw]
with open(os.path.join(STIM, "weights.i8.hex"), "w") as f:
    for co in range(NCH_OUT):
        for lane in range(N_LANE_FULL):
            f.write(f"{{int(W_flat[co,lane]) & 0xFF:02x}}\\n")

samples = [
    ("rand0","rand",0),("rand1","rand",1),
    ("half","half",0),("gradient","gradient",0),
    ("rand_low","rand_low",2),("rand_high","rand_high",3),
]

cache = []
pre_lo, pre_hi = 0.0, 0.0
for sname, kind, seed in samples:
    img = make_input(kind, seed)
    o = run_ort(img)
    ort_out, qx_u8, s_a, zp_a = o
    if s_a == 0.0: s_a = 1.0
    out, _, _, _, (pmin, pmax) = hw_reference(qx_u8, s_a, zp_a)
    pre_lo = min(pre_lo, pmin); pre_hi = max(pre_hi, pmax)
    cache.append((sname, ort_out, qx_u8, s_a, zp_a))

def pick(amp):
    raw = round(amp * 1.2 * 1000.0) / 1000.0
    return max(raw / 127.0, 2.0 / 127.0)
# For SILU=0, S_OUT_PRE doubles as the requant output scale; S_OUT_SILU is unused
# but the IP still routes the i8 stream through that scale grid.
S_OUT_PRE  = pick(max(abs(pre_lo), abs(pre_hi)))
S_OUT_SILU = S_OUT_PRE
print(f"observed range=[{{pre_lo:.3f}},{{pre_hi:.3f}}] picked S_OUT_PRE={{S_OUT_PRE:.6f}}")

# Emit SV package so the RTL shim/TB pick up the same scale via parameter override.
def _write_s_out_params_sv():
    txt = (
        "// Copyright (c) 2026 vibeyolo\\n"
        "// Auto-generated by extract.py (do not edit).\\n"
        "package s_out_params;\\n"
        f"  localparam real S_OUT_PRE_VAL  = {{S_OUT_PRE:.10e}};\\n"
        f"  localparam real S_OUT_SILU_VAL = {{S_OUT_SILU:.10e}};\\n"
        "endpackage\\n"
    )
    path = os.path.join(STIM, "s_out_params.sv")
    with open(path + ".tmp", "w") as f:
        f.write(txt)
    os.replace(path + ".tmp", path)
_write_s_out_params_sv()

manifest = {{
    "layer_idx": LAYER_IDX, "layer_name": LAYER_NAME,
    "out_h": OUT_H, "out_w": OUT_W, "nch_out": NCH_OUT, "nch_in": NCH_IN,
    "K": K, "stride": STRIDE, "pad": PAD, "n_lane_full": N_LANE_FULL,
    "roi_h": ROI_H, "roi_w": ROI_W, "pad_h": PAD_H, "pad_w": PAD_W,
    "R": R, "C": C, "s_w": s_w,
    "s_out_pre": S_OUT_PRE, "s_out_silu": S_OUT_SILU,
    "no_silu": True, "samples": [],
}}

for sname, ort_out, qx_u8, s_a, zp_a in cache:
    hw_out, bias_eff, x_i8, s_acc, _ = hw_reference(qx_u8, s_a, zp_a)
    ort_roi = ort_out[0, :, R//STRIDE:R//STRIDE+OUT_H, C//STRIDE:C//STRIDE+OUT_W]
    scale_fp16 = np.array([f32_to_fp16(s_acc/S_OUT_PRE)]*NCH_OUT, dtype=np.uint16)
    bias_fp16  = np.array([f32_to_fp16(bias_eff[co]/S_OUT_PRE) for co in range(NCH_OUT)], dtype=np.uint16)

    with open(os.path.join(STIM, f"{{sname}}.input_i8.hex"), "w") as f:
        for h in range(PAD_H):
            for w in range(PAD_W):
                for kc in range(NCH_IN):
                    f.write(f"{{int(x_i8[kc,h,w]) & 0xFF:02x}}\\n")
    with open(os.path.join(STIM, f"{{sname}}.scale_fp16.hex"), "w") as f:
        for co in range(NCH_OUT): f.write(f"{{int(scale_fp16[co]):04x}}\\n")
    with open(os.path.join(STIM, f"{{sname}}.bias_fp16.hex"), "w") as f:
        for co in range(NCH_OUT): f.write(f"{{int(bias_fp16[co]):04x}}\\n")
    with open(os.path.join(STIM, f"{{sname}}.ref_ort.f32.hex"), "w") as f:
        for oy in range(OUT_H):
            for ox in range(OUT_W):
                for co in range(NCH_OUT):
                    bits = np.float32(ort_roi[co,oy,ox]).view(np.uint32)
                    f.write(f"{{int(bits):08x}}\\n")
    with open(os.path.join(STIM, f"{{sname}}.ref_hw.f32.hex"), "w") as f:
        for oy in range(OUT_H):
            for ox in range(OUT_W):
                for co in range(NCH_OUT):
                    bits = np.float32(hw_out[co,oy,ox]).view(np.uint32)
                    f.write(f"{{int(bits):08x}}\\n")

    diff = ort_roi - hw_out
    mxe = float(np.max(np.abs(diff))); mae = float(np.mean(np.abs(diff)))
    out_range = float(ort_roi.max()-ort_roi.min())
    cos = float(np.sum(ort_roi*hw_out)/(np.linalg.norm(ort_roi)*np.linalg.norm(hw_out)+1e-12))
    print(f"[{{sname}}] s_a={{s_a:.5f}} zp_a={{zp_a}} ORT vs HW: max_abs={{mxe:.4f}} mae={{mae:.4f}} cos={{cos:.6f}}")
    manifest["samples"].append({{"name":sname,"s_a":s_a,"zp_a":zp_a,"s_acc":s_acc,
        "ort_vs_hw_max_abs":mxe,"ort_vs_hw_mae":mae,"ort_vs_hw_cos":cos,"out_range":out_range}})

with open(os.path.join(STIM, "manifest.json"), "w") as f:
    json.dump(manifest, f, indent=2)
print(f"Wrote stim+ref to {{STIM}}")
'''
    (ldir / "extract.py").write_text(text)
    os.chmod(ldir / "extract.py", 0o755)


def emit_conv_nosilu_test(lay, idx, ldir, p_pix, p_cout, p_cin):
    """Same as conv_silu_test but TOTAL_LAT excludes SILU_LAT."""
    if lay["group"] != 1:
        assert lay["group"] == lay["cin"] == lay["cout"]
    # Re-use silu test body but with adjusted TOTAL_LAT.
    roi_out_h = min(8, lay["H_out"]); roi_out_w = min(8, lay["W_out"])
    roi_h = roi_out_h * lay["sh"]; roi_w = roi_out_w * lay["sw"]
    pad_h = roi_h + 2 * lay["ph"]; pad_w = roi_w + 2 * lay["pw"]
    n_cout_tile = math.ceil(lay["cout"] / p_cout)
    n_cin_tile = math.ceil(lay["cin"] / p_cin)
    n_lane_tile = lay["kh"] * lay["kw"] * p_cin
    n_lane_full = lay["kh"] * lay["kw"] * lay["cin"]
    dot_lat = 1 + math.ceil(math.log2(max(2, n_lane_tile)))

    text = f'''// Generated by tools/layergen/layergen.py — conv DV (no SiLU tail).
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>
#include "Vlayer_{idx}_tb.h"
#include "sim_ctrl.h"
using DUT = Vlayer_{idx}_tb;

static constexpr int NCH_IN = {lay['cin']};
static constexpr int NCH_OUT = {lay['cout']};
static constexpr int K = {lay['kh']};
static constexpr int STRIDE = {lay['sh']};
static constexpr int PAD = {lay['ph']};
static constexpr int P_PIX = {p_pix};
static constexpr int P_COUT = {p_cout};
static constexpr int P_CIN = {p_cin};
static constexpr int N_COUT_TILE = {n_cout_tile};
static constexpr int N_CIN_TILE = {n_cin_tile};
static constexpr int N_LANE_TILE = {n_lane_tile};
static constexpr int N_LANE_FULL = {n_lane_full};
static constexpr int ROI_H = {roi_h};
static constexpr int ROI_W = {roi_w};
static constexpr int PAD_H = {pad_h};
static constexpr int PAD_W = {pad_w};
static constexpr int OUT_H = {roi_out_h};
static constexpr int OUT_W = {roi_out_w};
static constexpr int DOT_LAT = {dot_lat};
static constexpr int TOTAL_LAT = DOT_LAT + 1 + 1 + 7;  // no SILU/ADDRQ

static std::string stim_dir() {{ return std::string("../stim"); }}
static std::vector<uint32_t> load_hex(const std::string& p) {{
    std::ifstream f(p); std::vector<uint32_t> v; std::string s;
    if (!f) {{ fprintf(stderr,"open fail %s\\n", p.c_str()); std::exit(2); }}
    while (std::getline(f,s)) if (!s.empty()) v.push_back((uint32_t)std::stoul(s,nullptr,16));
    return v;
}}
static float u32_to_f32(uint32_t b) {{ float f; std::memcpy(&f,&b,4); return f; }}
static double load_manifest_number(const std::string& key) {{
    std::ifstream f(stim_dir()+"/manifest.json");
    std::string t((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    std::string nd = "\\"" + key + "\\"";
    size_t p = t.find(nd); if (p==std::string::npos) std::exit(2);
    p = t.find(':',p); char* e=nullptr;
    return std::strtod(t.c_str()+p+1, &e);
}}

struct Sample {{ std::string name; std::vector<int8_t> input_i8;
                std::vector<uint16_t> scale_fp16, bias_fp16;
                std::vector<float> ref_ort, ref_hw; }};
static Sample load_sample(const std::string& n) {{
    Sample s; s.name=n;
    auto v = load_hex(stim_dir()+"/"+n+".input_i8.hex");
    s.input_i8.resize(v.size());
    for (size_t i=0;i<v.size();i++) s.input_i8[i] = (int8_t)(uint8_t)(v[i]&0xFF);
    for (auto x : load_hex(stim_dir()+"/"+n+".scale_fp16.hex")) s.scale_fp16.push_back((uint16_t)x);
    for (auto x : load_hex(stim_dir()+"/"+n+".bias_fp16.hex"))  s.bias_fp16.push_back((uint16_t)x);
    for (auto x : load_hex(stim_dir()+"/"+n+".ref_ort.f32.hex")) s.ref_ort.push_back(u32_to_f32(x));
    for (auto x : load_hex(stim_dir()+"/"+n+".ref_hw.f32.hex"))  s.ref_hw.push_back(u32_to_f32(x));
    return s;
}}
template<typename T> static void pack_bytes(T& d, const int8_t* s, int n) {{
    auto* p = reinterpret_cast<uint32_t*>(&d);
    int nw=(n+3)/4; for(int i=0;i<nw;i++) p[i]=0;
    for(int i=0;i<n;i++) p[i/4] |= ((uint32_t)(uint8_t)s[i] << ((i%4)*8));
}}
template<typename T> static void pack_u16(T& d, const uint16_t* s, int n) {{
    auto* p = reinterpret_cast<uint32_t*>(&d);
    int nw=(n+1)/2; for(int i=0;i<nw;i++) p[i]=0;
    for(int i=0;i<n;i++) p[i/2] |= ((uint32_t)s[i] << ((i%2)*16));
}}
template<typename T> static void unpack_bytes(const T& s, int8_t* d, int n) {{
    auto* p = reinterpret_cast<const uint32_t*>(&s);
    for(int i=0;i<n;i++) d[i] = (int8_t)((p[i/4] >> ((i%4)*8)) & 0xFFu);
}}

static void build_window_tile(const Sample& s, int pix_base, int cit, int8_t* out) {{
    int kc_base = cit*P_CIN;
    for (int pp=0; pp<P_PIX; pp++) {{
        int pix = pix_base+pp;
        int oy = pix/OUT_W, ox = pix%OUT_W;
        for (int kh=0; kh<K; kh++) for (int kw=0; kw<K; kw++)
            for (int kc_local=0; kc_local<P_CIN; kc_local++) {{
                int kc = kc_base + kc_local;
                int tile_lane = (kh*K+kw)*P_CIN + kc_local;
                int off = pp*N_LANE_TILE + tile_lane;
                out[off] = 0;
                if (pix < OUT_H*OUT_W && kc < NCH_IN) {{
                    int h_idx = oy*STRIDE + kh, w_idx = ox*STRIDE + kw;
                    out[off] = s.input_i8[(h_idx*PAD_W + w_idx)*NCH_IN + kc];
                }}
            }}
    }}
}}
static void build_weight_tile(const std::vector<int8_t>& w, int ct, int cit, int8_t* out) {{
    int kc_base = cit*P_CIN, co_base = ct*P_COUT;
    for (int co_l=0; co_l<P_COUT; co_l++) {{
        int co = co_base+co_l;
        for (int kh=0; kh<K; kh++) for (int kw=0; kw<K; kw++)
            for (int kc_l=0; kc_l<P_CIN; kc_l++) {{
                int kc = kc_base+kc_l;
                int full = (kh*K+kw)*NCH_IN + kc;
                int tile = (kh*K+kw)*P_CIN + kc_l;
                int off = co_l*N_LANE_TILE + tile;
                out[off] = 0;
                if (co<NCH_OUT && kc<NCH_IN) out[off] = w[co*N_LANE_FULL + full];
            }}
    }}
}}

struct Stats {{ double max_abs=0, mae=0, cos=0, out_range=0; int n=0, worst_idx=-1;
                float worst_dut=0, worst_ref=0; }};
static Stats compute_stats(const std::vector<float>& d, const std::vector<float>& r) {{
    Stats st; st.n=d.size();
    double sa=0,dot=0,na=0,nb=0; float mn=1e30f,mx=-1e30f;
    for (int i=0;i<st.n;i++) {{
        double e = std::abs((double)d[i]-(double)r[i]); sa+=e;
        if (e>st.max_abs) {{ st.max_abs=e; st.worst_idx=i; st.worst_dut=d[i]; st.worst_ref=r[i]; }}
        dot+=(double)d[i]*r[i]; na+=(double)d[i]*d[i]; nb+=(double)r[i]*r[i];
        if (r[i]<mn) mn=r[i]; if (r[i]>mx) mx=r[i];
    }}
    st.mae = sa/std::max(1,st.n);
    st.cos = dot/(std::sqrt(na)*std::sqrt(nb)+1e-30);
    st.out_range = mx-mn; return st;
}}

int main(int argc, char** argv) {{
    SimCtrl<DUT> sim(argc, argv); sim.max_time = 80000000;
    auto w_raw = load_hex(stim_dir()+"/weights.i8.hex");
    std::vector<int8_t> w_full(NCH_OUT*N_LANE_FULL);
    for (int i=0;i<NCH_OUT*N_LANE_FULL;i++) w_full[i] = (int8_t)(uint8_t)(w_raw[i]&0xFF);

    sim.dut->valid_i=0; sim.dut->first_cin_i=0; sim.dut->last_cin_i=0; sim.dut->cout_tile_idx_i=0;
    std::vector<int8_t> zx(P_PIX*N_LANE_TILE,0), zw(P_COUT*N_LANE_TILE,0);
    std::vector<uint16_t> zq(P_COUT,0);
    pack_bytes(sim.dut->x_flat_i, zx.data(), P_PIX*N_LANE_TILE);
    pack_bytes(sim.dut->w_flat_i, zw.data(), P_COUT*N_LANE_TILE);
    pack_u16(sim.dut->scale_flat_i, zq.data(), P_COUT);
    pack_u16(sim.dut->bias_flat_i, zq.data(), P_COUT);
    sim.reset();

    const std::vector<std::string> names = {{"rand0","rand1","half","gradient","rand_low","rand_high"}};
    int n_pass=0; double agg_cos=0, agg_mae=0, agg_max=0;

    for (const auto& sname : names) {{
        Sample s = load_sample(sname);
        std::vector<int8_t> y_i8(OUT_H*OUT_W*NCH_OUT, 0);
        const int N_OUT = OUT_H*OUT_W;
        const int N_PIX_GROUP = (N_OUT+P_PIX-1)/P_PIX;
        const int BEATS_PER_GROUP = N_COUT_TILE*N_CIN_TILE;
        const int TOTAL_BEATS = N_PIX_GROUP*BEATS_PER_GROUP;
        struct Commit {{ int pix_base, ct; }};
        std::vector<Commit> inflight; inflight.reserve(N_PIX_GROUP*N_COUT_TILE+TOTAL_LAT);
        int produced=0;
        for (int step=0; step < TOTAL_BEATS + TOTAL_LAT + 8; step++) {{
            if (step < TOTAL_BEATS) {{
                int pg = step/BEATS_PER_GROUP; int rem = step%BEATS_PER_GROUP;
                int ct = rem/N_CIN_TILE; int cit = rem%N_CIN_TILE;
                int pix_base = pg*P_PIX;
                int8_t win[P_PIX*N_LANE_TILE]; build_window_tile(s, pix_base, cit, win);
                pack_bytes(sim.dut->x_flat_i, win, P_PIX*N_LANE_TILE);
                int8_t wt[P_COUT*N_LANE_TILE]; build_weight_tile(w_full, ct, cit, wt);
                pack_bytes(sim.dut->w_flat_i, wt, P_COUT*N_LANE_TILE);
                uint16_t sb[P_COUT], bb[P_COUT];
                int co_base = ct*P_COUT;
                for (int co=0; co<P_COUT; co++) {{
                    sb[co] = (co_base+co<NCH_OUT)? s.scale_fp16[co_base+co] : 0;
                    bb[co] = (co_base+co<NCH_OUT)? s.bias_fp16[co_base+co]  : 0;
                }}
                pack_u16(sim.dut->scale_flat_i, sb, P_COUT);
                pack_u16(sim.dut->bias_flat_i,  bb, P_COUT);
                sim.dut->valid_i=1;
                sim.dut->first_cin_i = (cit==0);
                sim.dut->last_cin_i  = (cit==N_CIN_TILE-1);
                sim.dut->cout_tile_idx_i = (uint8_t)ct;
                if (cit==N_CIN_TILE-1) inflight.push_back({{pix_base,ct}});
            }} else {{ sim.dut->valid_i=0; sim.dut->first_cin_i=0; sim.dut->last_cin_i=0; }}
            sim.tick();
            if (sim.dut->valid_o && produced < (int)inflight.size()) {{
                int pix_base = inflight[produced].pix_base;
                int ct = inflight[produced].ct;
                int8_t row[P_PIX*P_COUT];
                unpack_bytes(sim.dut->y_flat_o, row, P_PIX*P_COUT);
                int co_base = ct*P_COUT;
                for (int pp=0; pp<P_PIX; pp++) {{
                    int pix = pix_base+pp;
                    if (pix>=N_OUT) continue;
                    for (int co=0; co<P_COUT; co++) {{
                        int co_g = co_base+co;
                        if (co_g<NCH_OUT) y_i8[pix*NCH_OUT+co_g] = row[pp*P_COUT+co];
                    }}
                }}
                produced++;
            }}
        }}
        double s_out_silu = load_manifest_number("s_out_silu");
        std::vector<float> dut_f(N_OUT*NCH_OUT);
        for (int i=0;i<N_OUT*NCH_OUT;i++) dut_f[i] = (float)((double)y_i8[i]*s_out_silu);
        Stats st_ort = compute_stats(dut_f, s.ref_ort);
        Stats st_hw  = compute_stats(dut_f, s.ref_hw);
        double thr = 0.08 * std::max(1e-6, st_ort.out_range);
        bool pass = (st_ort.cos > 0.997) && (st_ort.mae < thr);
        printf("\\n--- sample %s ---\\n", sname.c_str());
        printf("  DUT vs ORT  : max=%.4f mae=%.4f cos=%.6f range=%.3f\\n",
               st_ort.max_abs, st_ort.mae, st_ort.cos, st_ort.out_range);
        printf("  DUT vs HW   : max=%.4f mae=%.4f cos=%.6f\\n",
               st_hw.max_abs, st_hw.mae, st_hw.cos);
        printf("  => %s\\n", pass?"PASS":"FAIL");
        sim.check(pass, std::string("sample ")+sname+" pass");
        if (pass) n_pass++;
        agg_cos += st_ort.cos; agg_mae += st_ort.mae; agg_max = std::max(agg_max, st_ort.max_abs);
    }}
    printf("\\nAggregate: %d/%zu samples passed; avg cos=%.6f\\n",
           n_pass, names.size(), agg_cos/names.size());
    printf("Layer {idx} NOSILU P_PIX=%d P_COUT=%d P_CIN=%d\\n", P_PIX, P_COUT, P_CIN);
    return sim.finish();
}}
'''
    (ldir / "dv" / f"layer_{idx}_test.cc").write_text(text)


def emit_layer_skeleton(layers, idx, out_root, scale_path, target_cycles, max_p_pix, cout_cap, cin_cap, features=None):
    if features is None:
        features = {"has_silu": True, "has_residual": False, "residual_src": None,
                    "silu_out": None, "add_out": None}
    lay = layers[idx]
    lname = f"layer_{idx}_{short_name(lay['name'])}"
    ldir = out_root / lname
    (ldir / "rtl").mkdir(parents=True, exist_ok=True)
    (ldir / "dv").mkdir(parents=True, exist_ok=True)
    (ldir / "stim").mkdir(parents=True, exist_ok=True)
    # Default s_out_params.sv so verilator can lint before `make stim` runs.
    # `make stim` (extract.py) overwrites with observed amplitude.
    write_s_out_params_placeholder(ldir)

    # Use the selected DV-ish factors for static Verilator wrapper widths. The
    # generated layer module itself remains parameterized by scale_pkg.
    p_pix, p_cout, p_cin, cycles, _, _ = choose_balanced(
        lay,
        target_cycles,
        max_p_pix=max_p_pix,
        cout_cap=16 if cout_cap is None else min(cout_cap, 16),
        cin_cap=8 if cin_cap is None else min(cin_cap, 8),
    )
    n_cout_tile = math.ceil(lay["cout"] / p_cout)
    cout_idx_w = sv_width(n_cout_tile)
    n_lane = lay["kh"] * lay["kw"] * p_cin
    x_bits = p_pix * n_lane * 8
    w_bits = p_cout * n_lane * 8
    q_bits = p_cout * 16
    y_bits = p_pix * p_cout * 8
    r_bits = p_pix * p_cout * 8

    has_resid = features["has_residual"]
    has_silu  = features["has_silu"]
    RESID_INT = 1 if has_resid else 0
    SILU_INT  = 1 if has_silu  else 0

    if has_resid:
        # RTL with residual side-band ports replicated per P_PIX.
        rtl = f"""// Copyright (c) 2026 vibeyolo
// SPDX-License-Identifier: Apache-2.0
//
// Generated skeleton for L{idx} (RESIDUAL=1): {lay['name']}

module layer_{idx}
  import scale_pkg::*;
#(
  parameter int  COUT       = LAYER_{idx}_COUT,
  parameter int  CIN        = LAYER_{idx}_CIN,
  parameter int  K          = LAYER_{idx}_K,
  parameter int  STRIDE     = LAYER_{idx}_STRIDE,
  parameter int  PAD        = LAYER_{idx}_PAD,
  parameter int  P_COUT     = LAYER_{idx}_P_COUT,
  parameter int  P_CIN      = LAYER_{idx}_P_CIN,
  parameter int  P_PIX      = LAYER_{idx}_P_PIX,
  parameter real S_OUT_PRE  = 8.0 / 127.0,
  parameter real S_OUT_SILU = 8.0 / 127.0
) (
  input  logic                                            clk_i,
  input  logic                                            rst_ni,
  input  logic                                            valid_i,
  input  logic                                            first_cin_i,
  input  logic                                            last_cin_i,
  input  logic [$clog2((COUT+P_COUT-1)/P_COUT < 2 ? 2 : (COUT+P_COUT-1)/P_COUT)-1:0]
                                                          cout_tile_idx_i,
  input  logic signed [P_PIX-1:0][K*K*P_CIN-1:0][7:0]     x_i,
  input  logic signed [P_COUT-1:0][K*K*P_CIN-1:0][7:0]    w_i,
  input  logic        [P_COUT-1:0][15:0]                  scale_i,
  input  logic        [P_COUT-1:0][15:0]                  bias_i,
  input  logic signed [P_PIX-1:0][P_COUT-1:0][7:0]        r_i,
  input  logic        [P_COUT-1:0][15:0]                  r_scale_i,
  input  logic        [P_COUT-1:0][15:0]                  r_bias_i,
  output logic                                            valid_o,
  output logic [$clog2((COUT+P_COUT-1)/P_COUT < 2 ? 2 : (COUT+P_COUT-1)/P_COUT)-1:0]
                                                          cout_tile_idx_o,
  output logic signed [P_PIX-1:0][P_COUT-1:0][7:0]        y_o
);

  logic        [P_PIX-1:0]        ready_unused;
  logic        [P_PIX-1:0]        valid_lane;
  logic        [P_PIX-1:0][$clog2((COUT+P_COUT-1)/P_COUT < 2 ? 2 : (COUT+P_COUT-1)/P_COUT)-1:0]
                                      cout_tile_idx_lane;
  logic        [P_PIX-1:0]        tag_match_lane;
  assign valid_o = (&valid_lane) & (&tag_match_lane);
  assign cout_tile_idx_o = cout_tile_idx_lane[0];

  for (genvar pix = 0; pix < P_PIX; pix++) begin : gen_pix
    assign tag_match_lane[pix] = (cout_tile_idx_lane[pix] == cout_tile_idx_lane[0]);

    conv_layer #(
      .CIN(CIN), .COUT(COUT), .K(K), .STRIDE(STRIDE), .PAD(PAD),
      .H_OUT(LAYER_{idx}_H), .W_OUT(LAYER_{idx}_W),
      .P_COUT(P_COUT), .P_CIN(P_CIN),
      .RESIDUAL({RESID_INT}), .SILU({SILU_INT}),
      .S_OUT_PRE(S_OUT_PRE), .S_OUT_SILU(S_OUT_SILU)
    ) u_conv (
      .clk_i(clk_i), .rst_ni(rst_ni), .valid_i(valid_i), .ready_o(ready_unused[pix]),
      .first_cin_i(first_cin_i), .last_cin_i(last_cin_i),
      .cout_tile_idx_i(cout_tile_idx_i),
      .x_i(x_i[pix]), .w_i(w_i), .scale_i(scale_i), .bias_i(bias_i),
      .r_i(r_i[pix]), .r_scale_i(r_scale_i), .r_bias_i(r_bias_i),
      .valid_o(valid_lane[pix]), .ready_i(1'b1),
      .cout_tile_idx_o(cout_tile_idx_lane[pix]), .y_o(y_o[pix])
    );
  end

endmodule
"""
        tb = f"""// Generated TB for L{idx} (RESIDUAL=1).
module layer_{idx}_tb
  import s_out_params::*;
(
  input  logic                  clk_i,
  input  logic                  rst_ni,
  input  logic                  valid_i,
  input  logic                  first_cin_i,
  input  logic                  last_cin_i,
  input  logic [{cout_idx_w-1}:0]      cout_tile_idx_i,
  input  logic [{x_bits-1}:0]     x_flat_i,
  input  logic [{w_bits-1}:0]     w_flat_i,
  input  logic [{q_bits-1}:0]     scale_flat_i,
  input  logic [{q_bits-1}:0]     bias_flat_i,
  input  logic [{r_bits-1}:0]     r_flat_i,
  input  logic [{q_bits-1}:0]     r_scale_flat_i,
  input  logic [{q_bits-1}:0]     r_bias_flat_i,
  output logic                  valid_o,
  output logic [{cout_idx_w-1}:0]      cout_tile_idx_o,
  output logic [{y_bits-1}:0]     y_flat_o
);
  logic signed [{p_pix-1}:0][{n_lane-1}:0][7:0]     x_w;
  logic signed [{p_cout-1}:0][{n_lane-1}:0][7:0]    w_w;
  logic        [{p_cout-1}:0][15:0]                 scale_w, bias_w, rs_w, rb_w;
  logic signed [{p_pix-1}:0][{p_cout-1}:0][7:0]     y_w, r_w;
  assign x_w = x_flat_i;
  assign w_w = w_flat_i;
  assign scale_w = scale_flat_i;
  assign bias_w  = bias_flat_i;
  assign r_w     = r_flat_i;
  assign rs_w    = r_scale_flat_i;
  assign rb_w    = r_bias_flat_i;
  assign y_flat_o = y_w;
  layer_{idx} #(.P_COUT({p_cout}), .P_CIN({p_cin}), .P_PIX({p_pix}),
                .S_OUT_PRE(S_OUT_PRE_VAL), .S_OUT_SILU(S_OUT_SILU_VAL)) u_dut (
    .clk_i(clk_i), .rst_ni(rst_ni),
    .valid_i(valid_i), .first_cin_i(first_cin_i), .last_cin_i(last_cin_i),
    .cout_tile_idx_i(cout_tile_idx_i),
    .x_i(x_w), .w_i(w_w), .scale_i(scale_w), .bias_i(bias_w),
    .r_i(r_w), .r_scale_i(rs_w), .r_bias_i(rb_w),
    .valid_o(valid_o), .cout_tile_idx_o(cout_tile_idx_o), .y_o(y_w)
  );
endmodule
"""
    else:
        # Non-residual path (may or may not have SiLU).
        rtl = f"""// Copyright (c) 2026 vibeyolo
// SPDX-License-Identifier: Apache-2.0
//
// Generated skeleton for L{idx}: {lay['name']}  (SILU={SILU_INT})

module layer_{idx}
  import scale_pkg::*;
#(
  parameter int  COUT       = LAYER_{idx}_COUT,
  parameter int  CIN        = LAYER_{idx}_CIN,
  parameter int  K          = LAYER_{idx}_K,
  parameter int  STRIDE     = LAYER_{idx}_STRIDE,
  parameter int  PAD        = LAYER_{idx}_PAD,
  parameter int  P_COUT     = LAYER_{idx}_P_COUT,
  parameter int  P_CIN      = LAYER_{idx}_P_CIN,
  parameter int  P_PIX      = LAYER_{idx}_P_PIX,
  parameter real S_OUT_PRE  = 8.0 / 127.0,
  parameter real S_OUT_SILU = 8.0 / 127.0
) (
  input  logic                                            clk_i,
  input  logic                                            rst_ni,
  input  logic                                            valid_i,
  input  logic                                            first_cin_i,
  input  logic                                            last_cin_i,
  input  logic [$clog2((COUT+P_COUT-1)/P_COUT < 2 ? 2 : (COUT+P_COUT-1)/P_COUT)-1:0]
                                                          cout_tile_idx_i,
  input  logic signed [P_PIX-1:0][K*K*P_CIN-1:0][7:0]     x_i,
  input  logic signed [P_COUT-1:0][K*K*P_CIN-1:0][7:0]    w_i,
  input  logic        [P_COUT-1:0][15:0]                  scale_i,
  input  logic        [P_COUT-1:0][15:0]                  bias_i,
  output logic                                            valid_o,
  output logic [$clog2((COUT+P_COUT-1)/P_COUT < 2 ? 2 : (COUT+P_COUT-1)/P_COUT)-1:0]
                                                          cout_tile_idx_o,
  output logic signed [P_PIX-1:0][P_COUT-1:0][7:0]        y_o
);

  logic signed [P_COUT-1:0][7:0]  r_tie;
  logic        [P_COUT-1:0][15:0] rs_tie, rb_tie;
  logic        [P_PIX-1:0]        ready_unused;
  logic        [P_PIX-1:0]        valid_lane;
  logic        [P_PIX-1:0][$clog2((COUT+P_COUT-1)/P_COUT < 2 ? 2 : (COUT+P_COUT-1)/P_COUT)-1:0]
                                      cout_tile_idx_lane;
  logic        [P_PIX-1:0]        tag_match_lane;
  assign r_tie  = '0;
  assign rs_tie = '0;
  assign rb_tie = '0;
  assign valid_o = (&valid_lane) & (&tag_match_lane);
  assign cout_tile_idx_o = cout_tile_idx_lane[0];

  for (genvar pix = 0; pix < P_PIX; pix++) begin : gen_pix
    assign tag_match_lane[pix] = (cout_tile_idx_lane[pix] == cout_tile_idx_lane[0]);

    conv_layer #(
      .CIN(CIN), .COUT(COUT), .K(K), .STRIDE(STRIDE), .PAD(PAD),
      .H_OUT(LAYER_{idx}_H), .W_OUT(LAYER_{idx}_W),
      .P_COUT(P_COUT), .P_CIN(P_CIN),
      .RESIDUAL(0), .SILU({SILU_INT}),
      .S_OUT_PRE(S_OUT_PRE), .S_OUT_SILU(S_OUT_SILU)
    ) u_conv (
      .clk_i(clk_i), .rst_ni(rst_ni), .valid_i(valid_i), .ready_o(ready_unused[pix]),
      .first_cin_i(first_cin_i), .last_cin_i(last_cin_i),
      .cout_tile_idx_i(cout_tile_idx_i),
      .x_i(x_i[pix]), .w_i(w_i), .scale_i(scale_i), .bias_i(bias_i),
      .r_i(r_tie), .r_scale_i(rs_tie), .r_bias_i(rb_tie),
      .valid_o(valid_lane[pix]), .ready_i(1'b1),
      .cout_tile_idx_o(cout_tile_idx_lane[pix]), .y_o(y_o[pix])
    );
  end

endmodule
"""
        tb = f"""// Generated TB for L{idx} (SILU={SILU_INT}).
module layer_{idx}_tb
  import s_out_params::*;
(
  input  logic                  clk_i,
  input  logic                  rst_ni,
  input  logic                  valid_i,
  input  logic                  first_cin_i,
  input  logic                  last_cin_i,
  input  logic [{cout_idx_w-1}:0]      cout_tile_idx_i,
  input  logic [{x_bits-1}:0]     x_flat_i,
  input  logic [{w_bits-1}:0]     w_flat_i,
  input  logic [{q_bits-1}:0]     scale_flat_i,
  input  logic [{q_bits-1}:0]     bias_flat_i,
  output logic                  valid_o,
  output logic [{cout_idx_w-1}:0]      cout_tile_idx_o,
  output logic [{y_bits-1}:0]     y_flat_o
);
  logic signed [{p_pix-1}:0][{n_lane-1}:0][7:0]     x_w;
  logic signed [{p_cout-1}:0][{n_lane-1}:0][7:0]    w_w;
  logic        [{p_cout-1}:0][15:0]                 scale_w, bias_w;
  logic signed [{p_pix-1}:0][{p_cout-1}:0][7:0]     y_w;
  assign x_w = x_flat_i;
  assign w_w = w_flat_i;
  assign scale_w = scale_flat_i;
  assign bias_w  = bias_flat_i;
  assign y_flat_o = y_w;
  layer_{idx} #(.P_COUT({p_cout}), .P_CIN({p_cin}), .P_PIX({p_pix}),
                .S_OUT_PRE(S_OUT_PRE_VAL), .S_OUT_SILU(S_OUT_SILU_VAL)) u_dut (
    .clk_i(clk_i), .rst_ni(rst_ni),
    .valid_i(valid_i), .first_cin_i(first_cin_i), .last_cin_i(last_cin_i),
    .cout_tile_idx_i(cout_tile_idx_i),
    .x_i(x_w), .w_i(w_w), .scale_i(scale_w), .bias_i(bias_w),
    .valid_o(valid_o), .cout_tile_idx_o(cout_tile_idx_o), .y_o(y_w)
  );
endmodule
"""

    (ldir / "rtl" / f"layer_{idx}.sv").write_text(rtl)
    (ldir / "dv" / f"layer_{idx}_tb.sv").write_text(tb)

    mk = f"""REPO_ROOT := $(shell git rev-parse --show-toplevel 2>/dev/null)
ifeq ($(REPO_ROOT),)
REPO_ROOT := $(abspath $(CURDIR)/../../../..)
endif

MAC_DIR := $(REPO_ROOT)/hw/ip/mac8/rtl
DOT_DIR := $(REPO_ROOT)/hw/ip/dotN/rtl
I2F_DIR := $(REPO_ROOT)/hw/ip/i32_to_fp16/rtl
FMA_DIR := $(REPO_ROOT)/hw/ip/fp16_fma/rtl
F2I_DIR := $(REPO_ROOT)/hw/ip/fp16_to_i8_sat/rtl
RQ_DIR  := $(REPO_ROOT)/hw/ip/requant/rtl
SIL_DIR := $(REPO_ROOT)/hw/ip/act_silu/rtl
ARQ_DIR := $(REPO_ROOT)/hw/ip/add_rq/rtl
CL_DIR  := $(REPO_ROOT)/hw/ip/conv_layer/rtl
L_DIR   := {make_ref(ldir)}/rtl

TB_TOP := layer_{idx}_tb

STIM_DIR := {make_ref(ldir)}/stim

RTL_SRCS := \\
  {make_ref(scale_path)} \\
  $(STIM_DIR)/s_out_params.sv \\
  $(MAC_DIR)/mac8.sv \\
  $(DOT_DIR)/dotN.sv \\
  $(I2F_DIR)/i32_to_fp16.sv \\
  $(FMA_DIR)/fp16_fma.sv \\
  $(F2I_DIR)/fp16_to_i8_sat.sv \\
  $(RQ_DIR)/requant.sv \\
  $(SIL_DIR)/act_silu.sv \\
  $(ARQ_DIR)/add_rq.sv \\
  $(CL_DIR)/conv_layer.sv \\
  $(L_DIR)/layer_{idx}.sv \\
  $(CURDIR)/layer_{idx}_tb.sv

CC_SRCS := $(CURDIR)/layer_{idx}_test.cc
LINT_FLAGS := -Wno-UNOPTFLAT -Wno-ASCRANGE -Wno-SELRANGE -Wno-WIDTHTRUNC

include $(REPO_ROOT)/mk/verilator.mk

.PHONY: stim
stim: $(STIM_DIR)/.stim.stamp

# Always re-run extract.py before building the verilator binary so that
# stim/s_out_params.sv is up-to-date. The stamp keeps `make` happy with
# the file-as-prereq pattern; the verilator binary also depends on the
# s_out_params.sv source file directly via RTL_SRCS.
$(STIM_DIR)/.stim.stamp: {make_ref(ldir / "extract.py")}
\tpython3 {make_ref(ldir / "extract.py")}
\ttouch $@

# Force the binary build to wait on extract.py completion. Without this,
# make -j can race: the binary starts building against the placeholder
# s_out_params.sv before extract.py rewrites it.
$(STIM_DIR)/s_out_params.sv: $(STIM_DIR)/.stim.stamp
\t@true

test: stim
"""
    (ldir / "dv" / "Makefile").write_text(mk)

    is_pure_depthwise = (lay["group"] != 1
                         and lay["group"] == lay["cin"] == lay["cout"])
    if lay["group"] == 1 or is_pure_depthwise:
        if has_resid:
            emit_conv_resid_extract(lay, idx, ldir, features)
            emit_conv_resid_test(lay, idx, ldir, p_pix, p_cout, p_cin, features=features)
        elif not has_silu:
            emit_conv_nosilu_extract(lay, idx, ldir, features)
            emit_conv_nosilu_test(lay, idx, ldir, p_pix, p_cout, p_cin)
        else:
            emit_conv_silu_extract(lay, idx, ldir)
            emit_conv_silu_test(lay, idx, ldir, p_pix, p_cout, p_cin)

    readme = f"""# {lname}

Generated ordinary Conv+SiLU integration layer for `{lay['name']}`.

- Shape: Cin={lay['cin']} Cout={lay['cout']} K={lay['kh']} stride={lay['sh']} pad={lay['ph']}
- Output: {lay['H_out']}x{lay['W_out']}
- Generated DV wrapper factors: P_PIX={p_pix}, P_COUT={p_cout}, P_CIN={p_cin}, cycles~{cycles}

This directory is generated under `integ/generated/` so it does not replace
the hand-built integration tests.

Current generator limitation: the emitted extract/DV path covers ordinary
`group=1` Conv+SiLU layers. Residual, grouped, and depthwise layers still need
dedicated generator support.
"""
    (ldir / "README.md").write_text(readme)
    return ldir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(REPO / "integ" / "yolo26n" / "model_int8.onnx"))
    ap.add_argument("--target-cycles", type=int, default=100000)
    ap.add_argument("--max-p-pix", type=int, default=2,
                    help=("Maximum pixel-parallel factor to model. Default 2. The "
                          "conv_layer leaf remains single-pixel; generated wrappers implement values >1 "
                          "by instantiating multiple conv_layer lanes."))
    ap.add_argument("--cout-cap", type=int, default=None)
    ap.add_argument("--cin-cap", type=int, default=None)
    ap.add_argument("--out-root", default=str(REPO / "integ" / "generated"))
    ap.add_argument("--emit-scale", action="store_true")
    ap.add_argument("--emit-dv-scale", action="store_true")
    ap.add_argument("--gen-layer", type=int, action="append", default=[])
    ap.add_argument("--gen-layer-range", type=str, default=None,
                    help="Inclusive range like '26-101'. Combined with --gen-layer.")
    args = ap.parse_args()

    layers = infer_layer_specs(args.model)
    features_all = analyze_topology(args.model)
    out_root = Path(args.out_root)
    scale_path = out_root / "scale" / "scale_pkg_balanced_dv.sv"

    if args.emit_scale:
        emit_scale_pkg(
            layers,
            out_root / "scale" / "scale_pkg_balanced.sv",
            out_root / "scale" / "scale_report_balanced.md",
            args.target_cycles,
            args.max_p_pix,
            args.cout_cap,
            args.cin_cap,
            dv=False,
        )
    if args.emit_dv_scale:
        emit_scale_pkg(
            layers,
            scale_path,
            out_root / "scale" / "scale_report_balanced_dv.md",
            args.target_cycles,
            args.max_p_pix,
            16 if args.cout_cap is None else min(args.cout_cap, 16),
            8 if args.cin_cap is None else min(args.cin_cap, 8),
            dv=True,
        )
    gen_ids = list(args.gen_layer)
    if args.gen_layer_range:
        lo, hi = args.gen_layer_range.split("-")
        gen_ids.extend(range(int(lo), int(hi) + 1))
    for idx in gen_ids:
        ldir = emit_layer_skeleton(
            layers, idx, out_root, scale_path,
            args.target_cycles, args.max_p_pix, args.cout_cap, args.cin_cap,
            features=features_all.get(idx))
        print(f"Wrote {ldir}")


if __name__ == "__main__":
    main()
