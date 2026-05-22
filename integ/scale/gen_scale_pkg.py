#!/usr/bin/env python3
# Copyright (c) 2026 vibeyolo
# SPDX-License-Identifier: Apache-2.0
#
# gen_scale_pkg.py — Generate the per-layer parallelism package for YOLO26n.
#
# For each ConvInteger layer in integ/yolo26n/model_int8.onnx, computes:
#   M_i  = Cout * Cin * K^2 * H_out * W_out   (total MACs/frame)
#   pick (P_PIX_i, P_COUT_i, P_CIN_i) such that
#       P_PIX_i * P_COUT_i * P_CIN_i  >=  ceil(M_i / T_FRAME)
#       cycles_i = ceil(M_i / (P_PIX * P_COUT * P_CIN))  <=  T_FRAME
#
# Heuristic (matches dataflow naturally):
#   1. Fully unroll P_COUT = Cout (cheap in dataflow, one dotN tree per Cout).
#   2. If still short, increase P_CIN inside the input MAC tap, capped at Cin*K^2.
#   3. Only last resort: pixel-parallel (P_PIX) — costs line-buffer area.
#
# Emits:
#   scale_pkg.sv    — SystemVerilog package with LAYER_<i>_* localparams.
#   scale_report.md — markdown summary table.

import argparse
import math
import os
import sys

import onnx
from onnx import shape_inference, numpy_helper


def infer_layer_specs(model_path):
    m = onnx.load(model_path)
    m_inf = shape_inference.infer_shapes(m)
    vi_by_name = {vi.name: vi for vi in
                  list(m_inf.graph.value_info) +
                  list(m_inf.graph.output) +
                  list(m_inf.graph.input)}

    def shape_of(name):
        if name not in vi_by_name:
            return None
        return [d.dim_value if d.HasField('dim_value') else 1
                for d in vi_by_name[name].type.tensor_type.shape.dim]

    init_by_name = {init.name: init for init in m.graph.initializer}

    layers = []
    for n in m.graph.node:
        if n.op_type != 'ConvInteger':
            continue
        w = numpy_helper.to_array(init_by_name[n.input[1]])
        attrs = {a.name: a for a in n.attribute}
        strides = list(attrs['strides'].ints) if 'strides' in attrs else [1, 1]
        pads = list(attrs['pads'].ints) if 'pads' in attrs else [0, 0, 0, 0]
        group = attrs['group'].i if 'group' in attrs else 1
        ks = list(attrs['kernel_shape'].ints) if 'kernel_shape' in attrs else [w.shape[2], w.shape[3]]
        in_shape = shape_of(n.input[0])
        out_shape = shape_of(n.output[0])
        # ConvInteger weights: (Cout, Cin/group, Kh, Kw)
        cout = int(w.shape[0])
        cin = int(w.shape[1]) * int(group)
        kh = int(ks[0])
        kw = int(ks[1])
        H_in = int(in_shape[2]); W_in = int(in_shape[3])
        H_out = int(out_shape[2]); W_out = int(out_shape[3])
        layers.append({
            'name': n.name,
            'cout': cout, 'cin': cin, 'kh': kh, 'kw': kw,
            'sh': int(strides[0]), 'sw': int(strides[1]),
            'ph': int(pads[0]), 'pw': int(pads[1]),
            'group': int(group),
            'H_in': H_in, 'W_in': W_in,
            'H_out': H_out, 'W_out': W_out,
        })
    return layers


def pick_parallelism(layer, T):
    """Return (P_PIX, P_COUT, P_CIN, cycles, macs).

    The tiled layer template instantiates P_COUT dotN cells, each with
    N_LANE = K*K*P_CIN. So per-cycle MAC throughput = P_PIX * P_COUT * K*K * P_CIN.
    P_CIN ranges over Cin only (K*K is always folded into the dotN reduction).
    """
    cout = layer['cout']
    cin = layer['cin']
    K2 = layer['kh'] * layer['kw']
    H = layer['H_out']; W = layer['W_out']
    macs = cout * cin * K2 * H * W
    # per-cycle MAC throughput = P_PIX * P_COUT * K2 * P_CIN
    needed_par_ex_k2 = math.ceil(macs / (T * K2))

    # Step 1: P_COUT — pick smallest divisor of Cout meeting need.
    cout_divs = [d for d in range(1, cout + 1) if cout % d == 0]
    P_COUT = next((d for d in cout_divs if d >= needed_par_ex_k2), cout)
    # Step 2: P_CIN — divisor of Cin meeting remaining need.
    remaining = math.ceil(needed_par_ex_k2 / P_COUT)
    cin_divs = [d for d in range(1, cin + 1) if cin % d == 0]
    P_CIN = next((d for d in cin_divs if d >= remaining), cin)
    # Step 3: P_PIX last resort (most layers won't need this).
    remaining2 = math.ceil(needed_par_ex_k2 / (P_COUT * P_CIN))
    P_PIX_max = H * W
    pix_divs = [d for d in range(1, P_PIX_max + 1) if P_PIX_max % d == 0]
    P_PIX = next((d for d in pix_divs if d >= remaining2), P_PIX_max)

    total_par = P_PIX * P_COUT * P_CIN * K2  # K2 folded into dotN
    cycles = math.ceil(macs / total_par)
    return P_PIX, P_COUT, P_CIN, cycles, macs


def clamp_params(P_PIX, P_COUT, P_CIN, K2, macs, dv_cout_cap, dv_cin_cap):
    """Apply DV clamps to parallelism factors and recompute cycles (K2 folded)."""
    P_COUT_c = min(P_COUT, dv_cout_cap)
    P_CIN_c = min(P_CIN, dv_cin_cap)
    total_par = P_PIX * P_COUT_c * P_CIN_c * K2
    cycles = math.ceil(macs / total_par)
    return P_PIX, P_COUT_c, P_CIN_c, cycles


def gen_sv(layers, T, out_path, dv=False, dv_cout_cap=16, dv_cin_cap=8):
    n = len(layers)
    lines = []
    lines.append('// Copyright (c) 2026 vibeyolo')
    lines.append('// SPDX-License-Identifier: Apache-2.0')
    lines.append('//')
    if dv:
        lines.append('// scale_pkg_dv.sv — DV-only variant with capped parallelism for fast verilator builds.')
        lines.append(f'// P_COUT capped at {dv_cout_cap}, P_CIN capped at {dv_cin_cap}; cycles recomputed.')
        lines.append('// Package name is still `scale_pkg` — drop this file in place of scale_pkg.sv for DV.')
    else:
        lines.append('// scale_pkg.sv — auto-generated per-layer parallelism + shape parameters.')
    lines.append(f'// Generated by integ/scale/gen_scale_pkg.py with T_FRAME = {T}.')
    lines.append(f'// Layer count: {n}')
    lines.append('//')
    lines.append('// For each layer i, exposes:')
    lines.append('//   P_PIX_i, P_COUT_i, P_CIN_i      parallelism factors')
    lines.append('//   H_i, W_i, K_i                   output H, W, kernel size (Kh==Kw)')
    lines.append('//   CIN_i, COUT_i                   channel counts')
    lines.append('//   STRIDE_i, PAD_i                 stride and pad (symmetric)')
    lines.append('')
    lines.append('// verilator lint_off UNUSEDPARAM')
    if dv:
        lines.append('// verilator lint_off DECLFILENAME')
    lines.append('package scale_pkg;')
    lines.append('')
    lines.append(f'  localparam int T_FRAME    = {T};')
    lines.append(f'  localparam int NUM_LAYERS = {n};')
    lines.append('')
    for i, lay in enumerate(layers):
        P_PIX, P_COUT, P_CIN, cycles, macs = pick_parallelism(lay, T)
        if dv:
            K2 = lay['kh'] * lay['kw']
            P_PIX, P_COUT, P_CIN, cycles = clamp_params(
                P_PIX, P_COUT, P_CIN, K2, macs, dv_cout_cap, dv_cin_cap)
        K = lay['kh']
        assert lay['kh'] == lay['kw'], f"non-square kernel layer {i}"
        assert lay['sh'] == lay['sw'], f"non-square stride layer {i}"
        assert lay['ph'] == lay['pw'], f"non-symmetric pad layer {i}"
        lines.append(f'  // L{i}: {lay["name"]}')
        lines.append(f'  //   shape  Cin={lay["cin"]} Cout={lay["cout"]} K={K} stride={lay["sh"]} pad={lay["ph"]} group={lay["group"]}')
        lines.append(f'  //   H_in={lay["H_in"]} W_in={lay["W_in"]} H_out={lay["H_out"]} W_out={lay["W_out"]}')
        lines.append(f'  //   M={macs}  cycles={cycles}  (target T={T}{" — DV clamped" if dv else ""})')
        lines.append(f'  localparam int LAYER_{i}_P_PIX  = {P_PIX};')
        lines.append(f'  localparam int LAYER_{i}_P_COUT = {P_COUT};')
        lines.append(f'  localparam int LAYER_{i}_P_CIN  = {P_CIN};')
        lines.append(f'  localparam int LAYER_{i}_H      = {lay["H_out"]};')
        lines.append(f'  localparam int LAYER_{i}_W      = {lay["W_out"]};')
        lines.append(f'  localparam int LAYER_{i}_K      = {K};')
        lines.append(f'  localparam int LAYER_{i}_CIN    = {lay["cin"]};')
        lines.append(f'  localparam int LAYER_{i}_COUT   = {lay["cout"]};')
        lines.append(f'  localparam int LAYER_{i}_STRIDE = {lay["sh"]};')
        lines.append(f'  localparam int LAYER_{i}_PAD    = {lay["ph"]};')
        lines.append(f'  localparam int LAYER_{i}_GROUP  = {lay["group"]};')
        lines.append(f'  localparam int LAYER_{i}_CYCLES = {cycles};')
        lines.append('')
    lines.append('endpackage')
    lines.append('// verilator lint_on UNUSEDPARAM')
    if dv:
        lines.append('// verilator lint_on DECLFILENAME')
    lines.append('')
    with open(out_path, 'w') as f:
        f.write('\n'.join(lines))


def gen_report(layers, T, out_path):
    lines = []
    lines.append(f'# Scale report — YOLO26n  (T_FRAME = {T} cyc/frame)')
    lines.append('')
    lines.append(f'Layer count: **{len(layers)}** ConvInteger ops.')
    lines.append('')
    lines.append('Heuristic: fully unroll P_COUT, then expand P_CIN inside the input window, then P_PIX.')
    lines.append('"MAC cost" = P_PIX * P_COUT * P_CIN (total int8 multipliers instantiated).')
    lines.append('')
    lines.append('| idx | name | Cin | Cout | K | s | H_out | W_out | M_i | P_PIX | P_COUT | P_CIN | MAC cost | cycles_i |')
    lines.append('|-----|------|----:|-----:|--:|--:|------:|------:|----:|------:|-------:|------:|---------:|---------:|')
    total_macs = 0
    total_macunits = 0
    worst_cyc = 0
    for i, lay in enumerate(layers):
        P_PIX, P_COUT, P_CIN, cycles, macs = pick_parallelism(lay, T)
        total_macs += macs
        total_macunits += P_PIX * P_COUT * P_CIN
        worst_cyc = max(worst_cyc, cycles)
        short = lay['name'].split('/')[-3] if '/' in lay['name'] else lay['name']
        lines.append(f'| {i} | {lay["name"]} | {lay["cin"]} | {lay["cout"]} | {lay["kh"]} | {lay["sh"]} | '
                     f'{lay["H_out"]} | {lay["W_out"]} | {macs} | {P_PIX} | {P_COUT} | {P_CIN} | '
                     f'{P_PIX*P_COUT*P_CIN} | {cycles} |')
    lines.append('')
    lines.append(f'**Total MACs/frame:** {total_macs:,}  (= {total_macs/1e9:.4f} GMAC).')
    lines.append(f'**Total MAC units instantiated:** {total_macunits:,}.')
    lines.append(f'**Worst-layer cycles_i:** {worst_cyc}  (target T = {T}).')
    if worst_cyc <= T:
        lines.append('All layers meet T_FRAME.')
    else:
        lines.append(f'**WARNING:** worst layer exceeds T_FRAME by factor {worst_cyc/T:.2f}.')
    lines.append('')
    with open(out_path, 'w') as f:
        f.write('\n'.join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--T', type=int, default=100000)
    ap.add_argument('--model', default=os.path.join(os.path.dirname(__file__),
                                                     '..', 'yolo26n', 'model_int8.onnx'))
    ap.add_argument('--out', default=None)
    ap.add_argument('--report', default=os.path.join(os.path.dirname(__file__), 'scale_report.md'))
    ap.add_argument('--dv', action='store_true',
                    help='Emit DV-only variant with capped parallelism (default: scale_pkg_dv.sv).')
    ap.add_argument('--dv-cout-cap', type=int, default=16)
    ap.add_argument('--dv-cin-cap', type=int, default=8)
    args = ap.parse_args()

    if args.out is None:
        fname = 'scale_pkg_dv.sv' if args.dv else 'scale_pkg.sv'
        args.out = os.path.join(os.path.dirname(__file__), fname)

    layers = infer_layer_specs(os.path.abspath(args.model))
    gen_sv(layers, args.T, os.path.abspath(args.out),
           dv=args.dv, dv_cout_cap=args.dv_cout_cap, dv_cin_cap=args.dv_cin_cap)
    print(f'Wrote {args.out}')
    if not args.dv:
        gen_report(layers, args.T, os.path.abspath(args.report))
        print(f'Wrote {args.report}')
    print(f'Layers: {len(layers)}')


if __name__ == '__main__':
    main()
