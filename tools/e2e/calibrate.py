#!/usr/bin/env python3
# Copyright (c) 2026 vibeyolo
# SPDX-License-Identifier: Apache-2.0
#
# calibrate.py — derive the chip's fixed per-layer scales from a real-image
# corpus, using the SAME numpy chip-model forward as chain.py (so the calibrated
# ranges match the conv_stage arithmetic). Replaces the old single-image,
# max-based calibration (dump_layers.py: amp*1.1/127 on bus.jpg alone) that left
# the 10 attn-internal convs un-calibrated and let RTL fp16 outliers hijack
# downstream max-based dynamic scales (→ collapse on some images).
#
# For each of the 102 convs we record, per image, the 99.9th-percentile of
# |input| / |pre-act| / |post-SiLU| (the percentile drops the top 0.1% spikes
# WITHIN an image — exactly the outliers that wreck a max-based scale), then take
# the MAX across images (cover the most-demanding frame) and set scale =
# agg*1.1/127, floored at 2/127. Output: calib_scales.json keyed by conv name,
# consumed by chain.py --scales fixed (covers ALL convs, dumped or not).
#
# Usage: python3 tools/e2e/calibrate.py "assets/calib/coco/*.jpg" "assets/calib/qoi/*.png" [--pct 99.9] [--out calib_scales.json]

import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import chain  # noqa: E402  (reuses chain.py's ONNX graph + conv/op machinery)
from preprocess import preprocess  # noqa: E402

G = chain.G
IN_NAME = [i.name for i in G.input if i.name not in chain.INIT][0]


def p_amp(t, pct):
    a = np.abs(np.asarray(t)).reshape(-1)
    return float(np.percentile(a, pct)) if a.size else 0.0


def forward_collect(x, stats, pct):
    """One numpy chip-model forward (dynamic scales), recording per-conv amps."""
    vals = {IN_NAME: x[0]}
    scales = {}
    for node in G.node:
        if node.name in chain.TRIGGER:
            p = chain.TRIGGER[node.name]
            fp_in = vals[p["fp_in"]]
            if fp_in.ndim == 4:
                fp_in = fp_in[0]
            s_in = scales.get(p["fp_in"])
            if s_in is None:
                s_in = chain.pick_s_out(float(np.max(np.abs(fp_in))))
            pre = chain.conv_pre(fp_in, s_in, p)
            s_pre = chain.pick_s_out(float(np.max(np.abs(pre))))
            pre_code = np.clip(np.round(pre / s_pre), -128, 127) * s_pre
            y = pre_code / (1.0 + np.exp(-pre_code)) if p["silu"] else pre_code
            s_silu = chain.pick_s_out(float(np.max(np.abs(y))))
            out = chain.conv_numpy(fp_in, s_in, p, s_pre, s_silu)
            vals[p["out"]] = out[None]
            scales[p["out"]] = s_silu
            nm = p["conv"].name
            st = stats.setdefault(nm, {"in": [], "pre": [], "post": [],
                                       "silu": bool(p["silu"]),
                                       "idx": chain.NAME2IDX.get(nm, -1)})
            st["in"].append(p_amp(fp_in, pct))
            st["pre"].append(p_amp(pre, pct))
            st["post"].append(p_amp(y, pct))
        elif node.name in chain.OWNED or node.op_type == "DynamicQuantizeLinear":
            continue
        else:
            for n2, v in zip(node.output, chain.run_op(node, vals)):
                vals[n2] = v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("globs", nargs="+", help="image globs (COCO + QOI + ...)")
    ap.add_argument("--pct", type=float, default=99.9)
    ap.add_argument("--limit", type=int, default=0,
                    help="evenly subsample to at most N images (0 = all)")
    ap.add_argument("--out", default="integ/generated/e2e/calib_scales.json")
    args = ap.parse_args()

    imgs = []
    for g in args.globs:
        gg = g if os.path.isabs(g) else os.path.join(chain.ROOT, g)
        imgs += sorted(glob.glob(gg, recursive=True))   # supports ** for nested QOI suite
    if args.limit:
        imgs = imgs[::max(1, len(imgs) // args.limit)][:args.limit]  # even subsample
    if not imgs:
        sys.exit("no images matched")
    print(f"calibrating over {len(imgs)} images (pct={args.pct})")

    stats = {}
    for i, ip in enumerate(imgs):
        try:
            x = preprocess(ip).astype(np.float32)
        except Exception as e:
            print(f"  skip {os.path.basename(ip)}: {e}"); continue
        forward_collect(x, stats, args.pct)
        if (i + 1) % 10 == 0 or i + 1 == len(imgs):
            print(f"  {i + 1}/{len(imgs)}")

    FLOOR = 2.0 / 127.0
    def scale(vals):
        return max(max(vals) * 1.1 / 127.0, FLOOR) if vals else FLOOR

    calib = {}
    for nm, st in stats.items():
        s_pre = scale(st["pre"])
        s_silu = scale(st["post"]) if st["silu"] else s_pre
        calib[nm] = dict(s_in=scale(st["in"]), s_out_pre=s_pre,
                         s_out_silu=s_silu, silu=st["silu"], idx=st["idx"],
                         n_img=len(st["pre"]))
    op = os.path.join(chain.ROOT, args.out)
    os.makedirs(os.path.dirname(op), exist_ok=True)
    json.dump(dict(pct=args.pct, n_images=len(imgs), images=[os.path.basename(p) for p in imgs],
                   scales=calib), open(op, "w"), indent=1)
    print(f"\nwrote {len(calib)} conv scales -> {args.out}")
    # show the 10 previously-uncalibrated attn-internal convs
    internal = [nm for nm, c in calib.items() if c["idx"] < 0]
    print(f"calibrated {len(internal)} attn-internal convs (were dynamic/uncalibrated):")
    for nm in internal:
        c = calib[nm]
        print(f"  {nm:48s} s_pre={c['s_out_pre']:.4f} s_silu={c['s_out_silu']:.4f}")


if __name__ == "__main__":
    main()
