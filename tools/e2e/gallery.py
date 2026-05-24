#!/usr/bin/env python3
# Copyright (c) 2026 vibeyolo
# SPDX-License-Identifier: Apache-2.0
#
# gallery.py — assemble the per-image ORT-vs-chip renders produced by
# sweep_images.sh into a single contact-sheet PNG, and emit a markdown report
# table from the per-image metrics JSON. Pure image/text assembly; no process
# management.
#
# Usage: python3 tools/e2e/gallery.py [gallery_dir]

import glob
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    gdir = sys.argv[1] if len(sys.argv) > 1 else "integ/generated/e2e/gallery"
    gdir = gdir if os.path.isabs(gdir) else os.path.join(ROOT, gdir)
    metas = sorted(glob.glob(os.path.join(gdir, "*.json")))
    if not metas:
        print(f"no metrics JSON in {gdir}"); return
    rows = [json.load(open(m)) for m in metas]

    # ---- contact sheet: each per-image render scaled to a common tile width ----
    TILE_W = 900            # downscale each side-by-side render to this width
    BAR = 22                # caption strip height under each tile
    tiles = []
    for r in rows:
        png = os.path.join(gdir, os.path.splitext(os.path.basename(r["image"]))[0] + ".png")
        if not os.path.exists(png):
            continue
        im = Image.open(png).convert("RGB")
        h = int(im.height * TILE_W / im.width)
        im = im.resize((TILE_W, h), Image.BILINEAR)
        tile = Image.new("RGB", (TILE_W, h + BAR), (15, 15, 15))
        tile.paste(im, (0, 0))
        d = ImageDraw.Draw(tile)
        # Color by DETECTION agreement, not logits_cos (632 showed cos can be
        # 0.987 with 0 detections — the negative-background logits dominate the
        # cosine and mask a positive-peak collapse). green=matched all ORT dets,
        # amber=missed some, red=missed most/all.
        miss = r["ort_dets"] - r["matched"]
        if r["ort_dets"] == 0 or miss == 0:
            col = (90, 230, 90)
        elif r["matched"] >= max(1, r["ort_dets"] // 2):
            col = (240, 180, 60)
        else:
            col = (240, 90, 70)
        cap = (f"{r['image']}  chip/ORT dets={r['chip_dets']}/{r['ort_dets']}  "
               f"matched(IoU>=.5)={r['matched']}  miss={miss}  "
               f"logits_cos={r['logits_cos']:.3f}")
        d.text((6, h + 5), cap, fill=col)
        tiles.append(tile)

    if tiles:
        W = TILE_W
        Hs = [t.height for t in tiles]
        sheet = Image.new("RGB", (W, sum(Hs) + 8 * len(tiles) + 8), (0, 0, 0))
        y = 8
        for t in tiles:
            sheet.paste(t, (0, y)); y += t.height + 8
        sheet_path = os.path.join(gdir, "GALLERY.png")
        sheet.save(sheet_path)
        print(f"contact sheet -> {os.path.relpath(sheet_path, ROOT)}  "
              f"({W}x{sheet.height}, {len(tiles)} images)")

    # ---- markdown report ----
    lc = np.array([r["logits_cos"] for r in rows])
    tot_ort = sum(r["ort_dets"] for r in rows)
    tot_match = sum(r["matched"] for r in rows)
    full = sum(1 for r in rows if r["ort_dets"] > 0 and r["matched"] == r["ort_dets"])
    failed = [r for r in rows if r["ort_dets"] > 0 and r["matched"] < max(1, r["ort_dets"] // 2)]
    rep = [f"# E2E image sweep — chip (full RTL) vs ORT\n",
           f"{len(rows)} images. Chip = chained run with **every conv on conv_stage "
           "Verilator + all 6 block IPs on real RTL**, driven by the chip's **fixed "
           "calibrated scales** (`--conv rtl --rtl-blocks all --scales fixed`) — i.e. "
           "the static-scale silicon as it would tape out.\n",
           "> **Read detection match, not cosine.** `logits_cos` is dominated by the "
           "~24k negative no-object logits and can stay ≥0.98 even when every positive "
           "detection peak collapses (see img 632). The trustworthy metric is "
           "`matched` (chip dets with an IoU≥0.5 same-class ORT box) vs `ORT dets`.\n",
           f"**Detection recall: {tot_match}/{tot_ort} ORT detections matched** across "
           f"all images. **{full}/{len(rows)} images fully matched** ORT. "
           f"**{len(failed)} images failed** (missed >half of ORT dets).\n",
           "| image | chip dets | ORT dets | matched | miss | median IoU | logits_cos | worst_conv |",
           "|---|---|---|---|---|---|---|---|"]
    # worst detection-agreement first
    for r in sorted(rows, key=lambda r: (r["matched"] - r["ort_dets"], r["logits_cos"])):
        rep.append(f"| {r['image']} | {r['chip_dets']} | {r['ort_dets']} | {r['matched']} | "
                   f"{r['ort_dets']-r['matched']} | {r['median_iou']:.2f} | "
                   f"{r['logits_cos']:.4f} | #{r['worst_conv_idx']} ({r['worst_conv_cos']:.3f}) |")
    rep += ["",
            f"**logits_cos** (secondary): min {lc.min():.4f}, median {np.median(lc):.4f}, "
            f"mean {lc.mean():.4f}, max {lc.max():.4f}"]
    if failed:
        rep += ["", "**Failed images** (calibration generalization — see "
                "project_recalibrate_scales_before_pnr): "
                + ", ".join(r["image"] for r in failed)]
    rep += ["", "![gallery](GALLERY.png)", ""]
    rep_path = os.path.join(gdir, "REPORT.md")
    open(rep_path, "w").write("\n".join(rep))
    print(f"report -> {os.path.relpath(rep_path, ROOT)}")


if __name__ == "__main__":
    main()
