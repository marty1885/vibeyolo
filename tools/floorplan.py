#!/usr/bin/env python3
# Copyright (c) 2026 vibeyolo
# SPDX-License-Identifier: Apache-2.0
#
# floorplan.py — render an AREA-PROPORTIONAL floorplan of the YOLO26n die on a
# chosen PDK. This is NOT a place-and-route result (no router was run) — it is a
# squarified treemap where every rectangle's area equals the block's estimated
# silicon area (same model as tools/chip_area.py). It answers "how big is each
# block, and roughly how would they pack" — useful before committing to P&R.
#
#   python3 tools/floorplan.py --pdk asap7
#   → reports/FLOORPLAN_<pdk>.png
#
# Hierarchy: die → {Logic (int8 MAC datapath, split by network region),
#                   fp16 datapath (attention macw array + requant fma, measured),
#                   Weight ROM, Activation SRAM}.

import os, re, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sram_model as sm
import sram_bw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Logic model (matches chip_area.py).
A8_UM2    = 62.9      # mac8 area on ASAP7 RVT (yosys+abc liberty), µm²
OVERHEAD  = 1.6       # adders + ctl/glue (the fp16 datapath is carved out below)
LOGIC_UTIL = 0.65     # synth→physical P&R utilisation

# Measured fp16 leaf areas on ASAP7 RVT (5-stage), µm² — carved out of the
# overhead and shown explicitly since they're the timing-critical blocks.
MACW_UM2  = 700.3
FFMA_UM2  = 592.8
# flash_attn instances: 2 blocks × (BR*BC=512 macw score/PV + BR=16 fma lupd).
ATTN_FP16_UM2 = 2 * (512 * MACW_UM2 + 16 * FFMA_UM2)


def region(name):
    """Bucket a conv layer into a coarse network region by its /model.N/ index."""
    m = re.search(r"/model\.(\d+)", name)
    n = int(m.group(1)) if m else 0
    if n <= 8:   return "Backbone (model.0-8)"
    if n <= 22:  return "Neck + PSA attn (model.9-22)"
    return "Detect head (model.23)"


# ───────────────────────── squarified treemap (Bruls et al.) ─────────────────
def _worst(row, length):
    s = sum(row); mx = max(row); mn = min(row)
    return max((length**2 * mx) / s**2, s**2 / (length**2 * mn))

def squarify(areas, x, y, w, h):
    """Lay out `areas` as rects filling [x,y,w,h] (area-true, near-square)."""
    areas = list(areas)
    scale = (w * h) / sum(areas)
    vals  = [a * scale for a in areas]
    rects, idx = [], list(range(len(vals)))
    rx, ry, rw, rh = x, y, w, h
    i = 0
    while i < len(vals):
        length = min(rw, rh)
        row = [vals[i]]; j = i + 1
        while j < len(vals):
            if _worst(row, length) >= _worst(row + [vals[j]], length):
                row.append(vals[j]); j += 1
            else:
                break
        rs = sum(row)
        if rw >= rh:                       # lay the row down the left edge
            cw = rs / rh; cy = ry
            for a in row:
                ch = a / cw
                rects.append((rx, cy, cw, ch)); cy += ch
            rx += cw; rw -= cw
        else:                              # lay the row along the bottom edge
            ch = rs / rw; cx = rx
            for a in row:
                cw = a / ch
                rects.append((cx, ry, cw, ch)); cx += cw
            ry += ch; rh -= ch
        i = j
    return rects


def main():
    ap = argparse.ArgumentParser(description="area-proportional die floorplan")
    ap.add_argument("--pdk", default="asap7", choices=list(sm.PDKS))
    ap.add_argument("--scale-report", default=sram_bw.BAL)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    pdk = sm.set_pdk(args.pdk); sm._sync()
    rows = sram_bw.parse_balanced(args.scale_report)

    per_mac = A8_UM2 * OVERHEAD / LOGIC_UTIL
    # int8 logic per region (carve the fp16 attn array out of the total later).
    reg_um2 = {}
    for r in rows:
        macs = r["ppix"] * r["pcout"] * (r["k"]**2 * r["pcin"])
        reg_um2[region(r["name"])] = reg_um2.get(region(r["name"]), 0) + macs * per_mac
    rom_um2  = sum(r["k"]**2 * r["cin"] * r["cout"] * 8 for r in rows) * pdk.rom_um2_per_bit
    sram_um2 = sum(t.area_um2 for t in
                   [sram_bw.tile(m) for m in sram_bw.enumerate_mems(rows)])
    # The fp16 attention array is real silicon but, in the chip_area.py model,
    # it lives inside Logic's ×1.6 overhead (it's fp16_macw, not mac8, so it's
    # NOT in tot_mac). Carve it OUT of the Neck region (PSA attention sits in
    # model.10/22) and show it explicitly — keeps the die total == chip_area's.
    attn_um2 = ATTN_FP16_UM2 / LOGIC_UTIL   # same P&R basis as the rest
    NECK = "Neck + PSA attn (model.9-22)"
    reg_um2[NECK] = max(reg_um2.get(NECK, 0) - attn_um2, 0.0)

    # ── block list (name, area_um2, colour, group) ─────────────────────────
    LOGIC_C, FP16_C, ROM_C, SRAM_C = "#cfe8ff", "#ffd9a8", "#d7f0c2", "#f4c7d0"
    blocks = []
    for name in ("Backbone (model.0-8)", NECK, "Detect head (model.23)"):
        if reg_um2.get(name):
            blocks.append((name, reg_um2[name], LOGIC_C))
    blocks.append(("Attention fp16 MAC\n(flash_attn macw×1024 + fma×32)", attn_um2, FP16_C))
    blocks.append(("Weight ROM\n(3.6 MB, frozen weights)", rom_um2, ROM_C))
    blocks.append(("Activation SRAM\n(4.3 MB, 215 banks)", sram_um2, SRAM_C))

    total_um2 = sum(b[1] for b in blocks)
    total_mm2 = total_um2 / 1e6
    side = (total_mm2) ** 0.5          # square die of equal area, in mm

    # sort big→small for a stable, readable packing
    blocks.sort(key=lambda b: -b[1])
    rects = squarify([b[1] for b in blocks], 0, 0, side, side)

    # ── render ──────────────────────────────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, ax = plt.subplots(figsize=(8, 8.8))
    for (name, a, col), (rx, ry, rw, rh) in zip(blocks, rects):
        ax.add_patch(Rectangle((rx, ry), rw, rh, facecolor=col,
                               edgecolor="#333", linewidth=1.4))
        mm2 = a / 1e6
        ax.text(rx + rw/2, ry + rh/2,
                f"{name}\n{mm2:.2f} mm²  ({mm2/total_mm2*100:.0f}%)",
                ha="center", va="center", fontsize=8.5, wrap=True)
    ax.set_xlim(-0.05*side, 1.05*side); ax.set_ylim(-0.05*side, 1.08*side)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(
        f"YOLO26n accelerator — area-proportional floorplan ({pdk.name} 7nm)\n"
        f"die ≈ {total_mm2:.1f} mm² ({side:.2f}×{side:.2f} mm equiv.)  ·  "
        f"synth-area estimate, NOT place-and-route",
        fontsize=11)
    out = args.out or os.path.join(ROOT, "reports", f"FLOORPLAN_{pdk.name}.png")
    fig.tight_layout(); fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"die {total_mm2:.2f} mm²  ({side:.2f} mm/side)")
    for (name, a, _), (rx, ry, rw, rh) in zip(blocks, rects):
        print(f"  {a/1e6:6.2f} mm²  {rw:.2f}×{rh:.2f} mm  {name.splitlines()[0]}")
    print(f"wrote {os.path.relpath(out, ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
