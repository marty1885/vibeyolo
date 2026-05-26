#!/usr/bin/env python3
# Copyright (c) 2026 vibeyolo
# SPDX-License-Identifier: Apache-2.0
#
# render_def.py — draw a placed-cell floorplan from an OpenROAD DEF + the cell
# LEF (real footprints). Flip-flops are highlighted so pipeline register banks
# are visible. This IS a real placement (post-OpenROAD), unlike floorplan.py's
# area-proportional estimate.
#
#   python3 tools/render_def.py <def> <cell.lef> [-o out.png] [--title T]

import re, sys, argparse


def lef_sizes(path):
    """MACRO name -> (w_um, h_um)."""
    sz, name = {}, None
    for ln in open(path):
        m = re.match(r"\s*MACRO\s+(\S+)", ln)
        if m: name = m.group(1); continue
        m = re.match(r"\s*SIZE\s+([\d.]+)\s+BY\s+([\d.]+)", ln)
        if m and name: sz[name] = (float(m.group(1)), float(m.group(2))); name = None
    return sz


def parse_def(path):
    die = None; comps = []; unit = 1000.0; incomp = False
    for ln in open(path):
        m = re.search(r"UNITS DISTANCE MICRONS\s+(\d+)", ln)
        if m: unit = float(m.group(1)); continue
        m = re.search(r"DIEAREA\s*\(\s*(-?\d+)\s+(-?\d+)\s*\)\s*\(\s*(-?\d+)\s+(-?\d+)\s*\)", ln)
        if m: die = tuple(int(x) for x in m.groups()); continue
        if ln.startswith("COMPONENTS"): incomp = True; continue
        if ln.startswith("END COMPONENTS"): incomp = False; continue
        if incomp:
            m = re.match(r"\s*-\s+(\S+)\s+(\S+).*?\(\s*(-?\d+)\s+(-?\d+)\s*\)\s+(\w+)", ln)
            if m:
                comps.append((m.group(1), m.group(2),
                              int(m.group(3)), int(m.group(4)), m.group(5)))
    return die, comps, unit


def family(macro):
    if re.search(r"DFF|SDF|LATCH|DLL", macro, re.I): return ("flip-flop", "#e8736b")
    if re.search(r"^BUF|CKBUF|BUFx", macro):          return ("buffer/CTS", "#7fb4e8")
    if re.search(r"^INV|^HB|^TIE", macro):            return ("inv/buf", "#cfe0f0")
    if re.search(r"AND|OR|MAJ|XOR|XNOR|AO|OA", macro):return ("logic (and/or/xor)", "#c2e0b0")
    return ("other", "#d9d9d9")

# distinct, readable palette for IP/instance coloring
_PALETTE = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759", "#76b7b2", "#edc948",
            "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac", "#86bcb6", "#d37295"]

def instance_owner(inst):
    """Top-of-hierarchy instance (the sub-IP), e.g. u_fma_out/_123_ -> u_fma_out.
    Cells with no hierarchy prefix (top-level glue) -> 'top glue'."""
    head = re.split(r"[/.]", inst, 1)[0]
    return head if head.startswith("u_") else "top glue"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("def_file"); ap.add_argument("lef_file")
    ap.add_argument("-o", "--out", default="floorplan_placed.png")
    ap.add_argument("--title", default="placed floorplan")
    ap.add_argument("--by-instance", action="store_true",
                    help="color cells by their owning sub-IP instance, not cell type")
    a = ap.parse_args()

    sizes = lef_sizes(a.lef_file)
    die, comps, unit = parse_def(a.def_file)
    dx0, dy0, dx1, dy1 = (d/unit for d in die)

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=(8.2, 8.6))
    ax.add_patch(Rectangle((dx0, dy0), dx1-dx0, dy1-dy0,
                           facecolor="white", edgecolor="#222", lw=1.6))
    # assign a stable colour per owning instance when --by-instance
    owners = sorted({instance_owner(c[0]) for c in comps}) if a.by_instance else []
    ocol = {o: _PALETTE[i % len(_PALETTE)] for i, o in enumerate(owners)}
    if "top glue" in ocol: ocol["top glue"] = "#d9d9d9"

    seen, nff = {}, 0
    for inst, macro, x, y, orient in comps:
        w, h = sizes.get(macro, (0.054, 0.270))
        if a.by_instance:
            lab = instance_owner(inst); col = ocol[lab]
        else:
            lab, col = family(macro)
            if lab == "flip-flop": nff += 1
        seen[lab] = col
        ax.add_patch(Rectangle((x/unit, y/unit), w, h,
                               facecolor=col, edgecolor="none"))
    ax.set_xlim(dx0-1, dx1+1); ax.set_ylim(dy0-1, dy1+3)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(f"{a.title}\n{(dx1-dx0):.1f}×{(dy1-dy0):.1f} µm die · "
                 f"{len(comps):,} cells ({nff} flip-flops)", fontsize=11)
    ax.legend(handles=[Line2D([0],[0], marker='s', color='w', markerfacecolor=c,
                              markersize=10, label=l) for l, c in sorted(seen.items())],
              loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=3, frameon=False,
              fontsize=9)
    fig.tight_layout(); fig.savefig(a.out, dpi=140, bbox_inches="tight")
    print(f"wrote {a.out}  ({len(comps):,} cells, {nff} FF, die {dx1-dx0:.1f}×{dy1-dy0:.1f} µm)")


if __name__ == "__main__":
    main()
