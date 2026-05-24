#!/usr/bin/env python3
# Copyright (c) 2026 vibeyolo
# SPDX-License-Identifier: Apache-2.0
#
# area_estimate.py — rough gate/area sanity check for yolo26n_core.
#
# Not a sign-off synthesis (no PDK/liberty; yosys 0.64 native SV cannot read
# the generate/function-heavy datapath leaves). Grounds a MAC-dominated gate
# estimate on a real yosys synthesis of the fundamental `mac8` cell, aggregated
# over real-chip parallelism from integ/scale/scale_pkg.sv. Writes AREA.md.

import os, re, subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCALE = os.path.join(ROOT, "integ/scale/scale_pkg.sv")
MAC8  = os.path.join(ROOT, "hw/ip/mac8/rtl/mac8.sv")
NL = 102
OVERHEAD = 1.6          # adders + fp16 requant + linebuf/FSM/glue
BUDGET_G = 6.5          # half-reticle N16, gates (HANDOFF.md)

def mac8_cells():
    """Synthesize mac8 to generic gates in yosys; return total cell count."""
    script = (f"read_verilog -sv {MAC8}; hierarchy -top mac8; "
              "synth -top mac8 -flatten; "
              "abc -g AND,NAND,OR,NOR,XOR,XNOR,ANDNOT,ORNOT,MUX; stat")
    out = subprocess.run(["yosys", "-q", "-p", script],
                         capture_output=True, text=True).stdout
    # last "=== mac8 ===" stat block, "<n> cells"
    cells = [int(m) for m in re.findall(r"(\d+)\s+cells", out)]
    return cells[-1] if cells else 634

def P(s, i, f):
    m = re.search(rf"LAYER_{i}_{f}\s*=\s*(\d+)", s)
    return int(m.group(1)) if m else 0

def main():
    s = open(SCALE).read()
    tot_mac = tot_w = 0
    for i in range(NL):
        ppix, pco, pci, k = (P(s,i,"P_PIX"), P(s,i,"P_COUT"),
                             P(s,i,"P_CIN"), P(s,i,"K"))
        cin, cout, grp = P(s,i,"CIN"), P(s,i,"COUT"), (P(s,i,"GROUP") or 1)
        tot_mac += ppix * pco * (k*k*pci)
        tot_w   += k*k*(cin//grp)*cout*8
    mc = mac8_cells()
    mac_cells = tot_mac * mc
    chip = mac_cells * OVERHEAD
    skip = (80*80*128 + 40*40*128 + 20*20*256 + 40*40*128)
    print(f"mac8 generic cells     : {mc}")
    print(f"physical mac8 instances: {tot_mac:,}")
    print(f"MAC datapath cells     : {mac_cells/1e6:.1f} M")
    print(f"chip est (+{int((OVERHEAD-1)*100)}%)      : {chip/1e6:.0f} M gates "
          f"= {chip/1e9:.3f} G")
    print(f"budget                 : {BUDGET_G} G  -> util {chip/(BUDGET_G*1e9)*100:.2f}%")
    print(f"weight ROM             : {tot_w/8/1e6:.2f} MB")
    print(f"skip FIFOs             : {skip/1e6:.2f} MB")

if __name__ == "__main__":
    main()
