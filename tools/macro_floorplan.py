#!/usr/bin/env python3
# Copyright (c) 2026 vibeyolo
# SPDX-License-Identifier: Apache-2.0
#
# macro_floorplan.py — top-level HIERARCHICAL macro floorplan of yolo26n_core.
#
# Full flat gate-level P&R of the whole chip is infeasible here (~73M gate-equiv,
# 215 SRAM macros, 4.3MB act / 3.6MB ROM -> RAM blows up). The industry-standard
# substitute is a hierarchical floorplan: every conv_stage / block / SRAM is a
# HARD MACRO sized by its silicon-area estimate, wired by the real gen_core
# dataflow (yolo26n_layers_pkg::L_SRC + the 6 named blocks + skip FIFOs), and
# placed by OpenROAD's macro placer to minimise dataflow wirelength.
#
# This module just builds the model + emits the artifacts:
#   pnr/top/blocks.lef    synthetic LEF abstract per macro (CLASS BLOCK)
#   pnr/top/top_macro.v   netlist: macro instances + 1-bit-per-edge dataflow nets
#   pnr/top/pnr.tcl       OpenROAD: floorplan + macro_placement + write_def
# Then: openroad -exit pnr/top/pnr.tcl ; tools/render_def.py (macro mode).
#
#   python3 tools/macro_floorplan.py [--util 0.55] [--pdk asap7]

import os, re, sys, math, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sram_model as sm
import sram_bw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG  = os.path.join(ROOT, "integ/generated/core/yolo26n_layers_pkg.sv")

# ── area model (identical basis to tools/chip_area.py + tools/floorplan.py) ──
A8_UM2     = 62.9      # mac8 on ASAP7 RVT (yosys+abc liberty), µm²
OVERHEAD   = 1.6       # per-conv adders + requant + ctl/glue (absorbs residuals)
LOGIC_UTIL = 0.65      # synth-cell -> physical-die utilisation
MACW_UM2   = 700.3     # fp16_macw (5-stage, wide acc) measured, µm²
FFMA_UM2   = 592.8     # fp16_fma  (5-stage) measured, µm²
# flash_attn per instance: BR*BC=512 macw (score+PV) + BR=16 fma (l-update).
ATTN_LOGIC_UM2 = (512 * MACW_UM2 + 16 * FFMA_UM2) / LOGIC_UTIL


def parse_pkg():
    """Pull the L_* arrays out of the generated layers package."""
    txt = open(PKG).read()
    def arr(name):
        m = re.search(rf"L_{name}\s*\[NL\]\s*=\s*'\{{([^}}]*)\}}", txt)
        return [int(x) for x in m.group(1).split(",")]
    return {k: arr(k) for k in
            ("CIN","COUT","K","STRIDE","PAD","HIN","WIN","PCOUT","PCIN","SRC")}


def conv_logic_um2(L, i):
    macs = L["PCOUT"][i] * L["K"][i]**2 * L["PCIN"][i]      # ppix folded below
    return macs * A8_UM2 * OVERHEAD / LOGIC_UTIL


def build(pdk):
    L = parse_pkg()
    NL = len(L["CIN"])
    rows = sram_bw.parse_balanced(sram_bw.BAL)
    ppix = {r["idx"]: r["ppix"] for r in rows}

    # SRAM tiles, attributed to their owning macro by name.
    tiles = [(m, sram_bw.tile(m)) for m in sram_bw.enumerate_mems(rows)]
    sram_by_owner = {}
    for m, t in tiles:
        n = m["name"]
        if m["kind"] == "linebuf":
            owner = f"conv{int(re.search(r'L(\d+)', n).group(1))}"
        elif m["kind"] == "skip":
            owner = f"skip{int(re.search(r'conv#(\d+)', n).group(1))}"
        elif n.startswith("SPPF"):        owner = "sppf"
        elif n.startswith("upsample11"):  owner = "ups11"
        elif n.startswith("upsample14"):  owner = "ups14"
        elif n.startswith("attn10"):      owner = "attn10"
        elif n.startswith("attn22"):      owner = "attn22"
        elif n.startswith("detect"):      owner = "detect"
        else:                             owner = "misc"
        sram_by_owner[owner] = sram_by_owner.get(owner, 0.0) + t.area_um2

    macros = {}   # name -> dict(area_um2, region, kind, logic, rom, sram)
    def add(name, region, kind, logic=0.0, rom=0.0, sram=0.0):
        macros[name] = dict(region=region, kind=kind, logic=logic, rom=rom,
                            sram=sram, area_um2=logic+rom+sram)

    def region(i):
        if i <= 8:  return "backbone"
        if i <= 95: return "neck"          # incl. PSA attn + neck convs
        return "head"

    # 102 conv macros: logic (×ppix) + own weight ROM + own line buffers.
    for i in range(NL):
        logic = conv_logic_um2(L, i) * ppix[i]
        rom   = L["K"][i]**2 * L["CIN"][i] * L["COUT"][i] * 8 * pdk.rom_um2_per_bit
        sram  = sram_by_owner.get(f"conv{i}", 0.0)
        add(f"conv{i}", region(i), "conv", logic, rom, sram)

    # 2 attention arrays (fp16) — logic carved out explicitly + Q/K/V/O SRAM.
    add("attn10", "neck", "attn", logic=ATTN_LOGIC_UM2, sram=sram_by_owner.get("attn10",0))
    add("attn22", "neck", "attn", logic=ATTN_LOGIC_UM2, sram=sram_by_owner.get("attn22",0))

    # SRAM/buffer-dominated blocks (logic absorbed in conv overhead).
    add("sppf",  "neck", "block", sram=sram_by_owner.get("sppf",0))
    add("ups11", "neck", "block", sram=sram_by_owner.get("ups11",0))
    add("ups14", "neck", "block", sram=sram_by_owner.get("ups14",0))
    add("detect","head", "block", logic=24*FFMA_UM2/LOGIC_UTIL,    # box_affine+decode
        sram=sram_by_owner.get("detect",0))

    # 4 skip-FIFO banks (the big distributed activation SRAM).
    for tap in (10, 20, 39, 48):
        add(f"skip{tap}", "neck", "skip", sram=sram_by_owner.get(f"skip{tap}",0))

    # ── dataflow edges (gen_core connectivity) ──────────────────────────────
    edges = []
    for i in range(NL):
        s = L["SRC"][i]
        if s >= 0: edges.append((f"conv{s}", f"conv{i}"))
    # named blocks (from yolo26n_core_impl.sv port connections)
    edges += [("conv31","sppf"),
              ("skip39","ups11"), ("skip20","ups11"),
              ("skip48","ups14"), ("skip10","ups14"),
              ("conv34","attn10"), ("conv35","attn10"),
              ("conv88","attn22"), ("conv89","attn22"),
              ("conv64","detect"), ("conv70","detect")]
    # skip taps fan out from their producer conv
    edges += [("conv10","skip10"), ("conv20","skip20"),
              ("conv39","skip39"), ("conv48","skip48")]
    # block outputs feed the next conv on the spine (approx insertion points)
    edges += [("sppf","conv32"), ("attn10","conv36"), ("attn22","conv90"),
              ("ups11","conv57"), ("ups14","conv49")]
    edges = [(a,b) for a,b in edges if a in macros and b in macros]
    return macros, edges, L


def _sane(n):
    return re.sub(r"[^A-Za-z0-9_]", "_", n)


def flow_pos(macros, L):
    """1-D dataflow coordinate per macro (topological), for snake placement."""
    pos = {}
    for n in macros:
        if n.startswith("conv"):  pos[n] = int(n[4:])
        elif n == "sppf":  pos[n] = 31.5
        elif n == "attn10":pos[n] = 35.5
        elif n == "attn22":pos[n] = 89.5
        elif n == "ups11": pos[n] = 40.5
        elif n == "ups14": pos[n] = 48.5
        elif n.startswith("skip"): pos[n] = int(n[4:]) + 0.4
        elif n == "detect":pos[n] = 102
        else: pos[n] = 50
    return pos


def place(macros, geom, L, util):
    """Skyline bottom-left packing in dataflow order — compact (~0.8+ util) so the
    die stays small (top-level route grid scales with die AREA). Macros are tried
    in topological order, each dropped at the lowest-then-leftmost skyline slot
    within the target width, so the spine still flows roughly left→right/up."""
    order = sorted(macros, key=lambda n: flow_pos(macros, L)[n])
    tot   = sum(macros[n]["area_um2"] for n in macros)
    Wt    = math.sqrt(tot/util)                       # target die width (µm)
    GAP   = 1.0
    sky   = [(0.0, Wt, 0.0)]                           # (x0, x1, height) segments
    pos   = {}
    def fit_at(i, w):
        """lowest top if rect of width w starts at segment i; None if runs off."""
        x0 = sky[i][0]; need = x0 + w
        if need > Wt + 1e-6: return None
        top = 0.0; j = i
        while j < len(sky) and sky[j][0] < need - 1e-6:
            top = max(top, sky[j][2]); j += 1
        return top
    def add_rect(x, y, w, h):
        x1 = x + w; ny = y + h
        ns = []
        for (s0, s1, sh) in sky:
            if s1 <= x + 1e-6 or s0 >= x1 - 1e-6:     # untouched segment
                ns.append((s0, s1, sh)); continue
            if s0 < x - 1e-6: ns.append((s0, x, sh))  # left remainder
            ns.append((max(s0, x), min(s1, x1), ny))  # covered → raised
            if s1 > x1 + 1e-6: ns.append((x1, s1, sh))# right remainder
        # merge equal-height neighbours
        m = [ns[0]]
        for seg in ns[1:]:
            if abs(seg[2]-m[-1][2])<1e-6 and abs(seg[0]-m[-1][1])<1e-6:
                m[-1]=(m[-1][0],seg[1],seg[2])
            else: m.append(seg)
        sky[:] = m
    for n in order:
        w, h = geom[n][0]+GAP, geom[n][1]+GAP
        best = None
        for i in range(len(sky)):
            top = fit_at(i, w)
            if top is None: continue
            key = (top, sky[i][0])                    # lowest, then leftmost
            if best is None or key < best[0]:
                best = (key, sky[i][0], top)
        if best is None:                              # overflow width → new col
            x = 0.0; top = max(s[2] for s in sky)
        else:
            _, x, top = best
        pos[n] = (x, top)
        add_rect(x, top, w, h)
    W = max((pos[n][0]+geom[n][0]) for n in pos)
    H = max((pos[n][1]+geom[n][1]) for n in pos)
    return pos, W, H, order


def sp_place(macros, geom, edges, util, iters=4000, seed=1):
    """Wirelength-driven floorplan via sequence-pair simulated annealing. A
    sequence pair (X,Y) encodes relative placement with NO overlap; we anneal it
    to minimise total net HPWL (+ a mild die-area term). This keeps dataflow
    neighbours adjacent — the skyline packer did not, so adjacent convs ended up
    mm apart. Returns the same (pos, W, H, order) tuple as place()."""
    import random
    rng = random.Random(seed)
    names = list(macros)
    GAP = 2.0
    W = {n: geom[n][0]+GAP for n in names}
    H = {n: geom[n][1]+GAP for n in names}

    def decode(X, Y):
        xi = {m:i for i,m in enumerate(X)}; yi = {m:i for i,m in enumerate(Y)}
        x = {}
        for b in X:                                   # X order → preds first
            xb = 0.0
            for a in X:
                if a is b: continue
                if xi[a] < xi[b] and yi[a] < yi[b]:    # a left of b
                    if x[a] + W[a] > xb: xb = x[a] + W[a]
            x[b] = xb
        y = {}
        for b in Y:                                   # Y order → preds first
            yb = 0.0
            for a in Y:
                if a is b: continue
                if xi[a] > xi[b] and yi[a] < yi[b]:    # a below b
                    if y[a] + H[a] > yb: yb = y[a] + H[a]
            y[b] = yb
        return x, y
    def cost(x, y):
        cx = {n: x[n]+W[n]/2 for n in names}; cy = {n: y[n]+H[n]/2 for n in names}
        hpwl = sum(abs(cx[a]-cx[b])+abs(cy[a]-cy[b]) for a,b in edges)
        Wd = max(x[n]+W[n] for n in names); Hd = max(y[n]+H[n] for n in names)
        # HPWL-dominated: tiny deadspace regulariser only (sequence-pair already
        # packs tightly); wirelength drives the anneal.
        dead = Wd*Hd - sum(W[n]*H[n] for n in names)
        return hpwl + 0.02*dead, Wd, Hd
    def move(X, Y):
        nX, nY = X[:], Y[:]; r = rng.random(); L = len(nX)
        if r < 0.55:                                  # insertion in both seqs
            i = rng.randrange(L); j = rng.randrange(L)
            m = nX.pop(i); nX.insert(j, m)
            m = nY.pop(rng.randrange(L)); nY.insert(rng.randrange(L), m)
        elif r < 0.75: i,j = rng.randrange(L), rng.randrange(L); nX[i],nX[j]=nX[j],nX[i]
        elif r < 0.95: i,j = rng.randrange(L), rng.randrange(L); nY[i],nY[j]=nY[j],nY[i]
        else:          i,j = rng.randrange(L), rng.randrange(L); nX[i],nX[j]=nX[j],nX[i]; nY[i],nY[j]=nY[j],nY[i]
        return nX, nY

    X = names[:]; Y = names[:]; rng.shuffle(X); rng.shuffle(Y)
    x, y = decode(X, Y); c, Wd, Hd = cost(x, y)
    best = (c, X[:], Y[:])
    T0 = c/len(names); Tf = T0*1e-3
    cool = (Tf/T0)**(1.0/max(iters,1))                # smooth cool over all iters
    T = T0
    for _ in range(iters):
        nX, nY = move(X, Y)
        x, y = decode(nX, nY); nc, nWd, nHd = cost(x, y)
        if nc < c or rng.random() < math.exp(-(nc-c)/max(T,1e-9)):
            X, Y, c, Wd, Hd = nX, nY, nc, nWd, nHd
            if c < best[0]: best = (c, X[:], Y[:])
        T *= cool
    _, X, Y = best
    x, y = decode(X, Y)
    pos = {n:(x[n], y[n]) for n in names}
    Wd = max(x[n]+W[n] for n in names); Hd = max(y[n]+H[n] for n in names)
    return pos, Wd, Hd, names


def write_def(macros, geom, pos, W, H, edges, path):
    UNIT = 1000
    nm = lambda v: int(round(v*UNIT))
    o = [f"VERSION 5.8 ;", "DIVIDERCHAR \"/\" ;", "BUSBITCHARS \"[]\" ;",
         "DESIGN yolo26n_top ;", f"UNITS DISTANCE MICRONS {UNIT} ;",
         f"DIEAREA ( 0 0 ) ( {nm(W)} {nm(H)} ) ;", ""]
    o.append(f"COMPONENTS {len(macros)} ;")
    for n in macros:
        x, y = pos[n]
        o.append(f"    - u_{n} {n} + PLACED ( {nm(x)} {nm(y)} ) N ;")
    o += ["END COMPONENTS", "", "END DESIGN"]
    open(path,"w").write("\n".join(o)+"\n")


def write_def_routable(macros, geom, pos, W, H, edges, path):
    """Full DEF (placed macros + 2-pin dataflow nets) for OpenROAD global_route.
    Clock/reset omitted — we route only the 121 inter-block dataflow edges."""
    UNIT = 1000; nm = lambda v: int(round(v*UNIT))
    o = ["VERSION 5.8 ;", "DIVIDERCHAR \"/\" ;", "BUSBITCHARS \"[]\" ;",
         "DESIGN yolo26n_top ;", f"UNITS DISTANCE MICRONS {UNIT} ;",
         f"DIEAREA ( 0 0 ) ( {nm(W)} {nm(H)} ) ;", ""]
    o.append(f"COMPONENTS {len(macros)} ;")
    for n in macros:
        x,y = pos[n]
        o.append(f"    - u_{n} {n} + PLACED ( {nm(x)} {nm(y)} ) N ;")
    o += ["END COMPONENTS", ""]
    o.append(f"NETS {len(edges)} ;")
    for k,(a,b) in enumerate(edges):
        o.append(f"    - e{k} ( u_{a} o_e{k} ) ( u_{b} i_e{k} ) ;")
    o += ["END NETS", "", "END DESIGN"]
    open(path,"w").write("\n".join(o)+"\n")


_REGION_COL = {"backbone":"#4e79a7", "neck":"#59a14f", "head":"#e15759"}
_KIND_COL   = {"conv":"#6aa6d8","attn":"#f28e2b","block":"#b07aa1",
               "skip":"#edc948"}

def render(macros, geom, pos, W, H, edges, out, color="kind"):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, FancyArrow
    from matplotlib.lines import Line2D
    cmap = _KIND_COL if color=="kind" else _REGION_COL
    key  = (lambda m: m["kind"]) if color=="kind" else (lambda m: m["region"])
    fig, ax = plt.subplots(figsize=(11, 11*H/W + 1.2))
    ax.add_patch(Rectangle((0,0), W, H, facecolor="#fafafa", edgecolor="#222", lw=1.5))
    # dataflow edges (light)
    for a,b in edges:
        if a in pos and b in pos:
            ax0,ay0 = pos[a][0]+geom[a][0]/2, pos[a][1]+geom[a][1]/2
            bx0,by0 = pos[b][0]+geom[b][0]/2, pos[b][1]+geom[b][1]/2
            ax.plot([ax0,bx0],[ay0,by0], color="#bbb", lw=0.4, zorder=1)
    seen={}
    for n,m in macros.items():
        w,h = geom[n]; x,y = pos[n]
        col = cmap.get(key(m), "#ccc"); seen[key(m)]=col
        ax.add_patch(Rectangle((x,y), w, h, facecolor=col, edgecolor="#333",
                               lw=0.6, zorder=2))
        if w>55 and h>30:
            ax.text(x+w/2, y+h/2, n.replace("conv","c"), ha="center",
                    va="center", fontsize=6.5, zorder=3)
    ax.set_xlim(-20, W+20); ax.set_ylim(-20, H+40); ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"yolo26n_core — top-level macro floorplan ({len(macros)} hard "
                 f"macros, dataflow-ordered)\n{W/1000:.2f}×{H/1000:.2f} mm "
                 f"core · {sum(m['area_um2'] for m in macros.values())/1e6:.1f} "
                 f"mm² cells · colour = {color}", fontsize=12)
    ax.legend(handles=[Line2D([0],[0],marker='s',color='w',markerfacecolor=c,
              markersize=11,label=l) for l,c in sorted(seen.items())],
              loc="upper center", bbox_to_anchor=(0.5,-0.01), ncol=4,
              frameon=False, fontsize=10)
    fig.tight_layout(); fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"  wrote {os.path.relpath(out, ROOT)}")


def emit(macros, edges, pdk, util, sfx=""):
    """Write pnr/top/{blocks<sfx>.lef, top_macro<sfx>.v, pnr<sfx>.tcl}. The LEF
    pin names (o_e<k>/i_e<k>) are numbered off THIS edge list, so they match the
    routable DEF emitted from the same list — required for subset routes."""
    work = os.path.join(ROOT, "pnr/top")
    os.makedirs(work, exist_ok=True)
    GRID = 0.005                                    # ASAP7 manufacturing grid (µm)
    snap = lambda v: round(round(v/GRID)*GRID, 3)

    # macro geometry: square-ish, area-true, snapped.
    geom = {}
    for n, m in macros.items():
        side = snap(max(math.sqrt(m["area_um2"]), 1.0))
        geom[n] = (side, side)

    # incident edges -> per-macro pin list (1 pin per edge + clk/rst).
    pins = {n: [] for n in macros}                  # name -> [(pin, net)]
    nets = []
    for k, (a, b) in enumerate(edges):
        net = f"e{k}"
        pa, pb = f"o_{net}", f"i_{net}"
        pins[a].append((pa, net)); pins[b].append((pb, net)); nets.append(net)

    # ── LEF ──────────────────────────────────────────────────────────────
    L = ["VERSION 5.8 ;", "BUSBITCHARS \"[]\" ;", "DIVIDERCHAR \"/\" ;", ""]
    for n, m in macros.items():
        w, h = geom[n]
        L += [f"MACRO {n}", "  CLASS BLOCK ;", "  ORIGIN 0 0 ;",
              f"  SIZE {w} BY {h} ;", "  SYMMETRY X Y R90 ;",
              f"  SITE asap7sc7p5t ;"]
        plist = [("clk_i","CLOCK","INPUT"), ("rst_ni","SIGNAL","INPUT")]
        plist += [(p, "SIGNAL", "INOUT") for p,_ in pins[n]]
        # pins on M4 (horizontal, pitch 0.048, offset 0.012) at on-grid y so the
        # router can access them; stride a few tracks apart down the left edge.
        ntrk = max(1, int((h-0.024)/0.048))
        for j,(p,use,dirn) in enumerate(plist):
            k  = 1 + (j % (ntrk-1) if ntrk>1 else 0)
            py = round(0.012 + k*0.048, 3)
            L += [f"  PIN {p}", f"    DIRECTION {dirn} ;", f"    USE {use} ;",
                  "    PORT", "      LAYER M4 ;",
                  f"        RECT 0.0 {round(py-0.012,3)} 0.144 {round(py+0.012,3)} ;",
                  "    END", f"  END {p}"]
        L += [f"END {n}", ""]
    open(os.path.join(work,f"blocks{sfx}.lef"),"w").write("\n".join(L))

    # ── netlist ──────────────────────────────────────────────────────────
    V = ["// GENERATED by tools/macro_floorplan.py — top macro connectivity.",
         "module yolo26n_top (input clk_i, input rst_ni);"]
    for net in nets: V.append(f"  wire {net};")
    for n, m in macros.items():
        conns = [".clk_i(clk_i)", ".rst_ni(rst_ni)"]
        conns += [f".{p}({net})" for p,net in pins[n]]
        V.append(f"  {n} u_{n} ( {', '.join(conns)} );")
    V.append("endmodule")
    open(os.path.join(work,f"top_macro{sfx}.v"),"w").write("\n".join(V)+"\n")

    # ── OpenROAD script ────────────────────────────────────────────────────
    tot = sum(m["area_um2"] for m in macros.values())
    die = math.sqrt(tot/util)                       # µm, square core
    pad = 5.0
    DX, DY = snap(die+2*pad), snap(die+2*pad)
    tcl = f"""# GENERATED by tools/macro_floorplan.py — top-level macro floorplan.
# Hard-macro placement of yolo26n_core (conv stages + blocks + skip FIFOs).
set PDK /home/marty/Documents/aif/pdk/asap7
read_lef     $PDK/lef/asap7_tech_1x.lef
read_lef     $PDK/lef/asap7_R_1x.lef
read_lef     pnr/top/blocks.lef
read_verilog pnr/top/top_macro.v
link_design  yolo26n_top

initialize_floorplan -die_area "0 0 {DX} {DY}" \\
                     -core_area "{pad} {pad} {snap(die+pad)} {snap(die+pad)}" \\
                     -site asap7sc7p5t
source $PDK/make_tracks.tcl

# place top-level I/O pins, then hard-macro placement (RTL-MP, mpl).
place_pins -hor_layers M4 -ver_layers M5
set_thread_count 8
rtl_macro_placer -halo_width 2.0 -target_util {util}
write_def pnr/top/yolo26n_top.def
report_design_area
exit
"""
    open(os.path.join(work,f"pnr{sfx}.tcl"),"w").write(tcl)
    print(f"  emitted pnr/top/{{blocks.lef,top_macro.v,pnr.tcl}}  "
          f"die {DX:.1f}×{DY:.1f} µm, {len(nets)} nets")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdk", default="asap7", choices=list(sm.PDKS))
    ap.add_argument("--util", type=float, default=0.55)
    ap.add_argument("--emit", action="store_true", help="write LEF/netlist/tcl")
    ap.add_argument("--top-frac", type=float, default=0.0,
                    help="keep only the top fraction of layers by dataflow order "
                         "(e.g. 0.25 = late-network partition: head + P5 neck)")
    ap.add_argument("--placer", choices=["sp","skyline"], default="sp",
                    help="sp = wirelength-driven sequence-pair SA (default); "
                         "skyline = density-only bottom-left pack")
    ap.add_argument("--sp-iters", type=int, default=4000)
    args = ap.parse_args()
    pdk = sm.set_pdk(args.pdk); sm._sync()

    macros, edges, L = build(pdk)
    sfx = ""
    if args.top_frac > 0:
        fp = flow_pos(macros, L)
        thr = sorted(fp.values())[int(len(fp)*(1-args.top_frac))]
        keep = {n for n in macros if fp[n] >= thr}
        macros = {n:m for n,m in macros.items() if n in keep}
        edges  = [(a,b) for a,b in edges if a in keep and b in keep]
        sfx = f"_top{int(args.top_frac*100)}"
        print(f"[subset {sfx}] kept {len(macros)} of-flow macros (pos>={thr})")
    tot = sum(m["area_um2"] for m in macros.values())
    by_region, by_kind = {}, {}
    for n, m in macros.items():
        by_region[m["region"]] = by_region.get(m["region"],0)+m["area_um2"]
        by_kind[m["kind"]]      = by_kind.get(m["kind"],0)+m["area_um2"]

    print(f"macros: {len(macros)}   edges: {len(edges)}   "
          f"total core-cell area: {tot/1e6:.2f} mm²")
    print("  by kind:   " + "  ".join(f"{k} {v/1e6:.2f}" for k,v in
          sorted(by_kind.items(), key=lambda x:-x[1])))
    print("  by region: " + "  ".join(f"{k} {v/1e6:.2f}" for k,v in
          sorted(by_region.items(), key=lambda x:-x[1])))
    big = sorted(macros.items(), key=lambda x:-x[1]["area_um2"])[:8]
    print("  largest:   " + "  ".join(f"{n} {m['area_um2']/1e6:.3f}" for n,m in big))
    # die at target utilisation
    die = math.sqrt(tot/args.util)/1000.0
    print(f"  -> die at util {args.util}: {die:.2f}×{die:.2f} mm "
          f"({tot/args.util/1e6:.2f} mm²)")

    if args.emit:
        emit(macros, edges, pdk, args.util, sfx)

    # geometry + dataflow-ordered placement + DEF + renders
    GRID = 0.005; snap = lambda v: round(round(v/GRID)*GRID, 3)
    geom = {n: (snap(max(math.sqrt(m["area_um2"]),1.0)),)*2
            for n,m in macros.items()}
    if args.placer == "sp":
        pos, W, Hd, order = sp_place(macros, geom, edges, args.util, args.sp_iters)
    else:
        pos, W, Hd, order = place(macros, geom, L, args.util)
    work = os.path.join(ROOT, "pnr/top"); os.makedirs(work, exist_ok=True)
    write_def(macros, geom, pos, W, Hd, edges, os.path.join(work,f"placed{sfx}.def"))
    write_def_routable(macros, geom, pos, W, Hd, edges,
                       os.path.join(work,f"placed_routable{sfx}.def"))
    # NOTE: do NOT global_route this at the top level. A multi-mm² die builds a
    # die-wide M2-M7 routing/congestion grid (~1e5 × 1e5 GCells) that OOMs the
    # machine — the same RAM wall as full-chip flat P&R, reached via the router.
    # Top-level wirelength is estimated analytically below (HPWL on the placement).
    # analytic dataflow wirelength (HPWL on centers) — safe substitute for the
    # top-level global_route that OOMs the machine.
    ctr = {n:(pos[n][0]+geom[n][0]/2, pos[n][1]+geom[n][1]/2) for n in macros}
    hpwl = sum(abs(ctr[a][0]-ctr[b][0])+abs(ctr[a][1]-ctr[b][1])
               for a,b in edges if a in ctr and b in ctr)
    print(f"  placed: {W/1000:.2f}×{Hd/1000:.2f} mm core "
          f"(util {sum(m['area_um2'] for m in macros.values())/(W*Hd):.2f}), "
          f"dataflow HPWL {hpwl/1000:.1f} mm over {len(edges)} edges")
    render(macros, geom, pos, W, Hd, edges,
           os.path.join(ROOT,f"reports/FLOORPLAN_top{sfx}_byregion.png"), "region")
    render(macros, geom, pos, W, Hd, edges,
           os.path.join(ROOT,f"reports/FLOORPLAN_top{sfx}_bykind.png"), "kind")
    return 0


if __name__ == "__main__":
    sys.exit(main())
