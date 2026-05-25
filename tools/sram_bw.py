#!/usr/bin/env python3
# Copyright (c) 2026 vibeyolo
# SPDX-License-Identifier: Apache-2.0
#
# sram_bw.py — inter-stage activation SRAM bandwidth / area estimate.
#
# Replaces the "magic memory" assumption (every activation bus is a whole
# C-channel pixel/cycle, zero-latency, unlimited ports) with what real compiled
# macros can deliver. For every on-die memory on the *activation path* it
# computes the required bandwidth (bits/cycle, read and write) and tiles real
# macros (tools/sram_model.py) to meet it, then reports total SRAM area, the
# worst fmax, and whether the throughput sign-off (tools/throughput.py) still
# holds once memory is no longer free.
#
# Scope (per the agreed plan): activation path only —
#   * 102 conv-stage line buffers
#   * 4 skip-FIFO banks (taps m.4/m.6/m.10/m.13 cv2 = conv #10/20/39/48)
#   * SPPF / 2× upsample frame stores, 2× attention Q/K/V/O, detect-head SRAMs
# Weight ROM is treated as dedicated wide ROM (NOT modeled here).
#
# Writes SRAM.md and prints a one-line verdict.

import os, re, sys, math, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sram_model as sm

ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BAL     = os.path.join(ROOT, "integ/generated/scale/scale_report_balanced.md")
T_FRAME = 100_000          # cyc/frame spec (matches throughput.py)
II_MAGIC = 82_951          # current bottleneck stage (throughput.py, free memory)
SKIP_TAPS = {10, 20, 39, 48}   # gen_core.py: m.4/m.6/m.10/m.13 cv2


# ───────────────────────── scale report ─────────────────────────
def parse_balanced(path):
    """-> list of dicts: idx,name,cin,cout,k,h,w,ppix,pcout,pcin."""
    rows = []
    for line in open(path):
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) != 12 or not cells[0].isdigit():
            continue
        h, w = map(int, cells[5].lower().split("x"))
        rows.append(dict(idx=int(cells[0]), name=cells[1],
                         cin=int(cells[2]), cout=int(cells[3]), k=int(cells[4]),
                         h=h, w=w, ppix=int(cells[6]),
                         pcout=int(cells[7]), pcin=int(cells[8]),
                         cycles=int(cells[10])))
    return rows


# ───────────────────────── memory enumeration ─────────────────────────
# A "memory" = (name, kind, depth_words, rd_bits/cyc, wr_bits/cyc, port).
# Bandwidth is bits accessed per cycle; the macro `port` flavor captures whether
# read and write must happen the same cycle.
def enumerate_mems(rows):
    mems = []

    # 1) conv-stage line buffers: K row-SRAMs, each W_in deep, Cin*8 wide.
    #    Per cycle: write 1 column (Cin B) to one row, read 1 column (Cin B)
    #    from each of K rows -> the K×K window is completed by shift registers.
    #    -> K banks of (depth=W_in, width=Cin*8, 1r1w). W_in≈W_out (stride-1
    #    lower bound; downsample layers store ~2× rows, immaterial vs skip FIFOs).
    for r in rows:
        w_in = r["w"]
        for _ in range(r["k"]):                      # K row banks
            mems.append(dict(
                name=f"linebuf L{r['idx']} (1 of {r['k']} rows)",
                kind="linebuf", depth=w_in,
                rd_bits=r["cin"]*8, wr_bits=r["cin"]*8, port="1r1w"))

    # 2) skip-FIFO banks. Full output feature map, raster fill-then-drain.
    #    As built (skip_buf) it is single-buffered -> R and W never overlap in
    #    one frame -> 1rw. (Frame-to-frame pipeline overlap would need 1r1w or a
    #    2× ping/pong; flagged in the report.)
    by_idx = {r["idx"]: r for r in rows}
    for idx in sorted(SKIP_TAPS):
        r = by_idx[idx]
        mems.append(dict(
            name=f"skip-FIFO tap conv#{idx} ({r['cout']}ch {r['h']}x{r['w']})",
            kind="skip", depth=r["h"]*r["w"],
            rd_bits=r["cout"]*8, wr_bits=r["cout"]*8, port="1rw"))

    # 3) block frame stores — analytic, clearly annotated (cf. throughput.py).
    #    (name, depth_words, word_bits, port, elem_bits)
    BLOCKS = [
        # SPPF /model.9: input frame store 20x20x128 int8 + 3 maxpool row bufs.
        ("SPPF frame store (20x20x128 i8)",      20*20,  128*8, "1r1w"),
        ("SPPF maxpool row buf x3 (k5, 20w)",       20,  128*8, "1r1w"),
        # upsample_concat: store the low-res map, re-read at 2× resolution.
        ("upsample11 store (20x20x256 i8)",      20*20,  256*8, "1r1w"),
        ("upsample14 store (40x40x128 i8)",      40*40,  128*8, "1r1w"),
        # attention PSA/A2C2f: Q,K,V,O fp16, HEADS2 N400 DIM_Q32 DIM_V64.
        # token-vector granularity per access; random tile access -> 2rw.
        ("attn10 Q  (2x400x32 fp16)",         2*400,   32*16, "2rw"),
        ("attn10 K  (2x400x32 fp16)",         2*400,   32*16, "2rw"),
        ("attn10 V  (2x400x64 fp16)",         2*400,   64*16, "2rw"),
        ("attn10 O  (2x400x64 fp16)",         2*400,   64*16, "2rw"),
        ("attn22 Q  (2x400x32 fp16)",         2*400,   32*16, "2rw"),
        ("attn22 K  (2x400x32 fp16)",         2*400,   32*16, "2rw"),
        ("attn22 V  (2x400x64 fp16)",         2*400,   64*16, "2rw"),
        ("attn22 O  (2x400x64 fp16)",         2*400,   64*16, "2rw"),
        # detect head: 8400 anchors. cls logits (80 fp16), decoded boxes
        # (4 fp16), reduced score (1 fp16). ~0.76 MB total.
        ("detect logits (8400x80 fp16)",       8400,   80*16, "1r1w"),
        ("detect boxes  (8400x4 fp16)",        8400,    4*16, "1r1w"),
        ("detect score  (8400x1 fp16)",        8400,    1*16, "1r1w"),
    ]
    for name, depth, wbits, port in BLOCKS:
        mems.append(dict(name=name, kind="block", depth=depth,
                         rd_bits=wbits, wr_bits=wbits, port=port))
    return mems


def tile(m):
    """Tile one enumerated memory; access width = max(rd,wr) bits/cycle."""
    width = max(m["rd_bits"], m["wr_bits"])
    t = sm.tile_memory(m["name"], width, m["depth"], m["port"])
    return t


# ───────────────────────── report ─────────────────────────
def main():
    ap = argparse.ArgumentParser(description="inter-stage activation SRAM estimate")
    ap.add_argument("--pdk", default=os.environ.get("PDK", "n16"),
                    choices=list(sm.PDKS), help="macro/density profile")
    ap.add_argument("--scale-report", default=BAL,
                    help="balanced scale report (swap for a YOLO variant)")
    args = ap.parse_args()
    sm.set_pdk(args.pdk); sm._sync()
    CLKG = sm.ACTIVE.clk_hz / 1e9            # chip clock in GHz, for labels
    clk_ok = lambda f: "✅" if f >= sm.ACTIVE.clk_hz else "❌"

    rep = args.scale_report
    if not os.path.exists(rep):
        sys.exit(f"missing scale report: {rep}")
    rows = parse_balanced(rep)
    if rep == BAL:
        assert len(rows) == 102, f"expected 102 conv layers, got {len(rows)}"
    mems = enumerate_mems(rows)
    tiles = [(m, tile(m)) for m in mems]

    # ---- "magic" bandwidth currently assumed (every edge, full pixel/cyc) ----
    # Each conv edge moves Cout bytes/cyc; report the demand we silently assume.
    edge_bw = [(r["name"], r["cout"]*8) for r in rows]
    max_edge = max(edge_bw, key=lambda e: e[1])
    sum_edge_bits = sum(b for _, b in edge_bw)

    # ---- aggregates ----
    tot_cap_bits = sum(t.cap_bits for _, t in tiles)
    tot_macro    = sum(t.n_macro  for _, t in tiles)
    tot_area_mm2 = sum(t.area_mm2 for _, t in tiles)
    worst_fmax   = min(t.fmax_hz for _, t in tiles)
    clk_viol     = [(m, t) for m, t in tiles if not t.meets_clk]

    by_kind = {}
    for m, t in tiles:
        k = by_kind.setdefault(m["kind"], dict(n=0, macro=0, area=0.0, cap=0))
        k["n"] += 1; k["macro"] += t.n_macro
        k["area"] += t.area_mm2; k["cap"] += t.cap_bits

    # ---- "width-capped" alternative: cost of NOT banking ----
    # If each memory is capped at ONE macro word (MAX_WIDTH bits/cyc), a wide
    # pixel write serializes. On a conv stage that stretches the per-output beat
    # by ceil(Cout*8 / MAX_WIDTH), so (upper bound) the stage's cycle count
    # scales by that factor. The capped II is the worst stage AFTER scaling —
    # computed per-stage from the report's own cycle column (no cross-multiply).
    capped = []
    for r in rows:
        factor = math.ceil(r["cout"]*8 / sm.MAX_WIDTH)
        capped.append((r["name"], r["cout"], factor, r["cycles"]*factor))
    worst = max(capped, key=lambda c: c[3])
    # blocks stay at their compiled (magic) cycles as a floor — their accesses
    # are narrow (fp16 token vectors), so the conv path dominates the cap.
    ii_capped  = max(worst[3], II_MAGIC)
    fps_capped = sm.CLK_HZ / ii_capped

    # ───────────────────────── emit SRAM.md ─────────────────────────
    L = []; P = L.append
    P("# YOLO26n — inter-stage activation SRAM bandwidth / area estimate\n")
    P("_Generated by `tools/sram_bw.py` using the placeholder macro model in "
      "`tools/sram_model.py`. Every macro number is an N16-ish PDK placeholder "
      "(`# PDK:` tags) to be replaced with MemoryCompiler output. Scope: "
      "activation path only — line buffers, skip FIFOs, block frame stores, "
      "attention & detect SRAMs. Weight ROM is treated as dedicated wide ROM "
      "and is **not** modeled here._\n")

    P(f"_PDK profile: **{sm.ACTIVE.name}** · chip clock {CLKG:g} GHz · SRAM "
      f"{sm.ACTIVE.um2_per_bit:g} µm²/bit · macro {sm.ACTIVE.max_width} b × "
      f"{sm.ACTIVE.max_depth} words._\n")

    P("## 1. The bandwidth we currently assume is free\n")
    P("Each activation bus carries a whole pixel (`logic [C-1:0][7:0]`), one "
      f"pixel/cycle, zero latency, unlimited ports. At {CLKG:g} GHz that is real "
      "traffic a memory must sustain:\n")
    P("| metric | value |")
    P("|--------|------:|")
    P(f"| widest single edge | `{max_edge[0]}` = **{max_edge[1]//8} B/cyc** "
      f"= {max_edge[1]/8*sm.CLK_HZ/1e9:.0f} GB/s |")
    P(f"| sum over all {len(edge_bw)} conv edges | {sum_edge_bits//8:,} B/cyc "
      f"= {sum_edge_bits/8*sm.CLK_HZ/1e12:.2f} TB/s |")
    P(f"| a real macro port | {sm.MAX_WIDTH} bits/cyc "
      f"= {sm.MAX_WIDTH//8} B/cyc |")
    P(f"\n> The widest edge alone demands "
      f"{math.ceil(max_edge[1]/sm.MAX_WIDTH)}× a single macro word — so it must "
      f"be **banked**, or it serializes.\n")

    P("## 2. If we meet the bandwidth by banking (area-priority)\n")
    P("Tile enough macros to keep 1 access/cycle on every memory. II is "
      "**unchanged** (memory never stalls the datapath); the cost is SRAM area "
      "and macro count.\n")
    P("| metric | value | note |")
    P("|--------|------:|------|")
    P(f"| total activation-SRAM capacity | {tot_cap_bits/8/1e6:.2f} MB | |")
    P(f"| total macro instances | {tot_macro:,} | |")
    P(f"| total SRAM area | **{tot_area_mm2:.2f} mm²** | placeholder density |")
    P(f"| worst macro fmax | {worst_fmax/1e9:.2f} GHz | "
      f"{clk_ok(worst_fmax)} vs {CLKG:g} GHz |")
    P(f"| memories below {CLKG:g} GHz | {len(clk_viol)} | "
      f"{'✅ none' if not clk_viol else '❌ see below'} |")
    P(f"| throughput verdict | II = {II_MAGIC:,} cyc held | "
      f"{sm.CLK_HZ/II_MAGIC:,.0f} FPS @ {CLKG:g} GHz |\n")

    P("### By category\n")
    P("| kind | #mems | #macros | capacity | area (mm²) |")
    P("|------|------:|--------:|---------:|-----------:|")
    for k in ("linebuf", "skip", "block"):
        d = by_kind[k]
        P(f"| {k} | {d['n']} | {d['macro']:,} | {d['cap']/8/1e6:.2f} MB | "
          f"{d['area']:.2f} |")
    P("")

    P("## 3. If we DON'T bank (width-capped — cost of refusing area)\n")
    P(f"Cap every memory at a single {sm.MAX_WIDTH}-bit macro word. Wide pixel "
      f"writes then serialize; the worst-hit conv `{worst[0]}` (Cout {worst[1]}) "
      f"stretches **{worst[2]}×** ({worst[3]:,} cyc), becoming the new "
      f"bottleneck:\n")
    P("| | banked (area-priority) | width-capped |")
    P("|--|--:|--:|")
    P(f"| II (cyc) | {II_MAGIC:,} | {ii_capped:,} |")
    P(f"| FPS @ {CLKG:g} GHz | {sm.CLK_HZ/II_MAGIC:,.0f} | {fps_capped:,.0f} |")
    P(f"| meets T_FRAME={T_FRAME:,} | ✅ | "
      f"{'✅' if ii_capped<=T_FRAME else '❌'} |\n")
    P("> This is the lever: spend SRAM area (banking) to keep throughput, or "
      "save area and lose it. Given the ~1 % gate budget (`AREA.md`), banking "
      "is the obvious choice — but it is no longer free.\n")

    if clk_viol:
        P("## ⚠️ Memories below target clock\n")
        for m, t in clk_viol:
            P(f"- {t.name}: fmax {t.fmax_hz/1e9:.2f} GHz "
              f"(depth {t.depth_words}, {t.n_macro} macros)")
        P("")

    P("## 4. Largest individual memories\n")
    P("| memory | kind | cap | rd B/cyc | wr B/cyc | port | macros | area mm² | fmax |")
    P("|--------|------|----:|---------:|---------:|------|-------:|---------:|-----:|")
    for m, t in sorted(tiles, key=lambda mt: -mt[1].area_um2)[:15]:
        P(f"| {t.name} | {m['kind']} | {t.cap_bits/8/1024:.0f} KB | "
          f"{m['rd_bits']//8} | {m['wr_bits']//8} | {t.port} | {t.n_macro} | "
          f"{t.area_mm2:.4f} | {t.fmax_hz/1e9:.2f} |")
    P("")
    P("---")
    P(f"_PDK profile `{sm.ACTIVE.name}` from `tools/sram_model.py:PDKS`. Point a "
      "new profile at a different `.lib` + SRAM macro (e.g. a TSMC node) to "
      "regenerate — no other change needed._")

    out = os.path.join(ROOT, "reports", f"SRAM_{sm.ACTIVE.name}.md")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    open(out, "w").write("\n".join(L) + "\n")

    print(f"[pdk={sm.ACTIVE.name} clk={CLKG:g}GHz]")
    print(f"widest edge        = {max_edge[1]//8} B/cyc "
          f"({max_edge[1]/8*sm.CLK_HZ/1e9:.0f} GB/s)  [{max_edge[0]}]")
    print(f"total activation SRAM = {tot_cap_bits/8/1e6:.2f} MB in "
          f"{tot_macro:,} macros = {tot_area_mm2:.2f} mm^2")
    print(f"worst macro fmax   = {worst_fmax/1e9:.2f} GHz  "
          f"({len(clk_viol)} below {CLKG:g} GHz)")
    print(f"banked  : II {II_MAGIC:,} -> {sm.CLK_HZ/II_MAGIC:,.0f} FPS (held)")
    print(f"capped  : II {ii_capped:,} -> {fps_capped:,.0f} FPS "
          f"({'meets' if ii_capped<=T_FRAME else 'MISSES'} T_FRAME)")
    print(f"wrote {os.path.relpath(out, ROOT)}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
