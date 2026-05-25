#!/usr/bin/env python3
# Copyright (c) 2026 vibeyolo
# SPDX-License-Identifier: Apache-2.0
#
# sram_model.py — parameterized SRAM macro cost model (PLACEHOLDER for a real
# foundry MemoryCompiler).
#
# The chip currently assumes "magic" memory: activation buses are full
# C-channel pixels per beat (logic [C-1:0][7:0]), read/written every cycle with
# zero latency and unlimited ports. Real on-die SRAM is compiled macros with a
# fixed word width, a small port count, a max depth/width per instance, and an
# access time that grows with depth. To deliver C bytes/cycle you must *tile*
# (bank) several macros side by side.
#
# This module is the single place those physical limits live. Every number
# below is a defensible N16-ish PLACEHOLDER, tagged `# PDK:` — swap each for the
# datasheet value when we actually run MemoryCompiler against the PDK. Nothing
# in the estimator hard-codes a macro fact; it all comes through tile_memory().

from dataclasses import dataclass, field
import math, os

# ───────────────────────── PDK profiles ─────────────────────────────────────
# Two profiles. `n16` is the original N16-ish PLACEHOLDER. `sky130` is REAL:
# every number is measured from the open SkyWater sky130A PDK pulled via volare
#   - um2_per_bit : sky130_sram_2kbyte_1rw1r_32x512_8 LEF SIZE 683.1×416.54 um
#                   = 284,558 um^2 / 16,384 bits = 17.37 um^2/bit (densest macro)
#   - max_width   : that macro's word = 32 bits (sky130 SRAM words are NARROW)
#   - max_depth   : 512 words
#   - fmax        : minimum_period 1.956 ns (clk0) -> 511 MHz
#   - mac8_um2    : yosys+abc map of hw/ip/mac8 to sky130_fd_sc_hd = 4510.58 um^2
# Density already amortizes per-macro periphery, so periph=0 and the fixed
# 1rw1r macro means port flavor does not change area (mult = 1 for all).

@dataclass
class PDK:
    name:        str
    um2_per_bit: float
    periph_um2:  float
    max_width:   int          # bits/word per macro
    max_depth:   int          # words per macro
    min_depth:   int
    fmax_hz:     float        # macro max clock (depth-independent if measured)
    clk_hz:      float        # target/achievable chip clock on this node
    mac8_um2:    float        # area of one mac8 (0 => synth from `liberty`, else gates)
    rom_um2_per_bit: float    # weight ROM density (read-only; usually denser)
    liberty:     str = ""     # stdcell .lib path; chip_area.py synths mac8 against it
    gate_um2:    float = 0.0  # fallback: area per generic gate (no liberty)
    # ── energy (for tools/power.py) — COARSE, scale ~ C·V²·node ─────────────
    e_mac_pj:    float = 0.0  # energy per int8 MAC (mult+acc)
    e_mem_pj_bit: float = 0.0 # energy per bit of SRAM/ROM access
    leak_mw_mm2: float = 0.0  # static leakage power density
    clk_frac:    float = 0.25 # clock-tree power as fraction of dynamic
    fmax_depth_model: bool = False   # True: fmax = 1/(t_base+t_log*log2(depth))
    t_base_ns:   float = 0.18
    t_log_ns:    float = 0.045
    port_area_mult: dict = field(default_factory=lambda: {"1rw":1.0,"1r1w":1.0,"2rw":1.0})

PDKS = {
    # PLACEHOLDER N16 (see git history for derivation).
    "n16": PDK(name="n16", um2_per_bit=0.14, periph_um2=1500.0,
               max_width=144, max_depth=4096, min_depth=64,
               fmax_hz=2.0e9, clk_hz=1.0e9, mac8_um2=0.0,
               # gate_um2: SYNTH cell area for a generic N16 cell. ~0.4 µm²
               # (NAND2 ~0.26 µm², mac8's adder/FF mix runs bigger). chip_area
               # then applies a P&R utilization derate. The OLD 0.066 came from
               # HANDOFF's 6.5 G-gate budget (~15 MGate/mm²) and was ~10× too
               # dense vs routed N16 (~1-2 MGate/mm²). PLACEHOLDER until a real
               # N16 .lib; sky130 is the grounded reference.
               rom_um2_per_bit=0.10, gate_um2=0.40, fmax_depth_model=True,
               e_mac_pj=0.10, e_mem_pj_bit=0.020, leak_mw_mm2=30.0,
               t_base_ns=0.18, t_log_ns=0.045,
               port_area_mult={"1rw":1.0,"1r1w":1.4,"2rw":2.0}),
    # REAL sky130 (open PDK via volare, all measured — see header). Set `liberty`
    # so chip_area.py re-synthesizes mac8 itself; mac8_um2 caches the result.
    "sky130": PDK(name="sky130", um2_per_bit=17.37, periph_um2=0.0,
                  max_width=32, max_depth=512, min_depth=8,
                  fmax_hz=511e6, clk_hz=511e6, mac8_um2=4510.58,
                  rom_um2_per_bit=17.37, fmax_depth_model=False,
                  # 130nm vs 16nm: ~5× V² (1.8 vs 0.8 V) × ~8× node C ≈ 40× energy.
                  e_mac_pj=4.0, e_mem_pj_bit=0.80, leak_mw_mm2=2.0,
                  liberty=os.environ.get("SKY130_HD_LIB",
                    "/home/marty/Documents/aif/pdk/sky130A/sky130A/libs.ref/"
                    "sky130_fd_sc_hd/lib/sky130_fd_sc_hd__tt_025C_1v80.lib")),
}

# Active PDK (override with env PDK=sky130 or set_pdk()).
ACTIVE = PDKS[os.environ.get("PDK", "n16")]

def set_pdk(name):
    global ACTIVE
    ACTIVE = PDKS[name]
    return ACTIVE

# Back-compat module-level names (reflect the ACTIVE pdk at import time; tools
# that switch PDK should read sm.ACTIVE.* or pass pdk= explicitly).
def _sync():
    g = globals(); p = ACTIVE
    g["UM2_PER_BIT"] = p.um2_per_bit; g["PERIPH_UM2"] = p.periph_um2
    g["MAX_DEPTH"]   = p.max_depth;   g["MAX_WIDTH"]  = p.max_width
    g["MIN_DEPTH"]   = p.min_depth;   g["CLK_HZ"]     = p.clk_hz
    g["PORT_AREA_MULT"] = p.port_area_mult
_sync()


@dataclass
class Tiling:
    """Physical realization of one logical memory."""
    name:        str
    width_bits:  int     # bits accessed per cycle per port (logical word)
    depth_words: int     # entries
    port:        str
    n_width:     int     # macros across (to make the word wide enough)
    n_depth:     int     # macros down  (to make it deep enough)
    n_macro:     int     # total instances = n_width * n_depth
    cap_bits:    int     # stored bits (width * depth)
    area_um2:    float
    fmax_hz:     float   # limited by the per-macro depth
    meets_clk:   bool    # fmax >= CLK_HZ
    notes:       str = ""

    @property
    def area_mm2(self):
        return self.area_um2 / 1e6


def macro_fmax(depth_words: int, pdk: PDK = None) -> float:
    pdk = pdk or ACTIVE
    if not pdk.fmax_depth_model:
        return pdk.fmax_hz                      # measured, depth-independent
    d = max(depth_words, 2)
    t_ns = pdk.t_base_ns + pdk.t_log_ns * math.log2(d)
    return 1e9 / t_ns


def tile_memory(name: str, width_bits: int, depth_words: int,
                port: str = "1rw", pdk: PDK = None) -> Tiling:
    """Tile a logical (width_bits × depth_words, `port`) memory into macros.

    width_bits is the per-cycle access word = the required bandwidth in
    bits/cycle on that port (the caller folds R+W concurrency into `port`).
    """
    pdk = pdk or ACTIVE
    assert port in pdk.port_area_mult, f"unknown port {port}"
    width_bits  = max(int(width_bits), 1)
    depth_words = max(int(depth_words), 1)

    n_width = math.ceil(width_bits  / pdk.max_width)
    n_depth = math.ceil(depth_words / pdk.max_depth)
    n_macro = n_width * n_depth

    cap_bits = width_bits * depth_words
    area = cap_bits * pdk.um2_per_bit * pdk.port_area_mult[port] \
         + n_macro * pdk.periph_um2

    per_macro_depth = math.ceil(depth_words / n_depth)
    fmax = macro_fmax(per_macro_depth, pdk)

    notes = []
    if depth_words < pdk.min_depth:
        notes.append(f"depth<{pdk.min_depth}: flops likely cheaper")
    if fmax < pdk.clk_hz:
        notes.append(f"fmax {fmax/1e9:.2f}GHz < clk")

    return Tiling(name, width_bits, depth_words, port, n_width, n_depth,
                  n_macro, cap_bits, area, fmax, fmax >= pdk.clk_hz,
                  "; ".join(notes))


if __name__ == "__main__":
    # quick self-check / illustration
    for w, d, p in [(512*8, 6400, "1r1w"),   # widest skip edge, 80x80x... -ish
                    (128*8, 1600, "1rw"),
                    (64,    4096, "1rw"),
                    (16,    256,  "1rw")]:
        t = tile_memory(f"w{w}xd{d}", w, d, p)
        print(f"{t.name:14s} {p:5s} {t.n_macro:3d} macros  "
              f"{t.area_mm2:7.4f} mm^2  fmax {t.fmax_hz/1e9:.2f}GHz  {t.notes}")
