# Area / gate-count sanity check — yolo26n_core

Rough gate budget check for the wired structural core. **Not a sign-off
synthesis** (no liberty/PDK available; yosys 0.64 native SV cannot read the
generate/function-heavy datapath leaves such as `dotN`/`conv_layer`). Instead:
synthesize the fundamental MAC cell to generic gates in yosys, then aggregate
over the real-chip parallelism from `integ/scale/scale_pkg.sv`.

## Method

1. **`mac8` → generic gates (yosys 0.64).**
   `synth -flatten; abc -g AND,NAND,OR,NOR,XOR,XNOR,ANDNOT,ORNOT,MUX`
   → **634 cells** per MAC (≈ 602 combinational gates + 32 flops).
2. **Physical MAC count** = Σ_layers `P_PIX · P_COUT · K² · P_CIN`
   over the 102 conv stages at *real-chip* parallelism (`scale_pkg.sv`, the
   synthesis target — not the DV-capped pkg used for lint elaboration).
3. Multiply, then add a flat **+60 %** for the per-`dotN` adder trees, the
   fp16 requant path (`i32_to_fp16`/`fp16_fma`/`fp16_to_i8_sat`), line buffers,
   tile FSMs and the glue/blocks. (Order-of-magnitude, deliberately generous.)

## Result

| quantity | value |
|---|---|
| physical `mac8` instances (real chip) | **71,741** |
| gates per `mac8` (yosys generic) | ~634 cells |
| MAC datapath | ~45.5 M cells |
| + 60 % (adders / fp16 requant / ctl / glue) | **~73 M gates** |
| **half-reticle N16 budget** | **~6.5 G gates** |
| **budget utilisation** | **~1.1 %** (≈ 90× headroom) |

Memories (map to compiled macros, not flops — see synth audit):

| memory | size (int8) |
|---|---|
| weight ROM (all layers, K²·Cin/grp·Cout) | ~2.4 MB |
| skip FIFOs (m.4/m.6/m.10/m.13 cv2 taps) | ~1.33 MB |
| detect head SRAMs (logits/boxes/score) | ~0.76 MB |
| SPPF frame stores (4×) | ~0.2 MB |

## Conclusion

The accelerator is **MAC-dominated and lands at ~1 % of the half-reticle gate
budget** — exactly the "area is not a constraint, bonkers throughput" regime the
spec calls for (`HANDOFF.md`). Nowhere near over budget; the design is
gate-bound by neither logic nor on-die memory. The real floorplanning lever is
SRAM macro placement (a few MB), not gate count.

Regenerate: `python3 tools/area_estimate.py` (writes this file).
