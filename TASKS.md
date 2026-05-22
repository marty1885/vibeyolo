# vibeyolo — Building Block Tasks

Fixed-function int8 YOLO26n inference accelerator. Per-layer dataflow, no area cap, ASIC-friendly SystemVerilog in core-et's `prim_*` style.

## Design contract

- **Activations:** int8 symmetric (zero-point = 0), per-tensor.
- **Weights:** int8 symmetric, per-output-channel.
- **Accumulator:** int32.
- **Scale:** fp16 per output channel.
- **Bias:** fp16, added **post-scale** (FMA: `fp16(acc_i32) * scale_fp16 + bias_fp16`).
- **Activation (SiLU/Sigmoid):** int8→int8 256-entry LUT (LUT contents baked from layer's input/output scales).
- **Reset:** active-low, async-assert / sync-deassert (`rst_ni`).
- **Handshake:** ready/valid where streaming.
- **DV pattern:** SystemVerilog behavioral `*_ref.sv` golden alongside DUT; C++/Verilator testbench drives identical stimulus to both, compares outputs each cycle.

## Repo layout (mirrors core-et)

```
hw/ip/<block>/
  rtl/<block>.sv
  rtl/<block>_ref.sv     # SV behavioral golden
  dv/Makefile
  dv/<block>_test.cc
  README.md
mk/                       # verilator.mk, prim.mk
dv/common/sim_ctrl.h
```

Reuse from core-et (vendored or imported): `prim_fifo_sync`, `prim_ram_1p`, `prim_ram_2p`, `prim_clk_gate`, `prim_rst_sync`, `dft_pkg`, `ram_cfg_pkg`.

## Build order

**Phase 0 — template (sequential, by main):**
- [x] **#18** Repo skeleton (mk, dv/common, top Makefile)
- [x] **#1** `mac8` — int8×int8 + int32 acc. 71/71 checks, 100% line/branch/expr coverage. DV pattern established.

**Phase 1 — independent leaves (parallel agents, after mac8 green):**
- [x] **#3** `i32_to_fp16` — 177/177 checks, lint clean
- [x] **#7** `act_silu` — 25/25 checks across two scale variants, lint clean
- [x] **#8** `act_sigmoid` — 538/538 checks, two scale variants, lint clean
- [x] **#10** `maxpool_kxk` — K=2,3,5 all pass, 100% line/branch/expr (K=2 has one unreachable branch in tree odd-carry path, expected)
- [x] **#11** `linebuf_kxk` — 23/23 × 4 configs (K=3/W=4 C=1, K=5/W=8 C=1, K=3/W=8 C=16, K=3/W=8 C=64), lint clean. Multi-channel supported (one int8 lane per channel per cycle; `K*K*Channels` patch out). Caller must assert `clr_i` between frames (no continuous-flow).
- [x] **#12** `upsample2` — W=4 and W=8 both pass 18/18, lint clean
- [x] **#13** `concat_mux` — 22/22 × 2 configs, lint clean
- [x] **#16** `weight_rom` — 90/90 checks, self-contained ROM (no prim_ram_1p dep yet), lint clean. Includes `gen_hex.py` for DV stimulus files.
- [x] **#17** `skip_buf` — 58/58 × 2 configs (Depth=64, 512), 100% coverage, lint clean

**Phase 2 — dependent leaves:**
- [x] **#2** `dotN` — 1887/1887 (N=4) + 1894/1894 (N=16), instantiates mac8s + pipelined adder tree, lint clean
- [x] **#4** `fp16_fma` — 77/77 checks, lint clean, 0 mismatches across 100k random fp16 triples vs REF and C++ __int128 shadow.
- [x] **#5** `fp16_to_i8_sat` — 106/106, full 65536-input fp16 sweep, 0 mismatches, lint clean

**Phase 3 — composites:**
- [x] **#6** `requant` — `i32_to_fp16 → fp16_fma(scale, bias) → fp16_to_i8_sat`. 23/23 checks pass (directed edges + 30k random vectors), 0 DUT-vs-REF and 0 DUT-vs-shadow mismatches; ref is a flat independent SV behavioral model and the C++ shadow uses an `__int128` fp16-FMA path. Lint clean.
- [x] **#9** `add_rq` — composes `i32_to_fp16 × 2 → fp16_fma × 2 (per-side scale) → fp16_fma (add) → fp16_fma (out scale + bias) → fp16_to_i8_sat`. 16/16 checks, 0 DUT-vs-REF and 0 DUT-vs-shadow mismatches across ~12.4k stimulus (directed edges + 12k random with NaN/Inf/sub coverage), lint clean.
- [x] **#14** `softmax16` — 21/21 checks, lint clean. 1024-entry fp16 exp LUT (step 1/64) + 1024-entry fp16 reciprocal-mantissa LUT + fp16 add tree + per-lane fp16_fma. 12-cycle pipeline. Tolerance ≤32 fp16 ULP per lane vs both SV `_ref` and an independent C++ double-precision shadow; observed worst case 25 ULP over 5000 random ±8 vectors + 1000 tight-range ±2 vectors.
- [x] **#15** `box_decode` — DFL weighted-sum → xyxy with grid + stride. 22/22 checks pass (reset + directed: uniform d=7.5, one-hot at i=0 and i=15, strides {8,16,32}, 2000 random vectors, 500 peaked-distribution vectors). 7-cycle pipeline, throughput 1 box/cycle, built from `fp16_fma` (64 mul + add tree per side + 4 final fma) and `i32_to_fp16` (cx, cy, stride). Tolerance: **16 fp16 ULP** with a cancellation-aware absolute fallback `8 * fp16_step(max(|cx_center|, |d*stride|))` for the inherently cancellation-prone `cx_center − d*stride` formulation. `ref` vs `shadow` 0 ULP everywhere; worst `dut` vs `ref/shadow` ≈ 284 ULP, always within the absolute fallback. Lint clean.

## Validation rule

A block is **only** checked off when its DV passes (`make -C hw/ip/<block>/dv test` returns 0). No exceptions, no "looks right".

## Live task tracking

Task IDs above match TaskList IDs for this session. Status there is authoritative for in-flight work; this file is the durable plan.

## Generator / scaling path

Non-destructive generator work lives under `tools/layergen/` and emits artifacts
under `integ/generated/` so the hand-built `integ/layer_*` reference code is not
overwritten.

- [x] Create `tools/layergen/layergen.py` with balanced scale package generation.
- [x] Emit `integ/generated/scale/scale_pkg_balanced.sv` and report with generated `P_PIX` wrapper support: area proxy 22,096, worst real-chip layer 80,000 cycles, 0 target misses at `T_FRAME=100000`.
- [x] Emit generated L0 and L21 skeletons at `integ/generated/layer_0_model0/` and `integ/generated/layer_21_model7/`.
- [x] Lint generated L0 and L21 skeletons with `make -C integ/generated/layer_<N>/dv lint`.
- [x] Verify target-cycle control with generated reports for 120k/100k/80k/64k. Results are discrete because legal factors are divisors; 100k and 80k currently choose the same factors for most layers.
- [x] Fix grouped/depthwise conv MAC accounting in `tools/layergen` (`Cin/group`).
- [x] Add generated `P_PIX>1` wrapper support by instantiating multiple lockstep `conv_layer` lanes.
- [x] Generate ORT `extract.py` automatically for ordinary non-residual Conv+SiLU layers, piloted on L21.
- [x] Generate a generic tiled C++ sample driver automatically, piloted on L21 with `P_PIX=2`.
- [x] Promote generated L21 from lint-clean skeleton to ORT-validated layer: `make -C integ/generated/layer_21_model7/dv test` passes 6/6, avg cos 0.999290.
- [x] Generalize generated Conv+SiLU DV across L22-L25 and batch-run the next wave.
- [x] Fix generated extractor branch target selection for parallel cv1/cv2 branches; L24 originally targeted L23's activation until the extractor matched the current conv prefix.

## YOLO26n layer build progress (target T=100,000 cyc/frame)

Per-layer DV against ORT subgraph slice. Each layer composes the
verified `hw/ip/*` blocks under `scale_pkg::LAYER_<i>_*` parameters.

- [x] L0   /model.0    /model.0                                                conv 3->16, s2, k3, 320² → cosine 0.9988, 73 cyc
- [x] L1   /model.1    /model.1                                                conv 16->32, s2, k3, 160² → cosine 0.9987, 76 cyc
- [x] L2   /model.2    /model.2/cv1                                            conv 32->32, s1, k1, 160²
- [x] L3   /model.2    /model.2/m.0/cv1                                        conv 16->8, s1, k3, 160²
- [x] L4   /model.2    /model.2/m.0/cv2                                        conv 8->16, s1, k3, 160²
- [x] L5   /model.2    /model.2/cv2                                            conv 48->64, s1, k1, 160²
- [x] L6   /model.3    /model.3                                                conv 64->64, s2, k3, 80²
- [x] L7   /model.4    /model.4/cv1                                            conv 64->64, s1, k1, 80²
- [x] L8   /model.4    /model.4/m.0/cv1                                        conv 32->16, s1, k3, 80²
- [x] L9   /model.4    /model.4/m.0/cv2                                        conv 16->32, s1, k3, 80²
- [x] L10  /model.4    /model.4/cv2                                            conv 96->128, s1, k1, 80²
- [x] L11  /model.5    /model.5                                                conv 128->128, s2, k3, 40²
- [x] L12  /model.6    /model.6/cv1                                            conv 128->128, s1, k1, 40²
- [x] L13  /model.6    /model.6/m.0/cv1                                        conv 64->32, s1, k1, 40²
- [x] L14  /model.6    /model.6/m.0/cv2                                        conv 64->32, s1, k1, 40²
- [x] L15  /model.6    /model.6/m.0/m/m.0/cv1                                  conv 32->32, s1, k3, 40²
- [x] L16  /model.6    /model.6/m.0/m/m.0/cv2                                  conv 32->32, s1, k3, 40²
- [x] L17  /model.6    /model.6/m.0/m/m.1/cv1                                  conv 32->32, s1, k3, 40²
- [x] L18  /model.6    /model.6/m.0/m/m.1/cv2                                  conv 32->32, s1, k3, 40²
- [x] L19  /model.6    /model.6/m.0/cv3                                        conv 64->64, s1, k1, 40²
- [x] L20  /model.6    /model.6/cv2                                            conv 192->128, s1, k1, 40² → cosine 0.9998, 8/8 samples
- [x] L21  /model.7    /model.7                                                conv 128->256, s2, k3, 20² → generated DV cosine 0.9993, 6/6 samples
- [x] L22  /model.8    /model.8/cv1                                            conv 256->256, s1, k1, 20² → generated DV cosine 0.9991, 6/6 samples
- [x] L23  /model.8    /model.8/m.0/cv1                                        conv 128->64, s1, k1, 20² → generated DV cosine 0.9977, 6/6 samples
- [x] L24  /model.8    /model.8/m.0/cv2                                        conv 128->64, s1, k1, 20² → generated DV cosine 0.9996, 6/6 samples
- [x] L25  /model.8    /model.8/m.0/m/m.0/cv1                                  conv 64->64, s1, k3, 20² → generated DV cosine 0.9984, 6/6 samples
- [ ] L26  /model.8    /model.8/m.0/m/m.0/cv2                                  conv 64->64, s1, k3, 20²
- [ ] L27  /model.8    /model.8/m.0/m/m.1/cv1                                  conv 64->64, s1, k3, 20²
- [ ] L28  /model.8    /model.8/m.0/m/m.1/cv2                                  conv 64->64, s1, k3, 20²
- [ ] L29  /model.8    /model.8/m.0/cv3                                        conv 128->128, s1, k1, 20²
- [ ] L30  /model.8    /model.8/cv2                                            conv 384->256, s1, k1, 20²
- [ ] L31  /model.9    /model.9/cv1                                            conv 256->128, s1, k1, 20²
- [ ] L32  /model.9    /model.9/cv2                                            conv 512->256, s1, k1, 20²
- [ ] L33  /model.10   /model.10/cv1                                           conv 256->256, s1, k1, 20²
- [ ] L34  /model.10   /model.10/m/m.0/attn/qkv                                conv 128->256, s1, k1, 20²
- [ ] L35  /model.10   /model.10/m/m.0/attn/pe                                 conv 128->128, s1, k3, 20²
- [ ] L36  /model.10   /model.10/m/m.0/attn/proj                               conv 128->128, s1, k1, 20²
- [ ] L37  /model.10   /model.10/m/m.0/ffn/ffn.0                               conv 128->256, s1, k1, 20²
- [ ] L38  /model.10   /model.10/m/m.0/ffn/ffn.1                               conv 256->128, s1, k1, 20²
- [ ] L39  /model.10   /model.10/cv2                                           conv 256->256, s1, k1, 20²
- [ ] L40  /model.13   /model.13/cv1                                           conv 384->128, s1, k1, 40²
- [ ] L41  /model.13   /model.13/m.0/cv1                                       conv 64->32, s1, k1, 40²
- [ ] L42  /model.13   /model.13/m.0/cv2                                       conv 64->32, s1, k1, 40²
- [ ] L43  /model.13   /model.13/m.0/m/m.0/cv1                                 conv 32->32, s1, k3, 40²
- [ ] L44  /model.13   /model.13/m.0/m/m.0/cv2                                 conv 32->32, s1, k3, 40²
- [ ] L45  /model.13   /model.13/m.0/m/m.1/cv1                                 conv 32->32, s1, k3, 40²
- [ ] L46  /model.13   /model.13/m.0/m/m.1/cv2                                 conv 32->32, s1, k3, 40²
- [ ] L47  /model.13   /model.13/m.0/cv3                                       conv 64->64, s1, k1, 40²
- [ ] L48  /model.13   /model.13/cv2                                           conv 192->128, s1, k1, 40²
- [ ] L49  /model.16   /model.16/cv1                                           conv 256->64, s1, k1, 80²
- [ ] L50  /model.16   /model.16/m.0/cv1                                       conv 32->16, s1, k1, 80²
- [ ] L51  /model.16   /model.16/m.0/cv2                                       conv 32->16, s1, k1, 80²
- [ ] L52  /model.16   /model.16/m.0/m/m.0/cv1                                 conv 16->16, s1, k3, 80²
- [ ] L53  /model.16   /model.16/m.0/m/m.0/cv2                                 conv 16->16, s1, k3, 80²
- [ ] L54  /model.16   /model.16/m.0/m/m.1/cv1                                 conv 16->16, s1, k3, 80²
- [ ] L55  /model.16   /model.16/m.0/m/m.1/cv2                                 conv 16->16, s1, k3, 80²
- [ ] L56  /model.16   /model.16/m.0/cv3                                       conv 32->32, s1, k1, 80²
- [ ] L57  /model.16   /model.16/cv2                                           conv 96->64, s1, k1, 80²
- [ ] L58  /model.17   /model.17                                               conv 64->64, s2, k3, 40²
- [ ] L59  /model.23   /model.23/one2one_cv2.0/one2one_cv2.0.0                 conv 64->16, s1, k3, 80²
- [ ] L60  /model.23   /model.23/one2one_cv3.0/one2one_cv3.0.0/one2one_cv3.0.0.0 conv 64->64, s1, k3, 80²
- [ ] L61  /model.23   /model.23/one2one_cv2.0/one2one_cv2.0.1                 conv 16->16, s1, k3, 80²
- [ ] L62  /model.23   /model.23/one2one_cv3.0/one2one_cv3.0.0/one2one_cv3.0.0.1 conv 64->80, s1, k1, 80²
- [ ] L63  /model.19   /model.19/cv1                                           conv 192->128, s1, k1, 40²
- [ ] L64  /model.23   /model.23/one2one_cv2.0/one2one_cv2.0.2                 conv 16->4, s1, k1, 80²
- [ ] L65  /model.23   /model.23/one2one_cv3.0/one2one_cv3.0.1/one2one_cv3.0.1.0 conv 80->80, s1, k3, 80²
- [ ] L66  /model.19   /model.19/m.0/cv1                                       conv 64->32, s1, k1, 40²
- [ ] L67  /model.19   /model.19/m.0/cv2                                       conv 64->32, s1, k1, 40²
- [ ] L68  /model.23   /model.23/one2one_cv3.0/one2one_cv3.0.1/one2one_cv3.0.1.1 conv 80->80, s1, k1, 80²
- [ ] L69  /model.19   /model.19/m.0/m/m.0/cv1                                 conv 32->32, s1, k3, 40²
- [ ] L70  /model.23   /model.23/one2one_cv3.0/one2one_cv3.0.2                 conv 80->80, s1, k1, 80²
- [ ] L71  /model.19   /model.19/m.0/m/m.0/cv2                                 conv 32->32, s1, k3, 40²
- [ ] L72  /model.19   /model.19/m.0/m/m.1/cv1                                 conv 32->32, s1, k3, 40²
- [ ] L73  /model.19   /model.19/m.0/m/m.1/cv2                                 conv 32->32, s1, k3, 40²
- [ ] L74  /model.19   /model.19/m.0/cv3                                       conv 64->64, s1, k1, 40²
- [ ] L75  /model.19   /model.19/cv2                                           conv 192->128, s1, k1, 40²
- [ ] L76  /model.20   /model.20                                               conv 128->128, s2, k3, 20²
- [ ] L77  /model.23   /model.23/one2one_cv2.1/one2one_cv2.1.0                 conv 128->16, s1, k3, 40²
- [ ] L78  /model.23   /model.23/one2one_cv3.1/one2one_cv3.1.0/one2one_cv3.1.0.0 conv 128->128, s1, k3, 40²
- [ ] L79  /model.23   /model.23/one2one_cv2.1/one2one_cv2.1.1                 conv 16->16, s1, k3, 40²
- [ ] L80  /model.23   /model.23/one2one_cv3.1/one2one_cv3.1.0/one2one_cv3.1.0.1 conv 128->80, s1, k1, 40²
- [ ] L81  /model.22   /model.22/cv1                                           conv 384->256, s1, k1, 20²
- [ ] L82  /model.23   /model.23/one2one_cv2.1/one2one_cv2.1.2                 conv 16->4, s1, k1, 40²
- [ ] L83  /model.23   /model.23/one2one_cv3.1/one2one_cv3.1.1/one2one_cv3.1.1.0 conv 80->80, s1, k3, 40²
- [ ] L84  /model.22   /model.22/m.0/m.0.0/cv1                                 conv 128->64, s1, k3, 20²
- [ ] L85  /model.23   /model.23/one2one_cv3.1/one2one_cv3.1.1/one2one_cv3.1.1.1 conv 80->80, s1, k1, 40²
- [ ] L86  /model.22   /model.22/m.0/m.0.0/cv2                                 conv 64->128, s1, k3, 20²
- [ ] L87  /model.23   /model.23/one2one_cv3.1/one2one_cv3.1.2                 conv 80->80, s1, k1, 40²
- [ ] L88  /model.22   /model.22/m.0/m.0.1/attn/qkv                            conv 128->256, s1, k1, 20²
- [ ] L89  /model.22   /model.22/m.0/m.0.1/attn/pe                             conv 128->128, s1, k3, 20²
- [ ] L90  /model.22   /model.22/m.0/m.0.1/attn/proj                           conv 128->128, s1, k1, 20²
- [ ] L91  /model.22   /model.22/m.0/m.0.1/ffn/ffn.0                           conv 128->256, s1, k1, 20²
- [ ] L92  /model.22   /model.22/m.0/m.0.1/ffn/ffn.1                           conv 256->128, s1, k1, 20²
- [ ] L93  /model.22   /model.22/cv2                                           conv 384->256, s1, k1, 20²
- [ ] L94  /model.23   /model.23/one2one_cv2.2/one2one_cv2.2.0                 conv 256->16, s1, k3, 20²
- [ ] L95  /model.23   /model.23/one2one_cv3.2/one2one_cv3.2.0/one2one_cv3.2.0.0 conv 256->256, s1, k3, 20²
- [ ] L96  /model.23   /model.23/one2one_cv2.2/one2one_cv2.2.1                 conv 16->16, s1, k3, 20²
- [ ] L97  /model.23   /model.23/one2one_cv3.2/one2one_cv3.2.0/one2one_cv3.2.0.1 conv 256->80, s1, k1, 20²
- [ ] L98  /model.23   /model.23/one2one_cv2.2/one2one_cv2.2.2                 conv 16->4, s1, k1, 20²
- [ ] L99  /model.23   /model.23/one2one_cv3.2/one2one_cv3.2.1/one2one_cv3.2.1.0 conv 80->80, s1, k3, 20²
- [ ] L100 /model.23   /model.23/one2one_cv3.2/one2one_cv3.2.1/one2one_cv3.2.1.1 conv 80->80, s1, k1, 20²
- [ ] L101 /model.23   /model.23/one2one_cv3.2/one2one_cv3.2.2                 conv 80->80, s1, k1, 20²
