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
- [x] Add generator residual-add support (r_bias fold per footgun #6); pilot L26 passed cos 0.999313.
- [x] Add generator no-activation tail support (detect-head outputs without SiLU).
- [x] Add generator depthwise support via zero-expanding `(cout,1,k,k)` weights to dense `(cout,cin,k,k)`; validated on all 8 depthwise sites.
- [x] Add dynamic `S_OUT` sizing — emit `stim/s_out_params.sv` SV package with `S_OUT_PRE_VAL` (pre-SiLU range) and `S_OUT_SILU_VAL` (post-SiLU range) consumed by both RTL shim and extract.py. Fixed L37/L43/L45/L50/L68/L85/L100 (cos 0.985→0.998+).
- [x] Add generator residual + no-SiLU combo (pin `S_OUT_SILU == S_OUT_PRE` per footgun #9); validated L36/L38/L90/L92.
- [x] Fix `mk/verilator.mk` so layer `LINT_FLAGS` reach the build step, not just lint.
- [x] Refresh generated L0 with current emitter; now passes cos 0.9998 (was stale lint-only skeleton).
- [x] Parallel batch DV runner via `xargs -P 6 VERILATOR_JOBS=2` — full sweep ~1m 47s on 32-core box.

## Generator path — open
- [x] Re-test L82, L98 after LINT_FLAGS fix; backported `-Wno-WIDTHTRUNC` waiver into generator template and added targeted `lint_off UNUSEDSIGNAL` around `dotN` N==1 degenerate `node` decl. Both pass 6/6.

**All 102 conv layers now ORT-validated. Full regression sweep 82/82 PASS at ~1m 47s wall.**

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
- [x] L26  /model.8    /model.8/m.0/m/m.0/cv2                                  conv 64->64, s1, k3, 20²
- [x] L27  /model.8    /model.8/m.0/m/m.1/cv1                                  conv 64->64, s1, k3, 20²
- [x] L28  /model.8    /model.8/m.0/m/m.1/cv2                                  conv 64->64, s1, k3, 20²
- [x] L29  /model.8    /model.8/m.0/cv3                                        conv 128->128, s1, k1, 20²
- [x] L30  /model.8    /model.8/cv2                                            conv 384->256, s1, k1, 20²
- [x] L31  /model.9    /model.9/cv1                                            conv 256->128, s1, k1, 20²
- [x] L32  /model.9    /model.9/cv2                                            conv 512->256, s1, k1, 20²
- [x] L33  /model.10   /model.10/cv1                                           conv 256->256, s1, k1, 20²
- [x] L34  /model.10   /model.10/m/m.0/attn/qkv                                conv 128->256, s1, k1, 20²
- [x] L35  /model.10   /model.10/m/m.0/attn/pe                                 conv 128->128, s1, k3, 20²
- [x] L36  /model.10   /model.10/m/m.0/attn/proj                               conv 128->128, s1, k1, 20²
- [x] L37  /model.10   /model.10/m/m.0/ffn/ffn.0                               conv 128->256, s1, k1, 20²
- [x] L38  /model.10   /model.10/m/m.0/ffn/ffn.1                               conv 256->128, s1, k1, 20²
- [x] L39  /model.10   /model.10/cv2                                           conv 256->256, s1, k1, 20²
- [x] L40  /model.13   /model.13/cv1                                           conv 384->128, s1, k1, 40²
- [x] L41  /model.13   /model.13/m.0/cv1                                       conv 64->32, s1, k1, 40²
- [x] L42  /model.13   /model.13/m.0/cv2                                       conv 64->32, s1, k1, 40²
- [x] L43  /model.13   /model.13/m.0/m/m.0/cv1                                 conv 32->32, s1, k3, 40²
- [x] L44  /model.13   /model.13/m.0/m/m.0/cv2                                 conv 32->32, s1, k3, 40²
- [x] L45  /model.13   /model.13/m.0/m/m.1/cv1                                 conv 32->32, s1, k3, 40²
- [x] L46  /model.13   /model.13/m.0/m/m.1/cv2                                 conv 32->32, s1, k3, 40²
- [x] L47  /model.13   /model.13/m.0/cv3                                       conv 64->64, s1, k1, 40²
- [x] L48  /model.13   /model.13/cv2                                           conv 192->128, s1, k1, 40²
- [x] L49  /model.16   /model.16/cv1                                           conv 256->64, s1, k1, 80²
- [x] L50  /model.16   /model.16/m.0/cv1                                       conv 32->16, s1, k1, 80²
- [x] L51  /model.16   /model.16/m.0/cv2                                       conv 32->16, s1, k1, 80²
- [x] L52  /model.16   /model.16/m.0/m/m.0/cv1                                 conv 16->16, s1, k3, 80²
- [x] L53  /model.16   /model.16/m.0/m/m.0/cv2                                 conv 16->16, s1, k3, 80²
- [x] L54  /model.16   /model.16/m.0/m/m.1/cv1                                 conv 16->16, s1, k3, 80²
- [x] L55  /model.16   /model.16/m.0/m/m.1/cv2                                 conv 16->16, s1, k3, 80²
- [x] L56  /model.16   /model.16/m.0/cv3                                       conv 32->32, s1, k1, 80²
- [x] L57  /model.16   /model.16/cv2                                           conv 96->64, s1, k1, 80²
- [x] L58  /model.17   /model.17                                               conv 64->64, s2, k3, 40²
- [x] L59  /model.23   /model.23/one2one_cv2.0/one2one_cv2.0.0                 conv 64->16, s1, k3, 80²
- [x] L60  /model.23   /model.23/one2one_cv3.0/one2one_cv3.0.0/one2one_cv3.0.0.0 conv 64->64, s1, k3, 80²
- [x] L61  /model.23   /model.23/one2one_cv2.0/one2one_cv2.0.1                 conv 16->16, s1, k3, 80²
- [x] L62  /model.23   /model.23/one2one_cv3.0/one2one_cv3.0.0/one2one_cv3.0.0.1 conv 64->80, s1, k1, 80²
- [x] L63  /model.19   /model.19/cv1                                           conv 192->128, s1, k1, 40²
- [x] L64  /model.23   /model.23/one2one_cv2.0/one2one_cv2.0.2                 conv 16->4, s1, k1, 80²
- [x] L65  /model.23   /model.23/one2one_cv3.0/one2one_cv3.0.1/one2one_cv3.0.1.0 conv 80->80, s1, k3, 80²
- [x] L66  /model.19   /model.19/m.0/cv1                                       conv 64->32, s1, k1, 40²
- [x] L67  /model.19   /model.19/m.0/cv2                                       conv 64->32, s1, k1, 40²
- [x] L68  /model.23   /model.23/one2one_cv3.0/one2one_cv3.0.1/one2one_cv3.0.1.1 conv 80->80, s1, k1, 80²
- [x] L69  /model.19   /model.19/m.0/m/m.0/cv1                                 conv 32->32, s1, k3, 40²
- [x] L70  /model.23   /model.23/one2one_cv3.0/one2one_cv3.0.2                 conv 80->80, s1, k1, 80²
- [x] L71  /model.19   /model.19/m.0/m/m.0/cv2                                 conv 32->32, s1, k3, 40²
- [x] L72  /model.19   /model.19/m.0/m/m.1/cv1                                 conv 32->32, s1, k3, 40²
- [x] L73  /model.19   /model.19/m.0/m/m.1/cv2                                 conv 32->32, s1, k3, 40²
- [x] L74  /model.19   /model.19/m.0/cv3                                       conv 64->64, s1, k1, 40²
- [x] L75  /model.19   /model.19/cv2                                           conv 192->128, s1, k1, 40²
- [x] L76  /model.20   /model.20                                               conv 128->128, s2, k3, 20²
- [x] L77  /model.23   /model.23/one2one_cv2.1/one2one_cv2.1.0                 conv 128->16, s1, k3, 40²
- [x] L78  /model.23   /model.23/one2one_cv3.1/one2one_cv3.1.0/one2one_cv3.1.0.0 conv 128->128, s1, k3, 40²
- [x] L79  /model.23   /model.23/one2one_cv2.1/one2one_cv2.1.1                 conv 16->16, s1, k3, 40²
- [x] L80  /model.23   /model.23/one2one_cv3.1/one2one_cv3.1.0/one2one_cv3.1.0.1 conv 128->80, s1, k1, 40²
- [x] L81  /model.22   /model.22/cv1                                           conv 384->256, s1, k1, 20²
- [x] L82  /model.23   /model.23/one2one_cv2.1/one2one_cv2.1.2                 conv 16->4, s1, k1, 40²
- [x] L83  /model.23   /model.23/one2one_cv3.1/one2one_cv3.1.1/one2one_cv3.1.1.0 conv 80->80, s1, k3, 40²
- [x] L84  /model.22   /model.22/m.0/m.0.0/cv1                                 conv 128->64, s1, k3, 20²
- [x] L85  /model.23   /model.23/one2one_cv3.1/one2one_cv3.1.1/one2one_cv3.1.1.1 conv 80->80, s1, k1, 40²
- [x] L86  /model.22   /model.22/m.0/m.0.0/cv2                                 conv 64->128, s1, k3, 20²
- [x] L87  /model.23   /model.23/one2one_cv3.1/one2one_cv3.1.2                 conv 80->80, s1, k1, 40²
- [x] L88  /model.22   /model.22/m.0/m.0.1/attn/qkv                            conv 128->256, s1, k1, 20²
- [x] L89  /model.22   /model.22/m.0/m.0.1/attn/pe                             conv 128->128, s1, k3, 20²
- [x] L90  /model.22   /model.22/m.0/m.0.1/attn/proj                           conv 128->128, s1, k1, 20²
- [x] L91  /model.22   /model.22/m.0/m.0.1/ffn/ffn.0                           conv 128->256, s1, k1, 20²
- [x] L92  /model.22   /model.22/m.0/m.0.1/ffn/ffn.1                           conv 256->128, s1, k1, 20²
- [x] L93  /model.22   /model.22/cv2                                           conv 384->256, s1, k1, 20²
- [x] L94  /model.23   /model.23/one2one_cv2.2/one2one_cv2.2.0                 conv 256->16, s1, k3, 20²
- [x] L95  /model.23   /model.23/one2one_cv3.2/one2one_cv3.2.0/one2one_cv3.2.0.0 conv 256->256, s1, k3, 20²
- [x] L96  /model.23   /model.23/one2one_cv2.2/one2one_cv2.2.1                 conv 16->16, s1, k3, 20²
- [x] L97  /model.23   /model.23/one2one_cv3.2/one2one_cv3.2.0/one2one_cv3.2.0.1 conv 256->80, s1, k1, 20²
- [x] L98  /model.23   /model.23/one2one_cv2.2/one2one_cv2.2.2                 conv 16->4, s1, k1, 20²
- [x] L99  /model.23   /model.23/one2one_cv3.2/one2one_cv3.2.1/one2one_cv3.2.1.0 conv 80->80, s1, k3, 20²
- [x] L100 /model.23   /model.23/one2one_cv3.2/one2one_cv3.2.1/one2one_cv3.2.1.1 conv 80->80, s1, k1, 20²
- [x] L101 /model.23   /model.23/one2one_cv3.2/one2one_cv3.2.2                 conv 80->80, s1, k1, 20²

## Block-level integration

- [x] **SPPF block** (`integ/sppf_model9/`) — between L31 and L32. Three sequential `maxpool_kxk` (K=5) stages over L31 output, then `concat_mux` of {L31, mp1, mp2, mp3} feeding L32. 10/10 checks, 5/5 ORT samples bit-exact vs SW ref, cos ≥ 0.99994 vs ORT (H=20 W=20 C=128 K=5).
- [x] **Upsample integration — P4→P3 (`/model.11` + `/model.12`)** (`integ/upsample_model11/`) — frame-store NN 2x upsample of /model.10/cv2 (256ch, 20²) followed by channel-concat with /model.6/cv2 P3 skip (128ch, 40²) → 384ch 40². 10/10 checks, 5/5 samples bit-exact vs SW int8 golden, cos vs ORT range 0.9995..0.9998 (3/3 random tiles ≥ 0.998).
- [x] **Upsample integration — P3→detect (`/model.14` + `/model.15`)** (`integ/upsample_model14/`) — parameterized clone of the model.11 block. Frame-store NN 2x upsample of /model.13/cv2 (128ch, 40²) then channel-concat with /model.4/cv2 P3 skip (128ch, 80²) → 256ch 80² feeding L49 (/model.16/cv1). 10/10 checks, 5/5 samples bit-exact vs SW int8 golden, cos vs ORT range 0.999642..0.999833 (3/3 random tiles ≥ 0.998).
- [~] **Attention block — `/model.10` PSA** (`integ/attn_model10/`) — **behavioral placeholder, NOT synthesizable.** Numerics validated end-to-end against ORT (10/10 checks, 5/5 samples, cos 0.99972..0.99984 vs ORT), but matmuls + 400-lane softmax are implemented in SV `real` arithmetic. Kept as a golden reference for the structural rebuild; do NOT push to PD or count toward area/cycle modeling.
- [ ] **`hw/ip/flash_attn/` — systolic flash-attention leaf IP** — NEW factored approach. Parameterized (HEADS, N, DIM_Q, DIM_V, BR, BC) systolic fp16 MAC mesh with online (tiled) softmax — never materializes full N×N scores tensor. Single reusable leaf IP that both PSA (`/model.10`) and A2C2f (`/model.22`) will instantiate. Standalone DV with `flash_attn_ref.sv` golden + random Q/K/V sweep. Budget: P≈512 cells → ~60k cyc/pass, fits T_FRAME=100k.
- [ ] **Attention integration shim — `/model.10` PSA** — thin shim around `flash_attn` (~80 lines RTL): QKV split off L34's output, drive flash_attn, pe-add post-attention, requant boundaries, residuals via `add_rq` around external proj/FFN convs. Replaces the behavioral `integ/attn_model10/` once flash_attn is green; re-runs the existing extract.py / TB against ORT.
- [ ] **Attention integration shim — `/model.22` A2C2f** — parametric clone of the `/model.10` shim. L88..L92 shapes are byte-identical to L34..L38 per ONNX. Trivial after the model.10 shim exists.
- [ ] **Detect head with learned top-k** — end-to-end, no NMS. Largest remaining technical risk.

## Chip top-level (frozen for PD)

- [x] `hw/ip/yolo26n_top/` — frozen chip boundary for PD handoff.
  - AXI4-Stream slave `s_axis_pix` (2 px/cycle RGB u8, SOF on TUSER[0], EOL on TLAST).
  - AXI4-Stream master `m_axis_det` (1 det/beat: cls/score/xyxy fp16, TLAST=EOF).
  - AXI4-Lite slave `s_axil` (12-bit addr, 32-bit data): CTRL/STATUS/TOPK/SCRATCH/ID/VERSION.
  - Single clock domain, async-assert / sync-deassert reset, level-high IRQ.
  - `yolo26n_top.sv` (boundary, FROZEN) + `yolo26n_csr.sv` (AXI-Lite regfile) + `yolo26n_core.sv` (skeletal shell — body grows as SPPF / upsample / attn / detect head / 102-layer wiring lands).
  - Lints clean under verilator. PD owns the boundary from here; integration owns `yolo26n_core`.
- [x] PD-friendliness pass on `yolo26n_top`:
  - Added DFT bundle (`scan_en`, `scan_mode`, `test_clk`, `scan_in/out[N_SCAN_CHAINS]`, `bist_run/done/fail`).
  - Internal `prim_rst_sync` (behavioral, PD swaps for tech cell) — boundary `rst_ni` is async-assert / sync-deassert.
  - Internal `prim_clk_gate` on the core clock with `scan_en` bypass and `clk_gate_en_i` strap; CSR + AXIS skid stages stay on ungated `clk_i`.
  - All AXIS pins flopped via `axis_skid` register slices (full-throughput 2-entry skid buffers, in + out).
  - `(* keep_hierarchy = "yes" *)` on `yolo26n_core`.
  - `MEMORIES.md` published — skip_p3 (410 KB) + skip_p4 (205 KB) flagged as the macros to commission first.
  - `constraints/yolo26n_top.sdc` starter — clock defs, scan mode, IO budget, CSR false paths, dont-touch on core.
