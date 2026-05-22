# YOLO26n HW Accelerator — Handoff Doc

## Goal

Build a fixed-function, pipelined feed-forward ASIC that runs **YOLO26n int8 inference**. Half-reticle N16 (~430 mm², ~6.5G gates), "bonkers throughput" — area is *not* a constraint. Target `T_FRAME = 100,000 cycles/frame` → ~10K FPS @ 1 GHz. Each layer dedicated dataflow, no shared MAC array.

The user assumes "magical SRAM": activations stream in/out from external SRAM that we don't model. Skip-connection FIFOs need sizing later.

## Repo layout

```
/home/marty/Documents/aif/vibeyolo/
├── hw/ip/                       shared, parameterized IPs (one per directory)
│   ├── mac8/                    int8×int8→i32 multiply
│   ├── dotN/                    N-lane dot-product with pipelined adder tree
│   ├── i32_to_fp16/             i32 → fp16 (with auto-shift, see footgun #1)
│   ├── fp16_fma/                fp16 fused multiply-add
│   ├── fp16_to_i8_sat/          fp16 → i8 saturating round
│   ├── requant/                 i32 + scale + bias → i8 (used in every conv layer)
│   ├── add_rq/                  int8 + int8 → int8 with per-stream scaling
│   ├── act_silu/                SiLU LUT (int8 in, int8 out)
│   ├── softmax16/               fp16 softmax (used by attention blocks later)
│   ├── box_decode/              detect-head box decoder
│   ├── linebuf_kxk/             K×K patch line-buffer (k=3/5)
│   └── conv_layer/              ★ THE BIG ONE — fully parameterized Conv+(Resid)+(SiLU)
├── integ/
│   ├── yolo26n/model_int8.onnx  source-of-truth quantized ONNX
│   ├── scale/
│   │   ├── gen_scale_pkg.py     generator for per-layer parallelism + shapes
│   │   ├── scale_pkg.sv         REAL-CHIP config (P_COUT/P_CIN for synthesis)
│   │   ├── scale_pkg_dv.sv      DV-CAPPED config (P_COUT≤16, P_CIN≤8 for fast verilator)
│   │   └── scale_report.md
│   ├── generated/               non-destructive generated scale/layer experiments
│   ├── _layer_template/         canonical tiled layer reference + README
│   ├── stem_l0/  layer_1..20_*  per-layer dirs: extract.py + rtl/ + dv/ + README
│   └── ...
├── tools/layergen/              new generator for balanced scale + layer skeletons
├── mk/verilator.mk              shared make recipe
├── dv/common/                   shared SimCtrl<T> C++ harness
└── TASKS.md                     long-running task log
```

## What's done (as of this handoff)

### Shared IPs (all DV passing)
- All listed above are validated with `make -C hw/ip/<name>/dv test`.
- `conv_layer` is the **central** IP — see its README. Tiled architecture: P_COUT dotN cells × N_LANE=K²·P_CIN, time-multiplexed over `N_COUT_TILE × N_CIN_TILE` phases per output pixel.
- DV runs ≥30k random vectors clean on the leaf IPs.

### Layers built and ORT-validated
**100 of 102 conv layers** done. Only **L82, L98** remain (dotN small-N Verilator lint waiver — a build-flag issue, not correctness).

Two tracks:
- **Hand-built** under `integ/layer_*` (L0–L25): canonical reference implementations.
  - L0 (stem), L1 (4/6 — known stim issue), L2–L20. Residual layers L4/L9/L16/L18.
  - L20 `/model.6/cv2` 192→128 k1 s1 at 40² is the newest hand-built; 8/8 samples avg cos 0.999791.
- **Auto-generated** under `integ/generated/` (L0, L21–L101): emitted by `tools/layergen/layergen.py` and ORT-validated. Generated L0 also passes (cos 0.9998) — used as a regression target for the generator itself.

Each layer = `extract.py` (ORT-driven stim/golden) + ~30-80 line `.sv` shim around `conv_layer` + TB + Makefile + `stim/s_out_params.sv` (auto-sized per-layer SiLU scales). Cosine ≥ 0.998 vs ORT on 6 random tiles.

### Generator path (now the production codegen)
- `tools/layergen/layergen.py` is non-destructive — writes under `integ/generated/`, does not overwrite hand-built `integ/layer_*`.
- Emits balanced scale packages, per-layer RTL shim, extract.py, C++ tiled driver, TB, Makefile.
- Balancing rule: minimize area proxy `P_PIX * P_COUT * P_CIN` subject to `cycles <= T_FRAME`. Grouped/depthwise convs count only `Cin/group` input channels.
- Current default is `max_p_pix=2`. Generated wrappers implement `P_PIX>1` by instantiating multiple lockstep `conv_layer` lanes.
- Current balanced real-chip report at `T_FRAME=100000`: total area proxy 22,096, worst layer 80,000 cycles, 0 target misses.
- Generated extractor target selection matches the current conv prefix to disambiguate parallel branches (cv1 vs cv2).

**Conv variants the generator supports:**
1. Ordinary group=1 Conv+SiLU
2. Group=1 Conv with **no activation** (detect-head tails)
3. **Residual-add** Conv+SiLU (L26 pilot; r_bias_eff fold per footgun #6)
4. **Depthwise** (group==cin==cout) — handled by zero-expanding weights to dense, reusing the dense driver
5. **Residual + no-SiLU** (attn proj / FFN tail) — requires `S_OUT_SILU == S_OUT_PRE` per new footgun #9

**Dynamic S_OUT sizing:** the generator emits `stim/s_out_params.sv` (an SV package with `S_OUT_PRE_VAL`/`S_OUT_SILU_VAL` as `localparam real`), sized from observed activation amplitudes (`max * 1.1 / 127`). Both the RTL shim and extract.py consume the same package. Critical: `S_OUT_PRE` covers pre-SiLU range, `S_OUT_SILU` covers post-SiLU range — they are NOT interchangeable.

### Batch DV
Run all generated layers in parallel:
```bash
ls -d integ/generated/layer_*/dv | xargs -I{} -P 6 bash -c \
  'timeout 180 env VERILATOR_JOBS=2 make -C {} test > /tmp/dv_$$.log 2>&1 \
   && echo "PASS {}" || echo "FAIL {}"'
```
~1m 47s wall time for 81 layers on a 32-core box. Memory peak ~12 GiB.

### Numbers
- **Real chip**: worst-layer 76,800 cyc → 13K FPS @ 1 GHz. Median 51,200 cyc.
- **Real chip total MAC units**: ~54K (rough). Total gates well within half-reticle.
- **DV builds**: <60 sec per layer with the DV-capped pkg + VERILATOR_JOBS=2-4. Full sweep ~2 min wall.

## What's next

All 102 conv layers are ORT-validated. The remaining work is block-level integration around graph boundaries (SPPF, upsamples, attention, detect head) and then top-level wiring.

### Block-level IPs

**Done:**
- [x] **SPPF integration block** (`integ/sppf_model9/`) — between L31 and L32. Three sequential `maxpool_kxk` (K=5) stages over L31's output, then `concat_mux` of {L31_out, mp1, mp2, mp3} feeding L32's input. 10/10 checks, 5/5 ORT samples bit-exact vs SW ref, cos ≥ 0.99994 vs ORT (H=20 W=20 C=128 K=5 ROI=8x8).
- [x] **Upsample integration — `/model.11` + `/model.12`** (`integ/upsample_model11/`) — P4→P3 neck join after L39. Nearest-neighbour 2x upsample of /model.10/cv2 (256ch, 20²) then channel-concat with /model.6/cv2 P3 skip (128ch, 40²) → 384ch 40² feeding L40 (/model.13/cv1). Frame-store + drain design (mirrors SPPF — does NOT use the byte-serial `concat_mux` or single-channel `upsample2` IPs; channel-parallel wide bus). Both inputs share a per-sample S_OUT covering the joint range (real chip would do an upstream per-stream fp16 rescale). 10/10 checks, 5/5 samples bit-exact vs SW int8 golden, cos vs ORT 0.9995..0.9998 across all samples (3/3 random tiles ≥ 0.998).
- [x] **Upsample integration — `/model.14` + `/model.15`** (`integ/upsample_model14/`) — P3→detect neck join after L48. Parameterized clone of the model.11 block: NN 2x upsample of /model.13/cv2 (128ch, 40²) then channel-concat with /model.4/cv2 P3 skip (128ch, 80²) → 256ch 80² feeding L49 (/model.16/cv1). Same frame-store + drain RTL, only parameters change (H_A/W_A=40, C_A=128, H_B/W_B=80, C_B=128). 10/10 checks, 5/5 samples bit-exact vs SW int8 golden, cos vs ORT 0.999642..0.999833 across all samples (3/3 random tiles ≥ 0.998).
- [~] **Attention block — `/model.10` PSA** (`integ/attn_model10/`) — **behavioral placeholder, not synthesizable.** Numerics validated end-to-end against ORT (10/10 checks, 5/5 samples, cos 0.99972..0.99984), but the matmuls + 400-lane softmax are implemented in SV `real` arithmetic inside `always_ff`. Useful as a golden reference; do NOT count toward PD area/cycle modeling. Will be replaced by a thin shim around the new `hw/ip/flash_attn/` leaf IP once that lands.

**Done since last handoff:**
- [~] **`hw/ip/flash_attn/` — flash-attention leaf IP (WIP, over budget)** — parameterized tile-streaming fp16 flash-attention with online softmax; algorithm + golden + DV scaffolding correct. 5/5 DV checks pass at small shapes (HEADS=1, N=8, DIM_Q=4, DIM_V=4); worst cos(dut, shadow)=0.999976. **BR=1 BC=1 → ~2M cyc/frame at YOLO26n shapes = 20× over T_FRAME=100k.** Parallelization push is the next-up task — boundary, golden, and DV are parametric and stay as-is. Resolved: `pe` lives outside the IP (integration shim adds pe(V) post-IP).

**To build (in recommended order):**
1. **`flash_attn` parallelization push** *(top priority)* — unroll BR/BC to hit T_FRAME=100k at YOLO26n shapes (target P_FMA ~512 → ~60k cyc/pass). Boundary/golden/DV are parametric and stay as-is; re-validate existing small-shape tests AND add a production-shape test.
2. **Attention integration shim — `/model.10` PSA** *(small, after flash_attn hits budget)* — replaces behavioral `integ/attn_model10/` with ~80-line shim instantiating `flash_attn` + an external pe(V) `conv_layer` + `add_rq` residuals + requant boundaries. Re-run existing extract.py / TB against ORT.
3. **Attention integration shim — `/model.22` A2C2f** *(trivial)* — parametric clone of `/model.10` shim. L88..L92 = L34..L38 shape-wise.
4. **Detect head with learned top-k** *(hardest)* — YOLO26 is end-to-end (no NMS). `box_decode` validated; novel work is the learned top-k selection across three scales (80², 40², 20²). Largest remaining technical risk.

### Top-level integration (after all blocks)
- `top.sv` that wires all 102+ layer instances in a streaming dataflow.
- **Skip-connection FIFOs**: P3 skip = 80×80×64 = ~410KB, P4 skip = 40×40×128 = ~205KB. Plan SRAM budget.
- End-to-end ORT validation: push a real 640×640 image through the whole DUT, compare detect output vs ORT. Target cos ≥ 0.99 on final box/cls tensors.

### Deferred / cleanup
- Generated L1 synthetic stim refresh when E2E pipeline exists (feed L0's real output instead of synthetic).

### Top-level integration (after all 102 conv layers + above IPs)
- `top.sv` that wires all 102+ layer instances in a streaming dataflow.
- **Skip-connection FIFOs**: P3 skip = 80×80×64 = ~410KB, P4 skip = 40×40×128 = ~205KB. Plan SRAM budget.
- End-to-end ORT validation: push a real 640×640 image through the whole DUT, compare detect output vs ORT. Target cos ≥ 0.99 on final box/cls tensors.

### Known caveats
- **L1's synthetic test** spans output range 7..60, no single S_OUT covers it. Fix when doing real-image E2E validation — feed L0's actual output as L1's stim.
- **L4 small-range tiles** (half/gradient) fall to cos ~0.997 due to SiLU LUT noise floor. DUT bit-exact vs HW-ref; these tiles aren't representative.

## Critical context (READ THIS)

### The `conv_layer` interface (per-tile streaming, NOT full-channel)

```
Input:  x_i[K*K*P_CIN]    — one cin-tile of the K×K patch
        w_i[P_COUT][K*K*P_CIN]
        scale_i[P_COUT], bias_i[P_COUT] — sampled at last_cin
        first_cin_i, last_cin_i        — phase markers
        cout_tile_idx_i                — which cout-tile this beat's weights are for
        (residual ports if RESIDUAL=1)

Output: y_o[P_COUT][7:0]
        cout_tile_idx_o                — tag passed through
```

**Driver pattern** (innermost to outermost): cin_tile → cout_tile → pixel. The same input patch is replayed once per cin_tile × cout_tile beat.

### The two scale_pkg variants

- `scale_pkg.sv` — REAL CHIP. P_COUT often = COUT (fully unrolled per cout). Synthesis target. **DO NOT use for verilator** — designs hit 100k+ MAC cells flat and OOM.
- `scale_pkg_dv.sv` — DV ONLY. P_COUT ≤ 16, P_CIN ≤ 8 by default. Same package name (`scale_pkg`), so RTL is unchanged — verilator picks whichever file is on the command line. **Per-layer Makefiles point at scale_pkg_dv.sv.**

Regenerate either with `python3 integ/scale/gen_scale_pkg.py [--dv]`. The new math credits the K×K fold inside dotN, so cycles = M / (P_PIX·P_COUT·K²·P_CIN).

### Balanced generated scale path

Use this for new experiments without touching the checked hand-built scale files:

```bash
python3 tools/layergen/layergen.py --emit-scale --emit-dv-scale --gen-layer 0 --gen-layer 21
make -C integ/generated/layer_0_model0/dv lint
make -C integ/generated/layer_21_model7/dv lint
```

Important:
- `scale_pkg_balanced*.sv` still declare package name `scale_pkg`, so include only one scale package per Verilator/synthesis command.
- Generated DV scale is intentionally capped (`P_COUT<=16`, `P_CIN<=8`) and can miss `T_FRAME`; it is for build speed, not real timing.
- Generated layer skeletons are not complete layer validations until `extract.py` and the C++ sample driver exist.
- Generated wrappers implement `P_PIX>1` as parallel `conv_layer` lanes sharing weights/scales. This increases area roughly linearly with `P_PIX` for that layer.

### How to add a new conv layer

```bash
mkdir integ/layer_N_<short>/{rtl,dv,stim}
# 1. Write extract.py — see integ/layer_11/extract.py
#    Pulls weights/bias/scale from integ/yolo26n/model_int8.onnx,
#    folds u8→i8 (bias_eff[c] += s_a·s_w·(128-zp_a)·Σw[c]),
#    runs a few input tiles through ORT, dumps to stim/.
# 2. Write rtl/layer_N.sv — ~30-80 line shim, imports scale_pkg, instantiates conv_layer.
#    See integ/layer_11/rtl/layer_11.sv as the canonical template.
# 3. Write dv/layer_N_tb.sv and dv/layer_N_test.cc — copy from integ/layer_11/dv/.
#    Driver feeds cin_tile innermost, cout_tile outer, pixel outermost.
# 4. Write dv/Makefile — copy from layer_11; uses scale_pkg_dv.sv and conv_layer.sv.
# 5. make -C integ/layer_N/dv test  → build <60s, cos ≥ 0.998 on ≥3 random tiles.
```

### Quantization scheme
- Weights: int8 symmetric (zp_w = 0).
- Activations: u8 dyn-quant from ORT (zp_a varies, scale s_a per tensor). **We bridge to i8-sym** via `i8 = u8 - 128` plus a bias rewrite `bias_eff[c] += s_a·s_w·(128 - zp_a)·Σw[c]`. Math is exact; no accuracy loss.
- Per-channel fp16 scale + per-channel fp16 bias post-scale: `out_i8 = sat_i8(round(fp16(acc_i32) * scale_fp16 + bias_fp16))`.

## Footguns / gotchas

### 1. `i32_to_fp16` threshold subtlety (FIXED, but be aware)
- Old version: shift=0 for `msb_pos <= 15`. Values in (65504, 65535] would RNE-round UP to +Inf (next fp16 step after 65504 is 65536). Then FMA propagates Inf → i8 sat → silent saturation in random output channels.
- Fixed in `hw/ip/i32_to_fp16/rtl/i32_to_fp16.sv` line 72 → `msb_pos <= 14`. Lost 1 bit of acc precision for |x| ∈ [32768, 65504], gained correctness for the overflow band.
- If you change this file, also update `i32_to_fp16_ref.sv` (line ~55) AND `dv/i32_to_fp16_test.cc` (line ~41 in shadow_fp16).

### 2. SystemVerilog packed-array signed shift
- `arr[i] >>> n` on `logic signed [NCH-1:0][31:0] arr` does a **logical** (not arithmetic) shift because indexing yields an *unsigned* slice. Wrap with `$signed(arr[i]) >>> n` for arith.
- conv_layer handles this internally; just be careful if writing new IPs.

### 3. SystemVerilog `always_comb` local variables
- `int diff = some_expr` inside `always_comb` infers a latch AND causes UNOPTFLAT warnings in verilator. Use `automatic int diff; diff = some_expr;` instead.
- Bit us in linebuf_kxk. Worth remembering.

### 4. Verilator build cost is per-layer-gate-count
- A layer instantiated with full `scale_pkg.sv` parallelism (e.g. P_COUT=128) hits ~150k MAC cells flat. Verilator OOMs or takes 6+ minutes.
- **Always use `scale_pkg_dv.sv` for DV.** Pin `VERILATOR_JOBS=4` if you have RAM constraints.
- Top-level "all 102 layers in one verilator design" is **not feasible**. Plan E2E validation with `_ref.sv` golden-only or a C++ chain.

### 5. SiLU LUT noise floor
- `S_OUT_SILU` chosen per layer trades range vs precision. Layers with small output range (~±2) want fine LUT (e.g. 4/127). Wide range (~±60, e.g. L1) wants coarse, but coarse loses precision on quiet tiles.
- **Empirical rule**: size `S_OUT = max_observed_post_silu * 1.1 / 127`. Both extract.py AND the SV shim's `S_OUT_PRE`/`S_OUT_SILU` params must match.
- ~32 LUT steps over output range is the floor for cos ≥ 0.998.

### 6. Residual layers need scale-folding into r_bias
- For RESIDUAL=1, the residual stream's u8→i8 fold goes into r_bias: `r_bias_eff[c] = bias_orig[c] + (128 - zp_r)·s_r/S_OUT_SILU`.
- L4, L9, L16, L18 are residual layers. See their extract.py for working examples.

### 7. requant latency = 4 (not 3)
- The auto-shift handling added one register stage. Any code hardcoding "3" for requant latency is broken. Always use `valid_o` for downstream alignment.

### 8. Don't trust the task brief on ONNX shapes
- Multiple times the task description claimed e.g. "8-ch residual" but ONNX shape_inference revealed 16-ch full-width. **Always verify via `onnx.shape_inference.infer_shapes()` before coding.**

### 9. Residual + no-SiLU requires `S_OUT_SILU == S_OUT_PRE`
- The `conv_layer` IP wires `add_rq`'s `scale_a_fp16 = real_to_fp16(S_OUT_SILU)` unconditionally. When `SILU=0`, the "a" input is `rq_y` (post-requant, scaled at `S_OUT_PRE`), so the two must be equal or the residual sum is silently mis-scaled.
- Generator handles this for L36/L38/L90/L92 by pinning `S_OUT_SILU = S_OUT_PRE = max(|pre|, |add|) * 1.1 / 127` in `stim/s_out_params.sv`. Don't change the IP — fix it at the extractor level.

### 10. S_OUT_PRE covers pre-SiLU, S_OUT_SILU covers post-SiLU
- They are NOT interchangeable. `S_OUT_PRE` is the SiLU LUT *input* grid (and the requant fp16 scale divisor in stim); too small → negative pre-acts saturate the LUT input to `silu(-128*S_OUT_PRE) ≈ -0.238` (for default 8/127) instead of ~0.
- `S_OUT_SILU` is the LUT *output* grid (and residual-add grid); too coarse → LUT noise floor (footgun #5).
- Generator sizes each from its own observed amplitude. Empirical rule for both: `max(|tensor|) * 1.1 / 127`.

### 11. Generator `make -j` race against stim regeneration
- The generator emits `stim/s_out_params.sv` (SV package) that both RTL shim and extract.py consume. extract.py rewrites it at the end of `make stim` with values from observed amplitudes.
- Without explicit Make ordering, `make -j` builds the Verilator binary against the placeholder before `make stim` runs. Result: silent wrong cosine, no build error.
- Generator emits `.stim.stamp` and explicit prereq; if you see "lint clean but DV cos is wrong" symptoms, check the stamp wiring.

### 13. `softmax16` lane count is fixed; attention softmax needs locally-implemented N-way fp16 softmax
- `hw/ip/softmax16/` is hardcoded for N=16 fp16 lanes (parallel-in / parallel-out, single valid). It targets the DFL detection head where N=16 is the bin count.
- PSA/A2C2f attention softmax is N=400 per row (HW×WW spatial). The 16-lane IP cannot drop in (16-element max + sum trees, ports declared as `[16]`).
- The model.10 attention block (`integ/attn_model10/`) implements the 400-lane softmax locally in its DUT — same algorithm shape as `softmax16` (row max, per-lane `exp(x − max)`, sum, reciprocal, multiply) but in behavioral `real` arithmetic with RNE-round-to-fp16 after every op (matching `softmax16_ref.sv`'s semantics). The `softmax16` leaf IP is left untouched.
- Reusable: `tools/layergen` could grow a `softmax_n` skeleton parameterized over N if more attention sites of differing N pop up in YOLO27/etc.

### 12. mk/verilator.mk LINT_FLAGS reached lint only, not build
- Layer Makefiles setting `LINT_FLAGS = -Wno-SELRANGE -Wno-ASCRANGE` saw waivers honored at `make lint` but ignored at `make test`. Build failed `-Wall` on legitimate edge cases (any layer with `P_COUT=1` — depthwise + detect-head `cv2.x.2` nosilu tails).
- Fixed by passing `$(LINT_FLAGS)` to the build verilator invocation too.

## Useful invocations

```bash
# Test a single IP
make -C hw/ip/<name>/dv test

# Test a single layer
make -C integ/layer_N/dv test

# Test with bounded RAM
VERILATOR_JOBS=4 make -C integ/layer_N/dv test

# Regenerate scale_pkg
python3 integ/scale/gen_scale_pkg.py --T 100000
python3 integ/scale/gen_scale_pkg.py --T 100000 --dv --dv-cout-cap 16 --dv-cin-cap 8

# Generate balanced experimental scale + L0/L21 skeletons
python3 tools/layergen/layergen.py --emit-scale --emit-dv-scale --gen-layer 0 --gen-layer 21
make -C integ/generated/layer_0_model0/dv lint
make -C integ/generated/layer_21_model7/dv lint

# Inspect ONNX
python3 -c "
import onnx, onnx.shape_inference as si
m = si.infer_shapes(onnx.load('integ/yolo26n/model_int8.onnx'))
for n in m.graph.node:
    if n.op_type == 'ConvInteger': print(n.name, [i.name for i in n.input])
"
```

## Where to look first

1. `integ/layer_11/` — canonical working layer using conv_layer shim.
2. `integ/layer_20_cv2/` — newest completed generated-pattern hand layer, useful for automating L21+.
3. `tools/layergen/` — non-destructive balanced scale/layer skeleton generator.
4. `hw/ip/conv_layer/README.md` — IP doc with parameter list and dataflow.
5. `integ/_layer_template/README.md` — original tiled template + retrofit recipe.
6. `integ/scale/scale_pkg.sv` (or `_dv.sv`) — per-layer P_COUT/P_CIN/CIN/COUT/K/etc constants.
7. `TASKS.md` — log of completed work.

## What I'd do next (concrete plan for the new AI)

1. Add residual/add support to `tools/layergen`, using L26 (`/model.8/m.0/m/m.0/cv2`) as the pilot.
2. Generate/validate L26-L30. L26 and L28 need residual/add handling; L27/L29/L30 should be ordinary Conv+SiLU once their inputs are identified.
3. Continue conv layers in generated waves, but stop at graph-boundary ops to add explicit support for concat, SPPF maxpool integration, upsample integration, attention, and detect head.
4. Build attention block IPs for A2C2f/PSA in deeper neck.
5. Top-level integration + E2E ORT validation on real images.

The hardest remaining technical risk is the **detect head** (end-to-end, learned top-k, no NMS). Look at the ONNX for `/model.23` and below before designing it.
