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
│   ├── _layer_template/         canonical tiled layer reference + README
│   ├── stem_l0/  layer_1..19_*  per-layer dirs: extract.py + rtl/ + dv/ + README
│   └── ...
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
**20 of 102 conv layers** done:
- L0 (stem), L1 (4/6 — known stim issue), L2, L3, L4 (+resid), L5, L6 (3×3 s2), L7, L8, L9 (+resid), L10, L11, L12, L13
- L14, L15, L16 (+resid), L17, L18 (+resid), L19 — model.6 C3k2 nested bottlenecks

Each layer = `extract.py` (ORT-driven stim/golden) + ~30-80 line `.sv` shim around `conv_layer` + TB + Makefile. Cosine ≥ 0.998 vs ORT on ≥3 random tiles.

### Numbers
- **Real chip**: worst-layer 76,800 cyc → 13K FPS @ 1 GHz. Median 51,200 cyc.
- **Real chip total MAC units**: ~54K (rough). Total gates well within half-reticle.
- **DV builds**: <60 sec per layer with the DV-capped pkg + VERILATOR_JOBS=4.

## What's next

### Immediate (conv layers, easy after the IP work)
1. **L20–L101**: 82 more conv layers. Each is mostly mechanical with `conv_layer` shim pattern. Spawn agents in waves of 4-8.
2. Layer naming heuristic: read `integ/scale/scale_pkg.sv` for the canonical L_N name → ONNX node mapping.

### New IPs needed (not yet built)
1. **`maxpool_5x5`** — for SPPF block (between L31 and L32). Stride-1, padding 2. Streaming K×K max — same linebuf shape, replace MAC tree with comparator tree.
2. **`upsample_2x`** — neck nearest-neighbor 2× (between backbone and head). Trivial: register every input twice per dim.
3. **Attention block** (`a2c2f` or `psa`) — deeper neck has attention. softmax16 already validated; need to compose Q/K/V projections + attention scores. Look at YOLO26n's PSA/A2C2f block structure in ONNX.
4. **Detect head** — YOLO26 is **end-to-end (no NMS)**, uses learned top-k. `box_decode` already validated; need to wire it up + top-k logic.

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
2. `hw/ip/conv_layer/README.md` — IP doc with parameter list and dataflow.
3. `integ/_layer_template/README.md` — original tiled template + retrofit recipe.
4. `integ/scale/scale_pkg.sv` (or `_dv.sv`) — per-layer P_COUT/P_CIN/CIN/COUT/K/etc constants.
5. `TASKS.md` — log of completed work.

## What I'd do next (concrete plan for the new AI)

1. **Wave 6**: Build L20–L25 using `conv_layer` shim. Reference integ/layer_11/. Spawn one agent for the whole wave; ~10 min wall-clock.
2. **Wave 7-22**: Continue L26–L101 in similar waves of 4-6 layers.
3. **Build `maxpool_5x5` IP** when reaching SPPF layers (L32-ish region).
4. **Build `upsample_2x` IP** when reaching neck (~L40+ region).
5. **Build attention block IPs** for A2C2f/PSA in deeper neck.
6. **Top-level integration** + E2E ORT validation on real images.

The hardest remaining technical risk is the **detect head** (end-to-end, learned top-k, no NMS). Look at the ONNX for `/model.23` and below before designing it.
