# layer_2_cv1 — micro-integration: YOLO26n `/model.2/cv1` Conv 1x1 + SiLU

End-to-end test that proves our SV building blocks correctly implement the
**entry of the first C3k2 block** (`/model.2/cv1`) of YOLO26n compared
against onnxruntime on the same model (`integ/yolo26n/model_int8.onnx`).

## Slice

ONNX nodes 18–26:

```
/model.1/act/Mul_output_0
 → DynamicQuantizeLinear  (u8, s_a, zp_a)        [node 18]
 → ConvInteger            (32 → 32, 1x1, stride 1, pad 0)  [node 19]
 → Cast i32 → fp32                              [node 20]
 → Mul (s_a · s_w)                              [nodes 21–22]
 → Add bias (fp32)                              [nodes 23–24]
 → Sigmoid → Mul (SiLU)                         [nodes 25–26]
```

Conv parameters: Cin=32, Cout=32, K=1, stride=1, pad=0, w_zp=0 (per-tensor
symmetric int8 weights). Input/output shape: 160×160.

For DV we drive an 8×8×32 ROI (no padding ring is needed because K=1) and
produce an 8×8×32 output ROI from each sample.

## Files

```
extract.py                # ORT-driven stimulus + reference generator (golden)
rtl/layer_2_cv1.sv        # DUT: 32 × dotN(N=16) → 2-phase accum → requant → SiLU
rtl/layer_2_cv1_ref.sv    # Behavioral SV golden (clocked real-math model)
dv/layer_2_cv1_tb.sv      # Verilator wrapper (flattened ports)
dv/layer_2_cv1_test.cc    # C++ test: loads stim, drives DUT, scores cos vs ORT
dv/Makefile               # build/run
stim/                     # generated stimulus + ORT/HW reference
```

## Design choices

- **Parameters from `scale_pkg::LAYER_2_*`:**
  `P_PIX=1, P_COUT=32, P_CIN=16`, `CIN=COUT=32`, `K=1`, `H=W=160`,
  `LAYER_2_CYCLES = 51200`.

- **MAC budget — exactly 512 units.**
  `P_PIX × P_COUT × P_CIN = 1 × 32 × 16 = 512` MAC units, instantiated as
  **32 × dotN(N=16)**. The full per-pixel reduction (Cin=32) is split into
  `N_PHASE = NCH_IN / N_LANE = 32/16 = 2` consecutive half-windows. A small
  per-channel 2-phase accumulator (1 cycle) sums the two phase outputs into
  one int32 accumulator, then drives requant.

- **1x1 conv ⇒ no spatial reuse.** The kernel window is exactly the
  per-pixel input vector, so `linebuf_kxk` collapses to a pass-through
  register and is **not instantiated** (the system-level integrator will
  bypass it for K=1 layers). The DV drives the per-pixel half-window
  directly from the C++ TB.

- **u8 → i8 fold (same algebraic rewrite as stem_l0/layer_1).**
  `i8 = u8 - 128`, `bias_eff[c] = bias[c] + s_acc · (128 - zp_a) · sum_w[c]`,
  with `s_acc = s_a · s_w` (per-tensor; `zp_w == 0`).
  HW-ref matches ORT to 0.0000 max-abs on all samples.

- **Architecture / cycle accounting.**
  | Stage             | Latency (cyc) |
  | ----------------- | ------------- |
  | dotN(N=16)        | 1 + ⌈log₂16⌉ = 5 |
  | 2-phase accum     | 1             |
  | requant           | 3             |
  | act_silu          | 1             |
  | output register   | 1             |
  | **Total**         | **11**        |
  Throughput: **1 output pixel / 2 cycles** (one phase per cycle).

  Full-frame: 25,600 output pixels × 2 cyc/pix = **51,200 cycles** (matches
  `scale_pkg::LAYER_2_CYCLES`, target `T_FRAME = 100,000`).

  Total MACs/frame: `H·W·Cin·Cout = 160·160·32·32 = 26,214,400`
  = `MAC_UNITS (512) × FRAME_CYCLES (51200)` exactly — the array is 100%
  utilised (no idle MAC cycles).

- **SiLU scales.** Empirical pre-SiLU range on normalised 0..1 inputs is
  ≈ −0.3 .. 32. We reuse the layer-1 choice `S_OUT_PRE = S_OUT_SILU =
  80/127 ≈ 0.630`; LUT step ≈ 0.63, which dominates the DUT-vs-ORT error
  for samples whose output dynamic range is small (`half`, `gradient`).
  Same caveat as layer_1.

- **Reused IPs (unmodified):** `dotN`, `requant` (`i32_to_fp16 → fp16_fma →
  fp16_to_i8_sat`), `act_silu`. `linebuf_kxk` not used (K=1 trivial case).
  `add_rq` not used at this layer.

## Comparison criterion

Random tiles (the spec's "≥3 random input tiles") are required to satisfy
`cosine ≥ 0.998` vs ORT. Low-dynamic-range corner tiles (`half`, `gradient`)
are reported informationally with `cosine ≥ 0.99` because the SiLU LUT
quantisation step (~0.6) dominates absolute error when the output range is
only a few units (mirrors the `integ/layer_1` analysis).

## Result (latest run)

8/8 samples PASS. 6 random tiles all clear cosine ≥ 0.998 vs ORT.

```
sample           cos        max_abs   mae      out_range   verdict
rand0            0.999615   0.5797    0.1319   28.660      PASS (required)
rand1            0.999589   0.5745    0.1319   22.396      PASS (required)
rand2            0.999551   0.6030    0.1331   32.138      PASS (required)
rand3            0.999495   0.5953    0.1410   22.592      PASS (required)
rand_low         0.999147   0.5777    0.1486   19.783      PASS (required)
rand_high        0.998937   0.5764    0.1540   21.689      PASS (required)
half             0.991970   0.5699    0.2430    3.887      PASS (info)
gradient         0.993911   0.5909    0.1998    4.922      PASS (info)
```

`DUT vs HW-ref` numbers are identical to `DUT vs ORT` to four decimal
places — the entire DUT-vs-ORT gap comes from the SiLU LUT quantisation
step, exactly as on layer_1.

## Running

```
make -C integ/layer_2_cv1/dv test          # builds verilator binary, runs test
make -C integ/layer_2_cv1/dv lint          # lint-only
```
