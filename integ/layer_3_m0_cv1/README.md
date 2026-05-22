# layer_3_m0_cv1 — micro-integration: YOLO26n `/model.2/m.0/cv1`

End-to-end test that proves our SV building blocks correctly implement the
first conv inside the C3k2 bottleneck of `/model.2` against onnxruntime on
the same model (`integ/yolo26n/model_int8.onnx`).

## Slice

```
/model.2/cv1/act/Mul_output_0
 → Split (axis=1, sizes 16/16)        ← C3k2 split
 → second half = /model.2/Slice_1_output_0
 → DynamicQuantizeLinear (u8, s_a, zp_a)
 → ConvInteger           (16 → 8, 3x3, stride 1, pad 1)
 → Cast i32 → fp32
 → Mul (s_a · s_w)
 → Add bias (fp32)
 → Sigmoid → Mul (SiLU)
```

Conv parameters: `Cin=16, Cout=8, K=3, stride=1, pad=1`. Spatial: 160×160
→ 160×160. We treat the 16-channel input as given (the upstream Split is
wired externally by the system integrator).

For DV we drive a 10×10×16 input ROI (with the 1-pixel pad ring of *real*
neighbour data taken straight from the ORT-quantised
`/model.2/Slice_1_output_0_quantized` tensor) and produce an 8×8×8 output
ROI. All scales/zero-points are pulled from ORT intermediate outputs.

## Files

```
golden.py                                 # ORT-driven stimulus + reference generator
rtl/layer_3_m0_cv1.sv                     # DUT: 8-way parallel dotN/requant/silu
dv/layer_3_m0_cv1_tb.sv                   # Verilator wrapper (flattened ports)
dv/layer_3_m0_cv1_test.cc                 # C++ test: loads stim, drives DUT, scores
dv/Makefile                               # build/run
stim/                                     # generated stimulus + ORT/HW reference
```

## Design choices

- **Parameters from `scale_pkg::LAYER_3_*`.** `NCH_OUT=LAYER_3_COUT=8`,
  `N_LANE=LAYER_3_CIN*K*K=16*9=144`. Full-frame parallelism target is
  `P_PIX=1, P_COUT=8, P_CIN=48 → 76 800 cycles/frame`, which matches the
  `T_FRAME=100 000` budget. Product `P_PIX·P_COUT·P_CIN = 384`.

- **Bridge u8 → i8 fold.** Same as stem_l0 / layer_1:
  `x_i8 = clip(x_u8 - 128, -128, 127)`, and we rewrite the bias as
  `bias_eff[c] = bias[c] + s_a · s_w · (128 - zp_a) · Σ_w[c]`. HW-ref
  matches ORT to 0.0000 max-abs across all 6 samples.

- **Architecture: 8 × dotN(N=144) + 8 × requant + 8 × act_silu.** One
  3×3×16 window per cycle drives all 8 output channels in parallel.
  Identical dataflow to layer_1, just narrower.

  | Stage             | Latency           |
  | ----------------- | ----------------- |
  | dotN              | 1 + ⌈log₂144⌉ = 9 |
  | requant           | 3                 |
  | act_silu          | 1                 |
  | **Total pipeline**| 13                |

  Throughput: 1 output pixel / cycle (8 channels in parallel). At
  P_CIN=48 the full-frame run takes 144/48 = 3 cycles/pixel × 25 600 pix
  = 76 800 cycles, matching `LAYER_3_CYCLES`.

- **SiLU scales.** Pre-SiLU range observed empirically on this layer is
  small (≈ -0.3 .. 1.0) — much tighter than the stem. We pick
  `S_OUT_PRE = S_OUT_SILU = 2.0 / 127 ≈ 0.01575`, giving an LUT step of
  ~0.016 that captures the dynamic range with negligible quantisation
  noise (max-abs DUT-vs-ORT is ~0.02 across all samples).

## linebuf_kxk note

This unit DV drives windows directly from the C++ TB rather than
using `linebuf_kxk` (Channels=1 only). When the system-level integrator
wires this layer to the upstream Split, it will need to either (a)
instantiate 16 parallel `linebuf_kxk` modules (one per input channel) or
(b) generalise `linebuf_kxk` to support multi-channel interleaved
streams — same situation as layer_1.

## Comparison criterion

Per sample: **PASS** iff `cosine_sim > 0.998` AND
`mean_abs_err < 5% · ORT output dynamic range`.

## Result (latest run)

All 6/6 samples PASS. Aggregate: avg cos = 0.999733, avg mae = 0.0046,
worst max-abs = 0.0216 (≈ 1.4 LUT steps at `S_OUT_SILU = 2/127`).

```
sample      max_abs   mae       cos          out_range   thresh    verdict
rand0       0.0216    0.0044    0.999758     1.127       0.0563    PASS
rand1       0.0132    0.0044    0.999753     1.037       0.0519    PASS
half        0.0108    0.0056    0.999713     0.591       0.0296    PASS
gradient    0.0139    0.0049    0.999785     0.668       0.0334    PASS
rand_low    0.0127    0.0041    0.999711     0.931       0.0465    PASS
rand_high   0.0131    0.0043    0.999676     0.801       0.0401    PASS
```

`DUT vs HW-ref` numbers match `DUT vs ORT` to four decimal places — the
entire residual is SiLU LUT quantisation noise.

## Running

```
make -C integ/layer_3_m0_cv1/dv test          # builds verilator binary, runs test
make -C integ/layer_3_m0_cv1/dv lint          # lint-only
```
