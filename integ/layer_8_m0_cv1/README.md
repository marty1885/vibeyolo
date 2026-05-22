# layer_8_m0_cv1 — micro-integration: YOLO26n `/model.4/m.0/cv1`

End-to-end test that the SV building blocks correctly implement the first
conv inside the C3k2 bottleneck of `/model.4` against onnxruntime on the
same model (`integ/yolo26n/model_int8.onnx`).

## Slice

```
/model.4/cv1/act/Mul_output_0
 → Split (axis=1, sizes 32/32)             ← C3k2 split
 → second half = /model.4/Slice_1_output_0
 → DynamicQuantizeLinear (u8, s_a, zp_a)
 → ConvInteger             (32 → 16, 3x3, stride 1, pad 1)
 → Cast i32 → fp32
 → Mul (s_a · s_w)
 → Add bias (fp32)
 → Sigmoid → Mul (SiLU)
```

ONNX node: `/model.4/m.0/cv1/conv/Conv_quant`. Initializers used:
`onnx::Conv_1675_quantized` (weights, (16,32,3,3)),
`onnx::Conv_1675_scale` (s_w), `onnx::Conv_1676` (bias).

Conv parameters: `Cin=32, Cout=16, K=3, stride=1, pad=1`. Spatial: 80×80
→ 80×80. The DV drives an 8×8 output ROI from a 10×10×32 input ROI with
the 1-pixel pad ring of real neighbour data taken from the
ORT-quantised `/model.4/Slice_1_output_0_quantized` tensor.

## Files

```
extract.py                                 # ORT-driven stimulus + reference generator
rtl/layer_8_m0_cv1.sv                      # DUT: 16-way parallel dotN/requant/silu
dv/layer_8_m0_cv1_tb.sv                    # Verilator wrapper (flattened ports)
dv/layer_8_m0_cv1_test.cc                  # C++ test: loads stim, drives DUT, scores
dv/Makefile                                # build/run
stim/                                      # generated stimulus + ORT/HW reference
```

## Design choices

- **Parameters from `scale_pkg::LAYER_8_*`.** `NCH_OUT=LAYER_8_COUT=16`,
  `N_LANE=LAYER_8_CIN*K*K=32*9=288`. Full-frame parallelism target is
  `P_PIX=1, P_COUT=16, P_CIN=24 → 76 800 cycles/frame`, within the
  `T_FRAME=100 000` budget. Product `P_PIX·P_COUT·P_CIN = 384`.

- **Bridge u8 → i8 fold.** Same as layer_3 / stem_l0:
  `x_i8 = clip(x_u8 - 128, -128, 127)`, and we rewrite the bias as
  `bias_eff[c] = bias[c] + s_a · s_w · (128 - zp_a) · Σ_w[c]`. HW-ref
  matches ORT to 0.0000 max-abs across all 6 samples.

- **Architecture: 16 × dotN(N=288) + 16 × requant + 16 × act_silu.** One
  3×3×32 window per cycle drives all 16 output channels in parallel.

  | Stage             | Latency           |
  | ----------------- | ----------------- |
  | dotN              | 1 + ⌈log₂288⌉ = 10|
  | requant           | 3                 |
  | act_silu          | 1                 |
  | **Total pipeline**| 14                |

  Throughput: 1 output pixel / cycle (16 channels in parallel). At
  P_CIN=24 the full-frame run takes 288/24 = 12 cycles/pixel × 6400 pix
  = 76 800 cycles, matching `LAYER_8_CYCLES` (≤ 100k cyc target).

- **Accumulator pre-shift (the layer_8 wrinkle).** With N_LANE=288 and
  strong filters, observed `|acc| ≤ 387 223` and was already > 65 504
  (fp16 max normal) on the `rand0` ROI at channel 4 (|acc| = 67 432).
  Naïve `i32 → fp16` saturates to ±Inf there. The DUT arithmetic
  right-shifts dot_acc by `ACC_SHIFT = 3` bits before requant and the
  host scales `scale_fp16` by `2**ACC_SHIFT = 8` to compensate. After
  the shift worst-case |acc| ≈ 48 400, comfortably inside fp16. The 7-LSB
  loss maps to ≤ 7 · s_a · s_w ≈ 7e-4 in pre-SiLU space, far below the
  S_OUT_PRE LUT step (~0.031).

  **L6 gotcha applied:** `arr[i] >>> n` on a packed signed array is a
  *logical* shift. The shift uses `$signed(dot_acc[gc]) >>> ACC_SHIFT`
  inside a per-channel generate block so sign-bits propagate correctly.

- **SiLU scales.** Observed pre-SiLU range on this layer is ≈ -0.3 .. +2.7
  (wider than layer_3 because Cin doubled and weights aren't proportionally
  smaller). We pick `S_OUT_PRE = S_OUT_SILU = 4.0 / 127 ≈ 0.0315` so the
  i8 LUT spans ±4.0 — covers the dynamic range with ≤ 1 LUT-step of
  rounding noise.

## linebuf_kxk note

The unit DV drives windows directly from the C++ TB. System-level
integration wires this layer to the upstream Split with either (a) 32
parallel `linebuf_kxk` instances (one per input channel, K=3, W=80) or
(b) a multi-channel linebuf — same situation as layer_1 / layer_3. The
shared `hw/ip/linebuf_kxk`, `hw/ip/requant`, and `hw/ip/act_silu`
modules are unmodified.

## Comparison criterion

Per sample: **PASS** iff `cosine_sim > 0.998` AND
`mean_abs_err < 5% · ORT output dynamic range`.

## Result (latest run)

All 6/6 samples PASS. Aggregate: avg cos = 0.999927, avg mae = 0.0102,
worst max-abs = 0.0332 (≈ 1 LUT step at `S_OUT_SILU = 4/127`).

```
sample      max_abs   mae       cos          out_range   thresh    verdict
rand0       0.0316    0.0103    0.999928     2.850       0.1425    PASS
rand1       0.0313    0.0100    0.999928     2.947       0.1473    PASS
half        0.0227    0.0109    0.999923     2.052       0.1026    PASS
gradient    0.0332    0.0095    0.999929     2.282       0.1141    PASS
rand_low    0.0316    0.0103    0.999925     2.720       0.1360    PASS
rand_high   0.0324    0.0101    0.999931     3.013       0.1507    PASS
```

`DUT vs HW-ref` matches `DUT vs ORT` to four decimal places — entire
residual is SiLU LUT + acc-shift quantisation noise.

## Running

```
make -C integ/layer_8_m0_cv1/dv test          # builds verilator binary, runs test
make -C integ/layer_8_m0_cv1/dv lint          # lint-only
```
