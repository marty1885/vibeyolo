# layer_1 — micro-integration: YOLO26n `/model.1` Conv-BN-SiLU

End-to-end test that proves our SV building blocks correctly implement the
second int8 conv block of YOLO26n compared against onnxruntime on the same
model (`integ/yolo26n/model_int8.onnx`).

## Slice

ONNX nodes 9–17:

```
/model.0/act/Mul_output_0
 → DynamicQuantizeLinear  (u8, s_a, zp_a)
 → ConvInteger            (16 → 32, 3x3, stride 2, pad 1)
 → Cast i32 → fp32
 → Mul (s_a · s_w)
 → Add bias (fp32)
 → Sigmoid
 → Mul (SiLU)
```

Conv parameters: Cin=16, Cout=32, K=3, stride=2, pad=1.
Input shape: 320×320, output: 160×160.

For DV we drive an 18×18×16 ROI (with the pad-1 ring of *real* neighbour data,
taken straight from the ORT-quantised `/model.0/act/Mul_output_0_quantized`)
and produce an 8×8×32 output ROI.

## Files

```
golden.py                # ORT-driven stimulus + reference generator
rtl/layer_1.sv           # DUT: 32-way parallel dotN/requant/silu pipeline
dv/layer_1_tb.sv         # Verilator wrapper (flattened ports)
dv/layer_1_test.cc       # C++ test: loads stim, drives DUT, scores
dv/Makefile              # build/run
stim/                    # generated stimulus + ORT/HW reference
```

## Design choices

- **Parameters from `scale_pkg::LAYER_1_*`.** `NCH_OUT=LAYER_1_COUT=32` and
  `N_LANE=LAYER_1_CIN*K*K=16*9=144`. The full-frame parallelism target is
  `P_PIX=1, P_COUT=32, P_CIN=48 → ~76800 cycles/frame` (see
  `integ/scale/scale_report.md` line `| 1 |`).

- **u8 → i8 fold.** Same as stem_l0: `i8 = u8 - 128`,
  `bias_eff[c] = bias[c] + s_a · s_w · (128 - zp_a) · sum_w[c]`.
  HW-ref matches ORT to 0.0000 max-abs.

- **Architecture: 32 × dotN(N=144) + 32 × requant + 32 × act_silu.** One
  3×3×16 window per cycle drives all 32 output channels simultaneously.
  This is the same dataflow as stem_l0, just wider. Cycle accounting:
  | Stage             | Latency |
  | ----------------- | ------- |
  | dotN              | 1 + ⌈log₂144⌉ = 9 |
  | requant           | 3 |
  | act_silu          | 1 |
  | **Total pipeline**| 13 |
  Throughput: 1 output pixel / cycle (32 channels in parallel).

- **SiLU scales.** Empirical pre-SiLU range on normalised 0..1 inputs is
  ≈ -0.3 .. 60. We pick `S_OUT_PRE = S_OUT_SILU = 80/127 ≈ 0.630` (covers
  ±80 in both pre- and post-SiLU). LUT step ≈ 0.63, which is the dominant
  observed DUT-vs-ORT error.

## linebuf_kxk note

This layer's unit DV drives windows directly from the C++ TB rather than
using `linebuf_kxk` (Channels=1 only, and we'd need 16 instances). When
the system-level integrator wires layer_1 to its upstream stage, it will
need to either (a) instantiate 16 parallel `linebuf_kxk` modules (one per
input channel) or (b) generalise `linebuf_kxk` to support multi-channel
interleaved streams. See `TASKS.md` open issue on `linebuf_kxk` (`clr_i`
between frames) — not exercised by this unit test.

## Comparison criterion

Per sample: **PASS** iff `cosine_sim > 0.99` AND `mean_abs_err < 5% · ORT
output dynamic range`.

## Result (latest run)

All 6 samples PASS. Aggregate: avg cos = 0.998687, avg mae = 0.149,
worst max-abs = 0.581 (≈ one LUT step at `S_OUT_SILU = 80/127`).

```
sample      max_abs   mae      cos        out_range   thresh   verdict
rand0       0.5739    0.1442   0.999895   59.654      2.983    PASS
rand1       0.5730    0.1350   0.999853   61.298      3.065    PASS
half        0.4862    0.1771   0.996076    7.444      0.372    PASS
gradient    0.5808    0.1685   0.997041    7.729      0.387    PASS
rand_low    0.5745    0.1387   0.999735   38.772      1.939    PASS
rand_high   0.5766    0.1452   0.999555   31.878      1.594    PASS
```

`DUT vs HW-ref` numbers are identical to `DUT vs ORT` to four decimal places
— the entire DUT-vs-ORT gap comes from the SiLU LUT quantisation step.

## Running

```
make -C integ/layer_1/dv test          # builds verilator binary, runs test
make -C integ/layer_1/dv lint          # lint-only
```
