# layer_6 — micro-integration: YOLO26n `/model.3/conv` Conv-BN-SiLU

End-to-end test that proves our SV building blocks correctly implement the
L6 int8 conv block of YOLO26n compared against onnxruntime on the same
model (`integ/yolo26n/model_int8.onnx`).

## Slice

```
/model.2/cv2/act/Mul_output_0
 → DynamicQuantizeLinear   (u8, s_a, zp_a)
 → ConvInteger             (64 → 64, 3×3, stride 2, pad 1)
 → Cast i32 → fp32
 → Mul (s_a · s_w)
 → Add bias                (fp32)
 → Sigmoid → Mul           (SiLU)
 → /model.3/act/Mul_output_0
```

Conv parameters: Cin=64, Cout=64, K=3, stride=2, pad=1.
Input plane 160×160 → output 80×80.

For DV we drive an 18×18×64 input ROI (with the pad-1 ring taken from real
neighbour data, i.e. the ORT-quantised `/model.2/cv2/act/Mul_output_0`
tensor) and produce an 8×8×64 output ROI starting at (R//2, C//2)=(16,16)
on the 80×80 output plane.

## Files

```
extract.py               # ORT-driven stimulus + reference generator
rtl/layer_6.sv           # DUT: 64-way parallel dotN/requant/silu pipeline
dv/layer_6_tb.sv         # Verilator wrapper (flattened ports)
dv/layer_6_test.cc       # C++ test: loads stim, drives DUT, scores
dv/Makefile              # build/run
stim/                    # generated stimulus + ORT/HW reference
```

## Design choices

- **Parameters from `scale_pkg::LAYER_6_*`.** `NCH_OUT=LAYER_6_COUT=64` and
  `N_LANE=LAYER_6_CIN*K*K=64*9=576`. The full-frame parallelism target is
  `P_PIX=1, P_COUT=64, P_CIN=48 → ~76800 cycles/frame` (matches
  `scale_pkg::LAYER_6_CYCLES`; ≤100k budget).

- **u8 → i8 fold.** Same as layer_1 / stem_l0: `i8 = u8 - 128`,
  `bias_eff[c] = bias[c] + s_a · s_w · (128 - zp_a) · sum_w[c]`.
  HW-ref matches ORT to 0.0000 max-abs across all samples.

- **Architecture: 64 × dotN(N=576) + 64 × requant + 64 × act_silu.** One
  3×3×64 window per cycle drives all 64 output channels simultaneously.
  Cycle accounting:
  | Stage             | Latency |
  | ----------------- | ------- |
  | dotN              | 1 + ⌈log₂576⌉ = 11 |
  | requant           | 3 |
  | act_silu          | 1 |
  | **Total pipeline**| 15 |
  Throughput: 1 output pixel / cycle (64 channels in parallel).

- **Stride-2.** The TB steps the window position by STRIDE=2 on each
  output pixel, so the dotN pipeline still runs at full rate and the
  decimation is implicit at the read side (same pattern as `layer_1`).

- **Accumulator pre-scale (ACC_SHIFT=3).** With N_LANE=576 the i32 dotN
  accumulator can reach ~115k for YOLO26n L6, which exceeds fp16's
  representable range (max ≈ 65504) and would saturate the
  `i32_to_fp16` front of `requant` to ±Inf. We arithmetic-shift-right
  `dot_acc` by 3 bits (÷8) inside `layer_6.sv` and pre-multiply the
  per-channel `scale_fp16` by 2³=8 in `extract.py` so the math is
  algebraically unchanged. The shift is implemented as
  `$signed(dot_acc[gc]) >>> ACC_SHIFT` — note the `$signed` cast is
  required because indexing a packed array yields an unsigned slice in
  SystemVerilog by default, which silently breaks ASR on negative
  accumulators.

- **SiLU scales.** Empirical pre-SiLU range on normalised 0..1 inputs is
  ≈ -0.3 .. 6.1, much tighter than layer_1. We pick
  `S_OUT_PRE = S_OUT_SILU = 8/127 ≈ 0.063` to span ±8 with LUT step ≈
  0.063 — small enough that the LUT quantisation contributes < 1% of
  the output dynamic range.

## linebuf_kxk note

This layer's unit DV drives windows directly from the C++ TB rather than
using `linebuf_kxk`. When the system-level integrator wires layer_6 to
its upstream stage, it will instantiate `linebuf_kxk` (K=3, C=64,
W=H=160) with stride-2 decimation at the read side (or 64 parallel
linebufs, depending on the multi-channel strategy already chosen for
upstream layers).

## Comparison criterion

Per sample: **PASS** iff `cosine_sim > 0.998` AND `mean_abs_err < 5% ·
ORT output dynamic range`.

## Result (latest run)

All 6 samples PASS. Aggregate: avg cos = 0.999792, avg mae = 0.0187,
worst max-abs = 0.065 (≈ one LUT step at `S_OUT_SILU = 8/127`).

```
sample      max_abs   mae      cos        out_range   verdict
rand0       0.064     0.019    0.99980    6.370       PASS
rand1       0.063     0.019    0.99979    6.064       PASS
half        0.056     0.018    0.99975    2.936       PASS
gradient    0.060     0.019    0.99974    3.043       PASS
rand_low    0.061     0.019    0.99981    5.649       PASS
rand_high   0.065     0.019    0.99978    5.219       PASS
```

`DUT vs HW-ref` numbers are identical to `DUT vs ORT` to four decimal
places — the entire DUT-vs-ORT gap comes from the SiLU LUT quantisation
step.

## Running

```
make -C integ/layer_6/dv test          # builds verilator binary, runs test
make -C integ/layer_6/dv lint          # lint-only
```
