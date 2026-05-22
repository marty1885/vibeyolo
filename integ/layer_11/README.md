# layer_11 — micro-integration: YOLO26n `/model.5/conv` Conv-BN-SiLU

End-to-end test that proves our SV building blocks correctly implement the
L11 int8 conv block of YOLO26n compared against onnxruntime on the same
model (`integ/yolo26n/model_int8.onnx`).

## Slice

```
/model.4/cv2/act/Mul_output_0
 -> DynamicQuantizeLinear   (u8, s_a, zp_a)
 -> ConvInteger             (128 -> 128, 3x3, stride 2, pad 1)
 -> Cast i32 -> fp32
 -> Mul (s_a * s_w)
 -> Add bias                (fp32)
 -> Sigmoid -> Mul          (SiLU)
 -> /model.5/act/Mul_output_0
```

Conv parameters: Cin=128, Cout=128, K=3, stride=2, pad=1.
Input plane 80x80 -> output 40x40. ONNX node `/model.5/conv/Conv_quant`
with initializers `onnx::Conv_1684_quantized` (weights),
`onnx::Conv_1684_scale`, `onnx::Conv_1684_zero_point`, `onnx::Conv_1685`
(bias).

For DV we drive an 18x18x128 input ROI (with the pad-1 ring taken from real
neighbour data, i.e. the ORT-quantised `/model.4/cv2/act/Mul_output_0`
tensor) and produce an 8x8x128 output ROI starting at (R//2, C//2)=(8,8)
on the 40x40 output plane.

## Files

```
extract.py               # ORT-driven stimulus + reference generator
rtl/layer_11.sv          # DUT: 128-way parallel dotN/requant/silu pipeline
dv/layer_11_tb.sv        # Verilator wrapper (flattened ports)
dv/layer_11_test.cc      # C++ test: loads stim, drives DUT, scores
dv/Makefile              # build/run
stim/                    # generated stimulus + ORT/HW reference
```

## Design choices

- **Parameters from `scale_pkg::LAYER_11_*`.** `NCH_OUT=LAYER_11_COUT=128`
  and `N_LANE=LAYER_11_CIN*K*K=128*9=1152`. The full-frame parallelism
  target is `P_PIX=1, P_COUT=128, P_CIN=24 -> 9*128/24 * 40*40 = 76800
  cycles/frame` (matches `scale_pkg::LAYER_11_CYCLES`; <= 100k budget).

- **u8 -> i8 fold.** Same as layer_1 / layer_6 / stem_l0: `i8 = u8 - 128`,
  `bias_eff[c] = bias[c] + s_a * s_w * (128 - zp_a) * sum_w[c]`.
  HW-ref matches ORT to 0.0000 max-abs across all samples.

- **Architecture: 128 x dotN(N=1152) + 128 x requant + 128 x act_silu.**
  One 3x3x128 window per cycle drives all 128 output channels
  simultaneously. Cycle accounting:
  | Stage             | Latency |
  | ----------------- | ------- |
  | dotN              | 1 + ceil(log2(1152)) = 12 |
  | requant           | 3 |
  | act_silu          | 1 |
  | **Total pipeline**| 16 |
  Throughput: 1 output pixel / cycle (128 channels in parallel).

- **Stride-2.** The TB steps the window position by STRIDE=2 on each
  output pixel, so the dotN pipeline still runs at full rate and the
  decimation is implicit at the read side (same pattern as layer_1 and
  layer_6).

- **Accumulator pre-scale (ACC_SHIFT=3).** With N_LANE=1152 the i32 dotN
  accumulator reaches ~107k on real YOLO26n L11 activations, which
  exceeds fp16's representable range (max ~65504) and would saturate the
  `i32_to_fp16` front of `requant` to +/-Inf. We arithmetic-shift-right
  `dot_acc` by 3 bits (/8) inside `layer_11.sv` and pre-multiply the
  per-channel `scale_fp16` by 2^3=8 in `extract.py` so the math is
  algebraically unchanged. After the shift, max |acc| ~ 13.4k, well
  inside fp16's representable range with comfortable margin. The shift
  is implemented as `$signed(dot_acc[gc]) >>> ACC_SHIFT` — the `$signed`
  cast is required because indexing a packed array yields an unsigned
  slice in SystemVerilog by default, which silently breaks ASR on
  negative accumulators.

- **SiLU scales.** Empirical pre-SiLU range on normalised 0..1 inputs is
  ~ -0.28 .. 2.95, much tighter than layer_1. We pick
  `S_OUT_PRE = S_OUT_SILU = 4/127 ~ 0.0315` to span +/-4 with LUT step ~
  0.0315 — well below 1% of the output dynamic range.

## linebuf_kxk note

This layer's unit DV drives windows directly from the C++ TB rather than
using `linebuf_kxk`. When the system-level integrator wires layer_11 to
its upstream stage, it will instantiate `linebuf_kxk` (K=3, C=128,
W=H=80) with stride-2 decimation at the read side (or 128 parallel
linebufs, depending on the multi-channel strategy chosen for upstream
layers).

## Comparison criterion

Per sample: **PASS** iff `cosine_sim > 0.998` AND `mean_abs_err < 5% *
ORT output dynamic range`.

## Result (latest run)

ORT-vs-HW-reference (i.e. the fp32 model of our SV pipeline) is
bit-identical to 4 decimals across all six samples (max_abs = 0.0000,
cos = 1.000000+/-eps), confirming the u8->i8 fold and the ACC_SHIFT
pre-scale are arithmetically lossless. The DUT-vs-ORT gap is dominated
by SiLU LUT quantisation at S_OUT_SILU = 4/127.

See the test stdout (printed during `make test`) for the DUT-vs-ORT
table.

## Running

```
make -C integ/layer_11/dv test    # builds verilator binary, runs test
make -C integ/layer_11/dv lint    # lint-only
```
