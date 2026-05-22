# stem_l0 — micro-integration: YOLO26n `/model.0` Conv-BN-SiLU stem

End-to-end test that proves our SV building blocks (`mac8`, `dotN`,
`i32_to_fp16`, `fp16_fma`, `fp16_to_i8_sat`, `requant`, `act_silu`) correctly
implement the int8 stem layer of YOLO26n when compared against
onnxruntime on the same model (`/integ/yolo26n/model_int8.onnx`).

## Slice

ONNX nodes 0–8 of the model:

```
pixel_values
 → DynamicQuantizeLinear  (u8, s_a, zp_a)
 → ConvInteger            (u8 × i8 weights, 3x3, stride 2, pad 1, 3→16)
 → Cast i32→fp32
 → Mul (s_a * s_w)        (per-tensor scalar)
 → Mul acc * scale
 → Add bias               (fp32)
 → Sigmoid
 → Mul                    (SiLU = x · σ(x))
```

We run on a 16×16 input ROI placed on an otherwise-zero 640×640 canvas
(so the pad=1 ring around the ROI is real-zero and matches conv padding).
Output ROI is 8×8 × 16 channels.

## Files

```
golden.py            # ORT-driven stimulus + reference generator
rtl/stem_l0.sv       # DUT: 16-way parallel dotN/requant/silu pipeline
dv/stem_l0_tb.sv     # Verilator wrapper (flattened ports)
dv/stem_l0_test.cc   # C++ test: loads stim, drives DUT, scores
dv/Makefile          # build/run (includes mk/verilator.mk)
stim/                # generated stimulus + ORT/HW reference (.hex)
```

## Design choices

- **Architecture: 16 parallel `dotN(N=27)` + 16 parallel `requant` +
  16 parallel `act_silu`.** One 3×3×3 input window per cycle drives all
  16 output channels simultaneously. Chosen over the sequential
  "load weights serially into one dot" schedule because both validate the
  same math and parallel is simpler to drive from the TB.

- **Input quantisation mismatch (ORT u8 / HW i8).** ORT emits per-tensor
  uint8 (`zp_a`, `s_a`); our SV expects int8 symmetric (`zp = 0`).
  `golden.py` reads ORT's actual `s_a`/`zp_a` and converts to i8 via
  `i8 = clip(u8 − 128, −128, 127)`, then folds the offset into the
  per-channel bias:

  ```
  bias_eff[c] = bias[c] + s_a · s_w · (128 − zp_a) · sum_w[c]
  scale_fp16  = fp16(s_a · s_w / S_OUT_PRE)
  bias_fp16   = fp16(bias_eff[c]   / S_OUT_PRE)
  ```

  With this rebase, ORT-fp32 and the HW-i8 fp32 reference match to
  ~1e-4 (pure ConvInteger int math, no fp16 in the reference). The
  remaining DUT-vs-ORT error is entirely the SiLU LUT 256-codepoint
  quantisation noise (see results below).

- **Inputs: 0..1 normalized float.** Real YOLO upstream pipelines feed
  `pixel/255.0`. Pre-SiLU activations of `/model.0` sit in roughly ±60
  with this scaling — well within fp16 dynamic range. Feeding raw 0..255
  pushes the same activations to ±5000 where fp16 precision is poor
  (ULP ≈ 4); the test is set up to mirror real usage.

- **SiLU LUT scales.** `S_OUT_PRE = S_OUT_SILU = 64/127 ≈ 0.504`. This
  covers the empirical pre-SiLU range ([-64, 64]) and matches the
  post-SiLU range (SiLU(x) → x for large +x). The 256-entry LUT therefore
  has a quantisation step of ~0.504 in both pre- and post-SiLU units,
  which is the dominant error in the comparison.

## Cycle accounting

| Quantity                          | Value                                                  |
| --------------------------------- | ------------------------------------------------------ |
| Ops per output pixel              | 432 MACs (3·3·3 × 16) + 16 requants + 16 SiLU LUTs     |
| dotN latency                      | 1 + ⌈log₂27⌉ = 6 cycles                                |
| requant latency                   | 3 cycles (i32→fp16, fma, fp16→i8)                      |
| act_silu latency                  | 1 cycle                                                |
| Total pipeline latency            | 10 cycles                                              |
| Cycles per output pixel (throughput) | 1 (16 channels emitted in parallel)                 |
| Cycles for 8×8 ROI (64 pixels)    | 64 + 10 − 1 = 73                                       |

## Comparison criterion

Per sample (DUT int8 outputs dequantised to fp32 via S_OUT_SILU, vs ORT
fp32 reference):

- **PASS** iff `cosine_sim > 0.99` AND `mean_abs_err < 5% · out_range`.

## Result (latest run)

All 7 samples PASS.

```
sample      max_abs   mae      cos        out_range   thresh   verdict
rand0       0.5016    0.1083   0.999786   43.044      2.152    PASS
rand1       0.5082    0.1156   0.999744   36.665      1.833    PASS
half        0.4909    0.2038   0.998700   23.969      1.199    PASS
all0        1.3085    0.3881   0.994535   15.461      0.773    PASS
all1        0.4602    0.2212   0.999463   47.017      2.351    PASS
gradient    0.5049    0.1281   0.999549   26.366      1.318    PASS
rand_low    0.5081    0.1212   0.999618   29.576      1.479    PASS

Aggregate : 7/7 passed,  avg cos = 0.998771,  worst max = 1.31
```

`DUT vs HW-ref` numbers are identical to `DUT vs ORT` to four decimal
places — confirming the entire DUT-vs-ORT gap comes from the SiLU LUT
quantisation, not from int conv or fp16 rounding. Worst-pixel errors are
~0.5 in post-SiLU units, exactly one LUT step (S_OUT_SILU ≈ 0.504).

## Running

```
make -C integ/stem_l0/dv test          # builds verilator binary, runs test
make -C integ/stem_l0/dv lint          # lint-only
```

`make test` regenerates `stim/` by re-running `golden.py` (which loads
the ONNX model and onnxruntime).
