# layer_9_m0_cv2 — micro-integration: YOLO26n `/model.4/m.0/cv2` Conv-BN-SiLU + bottleneck residual

End-to-end test that proves our shared SV building blocks correctly implement
**L9 = `/model.4/m.0/cv2`** — the second Conv-BN-SiLU inside the C3k2
bottleneck of `/model.4` — together with the bottleneck's post-SiLU residual
add, compared against onnxruntime on the same model
(`integ/yolo26n/model_int8.onnx`).

## Slice

```
/model.4/m.0/cv1/act/Mul_output_0      (16-ch, post-SiLU of cv1, 80×80)
 → DynamicQuantizeLinear  (u8, s_a, zp_a)
 → ConvInteger            (16 → 32, 3×3, stride 1, pad 1)
 → Cast i32 → fp32
 → Mul (s_a · s_w)
 → Add bias (fp32)
 → Sigmoid → Mul (SiLU)   → /model.4/m.0/cv2/act/Mul_output_0  (32-ch)
 → Add /model.4/Slice_1_output_0       (residual, 32-ch, indep. u8 dyn-quant)
                            → /model.4/m.0/Add_output_0
```

Conv parameters (`scale_pkg::LAYER_9_*`): `Cin=16, Cout=32, K=3, stride=1,
pad=1`, input/output `80×80`. Weight initializer `onnx::Conv_1678_quantized`
shape `(32,16,3,3)`. Target frame cycles = `LAYER_9_CYCLES = 76,800` at
`(P_PIX, P_COUT, P_CIN) = (1, 32, 12)`.

### Residual width verification

`shape_inference` on `model_int8.onnx`:

```
/model.4/cv1/act/Mul_output_0   shape [N, 64, 80, 80]
/model.4/Slice_1_output_0       shape [N, 32, 80, 80]    ← residual operand
/model.4/m.0/cv2/act/Mul_output_0 shape [N, 32, 80, 80]
```

The residual is the **full-width 32-channel** second half of the C3k2 cv1
output. The add is therefore a 32-channel elementwise add applied **after**
the cv2 SiLU. (Same pattern as L4 — confirmed against the live graph rather
than relying on the task description.)

## Files

```
extract.py                       # ORT-driven stimulus + reference generator
rtl/layer_9_m0_cv2.sv            # DUT: dotN/requant/silu + add_rq pipeline
dv/layer_9_m0_cv2_tb.sv          # Verilator wrapper (flattened ports)
dv/layer_9_m0_cv2_test.cc        # C++ test: loads stim, drives DUT, scores
dv/Makefile                      # build/run
stim/                            # generated stimulus + ORT/HW reference
```

## Design choices

- **Reuse-only.** No shared IPs were modified. Uses `dotN`, `requant`
  (internally `i32_to_fp16` + `fp16_fma` + `fp16_to_i8_sat`), `act_silu`,
  and `add_rq`. `linebuf_kxk` is **not** instantiated in this unit-level
  test (same as L4); the system integrator will wire 16 instances of
  `linebuf_kxk` (one per cv2 input channel) in front of this layer.

- **u8 → i8 fold.** `i8 = u8 − 128`, with the per-channel bias absorbing the
  zero-point shift:
  `bias_eff[c] = bias[c] + s_a · s_w · (128 − zp_a) · sum_w[c]`.

- **u8 → i8 bridge on the residual via bias rewrite.** The residual second
  operand also enters `add_rq` as `i8 = u8 − 128`; the `s_r · (128 − zp_r)`
  DC term is folded into `add_rq`'s post-`inv_out_scale` bias:
  `add_bias_fp16 = s_r · (128 − zp_r) / S_OUT_ADD`.

- **Architecture: 32 × dotN(N=144) + 32 × requant + 32 × act_silu + 32 ×
  add_rq.** One `3×3×16` window plus matching 32-ch residual sample per
  cycle drives all 32 output channels in parallel. Cycle accounting:

  | Stage             | Latency |
  | ----------------- | ------- |
  | dotN              | 1 + ⌈log₂144⌉ = 9 |
  | requant           | 3 |
  | act_silu          | 1 |
  | add_rq            | 5 |
  | **Total pipeline**| 18 |

  Throughput: 1 output pixel / cycle (32 channels in parallel). The residual
  arrives with `valid_i` and is delayed by `CV2_LAT = 13` cycles inside the
  DUT so it aligns with the cv2-post-SiLU stream at `add_rq.b`.

- **Scales.** L9 sees a much smaller dynamic range than L4 (pre-SiLU
  empirical ~−0.3..5.0; post-residual ~−0.6..5.3). We use
  `S_OUT_PRE = S_OUT_SILU = S_OUT_ADD = 6.5 / 127 ≈ 0.0512`. This brings the
  LUT step well below the per-pixel ORT error and gives the HW-ref `cos ≈
  0.99997` per tile. With L4-style ±80 scales the LUT step would dominate
  and tiles fall below the 0.998 bound.

- **Frame cycles.** 80×80 = 6400 pixels × 1 cyc/pix + 17 cyc fill ≈ 6,417
  cycles in this unit DUT. At full integration with `P_CIN = 12` (as
  prescribed by `scale_pkg`) the implied frame cost is
  `(Cin·K·K / P_CIN) · H · W = (144/12)·6400 = 76,800` cycles — i.e. the
  `LAYER_9_CYCLES` budget, well under the 100k target.

## Comparison criterion

Per sample: cosine vs ORT and DUT-vs-HW-ref reported. Overall **PASS**:
at least **3 of 6 tiles** meet `cos ≥ 0.998` vs the ORT
`/model.4/m.0/Add_output_0` reference (task spec).

## Result (latest run)

```
sample      max_abs   mae      cos        out_range   thresh   verdict
rand0       0.0771    0.0209   0.999907   5.720       0.286    PASS
rand1       0.0799    0.0207   0.999910   5.858       0.293    PASS
half        0.0416    0.0177   0.999939   3.656       0.183    PASS
gradient    0.0781    0.0208   0.999913   4.301       0.215    PASS
rand_low    0.0765    0.0205   0.999911   5.818       0.291    PASS
rand_high   0.0773    0.0207   0.999913   5.669       0.283    PASS

Aggregate: 6/6 tiles ≥ cos 0.998   →   meets spec
avg cos = 0.999916
worst max_abs = 0.0799
```

## Running

```
make -C integ/layer_9_m0_cv2/dv lint    # lint-only
make -C integ/layer_9_m0_cv2/dv test    # regenerates stim, builds, runs
```
