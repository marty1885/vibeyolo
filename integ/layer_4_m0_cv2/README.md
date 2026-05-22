# layer_4_m0_cv2 — micro-integration: YOLO26n `/model.2/m.0/cv2` Conv-BN-SiLU + bottleneck residual

End-to-end test that proves our shared SV building blocks correctly implement
**L4 = `/model.2/m.0/cv2`** — the second Conv-BN-SiLU inside the C3k2
bottleneck of `/model.2` — together with the bottleneck's post-SiLU residual
add, compared against onnxruntime on the same model (`integ/yolo26n/model_int8.onnx`).

## Slice

ONNX nodes 38–47:

```
/model.2/m.0/cv1/act/Mul_output_0      (8-ch, post-SiLU of cv1)
 → DynamicQuantizeLinear  (u8, s_a, zp_a)
 → ConvInteger            (8 → 16, 3x3, stride 1, pad 1)
 → Cast i32 → fp32
 → Mul (s_a · s_w)
 → Add bias (fp32)
 → Sigmoid → Mul (SiLU)   → /model.2/m.0/cv2/act/Mul_output_0  (16-ch)
 → Add /model.2/Slice_1_output_0      (residual, 16-ch, indep. u8 dyn-quant)
                            → /model.2/m.0/Add_output_0
```

Conv parameters (`scale_pkg::LAYER_4_*`): `Cin=8, Cout=16, K=3, stride=1,
pad=1`, input/output `160×160`. Target frame cycles = `LAYER_4_CYCLES =
76,800` at `(P_PIX, P_COUT, P_CIN) = (1, 16, 24)`.

The **residual** is `/model.2/Slice_1_output_0` — the *bottleneck's input*,
which the ONNX graph (verified with `shape_inference`) shows is **16 channels
wide**, not 8. The add is therefore a full 16-channel elementwise add applied
**after** the cv2 SiLU. (The task description suggested the operand was 8ch
on the first 8 output channels; the ONNX graph trumps that — see the
verification snippet in `extract.py`.)

For DV we drive an `18×18×8` ROI of cv2 input (the pad-1 ring sourced from
ORT's `/model.2/m.0/cv1/act/Mul_output_0_quantized` so the boundary uses
"real" neighbour data) and a `16×16×16` ROI of the residual (sourced from
ORT's `/model.2/Slice_1_output_0_quantized`). Output ROI is `16×16×16` at
position `(R, C) = (32, 32)`.

## Files

```
extract.py                       # ORT-driven stimulus + reference generator
rtl/layer_4_m0_cv2.sv            # DUT: dotN/requant/silu + add_rq pipeline
dv/layer_4_m0_cv2_tb.sv          # Verilator wrapper (flattened ports)
dv/layer_4_m0_cv2_test.cc        # C++ test: loads stim, drives DUT, scores
dv/Makefile                      # build/run
stim/                            # generated stimulus + ORT/HW reference
```

## Design choices

- **Reuse-only.** No shared IPs were modified. Uses `dotN`, `requant` (which
  internally uses `i32_to_fp16` + `fp16_fma` + `fp16_to_i8_sat`), `act_silu`,
  and `add_rq`. `linebuf_kxk` is **not** used by this unit-level test; the
  TB drives window samples directly (the system integrator that wires this
  layer into the streaming pipeline will need 8 instances of `linebuf_kxk`,
  same pattern as layer_1).

- **u8 → i8 fold.** `i8 = u8 - 128`, with the per-channel bias absorbing the
  zero-point shift:
  `bias_eff[c] = bias[c] + s_a · s_w · (128 - zp_a) · sum_w[c]`.
  HW-ref matches ORT to `cos ≥ 0.997` on every tile.

- **u8 dyn-quant bridge on the residual via bias rewrite.** The residual
  second operand also enters `add_rq` as `i8 = u8 - 128`; the `s_r·(128 -
  zp_r)` DC term is folded into `add_rq`'s post-`inv_out_scale` bias:
  `add_bias_fp16 = s_r · (128 - zp_r) / S_OUT_ADD`.

- **Architecture: 16 × dotN(N=72) + 16 × requant + 16 × act_silu + 16 ×
  add_rq.** One `3×3×8` window plus matching 16-ch residual sample per cycle
  drives all 16 output channels in parallel. Cycle accounting:
  | Stage             | Latency |
  | ----------------- | ------- |
  | dotN              | 1 + ⌈log₂72⌉ = 8 |
  | requant           | 3 |
  | act_silu          | 1 |
  | add_rq            | 5 |
  | **Total pipeline**| 17 |
  Throughput: 1 output pixel / cycle (16 channels in parallel). The residual
  arrives with `valid_i` and is delayed by `CV2_LAT = 12` cycles inside the
  DUT so it aligns with the cv2-post-SiLU stream at `add_rq.b`.

- **SiLU scales.** Empirical pre-SiLU range on normalised 0..1 inputs is
  ~−0.3 .. 21. We pick `S_OUT_PRE = S_OUT_SILU = 80/127 ≈ 0.630` (covers
  ±80 in both pre- and post-SiLU). LUT step is ≈ 0.63, which is the
  dominant DUT-vs-ORT error.

- **Output (post-residual) scale.** `S_OUT_ADD = 80/127`. The fp32
  reference `add_out` empirically spans ~ ±31, so the int8 grid leaves
  ample headroom while keeping ULP ≈ 0.63.

## linebuf_kxk note

Same as `layer_1`: not exercised by this unit DV. System integration will
instantiate 8 parallel `linebuf_kxk` (one per cv2 input channel) or extend
`linebuf_kxk` for multi-channel streams. The shared IP is **unmodified**.

## Comparison criterion

Per sample: cosine vs ORT and DUT-vs-HW-ref reported. Overall **PASS**: at
least **3 of 6 tiles** meet `cos ≥ 0.998` vs the ORT
`/model.2/m.0/Add_output_0` reference (task spec).

Small-dynamic-range tiles (`half`, `gradient`) may fall short of the strict
`0.998` cosine bound purely because of SiLU-LUT quantisation (the LUT step
of ~0.63 is a larger fraction of the small fp32 range). The HW-ref already
shows `cos ≈ 0.997` on `gradient` — the DUT is faithfully reproducing it.

## Result (latest run)

```
sample      max_abs   mae      cos        out_range   thresh   verdict
rand0       0.9321    0.2351   0.999361   31.109      1.555    PASS
rand1       0.8684    0.2361   0.999345   31.047      1.552    PASS
half        0.3830    0.1469   0.997206    6.463      0.323    FAIL (cos<0.998)
gradient    0.8518    0.2154   0.995059    7.747      0.387    FAIL (cos<0.998)
rand_low    0.8282    0.2458   0.998883   22.154      1.108    PASS
rand_high   0.8572    0.2520   0.998558   20.532      1.027    PASS

Aggregate: 4/6 tiles ≥ cos 0.998   →   meets spec (≥3 tiles)
avg cos = 0.998069
```

## Running

```
make -C integ/layer_4_m0_cv2/dv lint    # lint-only
make -C integ/layer_4_m0_cv2/dv test    # regenerates stim, builds, runs
```
