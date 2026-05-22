# layer_10_cv2 — YOLO26n /model.4/cv2 integration test

Conv 1×1, Cin=96 → Cout=128, stride=1, pad=0, plane 80×80, SiLU.
Exit of the second C3k2 block.

## Pipeline

```
x_i8 (u8-128) ─► dotN(N=P_CIN=8) ×NCH_OUT=128
              ─► N_PHASE=12-phase accumulator (per channel)
              ─► >>> ACC_SHIFT=2  (keeps |sum_q| ≤ ~20k inside fp16 range)
              ─► requant (i32→fp16 → fma(scale,bias) → i8_sat)
              ─► act_silu (LUT)
              ─► y_i8
```

## ONNX slice

```
/model.4/Concat_output_0  (fp32, 96ch, 80×80)
  → DynamicQuantizeLinear              (u8, s_a, zp_a)
  → ConvInteger                        (1x1, 96→128, w_zp=0)
  → Cast i32→fp32 → Mul(s_a·s_w) → Add bias
  → Sigmoid → Mul                      (SiLU)
  → /model.4/cv2/act/Mul_output_0      (fp32 reference)
```

- weight init : `onnx::Conv_1681_quantized`  (128,96,1,1)  (verified)
- weight scale: `onnx::Conv_1681_scale`     (per-tensor, zp=0)
- bias        : `onnx::Conv_1682`           (128,)

## Concat layout (verified via `onnx.shape_inference`)

`/model.4/Concat` (axis=1) takes:

| index | tensor                          | channels |
|-------|---------------------------------|----------|
| 0     | `/model.4/Slice_output_0`       | 32       |
| 1     | `/model.4/Slice_1_output_0`     | 32       |
| 2     | `/model.4/m.0/Add_output_0`     | 32       |

Total: 96. (YOLO26n c3k=False uses a single Bottleneck `m.0` with the
residual `Add`; the 32-ch sub-tensors come from cv1's 64-ch output being
split in half before the bottleneck.) For DV we treat the concat wiring as
external and consume the ORT-quantised 96-ch u8 tensor.

## Parallelism (scale_pkg::LAYER_10_*)

| Param   | Value | Notes |
|---------|-------|-------|
| P_PIX   | 1     | one output pixel per phase set |
| P_COUT  | 128   | full output-channel parallelism |
| P_CIN   | 8     | input-channel sub-window per cycle |
| N_PHASE | 12    | NCH_IN / P_CIN = 96/8 |
| MAC units | 1024 | NCH_OUT × P_CIN |

Cycles per frame: 80 × 80 × N_PHASE = **76 800** (matches
`LAYER_10_CYCLES`, target `T_FRAME = 100 000`).

## ACC_SHIFT (fp16 acc overflow gotcha)

With Cin=96 and the largest per-channel |sum_w| in the L10 weight tensor
(≈2228), the i32 accumulator after the full 96-lane reduction can swing
past the fp16 normal max (65 504). On the eight YOLO26n stimuli in
`extract.py` the empirical worst |sum_q| reaches **79 530**.

`extract.py` sweeps `ACC_SHIFT` from 0 upward until
`worst_sum_q >> ACC_SHIFT ≤ 32 768`; for L10 that gives **ACC_SHIFT = 2**.
The RTL arithmetic-shifts `sum_q` right by 2 (signed; `$signed(sum_q[gc])
>>> ACC_SHIFT` — note the `$signed` cast is required, L6 gotcha) and
`extract.py` pre-multiplies `scale_fp16` by `2^ACC_SHIFT = 4` so the
arithmetic is unchanged.

`manifest.json` records `acc_shift` and `worst_sum_q` for traceability.

## u8→i8 bridge

ORT's DynamicQuantizeLinear emits u8 with `zp_a != 0`. We subtract 128 to
get i8 and fold the zp shift into the per-channel bias:

    bias_eff[c] = bias[c] + s_a · s_w · (128 - zp_a) · sum_w[c]

## Reused IPs (unchanged)

- `hw/ip/dotN`        — 8-lane signed dot
- `hw/ip/requant`     — i32→fp16→fma(scale,bias)→i8 sat
- `hw/ip/act_silu`    — 256-entry LUT

No linebuf (K=1).

## Files

- `extract.py`             : pulls ORT initializers, builds ORT + HW-ref + stim
- `rtl/layer_10_cv2.sv`    : DUT
- `dv/layer_10_cv2_tb.sv`  : flat-port Verilator wrapper
- `dv/layer_10_cv2_test.cc`: C++ harness, drives 12 phases × ROI pixels
- `dv/Makefile`            : `make test` (regenerates stim, runs Verilator)

## Results (latest run)

| sample      | DUT vs ORT cosine | DUT vs HW-ref cosine | required |
|-------------|-------------------|----------------------|----------|
| rand0       | 0.999388          | 0.999388             | yes      |
| rand1       | 0.999378          | 0.999378             | yes      |
| rand2       | 0.999388          | 0.999388             | yes      |
| rand3       | 0.999356          | 0.999356             | yes      |
| rand_low    | 0.999360          | 0.999360             | yes      |
| rand_high   | 0.999360          | 0.999360             | yes      |
| half        | 0.999425          | 0.999425             | info     |
| gradient    | 0.999394          | 0.999394             | info     |

8/8 PASS. Avg cosine **0.99938**, worst max-abs **0.032** on a ~2.5
output range. Pipeline latency 10 cycles (DOT 4 + ACC 1 + RQ 3 + SiLU 1
+ output reg 1). Frame cycles 76 800 ≤ T_FRAME = 100 000.
