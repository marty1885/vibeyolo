# layer_7_cv1 — YOLO26n /model.4/cv1 integration test

Conv 1×1, Cin=64 → Cout=64, stride=1, pad=0, plane 80×80, SiLU.
Entry of the second C3k2 block.

## Pipeline

```
x_i8 (u8-128) ─► dotN(N=P_CIN=8) ×NCH_OUT=64
              ─► N_PHASE=8-phase accumulator (per channel)
              ─► >>> ACC_SHIFT=2  (keeps |acc| ≤ ~17k inside fp16 range)
              ─► requant (i32→fp16 → fma(scale,bias) → i8_sat)
              ─► act_silu (LUT)
              ─► y_i8
```

## Parallelism (scale_pkg::LAYER_7_*)

| Param   | Value | Notes |
|---------|-------|-------|
| P_PIX   | 1     | one output pixel per phase set |
| P_COUT  | 64    | full output-channel parallelism |
| P_CIN   | 8     | input-channel sub-window per cycle |
| N_PHASE | 8     | NCH_IN / P_CIN |
| MAC units | 512 | NCH_OUT × P_CIN |

Cycles per frame: 80 × 80 × N_PHASE = 51 200 (matches `LAYER_7_CYCLES`,
target `T_FRAME = 100 000`).

## ACC_SHIFT

Empirically, channel 60 has `sum_w = -582` (the largest |sum_w| in the
weight tensor). After the u8→i8 fold the raw int32 accumulator reaches
~68 k for that channel on YOLO26n random inputs — past fp16 max
(65 504). We shift the accumulator right by 2 bits (signed arithmetic
shift; see L6 gotcha) before requant and pre-multiply `scale_fp16` by
`2^ACC_SHIFT = 4` in `extract.py` so the math is unchanged.

## u8→i8 bridge

ORT's DynamicQuantizeLinear emits u8 with `zp_a != 0`. We subtract 128
to get i8 and fold the zp shift into the per-channel bias:

    bias_eff[c] = bias[c] + s_a · s_w · (128 - zp_a) · sum_w[c]

## Reused IPs (unchanged)

- `hw/ip/dotN`        — 8-lane signed dot
- `hw/ip/requant`     — i32→fp16→fma→i8 sat
- `hw/ip/act_silu`    — 256-entry LUT

No linebuf (K=1).

## Files

- `extract.py`            : pulls ORT initializers, builds ORI + HW-ref + stim
- `rtl/layer_7_cv1.sv`    : DUT
- `dv/layer_7_cv1_tb.sv`  : flat-port Verilator wrapper
- `dv/layer_7_cv1_test.cc`: C++ harness, drives 8 phases × ROI pixels
- `dv/Makefile`           : `make test`

## Results (latest run)

| sample      | DUT vs ORT cosine | DUT vs HW-ref cosine | required |
|-------------|-------------------|----------------------|----------|
| rand0       | 0.999898          | 0.999898             | yes      |
| rand1       | 0.999897          | 0.999897             | yes      |
| rand2       | 0.999898          | 0.999898             | yes      |
| rand3       | 0.999895          | 0.999895             | yes      |
| rand_low    | 0.999892          | 0.999892             | yes      |
| rand_high   | 0.999884          | 0.999884             | yes      |
| half        | 0.999893          | 0.999893             | info     |
| gradient    | 0.999871          | 0.999871             | info     |

8/8 PASS. Avg cosine 0.99989. Max abs error ≤ 0.046 on a ~3.4
output range. Pipeline latency 10 cycles (DOT 4 + ACC 1 + RQ 3 + SiLU 1
+ output reg 1).
