# layer_12_cv1 — YOLO26n /model.6/cv1 integration test

Conv 1×1, Cin=128 → Cout=128, stride=1, pad=0, plane 40×40, SiLU.
Entry of the third C3k2 block.

## Pipeline

```
x_i8 (u8-128) ─► dotN(N=P_CIN=4) ×NCH_OUT=128
              ─► N_PHASE=32-phase accumulator (per channel)
              ─► >>> ACC_SHIFT=3  (keeps |acc| ≤ ~15k inside fp16 range)
              ─► requant (i32→fp16 → fma(scale,bias) → i8_sat)
              ─► act_silu (LUT)
              ─► y_i8
```

## Parallelism (scale_pkg::LAYER_12_*)

| Param   | Value | Notes |
|---------|-------|-------|
| P_PIX   | 1     | one output pixel per phase set |
| P_COUT  | 128   | full output-channel parallelism |
| P_CIN   | 4     | input-channel sub-window per cycle |
| N_PHASE | 32    | NCH_IN / P_CIN |
| MAC units | 512 | NCH_OUT × P_CIN |

Cycles per frame: 40 × 40 × 32 = 51 200 (matches `LAYER_12_CYCLES`,
target `T_FRAME = 100 000`). Spatial area is only 1600 px so the same
512-MAC fabric (as L7) is used at a deeper N_PHASE.

## ACC_SHIFT

With C_IN doubled vs L7 and a worst-case |sum_w| ≈ 824 in the
weight tensor, the raw int32 accumulator reaches ~118 k on YOLO26n
random ROIs — well past fp16 max (65 504). `extract.py` empirically
probes every sample and picks the smallest shift that brings the
worst observed |acc| under fp16 max, with a floor of 3 for full-frame
headroom. The selected `ACC_SHIFT = 3` brings the worst probe
accumulator to ≈14.7 k.

The RTL shifts `sum_q` right by 3 bits (signed arithmetic shift; see
L6 gotcha — wrap with `$signed(arr[i]) >>> n`) before requant, and
`extract.py` pre-multiplies `scale_fp16` by `2^ACC_SHIFT = 8` so the
math is exactly compensated.

## u8→i8 bridge

ORT's DynamicQuantizeLinear emits u8 with `zp_a != 0`. We subtract 128
to get i8 and fold the zp shift into the per-channel bias:

    bias_eff[c] = bias[c] + s_a · s_w · (128 - zp_a) · sum_w[c]

## Reused IPs (unchanged)

- `hw/ip/dotN`        — 4-lane signed dot
- `hw/ip/requant`     — i32→fp16→fma→i8 sat
- `hw/ip/act_silu`    — 256-entry LUT

No linebuf (K=1).

## Files

- `extract.py`              : pulls ORT initializers, auto-probes ACC_SHIFT,
                              builds ORT ref + HW ref + stim
- `rtl/layer_12_cv1.sv`     : DUT
- `dv/layer_12_cv1_tb.sv`   : flat-port Verilator wrapper
- `dv/layer_12_cv1_test.cc` : C++ harness, drives 32 phases × ROI pixels
- `dv/Makefile`             : `make test`

## Results (latest run)

| sample      | DUT vs ORT cosine | max_abs | required |
|-------------|-------------------|---------|----------|
| rand0       | 0.999836          | 0.0363  | yes      |
| rand1       | 0.999838          | 0.0390  | yes      |
| rand2       | 0.999842          | 0.0336  | yes      |
| rand3       | 0.999843          | 0.0361  | yes      |
| rand_low    | 0.999827          | 0.0339  | yes      |
| rand_high   | 0.999817          | 0.0337  | yes      |
| half        | 0.999777          | 0.0257  | info     |
| gradient    | 0.999775          | 0.0345  | info     |

8/8 PASS. Avg cosine 0.99982 (well above 0.998 bar). Max abs error
≤ 0.039 on a ~3.4 output range. Pipeline latency 9 cycles
(DOT 3 + ACC 1 + RQ 3 + SiLU 1 + output reg 1).

## Build notes

The shared `requant` IP intentionally leaves `i32_to_fp16.shift_o`
unconnected, which Verilator flags as `PINMISSING`. Per the task rules
the shared IP must not be modified, so the Makefile appends
`-Wno-PINMISSING` to `COMMON_FLAGS` after including `mk/verilator.mk`.
