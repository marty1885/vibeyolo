# layer_13_m0_cv1 — YOLO26n /model.6/m.0/cv1

ONNX node: `/model.6/m.0/cv1/conv/Conv_quant` (ConvInteger), weight
initializer `onnx::Conv_1690_quantized` (verified: shape `[32, 64, 1, 1]`,
zp_w = 0), bias `onnx::Conv_1691` (shape `[32]`).

This is the **first conv of the bottleneck inside C3k2 block #3**. Because
YOLO26n's deeper stages use `c3k=False`, the bottleneck is a (1×1 + 1×1)
pair rather than (3×3 + 3×3). So unlike layer_8_m0_cv1, **this conv is
K = 1**.

## Geometry

| Param | Value |
| ----- | ----- |
| Cin   | 64 (second half of `/model.6/cv1` split, ORT name `/model.6/Slice_1_output_0`) |
| Cout  | 32 |
| K     | 1, stride 1, pad 0 |
| H = W | 40 |
| Act   | SiLU |

## Parallelism (scale_pkg::LAYER_13_*)

| Param      | Value |
| ---------- | ----- |
| P_PIX      | 1 |
| P_COUT     | 32 (full output-channel parallelism) |
| P_CIN = N_LANE | 2 |
| N_PHASE    | 64 / 2 = 32 |
| MAC units  | 32 × dotN(N=2) = **64** |

Each output pixel = 32 phase cycles, one phase committing every cycle.
Frame cycles = 32 × 40 × 40 = **51 200** (matches `LAYER_13_CYCLES`,
target T_FRAME=100 000). Pipeline latency from last phase to valid_o =
DOT(2) + commit(1) + requant(3) + silu(1) + outreg(1) = **8 cycles**.

## Pipeline

```
  x_i (2 i8)            w_i (32×2 i8)
        │                       │
        └────── 32 × dotN(N=2) ─┘     (latency = 1+clog2(2) = 2)
                  │ dot_acc i32
                  ▼
        32-phase per-channel accumulator   (commit on phase==31)
                  │ sum_q i32
                  ▼
            >>> ACC_SHIFT (=2)              (see overflow analysis)
                  │
                  ▼
        32 × requant (fp16 fma + sat → i8)
                  │
                  ▼
        32 × act_silu (LUT, i8 → i8)
                  │
                  ▼
            valid_o, y_o[32×i8]
```

## Accumulator overflow analysis

`extract.py` measured the per-pixel `|sum_q|` over the eight ROIs:

| sample     | max\|sum_q\| |
| ---------- | ------------ |
| rand0      | 67759 |
| rand1      | 65710 |
| rand2      | 65339 |
| rand3      | 67321 |
| rand_low   | 62350 |
| rand_high  | 62204 |
| half       | 44546 |
| gradient   | 47010 |

Worst observed = **67 759**, exceeding fp16 normal max (65 504). The
script picks the smallest `ACC_SHIFT` that brings `max|sum_q| >> shift`
under 32 768; result = **ACC_SHIFT = 2**. The RTL arithmetic-shifts
`sum_q` right by 2 before requant; `scale_fp16` is pre-multiplied by
`2**ACC_SHIFT` in `extract.py` so the dequantised math is equivalent.

## Reused IPs (unmodified)

- `hw/ip/dotN` (N=2)
- `hw/ip/requant` (`hw/ip/i32_to_fp16` + `hw/ip/fp16_fma` +
  `hw/ip/fp16_to_i8_sat`)
- `hw/ip/act_silu`
- `linebuf_kxk` is **not** used (K=1).

## Files

- `extract.py` — ORT-driven stimulus + reference generator.
- `rtl/layer_13_m0_cv1.sv` — datapath.
- `dv/layer_13_m0_cv1_tb.sv` — flattened Verilator wrapper.
- `dv/layer_13_m0_cv1_test.cc` — C++ harness (8 samples, cos threshold
  0.998 on the 6 random samples).
- `dv/Makefile` — uses `mk/verilator.mk`. `make test` builds + runs.

## Result (current)

```
Aggregate: 8/8 samples passed
  avg cos     = 0.999588
  avg mae     = 0.009128
  worst max   = 0.031733
Frame cycles = 51 200  (≤ 100 000 target ✓)
```

All six "required" random ROIs hit cos ≥ 0.998 vs the ORT fp32 reference;
the two "informational" samples (`half`, `gradient`) also pass with
cos ≈ 0.9996. DUT vs the bit-accurate HW reference (built in `extract.py`)
matches numerically modulo the SiLU LUT quantisation step.

## Build note

The shared `requant.sv` instantiates `i32_to_fp16` without binding the
`shift_o` debug output, which triggers a Verilator `PINMISSING` warning
with `-Wall`. The local `Makefile` appends `-Wno-PINMISSING` to
`COMMON_FLAGS` to downgrade it (and adds it to `LINT_FLAGS`) so the
shared IPs do not need any edits.
