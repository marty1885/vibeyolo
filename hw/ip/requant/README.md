# requant

Per-channel int32 → int8 requantization. Takes a signed int32 accumulator
output (post-MAC tree) and applies an fp16 scale + fp16 bias, saturating
back to signed int8. This is the standard YOLO-style activation requant:

```
y = sat_i8( round( fp16(acc) * scale + bias ) )
```

Phase 3 composite block: a thin shell over the three already-verified
sub-IPs, with pipeline alignment for the per-channel scale/bias inputs.

## Composition

| Stage | Sub-IP             | Latency | Function                          |
|-------|--------------------|---------|-----------------------------------|
| 1     | `i32_to_fp16`      | 1 cycle | int32 → fp16, RNE, sat to ±Inf    |
| 2     | `fp16_fma`         | 1 cycle | fp16 a*b + c, single rounding     |
| 3     | `fp16_to_i8_sat`   | 1 cycle | fp16 → int8, RNE, clamp [-128,127]|

Total latency: **3 cycles** from `acc_i`/`scale_fp16_i`/`bias_fp16_i`
sampled at posedge K to `y_o` available after posedge K+3.

The per-channel `scale_fp16_i` and `bias_fp16_i` are clocked through one
local pipeline register before being driven into the FMA so they remain
aligned with the `i32_to_fp16` output (which is itself one cycle behind
the corresponding `acc_i`).

## Ports

| Port            | Dir | Width  | Description                              |
|-----------------|-----|--------|------------------------------------------|
| `clk_i`         | in  | 1      | Clock, rising edge.                      |
| `rst_ni`        | in  | 1      | Async-assert / sync-deassert, active low.|
| `valid_i`       | in  | 1      | Input data valid; shifts through 3-deep SR. |
| `acc_i`         | in  | 32 (s) | Signed int32 accumulator value.          |
| `scale_fp16_i`  | in  | 16     | fp16 per-output-channel scale.           |
| `bias_fp16_i`   | in  | 16     | fp16 per-output-channel bias (post-scale).|
| `valid_o`       | out | 1      | Output valid, aligned with `y_o`.        |
| `y_o`           | out | 8  (s) | Saturated int8 result.                   |

Pure feed-forward pipeline; no back-pressure. The caller is responsible
for pacing upstream cadence (typical use: tied 1:1 to the MAC tree
output, which already throttles itself).

## Dependencies

- `hw/ip/i32_to_fp16/rtl/i32_to_fp16.sv`
- `hw/ip/fp16_fma/rtl/fp16_fma.sv`
- `hw/ip/fp16_to_i8_sat/rtl/fp16_to_i8_sat.sv`

## Running DV

```
make -C hw/ip/requant/dv test   # build + run
make -C hw/ip/requant/dv lint   # verilator lint
make -C hw/ip/requant/dv clean
```

The testbench wraps `requant` (DUT — composed of the three sub-IPs) and
`requant_ref` (an independent, flat behavioral SV model that does NOT
instantiate any of the sub-IPs) in `requant_tb`, drives identical
stimulus, and on every cycle checks:

1. DUT `y_o` and `valid_o` match REF (per-cycle `mismatch_o == 0`).
2. DUT `y_o` matches a third independent C++ `__int128` shadow that
   implements `int32 → fp16 → fp16-FMA → fp16 → int8` from scratch.

The fp16-FMA stage uses a 128-bit magnitude with single end-of-step
rounding (RNE) — fp32 is not wide enough to faithfully model fp16
alignment (~76 bits needed in the worst case, per the lesson from
`fp16_fma`'s test).
