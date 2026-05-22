# fp16_to_i8_sat

IEEE-754 binary16 (fp16) to signed int8 converter. Round-to-nearest-even
on the mantissa, saturating clamp to `[-128, +127]`, NaN inputs forced
to zero.

Used in the requant pipeline:
`i32 accumulator -> fp16 -> fma(scale, bias) -> fp16_to_i8_sat`.

## Ports

| Port     | Dir | Width  | Description                                   |
|----------|-----|--------|-----------------------------------------------|
| `clk_i`  | in  | 1      | Clock, rising edge.                           |
| `rst_ni` | in  | 1      | Async-assert, sync-deassert active-low reset. |
| `x_i`    | in  | 16     | IEEE-754 binary16 input.                      |
| `y_o`    | out | 8 (s)  | Signed int8 result (registered).              |

## Contract

The compute is combinational; the output is registered one cycle.

On each rising edge of `clk_i`:

1. If `!rst_ni`: `y_o <= 8'sd0`.
2. Else: `y_o <= int8_sat( rne( fp16_value(x_i) ) )`.

Special-case map:

| fp16 input              | int8 output |
|-------------------------|-------------|
| NaN (any payload, sign) | `0`         |
| `+Inf`                  | `+127`      |
| `-Inf`                  | `-128`      |
| `±0`                    | `0`         |
| subnormal               | `0`         |
| finite, `value >= 128`  | `+127`      |
| finite, `value <= -128` | `-128`      |
| finite, otherwise       | RNE round   |

Key rounding edge case: `+127.5` (RNE -> +128) saturates back to `+127`.
`-128.5` (not representable in fp16 anyway) and `-128.0` land exactly
at `-128` without saturating.

## Running DV

```
make -C hw/ip/fp16_to_i8_sat/dv test     # exhaustive 16-bit sweep + directed
make -C hw/ip/fp16_to_i8_sat/dv lint     # verilator lint
make -C hw/ip/fp16_to_i8_sat/dv clean
```

The DV wraps `fp16_to_i8_sat` (DUT) and `fp16_to_i8_sat_ref` (an
independently coded SV behavioral golden built as a 14-bit-fraction
fixed-point conversion followed by RNE-and-saturate) inside
`fp16_to_i8_sat_tb`. The C++ testbench iterates the full 65,536-element
fp16 input space, checking (a) DUT vs REF (`mismatch_o == 0`) and
(b) DUT vs an independent C++ shadow that performs the same conversion
in pure integer arithmetic with no host-float involvement.
