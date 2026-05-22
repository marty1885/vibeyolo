# i32_to_fp16

Signed int32 to IEEE-754 binary16 (fp16) converter. Round-to-nearest-even
(RNE) on the mantissa; magnitudes that exceed the largest representable
finite fp16 (and tie-round above it) saturate to signed infinity.

Used in the requant pipeline: `i32 accumulator -> fp16 -> fma(scale, bias)
-> int8 saturating cast`.

## Ports

| Port      | Dir | Width  | Description                                   |
|-----------|-----|--------|-----------------------------------------------|
| `clk_i`   | in  | 1      | Clock, rising edge.                           |
| `rst_ni`  | in  | 1      | Async-assert, sync-deassert active-low reset. |
| `x_i`     | in  | 32 (s) | Signed int32 input.                           |
| `y_o`     | out | 16     | fp16 result (registered).                     |

## Contract

The compute is combinational; the output is registered one cycle.

On each rising edge of `clk_i`:

1. If `!rst_ni`: `y_o <= 16'h0000`.
2. Else: `y_o <= fp16_rne(x_i)`.

Conversion rules:

- `x_i == 0` -> `0x0000` (+0).
- Otherwise the sign bit is `x_i[31]` and the magnitude is `|x_i|`
  (well-defined for `INT32_MIN` since `2^31` fits in uint32).
- The leading-one of the magnitude defines the unbiased exponent;
  mantissa is the next 10 bits below it with guard/round/sticky used
  for RNE.
- If RNE produces an exponent of 31 or more, the result saturates to
  signed infinity (`0x7C00` / `0xFC00`).
- Integer inputs never produce subnormals.

## Running DV

```
make -C hw/ip/i32_to_fp16/dv test     # build + run cycle-accurate compare
make -C hw/ip/i32_to_fp16/dv lint     # verilator lint
make -C hw/ip/i32_to_fp16/dv clean
```

The DV wraps `i32_to_fp16` (DUT) and `i32_to_fp16_ref` (an independently
coded SV behavioral golden that uses `$shortrealtobits` for an fp32
intermediate) in `i32_to_fp16_tb`. Each cycle the test asserts
(a) `mismatch_o == 0` between DUT and REF and (b) the DUT matches an
independent C++ shadow that performs the conversion in pure integer
arithmetic — so a shared SV bug between DUT and REF would still be caught.
