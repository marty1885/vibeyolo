# act_sigmoid

Int8 → int8 sigmoid activation realised as a 256-entry ROM lookup. The
contents are baked at elaboration from `InScale` (input float scale) and
`OutScale` (output float scale): every codepoint `i ∈ [-128, 127]` is
mapped to `sat_i8(round_half_to_even(sigmoid(i * InScale) / OutScale))`,
where `sigmoid(f) = 1 / (1 + exp(-f))`.

One cycle of latency: `y_o` is registered. `rst_ni` is active-low,
async-assert / sync-deassert and forces `y_o` to 0.

Sigmoid output is mathematically in `(0, 1)`, so a natural `OutScale` is
`1/128`: the full sigmoid range maps to the non-negative half of int8
(`[0, 127]`). `y_o` is declared `signed` to match the rest of the chip's
int8-symmetric contract, but its quantised value is always non-negative
in practice.

## Ports

| Port      | Dir | Width  | Description                                  |
|-----------|-----|--------|----------------------------------------------|
| `clk_i`   | in  | 1      | Clock, rising edge.                          |
| `rst_ni`  | in  | 1      | Async-assert, sync-deassert active-low reset.|
| `x_i`     | in  | 8 (s)  | Signed int8 input, scale `InScale`.          |
| `y_o`     | out | 8 (s)  | Signed int8 output, scale `OutScale`.        |

## Parameters

| Name       | Type | Default  | Description                            |
|------------|------|----------|----------------------------------------|
| `InScale`  | real | `1/16`   | Float scale of `x_i`.                  |
| `OutScale` | real | `1/128`  | Float scale of `y_o`.                  |

## Rounding

Round half to even (banker's rounding) on the magnitude, then re-sign
and clamp to `[-128, 127]`. The DUT, the SV REF, and the C++ shadow all
implement the same convention so any one of them catches a regression
in the others.

## Running DV

```
make -C hw/ip/act_sigmoid/dv test     # build + run cycle-accurate compare
make -C hw/ip/act_sigmoid/dv lint     # verilator lint
make -C hw/ip/act_sigmoid/dv clean
```

The TB instantiates two parameterisations simultaneously: a
non-saturating `(InScale, OutScale) = (1/16, 1/128)` pair (variant A)
and an upper-rail saturating `(1/16, 1/256)` pair (variant B). The C++
test drives identical stimulus into both, checks the SV mismatch flags
cycle-by-cycle, cross-checks the DUT against an independent
double-precision C++ shadow, and verifies global monotonicity of the
sigmoid output across all 256 codepoints.
