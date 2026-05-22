# act_silu

Int8 → int8 SiLU (Swish) activation realised as a 256-entry ROM lookup.
The contents are baked at elaboration from `InScale` (input float scale)
and `OutScale` (output float scale): every codepoint `i ∈ [-128, 127]` is
mapped to `sat_i8(round_half_to_even(silu(i * InScale) / OutScale))`,
where `silu(f) = f / (1 + exp(-f))`.

One cycle of latency: `y_o` is registered. `rst_ni` is active-low,
async-assert / sync-deassert and forces `y_o` to 0.

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
| `OutScale` | real | `1/16`   | Float scale of `y_o`.                  |

## Rounding

Round half to even (banker's rounding) on the magnitude, then re-sign and
clamp to `[-128, 127]`. The DUT, the SV REF, and the C++ shadow all
implement the same convention so any one of them catches a regression in
the others.

## Running DV

```
make -C hw/ip/act_silu/dv test     # build + run cycle-accurate compare
make -C hw/ip/act_silu/dv lint     # verilator lint
make -C hw/ip/act_silu/dv clean
```

The TB instantiates two parameterisations simultaneously: a balanced
`(InScale, OutScale) = (1/16, 1/16)` pair (variant A — no saturation) and
a saturating `(1/4, 1/64)` pair (variant B — both rails clip). The C++
test drives identical stimulus into both, checks the SV mismatch flags
cycle-by-cycle, and cross-checks the DUT against an independent
double-precision C++ shadow.
