# fp16_fma

IEEE-754 binary16 fused multiply-add: `y = a*b + c` with a single
rounding step at the end (RNE).

Used in the requant pipeline immediately after `i32_to_fp16`: applies
the per-channel scale and bias as `fp16(acc) * scale + bias` with no
double-rounding.

## Ports

| Port      | Dir | Width | Description                                  |
|-----------|-----|-------|----------------------------------------------|
| `clk_i`   | in  | 1     | Clock, rising edge.                          |
| `rst_ni`  | in  | 1     | Async-assert, sync-deassert active-low.      |
| `a_i`     | in  | 16    | fp16 multiplier operand A.                   |
| `b_i`     | in  | 16    | fp16 multiplier operand B.                   |
| `c_i`     | in  | 16    | fp16 addend C.                               |
| `y_o`     | out | 16    | fp16 result `a*b+c` (registered).            |

## Contract

The compute is combinational; the output is registered one cycle.

On each rising edge of `clk_i`:

1. If `!rst_ni`: `y_o <= 16'h0000`.
2. Else: `y_o <= fp16_fma_rne(a_i, b_i, c_i)`.

Semantics:

- Subnormal inputs and outputs are supported.
- **Underflow flushes to zero** (the smallest representable subnormal
  is `2^-24`; any product/sum smaller than that, after rounding, becomes
  ±0 with sign carried through). This matches the default behaviour
  expected by the YOLO requant pipeline; there is no separate FTZ pin.
- Overflow → signed ±Inf.
- NaN: any input NaN propagates → canonical `0x7E00`. `Inf*0` → NaN.
  `(Inf*x) + (-Inf*y)` (i.e., signed Infs adding to zero) → NaN.
- Signed zero: under default RNE, the FMA result `a*b + c` with
  `a*b == -0` and `c == +0` is `+0`. Two `-0`s sum to `-0`.
- Catastrophic cancellation (e.g., `1*x + (-x)`) yields exact `+0`.

## Implementation

The DUT computes the 22-bit product, then aligns both the product and
`c` into a 100-bit working register (50-bit accumulator window plus a
50-bit sticky region). It anchors on whichever operand has the higher
"top exponent" so the dominant operand's leading bit lands near the
top of the accumulator. Same-sign add or different-sign subtract is
performed in full 100-bit precision, leading-1 detect over 101 bits
selects the result exponent, then mantissa extraction with
guard/round/sticky drives RNE rounding. A separate subnormal-output
path right-shifts the normalised value before extraction so the same
rounding code handles normal and subnormal results.

## DV / cross-check

```
make -C hw/ip/fp16_fma/dv test     # build + run cycle-accurate compare
make -C hw/ip/fp16_fma/dv lint     # verilator lint
make -C hw/ip/fp16_fma/dv clean
```

The DV wraps `fp16_fma` (DUT) and `fp16_fma_ref` (an independently
coded SV behavioural golden — different alignment strategy: anchors on
the smaller exponent and shifts both into a 128-bit common-anchor
window; normalises by downward scan rather than leading-zero count) in
`fp16_fma_tb`. Each cycle the test asserts (a) `mismatch_o == 0`
between DUT and REF, and (b) the DUT matches an independent C++ shadow
that performs the fp16 FMA in pure integer arithmetic with a `__int128`
accumulator. Three independent integer implementations (DUT, REF,
shadow) make a shared algorithmic bug very unlikely.

The randomized stress runs 100,000 random fp16 triples and requires
zero mismatches against both REF and shadow.
