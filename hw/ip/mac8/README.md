# mac8

Atomic signed int8 × int8 multiply-accumulate. One signed 8×8 multiply per
cycle, accumulated into a signed int32. Synchronous clear-and-load,
enable-gated accumulation, async-assert / sync-deassert active-low reset.

This is the foundational compute leaf for the YOLO accelerator — every dot
product (`dotN`) is a tree of `mac8` taps.

## Ports

| Port      | Dir | Width  | Description                                   |
|-----------|-----|--------|-----------------------------------------------|
| `clk_i`   | in  | 1      | Clock, rising edge.                           |
| `rst_ni`  | in  | 1      | Async-assert, sync-deassert active-low reset. |
| `clr_i`   | in  | 1      | Synchronous clear-and-load.                   |
| `en_i`    | in  | 1      | Accumulate enable.                            |
| `a_i`     | in  | 8 (s)  | Signed int8 multiplicand.                     |
| `b_i`     | in  | 8 (s)  | Signed int8 multiplicand.                     |
| `acc_o`   | out | 32 (s) | Signed int32 accumulator.                     |

## Contract

On each rising edge of `clk_i`:

1. If `!rst_ni`: `acc_o <= 0`.
2. Else if `clr_i`: `acc_o <= signed(a_i) * signed(b_i)` (load product —
   enables "first-tap" semantics in chained dot products).
3. Else if `en_i`: `acc_o <= acc_o + signed(a_i) * signed(b_i)`.
4. Else: `acc_o` holds.

`clr_i` has priority over `en_i`. The accumulator wraps on int32 overflow
(no saturation); callers are responsible for sizing the chain so wrap
cannot occur in practice.

## Running DV

```
make -C hw/ip/mac8/dv test     # build + run cycle-accurate compare
make -C hw/ip/mac8/dv lint     # verilator lint
make -C hw/ip/mac8/dv clean
```

The DV wraps `mac8` (DUT) and `mac8_ref` (SV behavioral golden) in
`mac8_tb`, drives identical stimulus to both, and on every cycle asserts
(a) `mismatch_o == 0` between the two SV implementations and (b) the DUT
matches an independent C++ shadow accumulator (so a shared bug between
DUT and REF would still be caught).
