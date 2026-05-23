# reduce_max_n

Signed int8 **N-wide max reduction**, one element/cycle. All `N` candidates
arrive in parallel on `x_i`; on each cycle `en_i` is high the registered
output `y_o` updates to the maximum and `valid_o` pulses (en delayed one
cycle). Latency = 1.

## Why

The detect head (`/model.23`) reduces the 80 per-anchor class **logits** to a
single ranking **score** (`N=80`). All 80 class channels share one per-tensor
activation scale, so the max in int8 preserves the fp16 ordering — the dequant
to a common fp16 happens downstream on the single survivor, not on all N lanes.
This keeps the reduce cheap (int8 comparator tree) and exact for ranking.

## Interface

| port | dir | width | notes |
|---|---|---|---|
| `clk_i` / `rst_ni` | in | 1 | async-assert, sync-deassert active-low reset |
| `en_i` | in | 1 | accept a new vector this cycle |
| `x_i` | in | `N×8` | packed signed int8 lanes |
| `valid_o` | out | 1 | high the cycle `y_o` is valid (= `en_i` delayed 1) |
| `y_o` | out | 8 | signed int8 max; 0 after reset, holds when `en_i` low |

Parameter: `N` (default 80).

## Implementation

Balanced binary comparator tree via a recursive `automatic` function —
O(log₂N) depth, parameter-clean for any N (odd levels carry the trailing
element up unchanged, since max is associative and `max(a)==a`). The golden
`reduce_max_n_ref.sv` is an independently coded flat linear scan.

## DV

`make -C dv test` builds one binary per N. **N=80 / 16 / 4 all pass 82/82**
(reset, hold-on-`en=0`, directed corners, 10 000 random vectors vs both the SV
ref and an independent C++ `std::max_element` shadow, mid-stream reset). The
extra N=16/4 configs exercise odd/even tree depths and narrow ports to prove
the parameterization scales.
