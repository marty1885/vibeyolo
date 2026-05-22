# maxpool_kxk

Parameterized signed int8 K-by-K max-reduction tap. Consumes K*K parallel
int8 candidates and produces the signed maximum on a registered output.
Used by YOLO's SPPF stage (K=5) and any other pooling kernels (K=2, 3,
…) the model needs.

## Parameters

| Parameter | Default | Description                                |
|-----------|---------|--------------------------------------------|
| `K`       | 5       | Kernel side. Total inputs N = K*K.         |

## Ports

| Port      | Dir | Width        | Description                              |
|-----------|-----|--------------|------------------------------------------|
| `clk_i`   | in  | 1            | Clock, rising edge.                      |
| `rst_ni`  | in  | 1            | Async-assert, sync-deassert active-low.  |
| `en_i`    | in  | 1            | Update enable.                           |
| `x_i`     | in  | K*K * 8 (s)  | Flattened packed array of K*K int8s.     |
| `y_o`     | out | 8 (s)        | Registered signed int8 max of `x_i`.     |

## Contract

On each rising edge of `clk_i`:

1. If `!rst_ni`: `y_o <= 0`.
2. Else if `en_i`: `y_o <= max(x_i[0], x_i[1], ..., x_i[N-1])` (signed).
3. Else: `y_o` holds.

One-cycle latency, fully combinational reduction tree feeding the output
flop.

## Implementation notes

- DUT (`maxpool_kxk.sv`) uses a recursive `automatic` function that
  builds a balanced binary comparator tree level-by-level. Odd-length
  levels carry the trailing element up unchanged — equivalent to a
  power-of-two tree padded with -INF but without the dead comparators.
- REF (`maxpool_kxk_ref.sv`) is deliberately coded in a different style:
  a flat `always_comb` for-loop scan that tracks a running max
  initialised to `-128` (the int8 identity for `max`). Cross-checking
  these two styles makes a shared bug unlikely.

## Running DV

```
make -C hw/ip/maxpool_kxk/dv test     # builds + runs K=2, K=3, K=5
make -C hw/ip/maxpool_kxk/dv lint     # verilator lint
make -C hw/ip/maxpool_kxk/dv clean
```

The DV uses `mk/verilator.mk` multi-test mode: each K is a separate
elaboration (`-GK=<n>`) with a matching C++ define (`-DMAXPOOL_K=<n>`)
so the test driver lays out `x_i` correctly for that width. Every cycle
the testbench asserts (a) `mismatch_o == 0` between DUT and SV REF and
(b) the DUT matches an independent C++ shadow max
(`std::max_element`). Randomized stress: 10,000 patches per K.
