# linebuf_kxk

Streaming K x K sliding-window patch generator with zero-padding — the
im2col-style front-end for a convolution stage of the YOLO accelerator.

Consumes a H x W feature map of `Channels` int8 lanes in row-major raster
order (one pixel of all channels per cycle when valid) and emits H x W
K x K x Channels patches in raster order. For output position `(or, oc)`
and channel `c`, element `[(ky*K + kx)*Channels + c]` of the output patch
is the input pixel at `(or + ky - P, oc + kx - P)` on channel `c` (where
`P = (K-1)/2`), or zero if that position is out of frame (same-padding).

## Ports

| Port       | Dir | Width                | Description                                 |
|------------|-----|----------------------|---------------------------------------------|
| `clk_i`    | in  | 1                    | Clock, rising edge.                         |
| `rst_ni`   | in  | 1                    | Async-assert, sync-deassert active-low rst. |
| `clr_i`    | in  | 1                    | Synchronous clear — empties all state.      |
| `wvalid_i` | in  | 1                    | Input-stream valid.                         |
| `wready_o` | out | 1                    | Input-stream ready.                         |
| `wdata_i`  | in  | `Channels*8` (s)     | Input pixel (one int8 per channel).         |
| `rvalid_o` | out | 1                    | Output-stream valid.                        |
| `rready_i` | in  | 1                    | Output-stream ready.                        |
| `rdata_o`  | out | `K*K*Channels*8` (s) | Patch, indexed `[(ky*K+kx)*Channels + c]`.  |

Handshake is global: a single `wready_o`/`rvalid_o` pair gates all
channels in lock-step (one fire moves all `Channels` lanes at once).

## Parameters

| Parameter  | Default | Notes                                            |
|------------|---------|--------------------------------------------------|
| `K`        | 3       | Kernel size; must be odd (1, 3, 5, ...).         |
| `W`        | 16      | Frame width; must be `>= K`.                     |
| `H`        | 16      | Frame height; must be `>= K`.                    |
| `Channels` | 1       | Number of int8 lanes (any `>= 1`).               |

## Storage and emission model (DUT)

The DUT keeps, **per channel**, a circular bank of `K` row buffers, each
`W` int8 wide. All `Channels` banks share one handshake/control FSM —
they fire together. Input row `r` writes into bank `r mod K`. On each
output beat the K x K x Channels patch is assembled combinationally from
`K * K * Channels` reads against the banks with explicit out-of-frame
substitution (`0`).

`inputs_written` and `outputs_emitted` counters drive the handshake:

* `wready_o = inputs_written  < H*W`
* `rvalid_o = outputs_emitted < H*W AND inputs_written >= req_cnt`

`req_cnt` is the raster index just past the bottom-right in-frame pixel of
the current output window (`(min(out_row+P, H-1), min(out_col+P, W-1))`).
For interior output positions this is essentially "the window's
bottom-right pixel has arrived"; for output positions near the right or
bottom edge of the frame, the clipping lets emission start (or finish)
without waiting for pixels that would have been out of frame anyway.

After accepting `H*W` inputs the DUT keeps `wready_o = 0` and continues to
emit any remaining bottom-padded output patches.

## Reference model

`linebuf_kxk_ref` is an intentionally-different behavioral golden that
stores the entire `H*W` frame as a flat unpacked int8 array per channel
and, on each output beat, builds the `K*K*Channels` patch by explicit
signed-range checks against `H` and `W` with zero fill. Plain `int`
counters and no row-bank rotation — slow but obviously correct, and
decoupled from the DUT's incremental implementation so a shared coding
error cannot pass cycle-by-cycle.

## Contract

Standard ready/valid handshake on both ports. `clr_i` (or `!rst_ni`)
drops all internal state: `rvalid_o = 0`, `wready_o = 1`.

Output patch order matches im2col row-major within the K x K window,
channel-minor. For output position `(or, oc)` the element at
`[(ky*K + kx)*Channels + c]` corresponds to input row `or + ky - P`,
column `oc + kx - P`, channel `c`.

**Frame boundary:** the FSM accepts exactly `H*W` input pixel-vectors
per frame, then drops `wready_o` until the caller asserts `clr_i` (a
single-cycle pulse) to release for the next frame. Continuous-flow
(back-to-back frames without `clr_i`) is **not supported** in the
current implementation; callers must drive a `clr_i` between frames.

## Running DV

```
make -C hw/ip/linebuf_kxk/dv test     # build + run all 4 configs
make -C hw/ip/linebuf_kxk/dv lint     # verilator lint
make -C hw/ip/linebuf_kxk/dv clean
```

Configurations exercised:

| Tag       | K | W | H | Channels |
|-----------|---|---|---|----------|
| `k3w4`    | 3 | 4 | 4 | 1        |
| `k5w8`    | 5 | 8 | 8 | 1        |
| `k3w8c16` | 3 | 8 | 8 | 16       |
| `k3w8c64` | 3 | 8 | 8 | 64       |

Tests (per configuration):

1. Reset state.
2. Directed pattern (no back-pressure), patches checked against C++ shadow.
3. Random output back-pressure (`rready_i` stalled ~50% of cycles).
4. Slow producer (`wvalid_i` gapped ~50% of cycles).
5. Random stress, 3 frames, both sides gapped.
6. Mid-stream `clr_i` followed by a fresh frame.

Every cycle the wrapper's `mismatch_o` (sampled across `wready_o`,
`rvalid_o`, and `rdata_o` when both `rvalid_o` are high) must be 0, and
every emitted patch must match an independent C++ shadow that builds the
K x K window from the full input frame with explicit zero-padding.
