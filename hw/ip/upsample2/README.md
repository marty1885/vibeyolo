# upsample2

Streaming nearest-neighbor 2x upsample. Consumes a HxW int8 feature map in
row-major raster order (one pixel per cycle when valid) and emits a
(2H)x(2W) upsampled stream, where each input pixel is replicated into a
2x2 block of identical output pixels.

Used in the FPN/PAN neck of the YOLO accelerator. Pure timing and
replication — no arithmetic.

## Ports

| Port       | Dir | Width  | Description                                   |
|------------|-----|--------|-----------------------------------------------|
| `clk_i`    | in  | 1      | Clock, rising edge.                           |
| `rst_ni`   | in  | 1      | Async-assert, sync-deassert active-low reset. |
| `clr_i`    | in  | 1      | Synchronous clear — empties all state.        |
| `wvalid_i` | in  | 1      | Input-stream valid.                           |
| `wready_o` | out | 1      | Input-stream ready.                           |
| `wdata_i`  | in  | 8 (s)  | Input pixel (int8 symmetric).                 |
| `rvalid_o` | out | 1      | Output-stream valid.                          |
| `rready_i` | in  | 1      | Output-stream ready.                          |
| `rdata_o`  | out | 8 (s)  | Output pixel (int8 symmetric).                |

## Parameters

| Parameter  | Default | Notes                                              |
|------------|---------|----------------------------------------------------|
| `W`        | 16      | Input row width in pixels.                         |
| `Channels` | 1       | **Only `Channels == 1` is currently supported.**   |

The implementation does not know `H`; it simply alternates between consuming
one input row and replaying that row. The block can run forever.

## Contract

For each input row of `W` pixels the block emits two output rows of `2*W`
pixels:

1. As each input pixel is accepted it is held in a one-deep "current pixel"
   register and emitted twice horizontally into the first output row,
   then captured into a row buffer.
2. After all `W` input pixels of a row have been consumed, the block enters
   replay mode: it emits the captured row a second time (each entry again
   replicated horizontally) before accepting the next input row.

Standard AXI-Stream-style ready/valid: handshake fires when both `*valid`
and `*ready` are high. Producer must hold `wdata_i` stable while
`wvalid_i` is asserted and `wready_o` is low. The input is back-pressured
(`wready_o` = 0) throughout replay.

`clr_i` (or `!rst_ni`) drops all internal state. Immediately after reset:
`rvalid_o = 0`, `wready_o = 1`.

The row buffer is an inferred register array `logic signed [7:0] rowbuf[W]`.
For the modest line widths in the YOLO26n neck this synthesises to flops or
a small distributed RAM; a `prim_ram_1p` swap is straightforward if a
target technology requires it.

## Multi-channel (future work)

For multi-channel feature maps the cleanest extension is to interleave
channels at the pixel level: cycle pixel = `c0p0, c1p0, ..., c0p1, c1p1`,
and store `W * Channels` int8 entries in the row buffer. The current
implementation only supports `Channels == 1` and the `Channels` parameter
is included in the port list as a forward-compatibility hook; non-default
values trigger a simulation-time `$fatal`.

## Running DV

```
make -C hw/ip/upsample2/dv test     # build + run W=4 and W=8 cases
make -C hw/ip/upsample2/dv lint     # verilator lint
make -C hw/ip/upsample2/dv clean
```

DV instantiates `upsample2` (DUT) and `upsample2_ref` (SV behavioral
golden) side-by-side in `upsample2_tb` and drives identical stimulus to
both. Every cycle the wrapper's `mismatch_o` (sampled across handshake
signals and `rdata_o` when both `rvalid_o` are high) must be 0, and the
DUT's emitted stream must match an independent C++ nearest-neighbor-2x
shadow of the entire input sequence. The two SV implementations use
deliberately different idioms (enum FSM + array vs. integer mode + queue
+ holding-register) so a shared coding error cannot pass cycle-by-cycle.

Tests:
1. Reset state.
2. Directed `WxW` pattern with no back-pressure.
3. Random output back-pressure (`rready_i` stalled ~40% of cycles).
4. Slow producer (`wvalid_i` gapped ~40% of cycles).
5. Full random stress (16 rows, both sides gapped).
6. Mid-stream `clr_i` followed by a fresh stream.

All run at both `W = 4` and `W = 8`.
