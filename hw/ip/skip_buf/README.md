# skip_buf

Streaming int8 feature-map skip buffer for the FPN/PAN U-turn in the YOLO
neck. The producer writes a full `H x W x C` tensor in raster order
(channel-fastest, then width, then height), the buffer fills to its
`Depth = H*W*C` capacity, asserts `full_o`, and then drains in the same
order to the consumer. Reads and writes are phased: no interleaving in
this variant.

## Ports

| Port       | Dir | Width | Description                                          |
|------------|-----|-------|------------------------------------------------------|
| `clk_i`    | in  | 1     | Clock, rising edge.                                  |
| `rst_ni`   | in  | 1     | Async-assert, sync-deassert active-low reset.        |
| `clr_i`    | in  | 1     | Synchronous clear — returns the buffer to empty.     |
| `wvalid_i` | in  | 1     | Input-stream valid.                                  |
| `wready_o` | out | 1     | Input-stream ready (`= ~full_o`).                    |
| `wdata_i`  | in  | 8 (s) | Input byte (int8 symmetric).                         |
| `rvalid_o` | out | 1     | Output-stream valid (high only when full and not yet drained). |
| `rready_i` | in  | 1     | Output-stream ready.                                 |
| `rdata_o`  | out | 8 (s) | Output byte (int8 symmetric).                        |
| `full_o`   | out | 1     | Buffer holds `Depth` entries; further writes blocked.|
| `empty_o`  | out | 1     | Buffer is empty (`!full_o && wptr == 0`).            |

## Parameters

| Parameter | Default | Notes                                          |
|-----------|---------|------------------------------------------------|
| `H`       | 16      | Feature-map height.                            |
| `W`       | 16      | Feature-map width.                             |
| `C`       | 32      | Channel count.                                 |

`Depth = H * W * C` is the total int8 capacity. With the defaults that
is 8 KiB.

## Contract

- After reset / `clr_i`: `empty_o = 1`, `full_o = 0`, `wready_o = 1`,
  `rvalid_o = 0`.
- Writes fill linearly until `Depth` entries have been accepted, then
  `full_o` asserts and `wready_o` deasserts. Further `wvalid_i` writes
  are ignored.
- Reads are blocked (`rvalid_o = 0`) until `full_o` is set. Once full,
  reads drain in the same order as the writes were received.
- After `Depth` reads, `rvalid_o` deasserts. Re-use of the block then
  requires either a new fill cycle (after `clr_i`) or a fresh `clr_i` /
  reset.
- Standard AXI-Stream-style handshake: producer must hold `wdata_i`
  stable while `wvalid_i` is asserted and `wready_o` is low.

## Implementation notes

The storage element is a plain `logic [7:0] mem[Depth]` array.
Verilator and downstream synthesis tools infer a BRAM/SRAM for large
`Depth`; the swap to an explicit `prim_ram_1p` is a one-file change.
The read port is combinational on `rptr_q` — a registered-output SRAM
shim can be added when targeting a specific technology.

The DUT and the REF use deliberately different idioms:

- DUT: `mem[Depth]` array with `wptr_q` / `rptr_q` (each `PW = log2(Depth)+1`
  bits wide so the count `Depth` is representable) and a `full_q` latch.
- REF: SystemVerilog queue `byte q[$]` with `push_back` / `pop_front` and
  integer count registers. State (`full_now`, `drained_now`) is derived
  combinationally from the counts.

Both must produce identical `wready_o`, `rvalid_o`, `full_o`, `empty_o`,
and (when valid) `rdata_o` every cycle.

## Limitations / future work

- This variant reads back in **the same raster order as it was written
  (FIFO-shaped)**. The "read in a different order" capability called out
  in `TASKS.md` (e.g. configurable CHW vs HWC traversal at the read
  port) is left for a future variant — it will involve a programmable
  read-address generator on the same storage.
- Fill and drain are phased; no simultaneous read/write. A future
  variant could overlap once a fill-and-drain skid is acceptable.
- For large `Depth` (the default `16*16*32 = 8 KiB`) Verilator
  simulation of full-depth sweeps is slow. The DV picks a small
  configuration (`H = W = C = 4 → Depth = 64`) for fast functional
  coverage, plus one larger configuration (`H = W = C = 8 → Depth =
  512`) to stress the wider address decode.

## Running DV

```
make -C hw/ip/skip_buf/dv test    # build + run small (Depth=64) and large (Depth=512)
make -C hw/ip/skip_buf/dv lint    # verilator lint
make -C hw/ip/skip_buf/dv clean
```

DV instantiates `skip_buf` (DUT) and `skip_buf_ref` (SV behavioral
golden) side-by-side in `skip_buf_tb` and drives identical stimulus to
both. Every cycle the wrapper's `mismatch_o` (sampled across handshake
signals, status flags, and `rdata_o` when both `rvalid_o` are high) must
be 0, and the drained byte stream must match the C++ `std::vector<int8_t>`
shadow of the writes.

Tests:

1. Reset / `clr_i` semantics; reading while empty yields `rvalid_o = 0`.
2. Directed fill+drain, no back-pressure.
3. Writes while full are ignored; pending data drains intact.
4. Random read back-pressure (`rready_i` stalled ~50% of cycles).
5. Slow producer (`wvalid_i` gapped ~50% of cycles).
6. Mid-fill `clr_i` drops partial data; fresh stream completes cleanly.
7. Random fill+drain stress, three back-to-back cycles with both sides
   gapped.

All run at both the small (`H = W = C = 4`) and large (`H = W = C = 8`)
configurations.
