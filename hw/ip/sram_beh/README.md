# sram_beh — behavioral compiled-SRAM macro + banked wrapper

The chip has assumed **magic memory**: every activation bus is a whole
`logic [C-1:0][7:0]` pixel moved per cycle, zero latency, unlimited ports. Real
on-die memory is *compiled macros* — one word per port per cycle, fixed width,
fixed depth, fixed read latency. This IP is the RTL that makes that cost
explicit, and it is the synthesizable counterpart of the analytic estimate in
`tools/sram_bw.py` / `tools/sram_model.py`.

## Modules

| module | role |
|--------|------|
| `sram_beh` | one **1R1W** macro: `Width` bits/word, `Depth` words, `ReadLat` (1 or 2). Read-before-write on same address. Contains the `MEMORYCOMPILER SWAP POINT` — replace the storage with the foundry macro instance at tape-out. |
| `sram_beh_banked` | tiles a logical `LWidth × LDepth` 1R1W memory into `NW·ND` `sram_beh` macros (`NW = ⌈LWidth/MaxWidth⌉` across, `ND = ⌈LDepth/MaxDepth⌉` down) — the **drop-in replacement for a magic wide array**. Presents the same single-word-per-cycle port; its existence forces the bandwidth cost (you pay `NW` macros to move `LWidth` bits/cycle). |

`MaxWidth`/`MaxDepth` mirror `MAX_WIDTH`/`MAX_DEPTH` in `tools/sram_model.py`, so
the RTL banking and the area estimate count the same macros.

## Why both

To deliver more than one macro word per cycle you **bank** — you cannot widen a
single macro arbitrarily. That is precisely the limit the magic model hid. A
512-channel activation edge = 4096 bits/cycle = ⌈4096/144⌉ ≈ 29 macros wide.
`sram_bw.py` quantifies the area (and the throughput hit if you refuse to bank);
`sram_beh_banked` is how you actually instantiate it in RTL.

## Verification

`make -C hw/ip/sram_beh/dv test` — **dual-instance bit-exact** DV. One Verilator
design elaborates the banked DUT and a flat magic-array REF with matching
read-before-write + `ReadLat` semantics, driven by identical 1R1W stimulus
(continuous reads, random writes). Configs (both force `NW>1` *and* `ND>1`):

| cfg | LWidth × LDepth | MaxWidth × MaxDepth | ReadLat | result |
|-----|-----------------|---------------------|--------:|--------|
| lat1 | 256 × 96 | 144 × 32 | 1 | 50000/50000, 0 mismatch |
| lat2 | 256 × 96 | 144 × 32 | 2 | 50000/50000, 0 mismatch |

## Scope / next

v1 is 1R1W (1RW is the degenerate case; 2RW is a future flavor). The read-enable
*hold* path (`r_en=0`) is intentionally not exercised in DV — drive continuous
reads (the streaming use case) and treat `r_data_o` as valid `ReadLat` cycles
after `r_en`. Swapping the real MemoryCompiler macro in is a `sram_beh.sv`-local
change; nothing above it needs to know.
