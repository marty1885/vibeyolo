# weight_rom

Per-layer weight + scale + bias ROM for the int8 YOLO accelerator. Keyed
by output-channel index, holds:

1. **`Wq`** — int8 weights, packed `Kh*Kw*Ic` bytes per output channel.
2. **`scale`** — fp16, per output channel.
3. **`bias`** — fp16, per output channel.

Read-only single port, 1-cycle latency, hold-on-`!req_i`. Self-contained
for now (no `prim_ram_1p` / `dft_pkg` / `ram_cfg_pkg` dependency); the
backing store is a plain `logic [W-1:0] mem [Depth]` initialized via
`$readmemh`. Later we can swap to `prim_ram_1p` once it is vendored in.

## Parameters

| Parameter  | Default | Description                                    |
|------------|---------|------------------------------------------------|
| `Kh`       | 3       | Kernel height.                                 |
| `Kw`       | 3       | Kernel width.                                  |
| `Ic`       | 16      | Input channels.                                |
| `Oc`       | 32      | Output channels (= memory depth).              |
| `InitFile` | `""`    | Hex file prefix; empty = zero-init.            |

Derived: `RowLen = Kh*Kw*Ic`, `RowBits = RowLen*8`, `AddrW = $clog2(Oc)`.

## Ports

| Port         | Dir | Width        | Description                            |
|--------------|-----|--------------|----------------------------------------|
| `clk_i`      | in  | 1            | Clock, rising edge.                    |
| `rst_ni`     | in  | 1            | Async-assert/sync-deassert active-low. |
| `req_i`      | in  | 1            | Read request.                          |
| `oc_addr_i`  | in  | `AddrW`      | Output-channel address.                |
| `w_row_o`    | out | `RowBits`    | Weight row (registered, 1-cycle).      |
| `scale_o`    | out | 16           | fp16 scale (registered, 1-cycle).      |
| `bias_o`     | out | 16           | fp16 bias  (registered, 1-cycle).      |

On cycle T, assert `req_i` and `oc_addr_i`; on cycle T+1, `w_row_o`,
`scale_o`, `bias_o` are valid. With `req_i = 0`, outputs hold their
previous value (matches `prim_ram_1p` semantics).

## $readmemh file format

Three separate files are loaded, keyed off `InitFile`:

- `<InitFile>.w.hex` — one line per output channel; each line is a
  hex word of width `RowBits` (`Kh*Kw*Ic*2` hex chars), MSB-first, so
  byte index `b` of a row lives at bits `[b*8 +: 8]`.
- `<InitFile>.s.hex` — one line per output channel; 4 hex chars (fp16).
- `<InitFile>.b.hex` — one line per output channel; 4 hex chars (fp16).

When `InitFile == ""`, the `$readmemh` calls are skipped and the memory
is zero-init (silences lint about a missing default file).

A reference generator lives at `dv/gen_hex.py`; the DV Makefile invokes
it before elaboration. The pattern it writes (used by the test as a
shadow check) is:

```
row N, byte b: (N * RowLen + b) & 0xFF
scale[N]:      fp16(1.0 / (N + 1))
bias[N]:       fp16(N * 0.5)
```

## Running DV

```
make -C hw/ip/weight_rom/dv test
make -C hw/ip/weight_rom/dv lint
make -C hw/ip/weight_rom/dv clean
```

The DV wraps `weight_rom` (DUT) and `weight_rom_ref` (SV behavioral
golden, with a different indexing path) in `weight_rom_tb`, drives the
same `req_i`/`oc_addr_i` stimulus to both, and checks (a) `mismatch_o`
between the two SV implementations every cycle and (b) the DUT against
an independent C++ shadow computed from the same deterministic pattern
the Python generator writes — so a shared `$readmemh` indexing bug
would still surface.

Default DV configuration: `Kh=3, Kw=3, Ic=4, Oc=8`. Override on the
make line, e.g. `make WROM_OC=16 test`.
