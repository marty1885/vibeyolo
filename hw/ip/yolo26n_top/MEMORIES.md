# yolo26n — memory macro budget (early estimate)

This is the PD-facing list of SRAM/ROM macros the chip will eventually need.
Numbers are estimates derived from `scale_pkg.sv` and the YOLO26n graph; PD
should treat them as the **upper bound for memory compiler commissioning**
and we will tighten as the integration blocks land.

Conventions:
- All RAMs are single-clock (`clk_i` functional / `test_clk_i` for BIST).
- Wrap each macro in a `prim_ram_*` shim (already used by leaf IPs). PD only
  consumes the compiler view of the underlying macro.

## Skip-connection buffers (largest)

| Name        | Width | Depth   | Bytes   | Ports | Notes                                   |
| ----------- | ----- | ------- | ------- | ----- | --------------------------------------- |
| skip_p3     | 64 b  | 51 200  | 410 KB  | 1R1W  | 80×80×64 ch, neck P3 stream             |
| skip_p4     | 128 b | 12 800  | 205 KB  | 1R1W  | 40×40×128 ch, neck P4 stream            |

These dominate the on-die SRAM area and should be commissioned first.

## Weight ROMs (per layer)

102 conv layers. Each layer's weight ROM size = `Cout * Cin/group * K * K` bytes
(int8, symmetric). The 5 biggest sit at 256×256×3×3 = 576 KB each; the median
layer is ~16 KB. Total weight footprint ≈ 6.0 MB (regenerable via
`tools/layergen/`, will publish a per-layer table once weight broadcast
network is final).

PD: commission a generic ROM compiler view at 64b × {256, 1k, 4k, 16k, 64k}
depth as starter points.

## Activation LUTs (small)

| Name        | Width | Depth | Notes                              |
| ----------- | ----- | ----- | ---------------------------------- |
| silu_lut    | 8 b   | 256   | per-scale-variant, baked at gen    |
| sigmoid_lut | 8 b   | 256   | detect head                        |
| exp_lut     | 16 b  | 1024  | softmax16                          |
| recip_lut   | 16 b  | 1024  | softmax16                          |

Fit in logic gates today; can stay synthesised flops unless area pressure
calls for a tiny ROM compiler view.

## Skip / FIFO buffers (small)

`skip_buf` instances inside conv_layer + various pipeline FIFOs. Count and
depth depend on the final `P_PIX/P_COUT/P_CIN` choices; expect <64 KB
aggregate. Will inventory after block-level integration.

## TBD until block-level integration lands

- Detect-head top-K storage (sorted-K of ~8400 candidates across 3 scales).
- Attention QKV staging buffers (model.10, model.22).

## BIST

All RAMs are expected to be brought through a shared March-style BIST
controller driven by `bist_run_i` at the chip boundary. Pass/fail aggregates
into `bist_done_o` / `bist_fail_o`. Controller RTL lands with the first real
SRAM instantiation.
