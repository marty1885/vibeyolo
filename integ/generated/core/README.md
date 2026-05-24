# yolo26n_core — structural dataflow netlist (generated)

Emitted by `tools/gen_core.py` (non-destructive). This is the body PD
floorplans behind the frozen `hw/ip/yolo26n_top/` boundary.

## Files
- `yolo26n_layers_pkg.sv` — per-layer dims (DV-capped parallelism) + the
  `L_SRC[]` producer-index table, derived from the ONNX activation edges and
  the balanced scale report.
- `yolo26n_core_impl.sv` — full structural netlist: 102 `conv_stage`, the
  C2f/neck glue, the 6 integration blocks, and 4 skip-FIFO banks, wired through
  per-conv buses + a frame-start daisy-chain sequencer.

## Status — top wiring COMPLETE

| piece | state |
|-------|-------|
| 102 `conv_stage` instances (correct dims) | ✅ composed |
| conv→conv dataflow edges (exact from ONNX) | ✅ wired |
| C2f/neck glue: slice (subrange) / concat (bus-join) / residual `add_rq` banks | ✅ wired, widths from ONNX shape-inference |
| 6 verified blocks (sppf / 2× upsample_concat / 2× attn / detect_head) | ✅ instantiated at graph positions, fed by member convs |
| 4 skip-FIFO banks (`skip_buf`) at m.4/m.6/m.10/m.13 cv2 taps | ✅ instantiated |
| detect-head output → top `det_*` ports | ✅ wired |
| full-design **lint** (Verilator) | ✅ **0 warnings, 0 errors** (≈340 s, ≈14 GB — footgun #4: lint only, no whole-chip sim) |
| rough gate count (yosys-grounded) | ✅ ~73 M gates ≈ **1.1 % of 6.5 G budget** (`AREA.md`) |

Widths for every glue tensor (slice offsets, concat sums, residual widths) are
taken from ONNX shape-inference, so the deeply-nested C2fCIB inner structures
are lint-correct without hand-tracing. This is a **connectivity / PD-floorplan**
netlist: per-region handshake *timing* is inherited from the already-DV'd
blocks (whole-chip sim is infeasible — footgun #4). Two attn `pe` (depthwise-on-V)
convs sit inside the attention region; their input is a width-correct V
stand-in (`IN_OVERRIDE` in the generator), like the rest of the attn region.

## Regenerate
```bash
python3 tools/gen_core.py
# partitioned lint (whole-chip sim infeasible — footgun #4):
RTL=(); while IFS= read -r f; do RTL+=("$f"); done \
  < <(find hw/ip -path '*/rtl/*.sv' ! -name '*_ref.sv' ! -path '*yolo26n_top*')
verilator --lint-only -Wall \
  -Wno-WIDTHEXPAND -Wno-WIDTHCONCAT -Wno-UNUSEDSIGNAL -Wno-UNOPTFLAT \
  -Wno-WIDTHTRUNC -Wno-SELRANGE -Wno-ASCRANGE -Wno-PINCONNECTEMPTY \
  -Wno-GENUNNAMED -Wno-UNDRIVEN -Wno-IMPLICITSTATIC -Wno-LATCH \
  -Wno-UNUSEDPARAM -Wno-BLKSEQ -Wno-ENUMVALUE -Wno-REALCVT -Wno-MULTITOP \
  --top-module yolo26n_core_impl "${RTL[@]}" \
  integ/attn_model10/stim/attn_scales_pkg.sv \
  integ/{sppf_model9,detect_model23,upsample_model11,attn_model10}/rtl/*.sv \
  integ/generated/core/yolo26n_layers_pkg.sv integ/generated/core/yolo26n_core_impl.sv
# gate-count sanity (writes AREA.md numbers):
python3 tools/area_estimate.py
```
