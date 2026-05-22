# layergen

Non-destructive generator path for YOLO26n layer integration.

The hand-built `integ/layer_*` directories remain the golden reference. This
tool writes generated artifacts under `integ/generated/` by default.

## What It Does Today

- Reads `integ/yolo26n/model_int8.onnx` through the existing scale inference code.
- Emits balanced real-chip and DV scale packages.
- Emits conv-layer RTL/DV wrapper skeletons for selected layer indices,
  including `P_PIX>1` by instantiating multiple `conv_layer` lanes.
- Lints generated wrapper skeletons against the existing `conv_layer` primitive.

The balancing rule is area-first: minimize `P_PIX * P_COUT * P_CIN` while
keeping each layer under `T_FRAME`. Grouped/depthwise convs count only
`Cin/group` input channels. Ties prefer cycles closer to the target.

By default `--max-p-pix=2`. The `conv_layer` leaf remains single-pixel; generated
wrappers implement pixel parallelism by instantiating one `conv_layer` lane per
pixel and sharing weights/scales across those lanes.

## Example

```bash
python3 tools/layergen/layergen.py --emit-scale --emit-dv-scale --gen-layer 0 --gen-layer 21
make -C integ/generated/layer_0_model0/dv lint
make -C integ/generated/layer_21_model7/dv lint
```

Generated files:

```text
integ/generated/scale/scale_pkg_balanced.sv
integ/generated/scale/scale_pkg_balanced_dv.sv
integ/generated/scale/scale_report_balanced.md
integ/generated/scale/scale_report_balanced_dv.md
integ/generated/layer_0_model0/
integ/generated/layer_21_model7/
```

## Current Limitations

- Does not yet generate ORT stimulus/reference extraction.
- Does not yet generate the C++ sample driver.
- Generated shims assume non-residual Conv+SiLU. Residual/add/concat/attention
  boundaries still need explicit handling.
- DV scale reports may miss `T_FRAME` because caps intentionally limit
  Verilator size; real-chip balanced scale is the timing reference.
