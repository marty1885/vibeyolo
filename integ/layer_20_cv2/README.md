# layer_20_cv2 — YOLO26n `/model.6/cv2/conv` Conv-BN-SiLU

Integration test for L20, the non-residual exit convolution of the
`/model.6` C3k2 block.

## Slice

```
/model.6/Concat_output_0_quantized
 -> ConvInteger             (192 -> 128, 1x1, stride 1, pad 0)
 -> Cast i32 -> fp32
 -> Mul (s_a * s_w)
 -> Add bias
 -> Sigmoid -> Mul          (SiLU)
 -> /model.6/cv2/act/Mul_output_0
```

The DV test drives an 8x8 ROI on the 40x40 input/output plane. Stimulus
and references are generated from `integ/yolo26n/model_int8.onnx` by
`extract.py`.

## Files

```
extract.py
rtl/layer_20.sv
dv/layer_20_tb.sv
dv/layer_20_test.cc
dv/Makefile
stim/
```

## Running

```bash
make -C integ/layer_20_cv2/dv test
```

The Makefile intentionally compiles `integ/scale/scale_pkg_dv.sv`, not
the real-chip `scale_pkg.sv`, so Verilator uses the capped
`P_COUT=64, P_CIN=4` configuration.
