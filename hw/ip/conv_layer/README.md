# conv_layer — reusable tiled Conv (+ optional residual + optional SiLU)

`conv_layer` is the single, fully-parameterised IP behind every YOLO26n
Conv-BN-SiLU layer. A real layer becomes a ~30-line shim that:

1. imports `scale_pkg::LAYER_<i>_*`,
2. forwards those constants as parameters to `conv_layer`,
3. exposes whatever port shape the existing TB / next layer expects.

The architecture is identical to the canonical tile pattern documented in
`integ/_layer_template/README.md`: `P_COUT` parallel `dotN` cells, each
`N=K*K*P_CIN` wide, time-multiplexed over `N_COUT_TILE × N_CIN_TILE` beats
per output pixel.

## Parameters

| Name            | Type | Notes                                                                 |
| --------------- | ---- | --------------------------------------------------------------------- |
| `CIN`           | int  | input channels                                                        |
| `COUT`          | int  | output channels                                                       |
| `K`             | int  | kernel size (1 or 3 in YOLO26n)                                       |
| `STRIDE`        | int  | informational only; tiles are pixel-agnostic                          |
| `PAD`           | int  | informational only; padding is handled outside                        |
| `H_OUT`/`W_OUT` | int  | informational only                                                    |
| `P_COUT`        | int  | parallelism in output channels — must divide `COUT`                   |
| `P_CIN`         | int  | parallelism in input channels  — must divide `CIN`                    |
| `RESIDUAL`      | int  | 0/1. If 1, an `add_rq` cell folds `r_i` (int8) into the activation    |
| `SILU`          | int  | 0/1. If 0, the SiLU LUT is bypassed (used by some head layers)        |
| `S_OUT_PRE`     | real | SiLU input scale  (i8 → fp; default `4/127`)                          |
| `S_OUT_SILU`    | real | SiLU output scale (fp → i8; default `4/127`). Also used by add_rq.    |

Constraints:
- `COUT % P_COUT == 0` and `CIN % P_CIN == 0` (the driver may pad, but the
  IP itself assumes integer tile counts for `cout_tile_idx_i` width).
- `P_COUT` and `P_CIN` come from `gen_scale_pkg.py` and are sized so the
  per-frame cycle count fits the global `T_FRAME` budget.

## Ports (summary)

```
clk_i, rst_ni
valid_i, ready_o          // ready_o is always 1 (feed-forward pipeline)
first_cin_i, last_cin_i   // cin-tile phase markers
cout_tile_idx_i           // outer cout tile id
x_i      [K*K*P_CIN]      // i8 lanes for current cin tile of current patch
w_i      [P_COUT][K*K*P_CIN]  // i8 weights for (cout_tile, cin_tile)
scale_i  [P_COUT]         // fp16, gated by last_cin
bias_i   [P_COUT]         // fp16, gated by last_cin
r_i      [P_COUT]         // RESIDUAL only: int8 residual stream
r_scale_i[P_COUT]         // RESIDUAL only: fp16 residual scale (per cout)
r_bias_i [P_COUT]         // RESIDUAL only: fp16 post-add bias

valid_o, ready_i          // ready_i is currently observed but unused
cout_tile_idx_o           // tag passed through
y_o      [P_COUT]         // i8 activations
```

## Dataflow

Per output pixel `(oy, ox)`:

```
for cout_tile in 0 .. N_COUT_TILE-1:        (outer)
  acc[0..P_COUT-1] = 0
  for cin_tile in 0 .. N_CIN_TILE-1:        (inner)
    acc += dotN(x_tile, w_tile[cout_tile][cin_tile])
  y_pre   = requant(acc, scale[cout_tile], bias[cout_tile])
  y_silu  = SILU     ? silu(y_pre) : y_pre
  y_final = RESIDUAL ? add_rq(y_silu, r[cout_tile], r_scale, r_bias) : y_silu
  emit (cout_tile_idx=cout_tile, y_final)
```

Cycles per pixel = `N_COUT_TILE × N_CIN_TILE`.
Total pipeline latency = `1 + clog2(N_LANE)` (dotN) + 1 (acc) + 1 (rq_in)
+ 4 (requant) + `[1 if SILU]` + `[5 if RESIDUAL]`.

## Residual / S_R folding

The `add_rq` IP requires the caller to supply both per-stream fp16 scales
and a single `inv_out_scale`. `conv_layer` wires them as:

| add_rq port             | Source                                                        |
| ----------------------- | ------------------------------------------------------------- |
| `a_i8_i`                | post-(requant+SiLU) activation from the convolution           |
| `scale_a_fp16_i`        | **constant** fp16 of `S_OUT_SILU` (compile-time)              |
| `b_i8_i`                | `r_i`                                                         |
| `scale_b_fp16_i`        | `r_scale_i` (per-output-channel)                              |
| `inv_out_scale_fp16_i`  | **constant** fp16 of `1/S_OUT_SILU` (compile-time)            |
| `bias_fp16_i`           | `r_bias_i` (per-output-channel)                               |

That is, the conv output and the residual share the same output grid
(`S_OUT_SILU`). Any per-channel trim is folded into `r_bias_i` by the
extractor. Because `r_*` is per-(pixel,cout_tile) and not per cin-tile,
the IP samples it on the `last_cin` beat and re-aligns through an internal
shift register so it reaches `add_rq` together with the matching activation.

## How to write a new layer

1. **Extract**: `integ/layer_<i>/extract.py` pulls ORT weights/scales/biases
   per channel, dumps `.hex` golden tensors, and (for residual layers)
   computes `r_bias_fp16 = trained_bias - extra_shift`. Set `ACC_SHIFT=0`
   — `requant` autoscales.
2. **Shim**: copy `integ/layer_11/rtl/layer_11.sv` (a ~80-line wrapper) to
   `integ/layer_<i>/rtl/layer_<i>.sv` and rename. Bind `LAYER_<i>_*` to
   `conv_layer` parameters; pick `P_COUT`/`P_CIN` from `scale_pkg`.
   Set `RESIDUAL=1`/`SILU=0` if applicable. Tie unused residual ports to
   zero.
3. **TB**: copy `integ/layer_11/dv/layer_11_tb.sv` and adjust the flat port
   widths to the new `K*K*P_CIN`, `P_COUT*K*K*P_CIN`, etc.
4. **Test driver**: copy `integ/layer_11/dv/layer_11_test.cc` and
   re-parameterise the geometry constants. The driver pattern (cin
   innermost, cout outer, pixel outermost) is identical for every layer.
5. **Build & run**: `make -C integ/layer_<i>/dv test`. Pass criterion is
   cosine ≥ 0.998 vs the ORT golden on ≥ 3 random samples.

## DV

`hw/ip/conv_layer/dv` builds four standalone Verilator binaries (one per
config) and runs each on 4 random seeds. Each seed:

- Generates random i8 inputs/weights and small per-channel fp16 scales.
- Runs DUT and an independent SV reference (`conv_layer_ref.sv`) in
  lockstep — must be **bit-exact** (same sub-IPs, different dotN coding).
- Compares DUT to a C++ floating-point shadow — must reach cosine ≥ 0.999
  (no quantization noise floor; both sides use the same math).

| Config | Family   | K | STRIDE | CIN | COUT | P_COUT | P_CIN | RESIDUAL | SILU |
| ------ | -------- | - | ------ | --- | ---- | ------ | ----- | -------- | ---- |
| C1     | layer_2  | 1 | 1      | 32  | 32   | 16     | 8     | 0        | 1    |
| C2     | layer_3  | 3 | 1      | 16  | 8    | 4      | 4     | 0        | 1    |
| C3     | layer_11 | 3 | 2      | 128 | 128  | 32     | 16    | 0        | 1    |
| C4     | layer_4  | 3 | 1      | 16  | 32   | 8      | 4     | 1        | 1    |

Run:

```
make -C hw/ip/conv_layer/dv test       # all 4 configs, ~50s
make -C hw/ip/conv_layer/dv test-c1    # one config
make -C hw/ip/conv_layer/dv lint
```
