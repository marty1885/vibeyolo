# Layer template — tiled Conv-BN-SiLU pattern

This directory holds the canonical structure every YOLO26n Conv-BN-SiLU layer
should follow once it has been retrofitted to honour
`scale_pkg::P_COUT_<i>` and `scale_pkg::P_CIN_<i>`.

The reference implementation is `layer_template.sv`. The first real layer
retrofitted to this pattern is `integ/layer_11/` — use it as a working
example.

## Why tile?

The legacy layers fully unrolled both the output-channel and input-feature
dimensions:

```
NCH_OUT * (K*K*CIN)   MAC cells per layer
```

For L11 that is `128 * 1152 = 147 k` MAC cells. Verilator only marginally
compiles it, and STA on the full 102-layer network is a non-starter.

The tiled layer instead instantiates only `P_COUT * (K*K*P_CIN)` MAC cells
and time-multiplexes the rest over `N_COUT_TILE × N_CIN_TILE` cycles per
output pixel. For L11 with `P_COUT=16, P_CIN=8`:

```
16 * 72 = 1 152 MAC cells (≈ 130× smaller)
cycles/pixel = 8 * 16 = 128
frame cycles = 40 * 40 * 128 = 204 800   (well inside the 1.84 M scale_pkg budget)
```

## How to compute the tile dimensions

For layer *i* with parameters from `scale_pkg`:

```
N_COUT_TILE = ceil(LAYER_i_COUT / LAYER_i_P_COUT)
N_CIN_TILE  = ceil(LAYER_i_CIN  / LAYER_i_P_CIN)
N_LANE      = LAYER_i_K * LAYER_i_K * LAYER_i_P_CIN
beats/pixel = N_COUT_TILE * N_CIN_TILE
frame_cyc   = LAYER_i_H * LAYER_i_W * N_COUT_TILE * N_CIN_TILE
              (matches LAYER_i_CYCLES from gen_scale_pkg.py)
```

`gen_scale_pkg.py` chooses `P_COUT` and `P_CIN` to land inside `T_FRAME`; the
RTL does not need to know that — it just iterates `N_COUT_TILE × N_CIN_TILE`
phases.

## Module structure

See `layer_template.sv` for the canonical form. Stages:

| Stage | What | Latency |
| ----- | ---- | ------- |
| A | `P_COUT` parallel `dotN(N=K*K*P_CIN)` | `1 + clog2(N_LANE)` |
| B | i32 accumulator across `cin_tile` (P_COUT wide). `first_cin` clears, `last_cin` commits. | 1 |
| C | `P_COUT` parallel `requant` (internal i32-autoscale, no ACC_SHIFT trick required) | 4 |
| D | `P_COUT` parallel `act_silu` | 1 |

Output stream: `valid_o` + `y_o[P_COUT-1:0]` + `cout_tile_idx_o`. The
consumer stitches `N_COUT_TILE` beats per pixel to reconstruct the full
`COUT`-wide activation.

The legacy `ACC_SHIFT` workaround in L11 is **no longer needed**. The new
`requant.sv` autoscales i32→fp16 and bumps the per-channel `scale_fp16`
exponent to compensate; layers feed `scale_fp16` straight from the trained
model (no `2^ACC_SHIFT` premultiply). `extract.py` should set `ACC_SHIFT=0`
or remove the term entirely.

## Driver / TB pattern

The TB advances `cin_tile` innermost, `cout_tile` outermost, pixel
outermost-of-all:

```
for pix in 0..H_out*W_out-1:
  (oy, ox) = (pix / W_out, pix % W_out)
  for ct in 0..N_COUT_TILE-1:
    for cit in 0..N_CIN_TILE-1:
      x_i             = build_window_tile(input, oy, ox, cit)
      w_i             = build_weight_tile(weights_full, ct, cit)
      scale_i, bias_i = per-channel slice  [ct*P_COUT .. ct*P_COUT+P_COUT-1]
      first_cin_i     = (cit == 0)
      last_cin_i      = (cit == N_CIN_TILE - 1)
      cout_tile_idx_i = ct
      valid_i         = 1
      tick()
```

`build_window_tile(oy, ox, cit)` returns `K*K*P_CIN` i8 lanes. Lane index
`(kh*K + kw)*P_CIN + kc_local`, where `kc_local ∈ [0, P_CIN)` corresponds
to global input channel `cit*P_CIN + kc_local`.

`build_weight_tile(ct, cit)` returns `P_COUT * K*K*P_CIN` i8 weights with
the same tiling.

On every `valid_o` beat the testbench reads `y_o[P_COUT-1:0]` and
`cout_tile_idx_o`, and writes those 16 i8 values into the
`(pix, cout_tile_idx_o*P_COUT .. +P_COUT-1)` slot of the reconstructed
output. With `N_COUT_TILE` beats per pixel the full `COUT`-wide tensor is
recovered. Compare against the ORT reference as before.

## Retrofitting checklist (for the next layer agent)

1. Copy `layer_template.sv` to `integ/layer_<i>/rtl/layer_<i>.sv` and
   rename the module + replace the parameter defaults with
   `LAYER_<i>_*` from `scale_pkg`.
2. Update `dv/layer_<i>_tb.sv` to flatten the new ports: `x_flat_i` is now
   `K*K*P_CIN*8` bits, `w_flat_i` is `P_COUT*K*K*P_CIN*8` bits,
   `scale_flat_i`/`bias_flat_i` are `P_COUT*16` bits each.
3. Update `dv/layer_<i>_test.cc` to drive `(cin innermost, cout outer)`
   beats with `first_cin`/`last_cin`/`cout_tile_idx`, and reconstruct the
   full `COUT`-wide output from `N_COUT_TILE` tile beats per pixel.
4. Remove any `ACC_SHIFT` premultiply from `extract.py` (set to 0). The
   new requant.sv handles i32 autoscale internally.
5. Build target: `make -C integ/layer_<i>/dv test` should complete in
   well under a minute. If it takes longer, double-check that the layer is
   instantiating `P_COUT` dotN cells (not `COUT`).
6. Pass criterion: cosine ≥ 0.998 vs ORT on ≥ 3 random samples.
