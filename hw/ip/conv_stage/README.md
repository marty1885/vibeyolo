# conv_stage — synthesizable per-layer streaming wrapper

`conv_stage` is the per-layer datapath the **chip** instantiates. It turns a
raster int8 activation stream into the next layer's raster int8 stream, hiding
the per-tile dataflow that the per-layer DV C++ drivers used to perform in
software. It is the missing synthesizable glue between the verified
`conv_layer` compute core and a real top-level netlist.

```
ivalid/idata[CIN] ─▶ linebuf_kxk(K,W_IN,H_IN,CIN)   (zero same-padding)
                          │ K*K*CIN patch / input position
        stride decimate ──┤ keep positions where row%S==0 && col%S==0
                       patch_q ──┐  tile-sequencer FSM
   on-die ROMs ($readmemh):      │  per out-pixel: outer cout-tile (ct),
     WROM [COUT][K*K*CIN] i8     ├─▶  inner cin-tile (cit); first/last_cin,
     SROM [COUT] fp16 scale      │   cout_tile_idx ─▶ conv_layer
     BROM [COUT] fp16 bias       │  collect P_COUT lanes/valid_o beat keyed by
                                 └─  cout_tile_idx_o, reassemble COUT pixel
   ─▶ ovalid/odata[COUT];  done_o pulses after the last output pixel
```

## Interface

| group | signals |
|-------|---------|
| frame | `start_i` (pulse), `done_o` (pulse) |
| input | `ivalid_i`, `iready_o`, `idata_i[CIN][7:0]` — one input pixel/beat, raster |
| output| `ovalid_o`, `oready_i`, `odata_o[COUT][7:0]` — one output pixel/beat, raster |

Matches the streaming idiom of the integration blocks (sppf / upsample_concat /
attn / detect_head), so the top-level netlist wires conv stages and blocks the
same way.

## Parameters

`CIN, COUT, K, STRIDE, PAD, H_IN, W_IN, P_COUT, P_CIN, SILU, S_OUT_PRE,
S_OUT_SILU, WINIT, SINIT, BINIT`. Weights/scale/bias are layer-constant on-die
ROMs loaded via `$readmemh` (`WINIT`/`SINIT`/`BINIT`), matching the existing
`weight_rom` pattern — appropriate for this fixed-function, area-unconstrained
accelerator. Depthwise layers arrive pre-expanded to dense weights (generator
zero-expand) and need no special handling.

## Verification

`make -C hw/ip/conv_stage/dv test-all` — **dual-instance bit-exact** DV. One
Verilator design elaborates both the DUT (`conv_stage`, frame-stream driven) and
a bare `conv_layer` REF driven by the already-ORT/ref-validated software tile
schedule over a zero-padded copy of the frame. Both share the identical
`conv_layer` compute, so any mismatch is a `conv_stage` sequencing/plumbing bug.

Configs (all PASS, 0 mismatches):

| cfg | shape | exercises |
|-----|-------|-----------|
| a | k3 s1 8→8 6² | multi cin/cout tile |
| b | k3 s2 8→8 8² | stride decimation |
| c | k1 s1 8→8 5² | single cout tile |
| e | k3 s1 6→6 5² | partial tiles (COUT,CIN not multiples of P) |
| f | k1 s1 8→8 5² | no-SiLU tail |

## Scope / extensions

v1 is group=1 (depthwise handled via dense expansion), `RESIDUAL=0`, `P_PIX=1`.
Residual graph edges (C2f / attention `Add` nodes) are wired at the top level as
explicit `add_rq` stages (matching the ONNX `Add` nodes 1:1) rather than folded
into `conv_stage`. The single-output-hold backpressure trades a little
intra-stage overlap for a provably overflow-free design; this affects only sim
cycle count, not correctness — timing sign-off uses the analytic per-stage
cycles from the balanced scale report.
