# box_decode

DFL (Distribution Focal Loss) detection-head decode for YOLO26n. Converts
four softmaxed bin-probability vectors plus a grid cell (cx, cy) and
stride into an xyxy box in pixel units, all in IEEE-754 binary16.

## Math

For each side `s ∈ {l, t, r, b}`:

```
d[s] = Σ_{i=0..15} p[s][i] * i        (in bin units; multiply by stride for pixels)
```

Box xyxy (pixel units) at grid (cx, cy) with stride `S`:

```
x1 = (cx + 0.5 − d_l) * S          = cx_center − d_l * S
y1 = (cy + 0.5 − d_t) * S          = cy_center − d_t * S
x2 = (cx + 0.5 + d_r) * S          = cx_center + d_r * S
y2 = (cy + 0.5 + d_b) * S          = cy_center + d_b * S
```

where `cx_center = (cx + 0.5) * S` and likewise for `cy_center`.

## Interface

```sv
module box_decode (
  input  logic               clk_i, rst_ni,
  input  logic               valid_i,
  input  logic        [15:0] p_l_i [16],
  input  logic        [15:0] p_t_i [16],
  input  logic        [15:0] p_r_i [16],
  input  logic        [15:0] p_b_i [16],
  input  logic signed [15:0] cx_i,
  input  logic signed [15:0] cy_i,
  input  logic signed [15:0] stride_i,
  output logic               valid_o,
  output logic        [15:0] x1_o,
  output logic        [15:0] y1_o,
  output logic        [15:0] x2_o,
  output logic        [15:0] y2_o
);
```

* Inputs `p_*_i[16]` are fp16 probabilities (typically softmaxed; not
  required to sum to 1.0 — the module just does a weighted sum).
* `cx_i`, `cy_i`, `stride_i` are signed int16 grid coordinates and stride
  in pixel units. They are sign-extended to int32 and converted to fp16
  internally via `i32_to_fp16`.
* Bin indices 0..15 are baked as fp16 constants (`BIN_FP16`).

## Implementation

Fully pipelined, feed-forward, throughput = 1 box per cycle.

| Stage | Cycle | Operation                                                          |
|-------|-------|--------------------------------------------------------------------|
| S0    | 1     | Register inputs (probs, cx, cy, stride).                           |
| S1    | 2     | 64 parallel `fp16_fma` multiplies `m[s][i] = p[s][i] * bin_fp16[i]`. <br/> 3 × `i32_to_fp16` convert cx, cy, stride to fp16. |
| S2    | 3     | Per side: 8 × `fp16_fma` adds (a*1 + c). <br/> `half_stride = stride_fp16 * 0.5`. |
| S3    | 4     | Per side: 4 adds. <br/> `cx_center = cx_fp16 * stride_fp16 + half_stride`, same for cy. |
| S4    | 5     | Per side: 2 adds.                                                  |
| S5    | 6     | Per side: 1 add → `d[s]` ready.                                    |
| S6    | 7     | Per coord: `fp16_fma(d, ±stride, c_center)` → x1, y1, x2, y2.      |

Negation of stride for the x1/y1 path is a sign-bit flip on the fp16
value (no datapath cost). `cx_center`, `cy_center` and `stride_fp16` are
piped through S3 → S5 to align with the final stage.

All adders and multipliers are instances of `fp16_fma`; both pure
multiplies (c = 0) and pure adds (b = 1.0) reuse the same building block,
giving uniform RNE rounding semantics throughout.

**Latency:** 7 cycles from `valid_i` high to `valid_o` high.
**Throughput:** 1 box / cycle.

## Verification

`make -C hw/ip/box_decode/dv test` — runs 22 checks:

1. Reset behaviour (outputs deasserted).
2. Uniform `1/16` probabilities → `d = 7.5` per side; box centred on cell.
3. One-hot at `i = 0` → `d = 0`; box collapses to `(cx_center, cy_center)`.
4. One-hot at `i = 15` → `d = 15`; maximum-extent box at chosen stride.
5. 150 random vectors across strides {8, 16, 32} with random cx, cy.
6. 2000 random vectors with random strides and grid positions.
7. 500 peaked-distribution vectors (narrow softmax → near-one-hot).

The TB instantiates `box_decode` and `box_decode_ref` (SV behavioral
golden in `real` arithmetic) in lockstep. The C++ test also computes an
independent double-precision shadow. Per-output comparison reports
sign-magnitude fp16 ULP distance against both reference and shadow.

### Tolerance: 16 ULP + cancellation-aware absolute fallback

The `cx_center − d * stride` formulation is inherently
cancellation-prone in fp16 when the box edge lies near the cell centre.
ULP distance near zero exaggerates such cases — a 0.14-pixel absolute
error on a 0.5-pixel result is 284 ULPs but completely benign for box
decoding (an order of magnitude smaller than any meaningful detection).

The tolerance is therefore:

```
pass if  ulp_diff ≤ 16   OR   |dut − ref| ≤ 8 * fp16_step(max(|cx_center|, |d*stride|))
```

The absolute fallback bounds error by the fp16 quantisation of the
larger operand entering the final subtract — i.e. it accepts any result
that is "as accurate as the input operand magnitudes permit."

Empirically:

* `ref` vs `shadow`: **0 ULP** across all tests (both compute in
  double-precision-equivalent arithmetic and round identically to fp16
  at the output).
* `dut` vs `ref` worst observed: **284 ULPs** (random), all on
  cancellation-dominated cases that satisfy the absolute fallback;
  typical non-cancellation worst case ≈ 10 ULPs.
* `dut` vs `shadow`: identical pattern to dut-vs-ref.

All 22 checks pass.

### Coverage

| metric  | %    |
|---------|------|
| line    | 87.9 |
| toggle  | 76.3 |
| branch  | 71.6 |
| expr    | 55.6 |

The uncovered branches are largely inside the reused `fp16_fma` paths
(subnormal/Inf/NaN handling that the DFL inputs do not exercise — the
multiplicands `p[i] * i` are always finite small positives, and bin
indices are baked constants) and the `i32_to_fp16` overflow/saturation
paths (cx, cy, stride are small int16).

## Reused blocks

* `hw/ip/fp16_fma/` — 64 multipliers in S1 + 60 add/fma instances across
  S2 – S6 (15 add-tree stages × 4 sides) + 2 centre fma + 4 final fma +
  4 negation-of-stride is sign-flip-only.
* `hw/ip/i32_to_fp16/` — 3 instances (cx, cy, stride).
