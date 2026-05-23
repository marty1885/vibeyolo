# box_affine

Per-anchor **affine box decode** for the YOLO26n detect head (`/model.23`).

This export's box branch (`cv2.x.2`) regresses the four **ltrb distances
directly** as int8 — there is **no DFL / 16-bin distribution**, so `box_decode`
(the DFL decoder) does not apply here. `box_affine` is the simpler direct path.

## Math

```
d      = fp16(ltrb_i8) * S_box                 (grid units)
anchor = (col + 0.5, row + 0.5)                (grid units, counter-derived)
x1 = (col+0.5 - d_l)*stride = cx_center - d_l*stride
x2 = (col+0.5 + d_r)*stride = cx_center + d_r*stride     (cx_center=(col+0.5)*stride)
y1,y2 likewise with row, d_t, d_b
cx=(x1+x2)/2  cy=(y1+y2)/2  w=x2-x1  h=y2-y1    (pixel units)
{cx,cy,w,h} /= 640                              (normalized output)
```

Centers fold `/2` and `/640` into one `×(1/1280)`; sizes take `×(1/640)`. All
arithmetic is fp16 via `fp16_fma` (fused single-rounding). `col`/`row`/`stride`
arrive as small i32 (the head's anchor counter derives them — no ROM) and are
converted with `i32_to_fp16`.

## Interface

`valid_i` + {`l_i,t_i,r_i,b_i` (int8), `s_box_i` (fp16), `col_i,row_i,stride_i`
(i32)} → after **7 cycles** `valid_o` + {`cx_o,cy_o,w_o,h_o`} (normalized fp16).
Throughput 1 anchor/cycle.

## DV — fp16 ULP tolerance with cancellation-aware fallback

Lockstep DUT vs an independent SV `real` golden (`box_affine_ref`) and a C++
double shadow. **7/7 checks pass** (directed + 20 000 random anchors across all
3 scales). `ref`-vs-`shadow` is **0 ULP** everywhere; `dut`-vs-`ref/shadow` is
within `16 ULP` **or** an absolute fallback.

The fallback matters: `x1=(ax−d_l)·stride` and `x2−x1` **cancel** — at stride=32
the operands (`cx_center`, `d·stride`) can be ~500 while the result is ~3, so the
worst `dut`-vs-shadow ULP reaches **4556** purely because fp16 steps are tiny
near zero (absolute error only ~7e-4 ≈ 0.13 px). The fp16 rounding floor scales
with the **pre-cancellation pixel magnitude** `pmax`, not the cancelled result,
so the absolute tolerance is `8·fp16_step(pmax)/{1280,640}` for centers/sizes.
Verified to ~0.6× of that floor in a 60k-sample sweep. fp16 arithmetic ⇒
tolerance, not bit-exact — the head's integer stages (reduce_max/topk/gather)
remain bit-exact.
