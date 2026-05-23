# detect_model23 — YOLO26n detect head (end-to-end, no NMS)

The `/model.23` post-processing block. Ingests the six int8 conv-tail streams
(box `cv2.x.2` 4ch, cls `cv3.x.2` 80ch) for all 8400 anchors and emits the
model outputs: `pred_boxes[300][4]` (normalized cx,cy,w,h fp16) and
`logits[300][80]` (fp16), selected by `TopK(reduce_max(cls), k=300)`.

This export has **no DFL** (box branch regresses 4 ltrb directly) and **no
sigmoid** in hardware (output is raw logits; ranking on logits is monotone-
equivalent). Anchor grid `(col+.5,row+.5)` and stride `{8,16,32}` are counter-
derived — no ROM.

## Composition (`rtl/detect_head.sv`)

```
P_DECODE (8400 cyc): per anchor →
   box_affine  -> boxes_mem[idx]            (67 KB)
   cls_i       -> logits_mem[idx]           (672 KB)
   reduce_max_n -> dequant_n(N=1) -> score  -> score_mem[idx] (16.8 KB)
P_TOPK  (<=~76k): feed score_mem[0..8399] into topk_fp16 at its in_ready rate
                  (random-access read decouples from the heap's sift — no
                  backpressure FIFO needed)
P_GATHER (~302): per heap index → boxes_mem (direct) + logits_mem -> dequant_n
                 with the index's originating-scale S_cls -> stream out
```

Built from verified leaf IPs: `box_affine`, `reduce_max_n`, `dequant_n`,
`topk_fp16`, `i32_to_fp16`, `fp16_fma`. SRAMs are behavioral arrays (PD swaps
for macros). **Cross-scale care:** the three scales carry different per-tensor
cls `S_OUT`, so the score dequants to a common fp16 before TopK ranks across
all anchors, and gathered logits dequant with their originating scale's S_OUT.

## Cycle budget

Sequential phases: decode 8400 + topk + gather ~302. TopK is the tall pole.
- **Typical** (random scores, measured in DV): **~23.7k cyc/frame**.
- **Worst case** (topk's adversarial sorted-ascending feed, 75,981 standalone):
  8400 + 75,981 + 302 ≈ **84.7k < T_FRAME=100k**.

## DV (`make -C dv test`)

Staged, against the int8 reference (the real gate) with ORT for sanity — see
`extract.py` and the DV-contract notes. **7/7 checks pass** across 6 inputs:
- selected anchor **set exact** vs the int8 reference TopK (`common=300/300`),
- gathered logits **bit-exact** (`logit_mism=0`),
- gathered boxes within fp16 ULP (16 ULP or a cancellation-aware
  `8·fp16_step(pmax)/{1280,640}` floor; observed ≤217 ULP on near-zero `w`).

Per the locked contract, end-to-end `pred_boxes`-vs-ORT cosine is **not** gated
on these object-free inputs (tie-driven TopK selection churn) — that waits for
real-image E2E at top-level. The topology itself is proven: a float numpy model
of the whole tail reproduces ORT bit-for-bit (see `extract.py` self-check).

## Files
- `extract.py` — staged ORT extractor: int8 stim for the 6 conv tails + int8-
  reference goldens (`*_ref_*.bin`, the gate) + ORT goldens (`*_ort_*.bin`,
  sanity) + `scales.f32`. Built-in self-check proves topology and int8 fidelity.
- `rtl/detect_head.sv` — the composed head.
- `dv/` — TB + C++ test (stim path derived from `__FILE__`, portable) + Makefile.
