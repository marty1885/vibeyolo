#!/usr/bin/env python3
# Copyright (c) 2026 vibeyolo
# SPDX-License-Identifier: Apache-2.0
#
# Detect-head extractor for YOLO26n /model.23 (end-to-end, NO NMS).
#
# This block sits AFTER the six already-validated conv tails:
#   box (cv2.x.2, 4ch):  L64 (80x80), L82 (40x40), L98 (20x20)
#   cls (cv3.x.2, 80ch): L70 (80x80), L87 (40x40), L101 (20x20)
#
# The DUT (hw/ip/detect_head) ingests those six int8 conv outputs and
# reproduces the entire /model.23 post-processing tail:
#
#   BOX:  ltrb[4,8400] (3 scales flattened+concat, row-major)
#         x1y1 = anchor - lt ;  x2y2 = anchor + rb     (anchor = col+.5,row+.5)
#         xyxy *= stride                                (stride = {8,16,32})
#         cx=(x1+x2)/2  cy=(y1+y2)/2  w=x2-x1  h=y2-y1
#         boxes /= 640                                  -> boxes[8400,4]
#   CLS:  logits[80,8400] (3 scales flattened+concat)
#         score = ReduceMax over 80                     -> score[8400]
#   SEL:  idx = TopK(score, k=300)                       -> idx[300]
#   GATHER: pred_boxes = boxes[idx]  -> [300,4]
#           logits     = logits[idx] -> [300,80]
#
# Critical composition risk this extractor pins down (and the staged DV
# guards): the THREE scales have DIFFERENT per-tensor cls S_OUT. Reduce-max
# stays int8 within a scale, but the score fed to TopK must be dequantized
# to a common fp16 before ranking across all 8400 anchors, and gathered
# logits must dequant with their ORIGINATING scale's S_OUT.
#
# Self-check built in: a float numpy model must reproduce ORT's outputs
# (proves topology), then an int8 model sets the RTL accuracy target.

import json
import os
import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, TensorProto

HERE = os.path.dirname(os.path.abspath(__file__))
STIM = os.path.join(HERE, "stim")
os.makedirs(STIM, exist_ok=True)
MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    "/home/marty/Documents/aif/vibeyolo/integ/yolo26n/model_int8.onnx",
)

# ---- geometry ----------------------------------------------------------
SCALES = [
    # (name, grid, H=W, stride, n_anchors)
    ("s0", 80, 8,  6400),
    ("s1", 40, 16, 1600),
    ("s2", 20, 32, 400),
]
N_ANCHOR = 8400
N_CLS    = 80
K_TOPK   = 300
IMG      = 640.0

# ---- ONNX tensor names -------------------------------------------------
BOX_CONV = {  # cv2.x.2 -> 4ch ltrb (post-bias float)
    "s0": "/model.23/one2one_cv2.0/one2one_cv2.0.2/Conv_output_0",
    "s1": "/model.23/one2one_cv2.1/one2one_cv2.1.2/Conv_output_0",
    "s2": "/model.23/one2one_cv2.2/one2one_cv2.2.2/Conv_output_0",
}
CLS_CONV = {  # cv3.x.2 -> 80ch logits (post-bias float)
    "s0": "/model.23/one2one_cv3.0/one2one_cv3.0.2/Conv_output_0",
    "s1": "/model.23/one2one_cv3.1/one2one_cv3.1.2/Conv_output_0",
    "s2": "/model.23/one2one_cv3.2/one2one_cv3.2.2/Conv_output_0",
}
G_BOXES = "/model.23/Div_3_output_0"      # decoded boxes [1,8400,4] pre-gather
G_SCORE = "/model.23/ReduceMax_output_0"  # per-anchor score [1,8400]
G_TKVAL = "/model.23/TopK_output_0"       # [1,300]
G_TKIDX = "/model.23/TopK_output_1"       # [1,300]
G_LOGITS = "logits"                        # final [1,300,80]
G_PBOX   = "pred_boxes"                    # final [1,300,4]

OUTS = (list(BOX_CONV.values()) + list(CLS_CONV.values()) +
        [G_BOXES, G_SCORE, G_TKVAL, G_TKIDX, G_LOGITS, G_PBOX])


def build_session():
    m = onnx.load(MODEL_PATH)
    have = {o.name for o in m.graph.output}
    for nm in OUTS:
        if nm not in have:
            dt = TensorProto.INT64 if nm == G_TKIDX else TensorProto.FLOAT
            m.graph.output.append(
                helper.make_tensor_value_info(nm, dt, None))
    tmp = os.path.join(HERE, "_model_with_intermediate.onnx")
    onnx.save(m, tmp)
    so = ort.SessionOptions(); so.log_severity_level = 3
    return ort.InferenceSession(tmp, sess_options=so,
                                providers=["CPUExecutionProvider"])


def make_input(kind, seed=0):
    rng = np.random.RandomState(seed)
    img = np.zeros((1, 3, 640, 640), np.float32)
    if kind == "rand":
        img[0] = rng.uniform(0.0, 1.0, (3, 640, 640)).astype(np.float32)
    elif kind == "rand_low":
        img[0] = rng.uniform(0.1, 0.6, (3, 640, 640)).astype(np.float32)
    elif kind == "half":
        img[0] = 0.5
    elif kind == "gradient":
        gx = np.tile(np.linspace(0, 1, 640, np.float32), (640, 1))
        img[0, 0] = gx; img[0, 1] = gx.T; img[0, 2] = 0.5 * (gx + gx.T)
    else:
        raise ValueError(kind)
    return img


def quantize_i8(x, s):
    return np.clip(np.round(x / s), -128, 127).astype(np.int8)


def fp16(x):
    return np.float16(x).astype(np.float32)


# ---- anchor grid (counter-derived, must match ONNX exactly) -----------
def anchor_grid():
    ax = np.empty(N_ANCHOR, np.float32); ay = np.empty(N_ANCHOR, np.float32)
    st = np.empty(N_ANCHOR, np.float32)
    off = 0
    for _, g, stride, n in SCALES:
        cols = (np.arange(n) % g).astype(np.float32) + 0.5
        rows = (np.arange(n) // g).astype(np.float32) + 0.5
        ax[off:off+n] = cols; ay[off:off+n] = rows; st[off:off+n] = stride
        off += n
    return ax, ay, st


def sw_float_head(box_f, cls_f):
    """Float numpy model mirroring ORT ops. box_f[scale]->(4,H,W),
    cls_f[scale]->(80,H,W). Must reproduce ORT (topology proof)."""
    # flatten + concat, row-major per scale
    ltrb = np.concatenate([box_f[s[0]].reshape(4, -1) for s in SCALES], axis=1)   # (4,8400)
    logit = np.concatenate([cls_f[s[0]].reshape(N_CLS, -1) for s in SCALES], 1)   # (80,8400)
    ax, ay, st = anchor_grid()
    l, t, r, b = ltrb[0], ltrb[1], ltrb[2], ltrb[3]
    x1 = (ax - l) * st; y1 = (ay - t) * st
    x2 = (ax + r) * st; y2 = (ay + b) * st
    cx = (x1 + x2) / 2; cy = (y1 + y2) / 2; w = x2 - x1; h = y2 - y1
    boxes = np.stack([cx, cy, w, h], axis=1) / IMG          # (8400,4)
    score = logit.max(axis=0)                                # (8400,)
    idx = np.argsort(-score, kind="stable")[:K_TOPK]
    return boxes, score, idx, logit.T                        # logit.T (8400,80)


def sw_int8_head(box_i8, cls_i8, S_box, S_cls):
    """int8 model mirroring the DUT. Reduce-max on int8 within scale,
    dequant score to common fp16, topk, gather + per-scale dequant."""
    ax, ay, st = anchor_grid()
    # box: dequant ltrb -> fp16, affine in fp16
    ltrb = np.empty((4, N_ANCHOR), np.float32)
    off = 0
    for nm, g, stride, n in SCALES:
        ltrb[:, off:off+n] = fp16(box_i8[nm].reshape(4, -1).astype(np.float32) * S_box[nm])
        off += n
    l, t, r, b = ltrb
    x1 = fp16(fp16(ax - l) * st); y1 = fp16(fp16(ay - t) * st)
    x2 = fp16(fp16(ax + r) * st); y2 = fp16(fp16(ay + b) * st)
    cx = fp16(fp16(x1 + x2) / 2); cy = fp16(fp16(y1 + y2) / 2)
    w = fp16(x2 - x1); h = fp16(y2 - y1)
    inv640 = fp16(1.0 / IMG)
    boxes = np.stack([fp16(cx*inv640), fp16(cy*inv640),
                      fp16(w*inv640), fp16(h*inv640)], axis=1)   # (8400,4)
    # cls: store int8 logits per scale; reduce-max int8; score -> common fp16
    logit_i8 = np.empty((N_ANCHOR, N_CLS), np.int8)
    score_i8 = np.empty(N_ANCHOR, np.int8)
    scale_of = np.empty(N_ANCHOR, np.int32)
    off = 0
    for si, (nm, g, stride, n) in enumerate(SCALES):
        blk = cls_i8[nm].reshape(N_CLS, -1).T                    # (n,80)
        logit_i8[off:off+n] = blk
        score_i8[off:off+n] = blk.max(axis=1)
        scale_of[off:off+n] = si
        off += n
    # dequant score to common fp16 using each anchor's originating scale
    Scl = np.array([S_cls[s[0]] for s in SCALES], np.float32)
    score_f16 = fp16(score_i8.astype(np.float32) * Scl[scale_of])
    idx = topk_minheap(score_f16, K_TOPK)
    # gather
    gboxes = boxes[idx]                                          # (300,4)
    glogit = fp16(logit_i8[idx].astype(np.float32) * Scl[scale_of[idx]][:, None])
    return boxes, score_f16, idx, gboxes, glogit


def topk_minheap(vals_f16, k):
    """Mirror topk_fp16 IP: min-heap of k, replace-if-greater. Returns the
    selected indices in heap order (set semantics)."""
    v = vals_f16.astype(np.float16)
    heap_v = np.empty(k, np.float16); heap_i = np.empty(k, np.int64)
    sz = 0
    def sift_up(j):
        while j > 0:
            p = (j-1)//2
            if heap_v[j] < heap_v[p]:
                heap_v[[j,p]] = heap_v[[p,j]]; heap_i[[j,p]] = heap_i[[p,j]]; j = p
            else: break
    def sift_down(j):
        while True:
            l = 2*j+1; r = 2*j+2; sm = j
            if l < sz and heap_v[l] < heap_v[sm]: sm = l
            if r < sz and heap_v[r] < heap_v[sm]: sm = r
            if sm == j: break
            heap_v[[j,sm]] = heap_v[[sm,j]]; heap_i[[j,sm]] = heap_i[[sm,j]]; j = sm
    for i in range(len(v)):
        if sz < k:
            heap_v[sz] = v[i]; heap_i[sz] = i; sz += 1; sift_up(sz-1)
        elif v[i] > heap_v[0]:
            heap_v[0] = v[i]; heap_i[0] = i; sift_down(0)
    return heap_i.copy()


def cos(a, b):
    a = a.astype(np.float64).ravel(); b = b.astype(np.float64).ravel()
    d = np.linalg.norm(a)*np.linalg.norm(b)
    return float(a@b/d) if d else 1.0


def topk_set_match(idx_a, score, idx_b):
    """Tie-aware: selected score multisets must match exactly; index
    identity only required where the boundary score is unique."""
    sa = np.sort(score[idx_a])[::-1]; sb = np.sort(score[idx_b])[::-1]
    multiset_ok = np.array_equal(sa.astype(np.float16), sb.astype(np.float16))
    set_ok = set(idx_a.tolist()) == set(idx_b.tolist())
    return multiset_ok, set_ok


def main():
    sess = build_session()
    ipt = sess.get_inputs()[0].name
    samples = [("rand0","rand",0),("rand1","rand",1),("rand2","rand",2),
               ("low","rand_low",3),("half","half",0),("grad","gradient",0)]

    # gather ORT, compute per-tensor scales (max*1.1/127) across all samples
    cache = []
    mx_box = {s[0]:0.0 for s in SCALES}; mx_cls = {s[0]:0.0 for s in SCALES}
    for sname, kind, seed in samples:
        r = sess.run(OUTS, {ipt: make_input(kind, seed)})
        d = {nm: np.asarray(t) for nm, t in zip(OUTS, r)}
        cache.append((sname, d))
        for s in SCALES:
            mx_box[s[0]] = max(mx_box[s[0]], float(np.abs(d[BOX_CONV[s[0]]]).max()))
            mx_cls[s[0]] = max(mx_cls[s[0]], float(np.abs(d[CLS_CONV[s[0]]]).max()))
    S_box = {k: max(v*1.1/127, 1e-5) for k, v in mx_box.items()}
    S_cls = {k: max(v*1.1/127, 1e-5) for k, v in mx_cls.items()}
    print("Per-scale S_box:", {k: round(v,6) for k,v in S_box.items()})
    print("Per-scale S_cls:", {k: round(v,6) for k,v in S_cls.items()})

    manifest = {"N_ANCHOR": N_ANCHOR, "N_CLS": N_CLS, "K_TOPK": K_TOPK,
                "IMG": IMG, "scales": [list(s) for s in SCALES],
                "S_box": S_box, "S_cls": S_cls, "samples": []}

    print("\n=== SELF-CHECK: float numpy model vs ORT (topology proof) ===")
    worst_float = 1.0
    for sname, d in cache:
        box_f = {s[0]: d[BOX_CONV[s[0]]][0] for s in SCALES}
        cls_f = {s[0]: d[CLS_CONV[s[0]]][0] for s in SCALES}
        boxes, score, idx, logitT = sw_float_head(box_f, cls_f)
        cb = cos(boxes, d[G_BOXES][0]); cs = cos(score, d[G_SCORE][0])
        ms, ss = topk_set_match(idx, d[G_SCORE][0], d[G_TKIDX][0].astype(int))
        worst_float = min(worst_float, cb, cs)
        print(f"  {sname:6s} boxes_cos={cb:.6f} score_cos={cs:.6f} "
              f"topk_multiset={ms} topk_set={ss}")

    print("\n=== int8 reference faithfulness vs ORT ===")
    print("  stable stages (gate the int8 ref) | gather: same-idx datapath | "
          "selection: informational (tie-churn on object-free inputs)")
    worst = {"boxes":1.0,"score":1.0,"gather_didx":1.0}
    for sname, d in cache:
        box_i8 = {s[0]: quantize_i8(d[BOX_CONV[s[0]]][0], S_box[s[0]]) for s in SCALES}
        cls_i8 = {s[0]: quantize_i8(d[CLS_CONV[s[0]]][0], S_cls[s[0]]) for s in SCALES}
        boxes, score_f16, idx, gboxes, glogit = sw_int8_head(box_i8, cls_i8, S_box, S_cls)
        cb = cos(boxes, d[G_BOXES][0]); cs = cos(score_f16, d[G_SCORE][0])
        # gather datapath isolated from selection: gather ORT boxes by the
        # SAME indices the int8 model chose.
        cgd = cos(gboxes, d[G_BOXES][0][idx])
        common = len(set(idx.tolist()) & set(d[G_TKIDX][0].astype(int).tolist()))
        worst["boxes"]=min(worst["boxes"],cb); worst["score"]=min(worst["score"],cs)
        worst["gather_didx"]=min(worst["gather_didx"],cgd)
        print(f"  {sname:6s} boxes={cb:.5f} score={cs:.5f} gather_sameidx={cgd:.6f} "
              f"sel_common={common}/300")

        # dump int8 stim per scale (channel-major, as conv emits)
        for s in SCALES:
            box_i8[s[0]].astype(np.int8).tofile(os.path.join(STIM, f"{sname}_box_{s[0]}.bin"))
            cls_i8[s[0]].astype(np.int8).tofile(os.path.join(STIM, f"{sname}_cls_{s[0]}.bin"))
        # PRIMARY DV GATE: int8-reference goldens (DUT must match bit-exactly,
        # input-independent). fp16 stored as raw uint16 bit patterns.
        boxes.astype(np.float16).view(np.uint16).tofile(os.path.join(STIM, f"{sname}_ref_boxes.bin"))
        score_f16.astype(np.float16).view(np.uint16).tofile(os.path.join(STIM, f"{sname}_ref_score.bin"))
        idx.astype(np.uint16).tofile(os.path.join(STIM, f"{sname}_ref_idx.bin"))
        gboxes.astype(np.float16).view(np.uint16).tofile(os.path.join(STIM, f"{sname}_ref_gboxes.bin"))
        glogit.astype(np.float16).view(np.uint16).tofile(os.path.join(STIM, f"{sname}_ref_glogit.bin"))
        # SANITY goldens: ORT float, used only at selection-stable stages.
        d[G_BOXES][0].astype(np.float32).tofile(os.path.join(STIM, f"{sname}_ort_boxes.bin"))
        d[G_SCORE][0].astype(np.float32).tofile(os.path.join(STIM, f"{sname}_ort_score.bin"))
        manifest["samples"].append(sname)

    with open(os.path.join(STIM, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nWorst float-vs-ORT cos: {worst_float:.6f}  (≈1.0 ⇒ topology correct)")
    print("Worst int8-ref faithfulness:", {k: round(v,6) for k,v in worst.items()})
    print("NOTE: end-to-end pred_boxes-vs-ORT is NOT a gate on object-free inputs "
          "(tie-churn). DUT is gated bit-exactly against the int8 reference goldens "
          "(*_ref_*.bin); ORT goldens (*_ort_*.bin) sanity the stable stages.")


if __name__ == "__main__":
    main()
