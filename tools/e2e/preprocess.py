#!/usr/bin/env python3
# Copyright (c) 2026 vibeyolo
# SPDX-License-Identifier: Apache-2.0
#
# preprocess.py — turn a real image into the model's `pixel_values`
# [1,3,640,640] float32 tensor (letterbox to 640, RGB, /255, CHW).
#
# Usage: python3 tools/e2e/preprocess.py assets/bus.jpg [out.npy]

import os
import sys
import numpy as np
from PIL import Image

IMG_SIZE = 640


def letterbox(img: np.ndarray, new=IMG_SIZE, color=114):
    """Resize HWC uint8 image to (new,new) preserving aspect, pad with `color`."""
    h, w = img.shape[:2]
    r = min(new / h, new / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = np.asarray(
        Image.fromarray(img).resize((nw, nh), Image.BILINEAR), dtype=np.uint8
    )
    out = np.full((new, new, 3), color, dtype=np.uint8)
    top = (new - nh) // 2
    left = (new - nw) // 2
    out[top:top + nh, left:left + nw] = resized
    return out


def preprocess(path: str) -> np.ndarray:
    img = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)  # HWC RGB
    lb = letterbox(img)                       # 640x640x3 RGB uint8
    x = lb.astype(np.float32) / 255.0         # 0..1
    x = np.transpose(x, (2, 0, 1))            # CHW
    return x[None].astype(np.float32)         # 1,3,640,640


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "assets/bus.jpg"
    dst = sys.argv[2] if len(sys.argv) > 2 else "assets/pixel_values.npy"
    x = preprocess(src)
    np.save(dst, x)
    print(f"{src} -> {dst}  shape={x.shape} dtype={x.dtype} "
          f"min={x.min():.3f} max={x.max():.3f} mean={x.mean():.3f}")
