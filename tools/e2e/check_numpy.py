#!/usr/bin/env python3
# Copyright (c) 2026 vibeyolo
# SPDX-License-Identifier: Apache-2.0
#
# check_numpy.py — numpy reconstruction of one dumped layer straight from the
# RTL stim hex (input_i8, weights, scale, bias, S_OUT), compared to ref_ort.npy.
# Validates the dump's fold/layout/S_OUT math independently of Verilator.
#
#   python3 tools/e2e/check_numpy.py integ/generated/e2e/layer_011

import json
import os
import sys
import numpy as np


def fp16_bits_to_f32(b):
    return np.array(b, dtype=np.uint16).view(np.float16).astype(np.float32)


def load_hex(path, dtype_bits):
    vals = [int(l, 16) for l in open(path) if l.strip()]
    return np.array(vals, dtype=np.int64)


def cos(a, b):
    a = a.reshape(-1).astype(np.float64); b = b.reshape(-1).astype(np.float64)
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / d) if d else 0.0


def main():
    d = sys.argv[1]
    m = json.load(open(os.path.join(d, "meta.json")))
    CIN, COUT, K, S, PAD = m["CIN"], m["COUT"], m["K"], m["STRIDE"], m["PAD"]
    H_IN, W_IN, H_OUT, W_OUT = m["H_IN"], m["W_IN"], m["H_OUT"], m["W_OUT"]

    x = np.load(os.path.join(d, "input_i8.npy")).astype(np.int32)  # (CIN,H,W)
    w_flat = load_hex(os.path.join(d, "weights.i8.hex"), 8)
    w_flat = ((w_flat + 128) % 256 - 128).astype(np.int32)         # i8
    # WROM[COUT][K*K*CIN], lane=(kh*K+kw)*CIN+kc
    W = w_flat.reshape(COUT, K, K, CIN).transpose(0, 3, 1, 2)      # (COUT,CIN,K,K)
    # ROMs hold scale/bias already divided by S_OUT_PRE (requant emits int8 in
    # S_OUT_PRE units; act_silu InScale=S_OUT_PRE reconstructs).
    scale = fp16_bits_to_f32(load_hex(os.path.join(d, "scale_fp16.hex"), 16))
    bias = fp16_bits_to_f32(load_hex(os.path.join(d, "bias_fp16.hex"), 16))
    s_pre = m["s_out_pre"]
    s_silu = m["s_out_silu"]

    xp = np.zeros((CIN, H_IN + 2 * PAD, W_IN + 2 * PAD), dtype=np.int32)
    xp[:, PAD:PAD + H_IN, PAD:PAD + W_IN] = x

    out = np.zeros((COUT, H_OUT, W_OUT), dtype=np.float32)
    for oy in range(H_OUT):
        for ox in range(W_OUT):
            win = xp[:, oy * S:oy * S + K, ox * S:ox * S + K]       # (CIN,K,K)
            acc = np.einsum("ockl,ckl->o", W, win)                  # (COUT,)
            # requant -> int8 pre-code in S_OUT_PRE units
            pre_code = np.clip(np.round(acc * scale + bias), -128, 127)
            pre = pre_code * s_pre
            y = pre / (1.0 + np.exp(-pre)) if m["silu"] else pre
            # act_silu / no-act output requant -> int8 in S_OUT_SILU units
            yq = np.clip(np.round(y / s_silu), -128, 127)
            out[:, oy, ox] = yq * s_silu

    ref = np.load(os.path.join(d, "ref_ort.npy"))
    c = cos(out, ref)
    mae = float(np.mean(np.abs(out - ref)))
    print(f"{os.path.basename(d)}: cos(np_from_hex, ORT)={c:.6f}  MAE={mae:.5f}  "
          f"ref|max|={np.abs(ref).max():.3f} s_silu={s_silu:.5f}")
    return c


if __name__ == "__main__":
    sys.exit(0 if main() > 0.99 else 1)
