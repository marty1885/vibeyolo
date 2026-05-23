#!/usr/bin/env python3
# Copyright (c) 2026 vibeyolo
# SPDX-License-Identifier: Apache-2.0
#
# run_layer.py — build + run one conv_stage over its real dumped frame, then
# compare the RTL int8 output (x s_out_silu) to the ORT fp32 golden.
#
#   python3 tools/e2e/run_layer.py 11          # layer index
#   python3 tools/e2e/run_layer.py integ/generated/e2e/layer_011
#
# Exit 0 iff cos >= threshold (default 0.99).

import json
import os
import subprocess
import sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
COSIM = os.path.join(ROOT, "tools/e2e/cosim")
E2E = os.path.join(ROOT, "integ/generated/e2e")


def cos(a, b):
    a = a.reshape(-1).astype(np.float64); b = b.reshape(-1).astype(np.float64)
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / d) if d else 0.0


def run(layer_arg, thr=0.99, jobs=None, quiet=False):
    if os.path.isdir(str(layer_arg)):
        ldir = layer_arg
    else:
        ldir = os.path.join(E2E, f"layer_{int(layer_arg):03d}")
    m = json.load(open(os.path.join(ldir, "meta.json")))
    name = f"layer_{m['idx']:03d}"
    # dotN degenerates at N_LANE = K*K*P_CIN == 1 (valid_sr[-1:0]); conv_layer
    # degenerates at P_COUT == 1 ([P_COUT-2:0] lint taps). P_COUT/P_CIN only set
    # parallelism, not the math, so bump them out of the degenerate corner.
    p_cin = m["P_CIN"]
    if m["K"] * m["K"] * p_cin < 2:
        p_cin = min(m["CIN"], 2)
    p_cout = max(m["P_COUT"], min(m["COUT"], 2))
    env = dict(os.environ)
    env.update(dict(
        CIN=str(m["CIN"]), COUT=str(m["COUT"]), K=str(m["K"]),
        STRIDE=str(m["STRIDE"]), PAD=str(m["PAD"]),
        H_IN=str(m["H_IN"]), W_IN=str(m["W_IN"]),
        P_COUT=str(p_cout), P_CIN=str(p_cin), SILU=str(m["silu"]),
        S_OUT_PRE=repr(m["s_out_pre"]), S_OUT_SILU=repr(m["s_out_silu"]),
        STIM=ldir, LAYER=name))
    if jobs:
        env["VERILATOR_JOBS"] = str(jobs)
    log = subprocess.run(["make", "-C", COSIM, "test"], env=env,
                         capture_output=True, text=True)
    if log.returncode != 0:
        if not quiet:
            print(log.stdout[-2000:]); print(log.stderr[-2000:])
        print(f"{name}: BUILD/RUN FAIL ({m['node']})")
        return None

    out = np.array([int(l, 16) for l in open(os.path.join(ldir, "out_i8.hex"))],
                   dtype=np.int32)
    out = ((out + 128) % 256 - 128).astype(np.float32)        # i8
    COUT, H_OUT, W_OUT = m["COUT"], m["H_OUT"], m["W_OUT"]
    out = out.reshape(H_OUT, W_OUT, COUT).transpose(2, 0, 1)  # (COUT,H,W)
    rtl_f32 = out * m["s_out_silu"]
    ref = np.load(os.path.join(ldir, "ref_ort.npy"))
    c = cos(rtl_f32, ref)
    mae = float(np.mean(np.abs(rtl_f32 - ref)))
    tag = "PASS" if c >= thr else "FAIL"
    print(f"{name:11s} {tag}  cos={c:.6f} MAE={mae:.4f}  "
          f"{m['node']}  {m['CIN']}->{m['COUT']} k{m['K']}s{m['STRIDE']} "
          f"{m['H_IN']}x{m['W_IN']} {'silu' if m['silu'] else 'noact'}")
    return c


if __name__ == "__main__":
    thr = float(os.environ.get("THR", "0.99"))
    c = run(sys.argv[1], thr=thr, quiet=bool(int(os.environ.get("QUIET", "0"))))
    sys.exit(0 if (c is not None and c >= thr) else 1)
