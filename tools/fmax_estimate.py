#!/usr/bin/env python3
# Copyright (c) 2026 vibeyolo
# SPDX-License-Identifier: Apache-2.0
#
# fmax_estimate.py — per-leaf Fmax against a real PDK liberty (default ASAP7).
#
# For each timing-relevant leaf IP: sv2v -> yosys synth -> dfflibmap+abc map to
# the PDK stdcells -> abc `stime` reports the longest register-to-register
# combinational delay. Fmax = 1/delay. This is a SYNTHESIS-level estimate
# (WireLoad="none": no routing/clock-tree/SI margin), so real post-P&R Fmax is
# lower — treat these as optimistic upper bounds, useful for *ranking* stages
# and finding the bottleneck. The bottleneck is what matters: it sets the chip
# clock, hence FPS = clk / II.
#
#   python3 tools/fmax_estimate.py                 # ASAP7 RVT/TT, all leaves
#   python3 tools/fmax_estimate.py --lib <path>    # any merged stdcell .lib

import os, re, sys, subprocess, argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HW   = os.path.join(ROOT, "hw/ip")
ASAP7 = "/home/marty/Documents/aif/pdk/asap7/lib/asap7sc7p5t_RVT_TT_merged.lib"

# Timing-relevant leaves. fp16_* need sv2v (return-in-function). mac8/dotN are
# the int8 datapath; the fp16 requant chain is the suspected critical path.
LEAVES = ["mac8", "fp16_fma", "i32_to_fp16", "fp16_to_i8_sat", "fp16_macw"]

ABC_SCRIPT = "strash\ndch -f\nmap -B 0.9\ntopo\nupsize\ndnsize\nstime\n"


def needs_sv2v(src):
    t = open(src).read()
    return "return" in t or "function" in t          # yosys 0.64 native chokes


def delay_ps(mod, lib):
    src = os.path.join(HW, mod, "rtl", f"{mod}.sv")
    if not os.path.exists(src):
        return None, "no rtl"
    if needs_sv2v(mod and src):
        v = f"/tmp/{mod}.v"
        r = subprocess.run(["sv2v", src], capture_output=True, text=True)
        if r.returncode != 0:
            return None, "sv2v fail"
        open(v, "w").write(r.stdout)
        read = f"read_verilog {v}"
    else:
        read = f"read_verilog -sv {src}"
    scr = f"/tmp/abc_{mod}.script"
    open(scr, "w").write(ABC_SCRIPT)
    ys = (f"{read}; synth -top {mod} -flatten; dfflibmap -liberty {lib}; "
          f"abc -liberty {lib} -script {scr}; stat -liberty {lib}")
    out = subprocess.run(["yosys", "-p", ys], capture_output=True, text=True).stdout
    d = re.findall(r"Delay\s*=\s*([\d.]+)\s*ps", out)
    a = re.findall(r"Chip area for module.*?:\s*([\d.]+)", out)
    return (float(d[-1]) if d else None,
            float(a[-1]) if a else None)


def main():
    ap = argparse.ArgumentParser(description="per-leaf Fmax on a PDK liberty")
    ap.add_argument("--lib", default=os.environ.get("ASAP7_LIB", ASAP7))
    ap.add_argument("--leaves", nargs="*", default=LEAVES)
    args = ap.parse_args()

    rows = []
    for m in args.leaves:
        d, a = delay_ps(m, args.lib)
        rows.append((m, d, a))
        ghz = f"{1e3/d:.3f}" if isinstance(d, float) else "—"
        ps  = f"{d:.1f}" if isinstance(d, float) else (a or "?")
        print(f"  {m:16s} delay {ps:>8} ps   fmax {ghz:>6} GHz")

    timed = [(m, d) for m, d, _ in rows if isinstance(d, float)]
    if timed:
        worst = max(timed, key=lambda x: x[1])
        print(f"\n  bottleneck: {worst[0]} @ {1e3/worst[1]:.3f} GHz "
              f"({worst[1]:.0f} ps)")
        print(f"  → chip Fmax (delay-mode, no wireload) ≈ {1e3/worst[1]:.2f} GHz")
    return 0


if __name__ == "__main__":
    sys.exit(main())
