#!/usr/bin/env bash
# Memory-capped coarse top-level route. The ulimit ensures a routing-grid blowup
# kills THIS process, not the machine. Cap ~21 GB (box has 30 GB; leaves OS room).
set -u
cd "$(git rev-parse --show-toplevel)"
CAP_KB=${CAP_KB:-22000000}            # virtual-mem cap (KB). Override: CAP_KB=...
echo "coarse top route, capped at ~$((CAP_KB/1000000)) GB virtual..."
( ulimit -v $CAP_KB; openroad -exit pnr/top/route_coarse.tcl ) \
    2>&1 | tee pnr/top/route_coarse.log
rc=${PIPESTATUS[0]}
echo "exit rc=$rc"
[ $rc -ne 0 ] && echo "(non-zero: hit the memory cap or a router error — safe, machine intact)"
exit 0
