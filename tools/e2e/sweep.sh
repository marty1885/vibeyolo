#!/usr/bin/env bash
# Copyright (c) 2026 vibeyolo
# SPDX-License-Identifier: Apache-2.0
#
# sweep.sh — run every dumped conv_stage layer over its real-image frame and
# tally cos vs ORT. Static parallel cap (no orchestration), per-process
# VERILATOR_JOBS kept low so concurrent builds don't blow up RAM.
#
#   tools/e2e/sweep.sh [PAR] [VJOBS]

set -u
ROOT="$(git rev-parse --show-toplevel)"
E2E="$ROOT/integ/generated/e2e"
PAR="${1:-6}"
export VERILATOR_JOBS="${2:-2}"
export QUIET=1
REPORT="$E2E/E2E_REPORT.md"

layers=$(ls -d "$E2E"/layer_* 2>/dev/null | sort)
echo "sweeping $(echo "$layers" | wc -l) layers, PAR=$PAR VERILATOR_JOBS=$VERILATOR_JOBS"

printf '%s\n' $layers | xargs -P "$PAR" -I{} bash -c \
  'python3 "'"$ROOT"'/tools/e2e/run_layer.py" "{}" 2>/dev/null || echo "$(basename {}) ERROR"' \
  | sort > "$E2E/_sweep_raw.txt"

{
  echo "# Real-image per-layer conv_stage vs ORT (bus.jpg)"
  echo
  echo "Generated $(date -u +%Y-%m-%dT%H:%M:%SZ). Each layer: real ORT activation"
  echo "(symmetric int8) streamed through one \`conv_stage\`; RTL int8 output x"
  echo "S_OUT_SILU compared to the ORT fp32 activation."
  echo
  echo '```'
  cat "$E2E/_sweep_raw.txt"
  echo '```'
  echo
  pass=$(grep -c ' PASS ' "$E2E/_sweep_raw.txt")
  fail=$(grep -c ' FAIL ' "$E2E/_sweep_raw.txt")
  err=$(grep -c 'ERROR\|FAIL (' "$E2E/_sweep_raw.txt")
  tot=$(wc -l < "$E2E/_sweep_raw.txt")
  echo "**$pass PASS / $fail below-threshold / build-errors in $tot layers (cos>=0.99)**"
} > "$REPORT"
echo "----"; tail -5 "$REPORT"
echo "report: $REPORT"
