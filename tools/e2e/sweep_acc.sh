#!/usr/bin/env bash
# Sweep the flash_attn wide-accumulator mantissa width and measure, on the
# real bus.jpg frame, the attn-block cosine vs ORT plus the chained final
# detection count. ACC_MANT=10 (≈fp16-precision accumulation) is the baseline;
# wider should track ORT more closely. Decided empirically per the brief.
set -u
cd "$(git rev-parse --show-toplevel)"
WIDTHS="${1:-10 13 16 21 26}"
for M in $WIDTHS; do
  echo "######## ACC_MANT=$M (ACC_EXP=8) ########"
  make -C integ/attn_model10/dv clean >/dev/null 2>&1
  make -C integ/attn_model22/dv clean >/dev/null 2>&1
  ACC_MANT=$M ACC_EXP=8 python3 tools/e2e/chain.py --rtl-blocks attn --render '' 2>&1 \
    | grep -E "block\[attn\]|^logits:|^pred_boxes:|detections \(conf|conf=" \
    | sed 's/^/  /'
  echo
done
