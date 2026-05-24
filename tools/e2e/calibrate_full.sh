#!/usr/bin/env bash
# calibrate_full.sh — long-run percentile calibration over the FULL corpus:
# COCO val2017 (~5000 imgs, in-distribution) + the QOI benchmark suite
# (~2800 imgs across textures/photo/screenshot/game/icon/pngimg — wide
# out-of-COCO variance). Run when the machine is idle; it's a single-threaded
# numpy chip-model pass (~1.4s/img → a few hours for the whole corpus).
#
# Produces calib_scales_full.json. To adopt it as the chip's scales, point
# chain.py at it (cp over calib_scales.json) and re-run the RTL sweep.
#
#   tools/e2e/calibrate_full.sh [--limit N] [--pct 99.9]
# e.g. a quick ~800-image pass:  tools/e2e/calibrate_full.sh --limit 800
set -u
cd "$(git rev-parse --show-toplevel)"
EXTRA="$*"
COCO="assets/calib/coco_val2017/val2017/*.jpg"
QOI="assets/calib/qoi_bench/**/*.png"
SMALL_COCO="assets/calib/coco/*.jpg"          # the original ~117 low-id COCO
SMALL_QOI="assets/calib/qoi/qoi_test_images/*.png"
echo "full corpus calibration -> integ/generated/e2e/calib_scales_full.json"
python3 tools/e2e/calibrate.py "$COCO" "$QOI" "$SMALL_COCO" "$SMALL_QOI" \
    --out integ/generated/e2e/calib_scales_full.json $EXTRA
echo
echo "to adopt:  cp integ/generated/e2e/calib_scales_full.json integ/generated/e2e/calib_scales.json"
echo "then:      tools/e2e/sweep_images.sh   (rebuilds RTL with the new baked scales)"
