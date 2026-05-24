#!/usr/bin/env bash
# sweep_images.sh — run the full chained "chip" (real RTL: every conv on
# conv_stage Verilator + all 6 block IPs) against ORT across a whole image set,
# render each ORT-vs-chip side-by-side, and collect per-image metrics JSON.
#
# MUST be sequential: chain.py's RTL backends write per-layer/per-block input &
# scale hex into SHARED dirs that the cached Verilator binaries read at runtime,
# so two images in flight would clobber each other. The only parallelism is
# VERILATOR_JOBS inside a single build. (See e2e_real_image_harness memory.)
#
# Usage: tools/e2e/sweep_images.sh [out_dir]
set -u
cd "$(git rev-parse --show-toplevel)"
OUT="${1:-integ/generated/e2e/gallery}"
mkdir -p "$OUT"
export VERILATOR_JOBS="${VERILATOR_JOBS:-4}"

# conv_stage bakes S_OUT_PRE/S_OUT_SILU as Verilator -G params into the per-layer
# build dir. We run --scales fixed, so those values are now image-INDEPENDENT
# (the chip's calibrated meta.json scales) — but any binaries cached from an
# earlier --scales dynamic run hold stale per-image values. Clear the conv build
# cache once so image #1 rebuilds with the fixed scales; #2..N then reuse it
# correctly. (Blocks use manifest/runtime scales — no rebuild needed.)
rm -rf tools/e2e/cosim/build/layer_*

# image set: the original bus.jpg + the downloaded testset
IMAGES=(assets/bus.jpg)
for f in assets/testset/*.jpg; do IMAGES+=("$f"); done

echo "sweep: ${#IMAGES[@]} images, full RTL (--conv rtl --rtl-blocks all) -> $OUT"
i=0
for img in "${IMAGES[@]}"; do
  i=$((i+1))
  name="$(basename "${img%.*}")"
  npy="$OUT/$name.pixels.npy"
  png="$OUT/$name.png"
  js="$OUT/$name.json"
  log="$OUT/$name.log"
  echo "[$i/${#IMAGES[@]}] $img"
  python3 tools/e2e/preprocess.py "$img" "$npy" >/dev/null 2>&1 || { echo "  PREPROCESS FAIL"; continue; }
  t0=$(date +%s)
  python3 tools/e2e/chain.py --conv rtl --rtl-blocks all --scales fixed \
      --pixels "$npy" --image "$img" --render "$png" --metrics-json "$js" \
      >"$log" 2>&1
  rc=$?
  t1=$(date +%s)
  if [ $rc -ne 0 ]; then
    echo "  FAIL rc=$rc ($((t1-t0))s) — see $log"; tail -5 "$log" | sed 's/^/    /'
  else
    grep -E "^logits:|^pred_boxes:|detections \(conf" "$log" | sed 's/^/    /'
    echo "    done in $((t1-t0))s -> $png"
  fi
done
echo "sweep complete -> $OUT"
python3 tools/e2e/gallery.py "$OUT"
