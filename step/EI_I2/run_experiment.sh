#!/bin/bash
# Innovation II — One-click experiment runner
# Runs all 4 groups in sequence and prints the comparison table.
#
# Copy this entire EI_I2/ folder to the RDK X5 board, then:
#   chmod +x run_experiment.sh
#   ./run_experiment.sh

set -e

MODEL="${1:-fire_detect_bayese_640x640_nv12.bin}"
IMG_DIR="${2:-./test_images}"
CONF="${3:-0.25}"
IOU="${4:-0.45}"

echo "============================================================"
echo " Innovation II: Output Head Matching Experiment"
echo " Model   : $MODEL"
echo " Images  : $IMG_DIR"
echo " conf=$CONF  iou=$IOU"
echo "============================================================"
echo ""

if [ ! -f "$MODEL" ]; then
    echo "[ERROR] Model not found: $MODEL"
    echo "Usage: ./run_experiment.sh <model.bin> <img_dir> [conf] [iou]"
    exit 1
fi

if [ ! -d "$IMG_DIR" ]; then
    echo "[ERROR] Image directory not found: $IMG_DIR"
    exit 1
fi

# ── Group 1: Shape-driven, natural order ──────────────────────────────────────
echo ">>> [1/4] Shape-driven  |  natural output order"
python3 detect_shape_driven.py \
    --model "$MODEL" --img_dir "$IMG_DIR" \
    --out_dir ./results/shape_driven_natural \
    --conf "$CONF" --iou "$IOU"
echo ""

# ── Group 2: Shape-driven, simulated reorder ──────────────────────────────────
echo ">>> [2/4] Shape-driven  |  simulated reorder"
python3 detect_shape_driven.py \
    --model "$MODEL" --img_dir "$IMG_DIR" \
    --out_dir ./results/shape_driven_reordered \
    --conf "$CONF" --iou "$IOU" \
    --simulate-reorder
echo ""

# ── Group 3: Fixed-index, natural order ───────────────────────────────────────
echo ">>> [3/4] Fixed-index   |  natural output order"
python3 detect_fixed_index.py \
    --model "$MODEL" --img_dir "$IMG_DIR" \
    --out_dir ./results/fixed_index_natural \
    --conf "$CONF" --iou "$IOU"
echo ""

# ── Group 4: Fixed-index, simulated reorder (expected failures) ───────────────
echo ">>> [4/4] Fixed-index   |  simulated reorder  (expect DECODE FAIL)"
# Run without set -e for this group since failures are expected
set +e
python3 detect_fixed_index.py \
    --model "$MODEL" --img_dir "$IMG_DIR" \
    --out_dir ./results/fixed_index_reordered \
    --conf "$CONF" --iou "$IOU" \
    --simulate-reorder
set -e
echo ""

# ── Summary table ─────────────────────────────────────────────────────────────
echo "============================================================"
echo " Experiment complete. Generating comparison table ..."
echo "============================================================"
echo ""
python3 analyze_results.py --results_dir ./results

echo ""
echo "All results saved to: $(pwd)/results/"
