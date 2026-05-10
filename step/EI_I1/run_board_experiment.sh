#!/bin/bash
# EI_I1 — 板端一键实验脚本
# 对四个模型（G1/G2/G3/G4）依次提取 logits + 统计检测结果
#
# 使用前准备:
#   1. 将三个新 .bin 传到板端同目录:
#      scp ubuntu:/data/output_g1_wrong_norm/fire_detect_g1_wrong_norm.bin ./
#      scp ubuntu:/data/output_g2_no_fire/fire_detect_g2_no_fire.bin       ./
#      scp ubuntu:/data/output_g3_int8_cls/fire_detect_g3_int8_cls.bin     ./
#   2. G4（完整修复）使用 step5 已有模型:
#      cp /home/sunrise/step5/fire_detect_bayese_640x640_nv12.bin ./fire_detect_g4_full_fix.bin
#   3. 准备测试图片: ./test_images/
#
# 运行:
#   chmod +x run_board_experiment.sh
#   ./run_board_experiment.sh

set -e

IMG_DIR="${1:-./test_images}"
CONF="${2:-0.10}"

MODELS=(
    "fire_detect_g1_wrong_norm.bin"
    "fire_detect_g2_no_fire.bin"
    "fire_detect_g3_int8_cls.bin"
    "fire_detect_g4_full_fix.bin"
)

echo "============================================================"
echo " EI_I1 板端实验: 三层校准失配诊断"
echo " 图片目录: $IMG_DIR   conf=$CONF"
echo "============================================================"
echo ""

# 检查图片目录
[ -d "$IMG_DIR" ] || { echo "[ERROR] 图片目录不存在: $IMG_DIR"; exit 1; }

missing=0
for m in "${MODELS[@]}"; do
    [ -f "$m" ] || { echo "[WARN] 模型不存在，跳过: $m"; missing=$((missing+1)); }
done
[ $missing -eq ${#MODELS[@]} ] && { echo "[ERROR] 所有模型均缺失"; exit 1; }

echo ""

for m in "${MODELS[@]}"; do
    [ -f "$m" ] || { echo "[SKIP] $m"; continue; }
    echo ">>> 提取 logits: $m"
    python3 extract_logits_bpu.py \
        --model "$m" \
        --img_dir "$IMG_DIR" \
        --out_dir ./results \
        --conf "$CONF"
    echo ""
done

echo "============================================================"
echo " 所有模型推理完成，汇总 logit 统计:"
echo "============================================================"
for m in "${MODELS[@]}"; do
    stem="${m%.bin}"
    rp="./results/${stem}/logit_report.json"
    if [ -f "$rp" ]; then
        echo ""
        echo "  [$stem]"
        python3 -c "
import json
d = json.load(open('$rp'))
print(f\"    logit_mean={d['logit_mean']:.4f}  logit_max={d['logit_max']:.4f}\")
print(f\"    positive%={d['logit_pct_positive']:.2f}%  total_det={d['total_det']}\")
"
    fi
done
echo ""
echo " logits .npy 文件位于 ./results/*/cls_logits_all.npy"
echo " 将 results/ 拷回 PC 后运行: python3 plot_diagnosis.py"
echo "============================================================"
