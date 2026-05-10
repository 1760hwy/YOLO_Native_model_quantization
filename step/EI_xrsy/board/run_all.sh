#!/bin/bash
# 板端一键运行所有消融配置推理
# 使用前修改下面的模型路径
#
# 用法：
#   chmod +x run_all.sh
#   ./run_all.sh
#
# 运行完毕后，在 PC 端执行：
#   scp -r sunrise@192.168.0.106:/home/sunrise/EI_xrsy/results  step/EI_xrsy/

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGES_DIR="$(dirname "$SCRIPT_DIR")/test_images"
RESULTS_DIR="$(dirname "$SCRIPT_DIR")/results"
PYTHON=python3

# ── 修改为你的实际模型路径 ────────────────────────────────────
MODEL_WRONG_INT8="/home/sunrise/step5/fire_detect_g1_wrong_norm.bin"      # Config A
MODEL_WRONG_INT16="/home/sunrise/step5/fire_detect_g2_no_fire.bin"        # Config B/C
MODEL_CORRECT_INT16="/home/sunrise/step5/fire_detect_bayese_640x640_nv12.bin"  # Config D/E
# ──────────────────────────────────────────────────────────────

mkdir -p "$RESULTS_DIR"

echo "================================================================"
echo " 消融实验板端推理  —  RDK X5 BPU"
echo " 测试图片：$IMAGES_DIR"
echo " 输出目录：$RESULTS_DIR"
echo "================================================================"

run_config() {
    local cfg=$1
    local model=$2
    echo ""
    echo "── Config $cfg ──────────────────────────────────────────"
    $PYTHON "$SCRIPT_DIR/run_board_inference.py" \
        --config "$cfg" \
        --model  "$model" \
        --images "$IMAGES_DIR" \
        --out    "$RESULTS_DIR"
}

# Config A：固定索引 + INT8 Softmax + 错误校准（基线）
run_config A "$MODEL_WRONG_INT8"

# Config B：固定索引 + INT16 Softmax + 错误校准
run_config B "$MODEL_WRONG_INT16"

# Config C：形状驱动 + INT16 Softmax + 错误校准
run_config C "$MODEL_WRONG_INT16"

# Config D：固定索引 + INT16 Softmax + 正确校准
run_config D "$MODEL_CORRECT_INT16"

# Config E：形状驱动 + INT16 Softmax + 正确校准（完整方案）
run_config E "$MODEL_CORRECT_INT16"

echo ""
echo "================================================================"
echo " 全部完成！目录结构："
ls "$RESULTS_DIR"
echo ""
echo " 在 PC 端拉取结果："
echo "   scp -r sunrise@192.168.0.106:$(dirname "$SCRIPT_DIR")/results  step/EI_xrsy/"
echo "================================================================"
