#!/bin/bash
# ============================================================
# RDK X5 火焰检测一键运行脚本 v2 (NV12 输入)
# 使用方法:
#   chmod +x run_detect.sh
#   ./run_detect.sh
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL="${SCRIPT_DIR}/fire_detect_bayese_640x640_nv12.bin"
IMG_DIR="${SCRIPT_DIR}/test_images"
RESULT_DIR="${SCRIPT_DIR}/results"
LOG_FILE="${SCRIPT_DIR}/detect_bpu.log"

# Set DEBUG_FLAG="--debug" to enable diagnostic output for first image
DEBUG_FLAG=""
# DEBUG_FLAG="--debug"

echo "============================================================"
echo "RDK X5 BPU 火焰检测"
echo "  模型:     ${MODEL}"
echo "  图片目录: ${IMG_DIR}"
echo "  结果目录: ${RESULT_DIR}"
echo "  日志:     ${LOG_FILE}"
echo "============================================================"

# ── 检查文件 ────────────────────────────────────────────────────────────────
if [ ! -f "${MODEL}" ]; then
    echo "[ERROR] 找不到模型文件: ${MODEL}"
    exit 1
fi

if [ ! -d "${IMG_DIR}" ] || [ -z "$(ls -A "${IMG_DIR}" 2>/dev/null)" ]; then
    echo "[ERROR] 测试图片目录为空或不存在: ${IMG_DIR}"
    exit 1
fi

IMG_COUNT=$(ls "${IMG_DIR}"/*.{jpg,jpeg,png,bmp} 2>/dev/null | wc -l)
echo "[信息] 找到 ${IMG_COUNT} 张图片"

# ── 检查模型信息 ─────────────────────────────────────────────────────────────
echo ""
echo "[模型信息]"
if command -v hrt_model_exec &>/dev/null; then
    hrt_model_exec model_info --model_file "${MODEL}" 2>&1 | grep -E "Model Name|Architecture|Subgraph|Input|Output" || true
else
    echo "  hrt_model_exec 未找到，跳过模型信息检查"
fi

# ── 运行推理 ─────────────────────────────────────────────────────────────────
echo ""
echo "[推理] 开始运行检测..."
echo "  时间: $(date)"
echo ""

mkdir -p "${RESULT_DIR}"

python3 "${SCRIPT_DIR}/detect_bpu.py" \
    --model   "${MODEL}" \
    --img_dir "${IMG_DIR}" \
    --out_dir "${RESULT_DIR}" \
    --conf 0.25 \
    --iou  0.45 \
    ${DEBUG_FLAG} \
    2>&1 | tee "${LOG_FILE}"

EXIT_CODE=${PIPESTATUS[0]}

echo ""
if [ ${EXIT_CODE} -eq 0 ]; then
    echo "============================================================"
    echo "检测完成!"
    echo "  结果目录: ${RESULT_DIR}"
    echo "  可视化图片: $(ls "${RESULT_DIR}"/*.jpg 2>/dev/null | wc -l) 张"
    echo "  报告文件: ${RESULT_DIR}/detection_report.json"
    echo ""
    echo "下一步: 在 Windows 上执行以下命令提取结果:"
    echo "  scp -r sunrise@192.168.0.106:$(echo "${RESULT_DIR}" | sed 's|/home/sunrise||') C:\\e\\workspace\\experiment\\EI_yolo\\step\\step6\\"
    echo "============================================================"
else
    echo "[ERROR] 检测脚本运行失败 (退出码: ${EXIT_CODE})"
    echo "请查看日志: ${LOG_FILE}"
    exit ${EXIT_CODE}
fi
