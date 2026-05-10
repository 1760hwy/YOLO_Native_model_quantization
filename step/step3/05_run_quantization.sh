#!/bin/bash
# ============================================================
# Step 3-5: 执行 hb_mapper 量化编译 (Ubuntu Docker 容器内)
# 运行环境: OpenExplorer v1.2.8 GPU 容器
# ============================================================
set -euo pipefail

LOG_DIR="/data/step3/logs"
LOG_FILE="${LOG_DIR}/step3_05_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "${LOG_DIR}"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "${LOG_FILE}"; }

log "============================================================"
log "STEP 3-5: hb_mapper 量化编译"
log "============================================================"

# ── 配置路径 ─────────────────────────────────────────────────
ONNX_FILE="/data/fire_detect.onnx"
CONFIG_FILE="/data/step3/04_quantization_config.yaml"
OUTPUT_DIR="/data/output_fire_detect"
MODEL_PREFIX="fire_detect_bayese_640x640_nv12"

# ── BP-A: 前置检查 ───────────────────────────────────────────
log "[BP-A] 前置检查 ..."
[ -f "${ONNX_FILE}" ] || { log "ERROR: ONNX 文件不存在: ${ONNX_FILE}"; exit 1; }
[ -f "${CONFIG_FILE}" ] || { log "ERROR: 配置文件不存在: ${CONFIG_FILE}"; exit 1; }
[ -d "/data/calibration_f32" ] || { log "ERROR: 校准数据目录不存在，请先运行 step3-1"; exit 1; }

CAL_COUNT=$(ls /data/calibration_f32/*.f32 2>/dev/null | wc -l)
[ "${CAL_COUNT}" -gt 0 ] || { log "ERROR: 校准数据目录为空"; exit 1; }
log "[BP-A] ONNX: ${ONNX_FILE} ✓"
log "[BP-A] 配置: ${CONFIG_FILE} ✓"
log "[BP-A] 校准图片: ${CAL_COUNT} 个 ✓"

# ── BP-B: 用 hb_mapper checker 预检算子兼容性 ────────────────
log ""
log "[BP-B] hb_mapper checker 算子预检 ..."
hb_mapper checker \
    --model-type onnx \
    --proto "${ONNX_FILE}" \
    2>&1 | tee -a "${LOG_FILE}"

log "[BP-B] 请检查上方输出中是否有 CPU 节点"
log "       如有 Softmax 在 CPU，确认 YAML node_info 配置正确"
log ""

# ── BP-C: 执行量化 ───────────────────────────────────────────
log "[BP-C] 开始 hb_mapper makertbin ..."
log "       预计耗时: 15~40 分钟 (校准 100 张)"
log ""

hb_mapper makertbin \
    --model-type onnx \
    --config "${CONFIG_FILE}" \
    2>&1 | tee -a "${LOG_FILE}"

log ""
log "[BP-C] hb_mapper 执行完成"

# ── BP-D: 验证输出文件 ───────────────────────────────────────
log "[BP-D] 检查输出文件 ..."
BIN_FILE="${OUTPUT_DIR}/${MODEL_PREFIX}.bin"

if [ ! -f "${BIN_FILE}" ]; then
    log "ERROR: 未找到 .bin 文件: ${BIN_FILE}"
    log "       请查看上方日志，查找 ERROR 关键字"
    exit 1
fi

BIN_SIZE=$(du -sh "${BIN_FILE}" | cut -f1)
log "[BP-D] .bin 模型: ${BIN_FILE} (${BIN_SIZE}) ✓"
log "[BP-D] 输出文件列表:"
ls -lh "${OUTPUT_DIR}/" | tee -a "${LOG_FILE}"

# ── BP-E: 模型信息验证 ───────────────────────────────────────
log ""
log "[BP-E] 验证模型信息 (hrt_model_exec) ..."
if command -v hrt_model_exec &>/dev/null; then
    hrt_model_exec model_info --model_file "${BIN_FILE}" 2>&1 | tee -a "${LOG_FILE}"
    log "[BP-E] 请确认:"
    log "       BPU Architecture: bayes-e"
    log "       BPU Subgraph Count: 1  (若>1说明 Softmax 未完全优化)"
    log "       Input: images [nv12]"
    log "       Output: 6 个输出"
else
    log "[BP-E] hrt_model_exec 不可用，跳过模型信息验证"
fi

log ""
log "============================================================"
log "✅ STEP 3-5 完成!"
log "   .bin 文件: ${BIN_FILE}"
log "   传输到 RDK X5 后使用 run_on_rdkx5.py 部署"
log "============================================================"
