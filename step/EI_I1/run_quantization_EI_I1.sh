#!/bin/bash
# EI_I1 — 在 Ubuntu Docker 内量化三个对照组
# 使用方法（宿主机执行）:
#   docker run -it --rm \
#     -v /path/to/data:/data \
#     openexplorer/ai_toolchain_ubuntu_20_x86:v1.2.8 \
#     bash /data/EI_I1/run_quantization_EI_I1.sh

set -e

MAPPER="hb_mapper makertbin --model-type onnx"
LOG_DIR="/data/EI_I1/quant_logs"
mkdir -p "$LOG_DIR"

echo "============================================================"
echo " EI_I1 量化实验: 三层校准失配对照"
echo " G1: 双重归一化  G2: 无火焰校准集  G3: INT8 cls"
echo "============================================================"
echo ""

# ── 前置检查 ──────────────────────────────────────────────────
for f in /data/fire_detect_gap.onnx \
          /data/calibration_g1_wrong_norm_f32 \
          /data/calibration_g2_no_fire_f32 \
          /data/calibration_fire_f32 \
          /data/EI_I1/quantization_g1_wrong_norm.yaml \
          /data/EI_I1/quantization_g2_no_fire.yaml \
          /data/EI_I1/quantization_g3_int8_cls.yaml; do
    [ -e "$f" ] || { echo "[ERROR] 缺少: $f"; exit 1; }
done
echo "[OK] 前置文件检查通过"
echo ""

# ── G1: 双重归一化 ─────────────────────────────────────────────
echo ">>> [G1] 双重归一化量化 ..."
$MAPPER --config /data/EI_I1/quantization_g1_wrong_norm.yaml \
    2>&1 | tee "$LOG_DIR/g1_wrong_norm.log"
echo "[G1] 完成 → /data/output_g1_wrong_norm/"
echo ""

# ── G2: 无火焰校准集 ───────────────────────────────────────────
echo ">>> [G2] 无火焰校准集量化 ..."
$MAPPER --config /data/EI_I1/quantization_g2_no_fire.yaml \
    2>&1 | tee "$LOG_DIR/g2_no_fire.log"
echo "[G2] 完成 → /data/output_g2_no_fire/"
echo ""

# ── G3: INT8 cls ───────────────────────────────────────────────
echo ">>> [G3] INT8 cls 量化 ..."
$MAPPER --config /data/EI_I1/quantization_g3_int8_cls.yaml \
    2>&1 | tee "$LOG_DIR/g3_int8_cls.log"
echo "[G3] 完成 → /data/output_g3_int8_cls/"
echo ""

echo "============================================================"
echo " 三组量化完成，生成的 .bin 文件:"
ls /data/output_g1_wrong_norm/*.bin 2>/dev/null && \
ls /data/output_g2_no_fire/*.bin    2>/dev/null && \
ls /data/output_g3_int8_cls/*.bin   2>/dev/null
echo ""
echo " 下一步: 将三个 .bin 传到 RDK X5，运行 run_board_experiment.sh"
echo "============================================================"
