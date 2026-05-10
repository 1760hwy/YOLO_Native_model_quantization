#!/bin/bash
# 板端评估：对 946 张 valid 图跑 A/B/C/D/E 五个配置，输出预测 txt
SCRIPT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/board_eval.py"
IMAGES="/home/sunrise/EI_xrsy/valid_images"
PREDS="/home/sunrise/EI_xrsy/preds"

M_A="/home/sunrise/step5/fire_detect_g1_wrong_norm.bin"
M_BC="/home/sunrise/step5/fire_detect_g2_no_fire.bin"
M_DE="/home/sunrise/step5/fire_detect_bayese_640x640_nv12.bin"

python3 "$SCRIPT" --config A --model "$M_A"  --images "$IMAGES" --out "$PREDS"
python3 "$SCRIPT" --config B --model "$M_BC" --images "$IMAGES" --out "$PREDS"
python3 "$SCRIPT" --config C --model "$M_BC" --images "$IMAGES" --out "$PREDS"
python3 "$SCRIPT" --config D --model "$M_DE" --images "$IMAGES" --out "$PREDS"
python3 "$SCRIPT" --config E --model "$M_DE" --images "$IMAGES" --out "$PREDS"

echo "=== 全部完成，preds 目录已生成 ==="
ls "$PREDS"
