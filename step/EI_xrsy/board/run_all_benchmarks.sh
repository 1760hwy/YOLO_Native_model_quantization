#!/bin/bash
# 全量实验复现：FPS基准 + USB摄像头 + RTSP模拟
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGES="/home/sunrise/EI_xrsy/valid_images"
M_A="/home/sunrise/step5/fire_detect_g1_wrong_norm.bin"
M_BC="/home/sunrise/step5/fire_detect_g2_no_fire.bin"
M_DE="/home/sunrise/step5/fire_detect_bayese_640x640_nv12.bin"

echo "============================================================"
echo " 1/3  FPS基准测试（对应论文表5和表9 FPS列）"
echo "============================================================"
# Config A: G1 wrong norm, fixed-index
python3 "$SCRIPT_DIR/fps_benchmark.py" --config A --model "$M_A" --images "$IMAGES" --n 200
# Config B: G2 no_fire, fixed-index
python3 "$SCRIPT_DIR/fps_benchmark.py" --config B --model "$M_BC" --images "$IMAGES" --n 200
# Config C: G2 no_fire, shape-driven (注: board_eval.py use_shape=True for C/E)
python3 "$SCRIPT_DIR/fps_benchmark.py" --config C --model "$M_BC" --images "$IMAGES" --n 200
# Config D: G4 bayese, fixed-index
python3 "$SCRIPT_DIR/fps_benchmark.py" --config D --model "$M_DE" --images "$IMAGES" --n 200
# Config E: G4 bayese, shape-driven
python3 "$SCRIPT_DIR/fps_benchmark.py" --config E --model "$M_DE" --images "$IMAGES" --n 200

echo ""
echo "============================================================"
echo " 2/3  USB摄像头测试（对应论文表8 USB行）"
echo "============================================================"
python3 "$SCRIPT_DIR/usb_cam_test.py"

echo ""
echo "============================================================"
echo " 3/3  RTSP模拟测试（对应论文表8 RTSP行）"
echo "============================================================"
python3 "$SCRIPT_DIR/rtsp_sim_test.py"

echo ""
echo "=== 全部测试完成 ==="
