# YOLO11 Fire Detection on RDK X5

**Deploy custom YOLO11 models on Horizon Robotics RDK X5 (Bayes-e BPU, 10 TOPS INT8) for real-time object detection.**

[**中文说明**](README_CN.md)

<p align="center">
  <img src="docs/pipeline.png" alt="Pipeline" width="800">
</p>

## Highlights

- Complete pipeline: Training → ONNX Export → Quantization → Edge Deployment
- Real-time fire detection via USB camera or RTSP stream
- Detailed pitfall documentation with solutions (especially calibration data issues)
- V7 inference script with auto-adaptive NCHW/NHWC output handling
- Compatible with any YOLO11 single-class or multi-class custom model

## Architecture

```
Windows/Linux (Training)          Docker (Quantization)         RDK X5 (Inference)
┌─────────────────────┐     ┌──────────────────────┐     ┌─────────────────────┐
│  1. Train YOLO11    │     │  4. hb_mapper checker │     │  7. Load .bin model │
│  2. Export ONNX     │────▶│  5. hb_mapper makertbin────▶│  8. NV12 preprocess │
│  3. Prepare calib   │     │  6. Verify metrics    │     │  9. BPU inference   │
│     data            │     │                       │     │ 10. DFL + NMS       │
└─────────────────────┘     └──────────────────────┘     └─────────────────────┘
```

## Quick Start

### Prerequisites

| Component | Requirement |
|-----------|------------|
| Dev Board | RDK X5 (Bayes-e BPU) |
| Training  | NVIDIA GPU + Python 3.8+ |
| Docker    | Horizon OE Docker image |
| Ultralytics | ≥ 8.1.0 |

### Step 1: Train Your Model

```bash
python scripts/train.py
```

> **Important**: Train for at least 100 epochs. Models trained for only 10 epochs (mAP50≈0.54) will produce zero detections after quantization.

### Step 2: Export ONNX (Critical)

```bash
python scripts/export_for_rdkx5.py
```

This exports 6 raw outputs in **NCHW format** (no DFL, no sigmoid):

| Index | Name | Shape | Description |
|-------|------|-------|-------------|
| 0 | bbox_P3 | (1, 64, 80, 80) | Bbox regression, stride 8 |
| 1 | cls_P3 | (1, nc, 80, 80) | Classification logits, stride 8 |
| 2 | bbox_P4 | (1, 64, 40, 40) | Bbox regression, stride 16 |
| 3 | cls_P4 | (1, nc, 40, 40) | Classification logits, stride 16 |
| 4 | bbox_P5 | (1, 64, 20, 20) | Bbox regression, stride 32 |
| 5 | cls_P5 | (1, nc, 20, 20) | Classification logits, stride 32 |

> **Why not use `model.export()`?** The default Ultralytics export includes DFL Softmax inside the graph, which causes the BPU to split the model into 10+ subgraphs, dropping FPS from 100+ to single digits.

### Step 3: Prepare Calibration Data (Most Common Pitfall!)

```bash
python scripts/prepare_calibration.py ./dataset/images/train ./calibration_f32 640 100
```

**⚠️ CRITICAL**: Calibration data must be in **0~255 range**, NOT normalized to 0~1!

The quantization config includes `scale_value: 0.003921568627451` (= 1/255), which means the BPU will handle normalization automatically. If you also normalize in the calibration script, you get double normalization (pixel / 255 / 255 = pixel / 65025), causing all cls outputs to collapse to strong negatives.

```python
# ❌ WRONG - double normalization
rgb_f32 = rgb.astype(np.float32) / 255.0  # DON'T DO THIS

# ✅ CORRECT - keep 0~255 range
rgb_f32 = rgb.astype(np.float32)           # BPU's scale_value handles /255
```

Verify your calibration data:
```bash
python scripts/verify_calibration.py ./calibration_f32
```

### Step 4: Quantize in Docker

```bash
# Enter Docker container
docker run -it --rm -v $(pwd):/fire_quant openexplorer/ai_toolchain:latest /bin/bash

# Inside Docker
cd /fire_quant

# Check model compatibility
hb_mapper checker --model-type onnx --march bayes-e --model fire_detect.onnx

# Quantize
hb_mapper makertbin --model-type onnx --config fire_detect_config.yaml

# Output: model_output/fire_detect.bin
```

### Step 5: Deploy on RDK X5

```bash
# Copy model to board
scp model_output/fire_detect.bin sunrise@<BOARD_IP>:/home/sunrise/

# Copy inference script
scp scripts/rtsp_fire_v7.py sunrise@<BOARD_IP>:/home/sunrise/

# Run on board
ssh sunrise@<BOARD_IP>

# USB camera
python3 rtsp_fire_v7.py --source 0 --conf 0.3

# RTSP camera
python3 rtsp_fire_v7.py --source "rtsp://user:pass@192.168.1.100:554/stream1" --conf 0.3
```

## Troubleshooting Guide

### Problem: Zero detections (cls outputs all strong negatives)

**Symptoms**: All cls output max values are around -10 to -21, sigmoid ≈ 0.

**Diagnosis**:
```python
# Run on board
from hobot_dnn import pyeasy_dnn as dnn
import numpy as np

models = dnn.load('fire_detect.bin')
# ... (preprocess and forward) ...
for i, out in enumerate(outputs):
    buf = np.array(out.buffer)
    print(f"Output[{i}]: shape={buf.shape}, min={buf.min():.4f}, max={buf.max():.4f}")
```

**Root Causes** (in order of likelihood):

| # | Cause | Fix |
|---|-------|-----|
| 1 | **Double normalization** in calibration data | Keep 0~255 range, don't /255 |
| 2 | **Insufficient training** (< 50 epochs) | Train for 100+ epochs |
| 3 | **BGR instead of RGB** in calibration data | Use `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)` |
| 4 | **NHWC export** instead of NCHW | Remove `.permute()` from export |
| 5 | **Wrong Softmax node** in config | Verify node name with ONNX inspection |

### Problem: Very low FPS (< 10)

**Cause**: Attention Softmax running on CPU, splitting model into many subgraphs.

**Fix**: Add `node_info` to quantization config:
```yaml
node_info: {
  "/model.10/m/m.0/attn/Softmax": {
    'ON': 'BPU', 'InputType': 'int16', 'OutputType': 'int16'
  }
}
```

### Problem: Output format mismatch

**Symptoms**: Garbled or incorrect detections.

**Check**: Compare your model outputs with a known working model:
```python
for i, out in enumerate(model.outputs):
    print(f"Output[{i}]: shape={out.properties.shape}, layout={out.properties.layout}")

# Expected (NCHW):
# (1, 64, 80, 80) - bbox
# (1, nc, 80, 80) - cls
# NOT (1, 80, 80, 64) - this is NHWC!
```

### Problem: Shared Conv quantization failure

If using custom Detect heads with shared convolutions (e.g., Detect_LSDECD), a single INT8 scale cannot handle three different feature distributions (P3/P4/P5).

**Fix**: Use standard YOLO11 Detect head with `ModuleList` (independent Conv per level).

## File Structure

```
.
├── README.md                    # English documentation (this file)
├── README_CN.md                 # Chinese documentation
├── scripts/
│   ├── train.py                 # Training script
│   ├── export_for_rdkx5.py     # ONNX export (6 outputs, NCHW, no DFL)
│   ├── prepare_calibration.py  # Calibration data generator
│   ├── verify_calibration.py   # Calibration data validator
│   ├── docker_verify.py        # Float verification in Docker
│   ├── diagnose_model.py       # Board-side model diagnostics
│   └── rtsp_fire_v7.py         # V7 inference (USB + RTSP)
├── configs/
│   ├── fire_detect_config.yaml # Quantization config
│   └── forestfire.yaml         # Dataset config example
└── docs/
    └── pipeline.png            # Architecture diagram
```

## Key Lessons Learned

1. **Calibration data is the #1 source of silent failures.** Quantization won't error out even with wrong data — you'll only discover the problem at inference time.

2. **Keep output format consistent with rdk_model_zoo** — NCHW, 6 outputs, no DFL/sigmoid in graph.

3. **Train sufficiently** — 10 epochs may make you think it's a quantization or code bug, when it's actually model accuracy.

4. **Verify at every step**: training → export → calibration → quantization → transfer → inference. Any step failing silently breaks the final result.

5. **Always verify file integrity** with `md5sum` after every transfer between machines.

## References

- [D-Robotics RDK Model Zoo](https://github.com/D-Robotics/rdk_model_zoo)
- [Horizon OpenExplorer Documentation](https://developer.d-robotics.cc/)
- [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics)

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

Special thanks to the RDK X5 community and D-Robotics for the toolchain and documentation.

---

**If this project helps you, please give it a ⭐ Star!**
