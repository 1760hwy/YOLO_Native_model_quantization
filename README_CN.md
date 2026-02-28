# YOLO11 火焰检测模型 RDK X5 部署教程

**将自定义训练的YOLO11模型部署到地平线RDK X5开发板（Bayes-e BPU, 10 TOPS INT8），实现实时目标检测。**

[**English**](README.md)

## 项目亮点

- 完整流程：训练 → ONNX导出 → 量化 → 边缘部署
- 支持USB摄像头和RTSP网络摄像头实时火焰检测
- **详细的踩坑记录和解决方案**（尤其是校准数据问题）
- V7推理脚本，自动适配NCHW/NHWC输出格式
- 兼容任何YOLO11单类/多类自定义模型

## 整体流程

```
Windows/Linux (训练)              Docker (量化)                  RDK X5 (推理)
┌─────────────────────┐     ┌──────────────────────┐     ┌─────────────────────┐
│  1. 训练YOLO11      │     │  4. hb_mapper 检查   │     │  7. 加载.bin模型    │
│  2. 导出ONNX        │────▶│  5. hb_mapper 量化  ─────▶│  8. NV12预处理      │
│  3. 准备校准数据    │     │  6. 验证量化指标     │     │  9. BPU推理         │
│                     │     │                      │     │ 10. DFL解码+NMS     │
└─────────────────────┘     └──────────────────────┘     └─────────────────────┘
```

## 快速开始

### 环境要求

| 项目 | 要求 |
|------|------|
| 开发板 | RDK X5（Bayes-e BPU） |
| 训练 | NVIDIA GPU + Python 3.8+ |
| Docker | 地平线OE Docker镜像 |
| Ultralytics | ≥ 8.1.0 |

### 第一步：训练模型

```bash
python scripts/train.py
```

> **重要**：至少训练100个epoch。仅训练10个epoch的模型（mAP50≈0.54）量化后将完全无法检测。

### 第二步：导出ONNX（关键步骤）

```bash
python scripts/export_for_rdkx5.py
```

导出6个原始输出，**NCHW格式**（不含DFL、不含sigmoid）：

| 索引 | 名称 | Shape | 说明 |
|------|------|-------|------|
| 0 | bbox_P3 | (1, 64, 80, 80) | 边框回归，stride 8 |
| 1 | cls_P3 | (1, nc, 80, 80) | 分类logit，stride 8 |
| 2 | bbox_P4 | (1, 64, 40, 40) | 边框回归，stride 16 |
| 3 | cls_P4 | (1, nc, 40, 40) | 分类logit，stride 16 |
| 4 | bbox_P5 | (1, 64, 20, 20) | 边框回归，stride 32 |
| 5 | cls_P5 | (1, nc, 20, 20) | 分类logit，stride 32 |

> **为什么不直接用 `model.export()`？** 默认导出会把DFL的Softmax包含在图中，导致BPU将模型拆分为10+子图，FPS从100+暴跌到个位数。

### 第三步：准备校准数据（最容易出错的环节！）

```bash
python scripts/prepare_calibration.py ./dataset/images/train ./calibration_f32 640 100
```

**⚠️ 最关键的注意事项**：校准数据必须是 **0~255范围**，**绝对不能归一化到0~1**！

量化配置中包含 `scale_value: 0.003921568627451`（= 1/255），BPU会自动做归一化。如果校准脚本中也做了 /255，就会导致**双重归一化**（pixel / 255 / 255），cls输出全部崩塌。

```python
# ❌ 错误 - 双重归一化
rgb_f32 = rgb.astype(np.float32) / 255.0  # 千万不要这样做！

# ✅ 正确 - 保持0~255
rgb_f32 = rgb.astype(np.float32)           # BPU的scale_value会自动/255
```

验证校准数据：
```bash
python scripts/verify_calibration.py ./calibration_f32
```

### 第四步：Docker中量化

```bash
# 进入Docker容器
docker run -it --rm -v $(pwd):/fire_quant openexplorer/ai_toolchain:latest /bin/bash

# 在Docker内执行
cd /fire_quant

# 检查模型兼容性
hb_mapper checker --model-type onnx --march bayes-e --model fire_detect.onnx

# 执行量化
hb_mapper makertbin --model-type onnx --config fire_detect_config.yaml

# 输出文件: model_output/fire_detect.bin
```

### 第五步：板端部署

```bash
# 传输模型到板子
scp model_output/fire_detect.bin sunrise@<板子IP>:/home/sunrise/

# 传输推理脚本
scp scripts/rtsp_fire_v7.py sunrise@<板子IP>:/home/sunrise/

# SSH登录板子运行
ssh sunrise@<板子IP>

# USB摄像头模式
python3 rtsp_fire_v7.py --source 0 --conf 0.3

# RTSP摄像头模式
python3 rtsp_fire_v7.py --source "rtsp://admin:password@192.168.1.100:554/stream1" --conf 0.3
```

## 踩坑指南

### 坑1：零检测（cls输出全为强负数）——最致命的问题

**症状**：所有cls输出的最大值在-10到-21之间，sigmoid后接近0，一个检测框都出不来。

**诊断方法**：
```python
# 在板子上运行
from hobot_dnn import pyeasy_dnn as dnn
import numpy as np

models = dnn.load('fire_detect.bin')
# ... 预处理并推理 ...
for i, out in enumerate(outputs):
    buf = np.array(out.buffer)
    print(f"Output[{i}]: shape={buf.shape}, min={buf.min():.4f}, max={buf.max():.4f}")
```

**根因排查（按可能性排序）**：

| # | 原因 | 解决方案 |
|---|------|----------|
| 1 | **校准数据双重归一化** | 保持0~255范围，不要/255 |
| 2 | **训练不足**（< 50 epoch） | 训练100+ epoch |
| 3 | **校准数据用了BGR**而非RGB | 加 `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)` |
| 4 | **导出时用了NHWC**而非NCHW | 去掉导出脚本中的 `.permute()` |
| 5 | **Softmax节点名写错** | 用ONNX工具检查实际节点名 |

### 坑2：FPS极低（< 10）

**原因**：Attention中的Softmax跑在CPU上，模型被拆成多个子图。

**解决**：量化配置中加 `node_info` 强制Softmax上BPU：
```yaml
node_info: {
  "/model.10/m/m.0/attn/Softmax": {
    'ON': 'BPU', 'InputType': 'int16', 'OutputType': 'int16'
  }
}
```

### 坑3：输出格式不匹配

**症状**：检测结果混乱或完全不出框。

**排查**：对比你的模型和官方模型的输出格式：
```python
for i, out in enumerate(model.outputs):
    print(f"Output[{i}]: shape={out.properties.shape}, layout={out.properties.layout}")

# 期望（NCHW格式）：
# (1, 64, 80, 80) - bbox
# (1, nc, 80, 80) - cls
# 不应该是 (1, 80, 80, 64) - 那是NHWC！
```

### 坑4：共享卷积量化失败

如果使用了自定义Detect头（如Detect_LSDECD）且卷积是共享的，单个INT8 scale无法覆盖P3/P4/P5三种不同的特征分布。

**解决**：使用标准YOLO11 Detect头（ModuleList，每级独立Conv）。

### 坑5：文件传输时MD5不一致

**解决**：每次传输后验证md5：
```bash
# Docker中
md5sum model_output/fire_detect.bin
# 板端
md5sum /home/sunrise/fire_detect.bin
# 两个值必须完全一致
```

## 文件结构

```
.
├── README.md                    # 英文文档
├── README_CN.md                 # 中文文档（本文件）
├── scripts/
│   ├── train.py                 # 训练脚本
│   ├── export_for_rdkx5.py     # ONNX导出（6输出，NCHW，无DFL）
│   ├── prepare_calibration.py  # 校准数据生成器
│   ├── verify_calibration.py   # 校准数据验证
│   ├── docker_verify.py        # Docker中float验证
│   ├── diagnose_model.py       # 板端模型诊断
│   └── rtsp_fire_v7.py         # V7推理脚本（USB + RTSP）
├── configs/
│   ├── fire_detect_config.yaml # 量化配置
│   └── forestfire.yaml         # 数据集配置示例
└── docs/
    └── pipeline.png            # 架构图
```

## 核心经验总结

1. **校准数据是最容易出错也最难排查的环节**。量化不报错不代表量化正确。
2. **保持和rdk_model_zoo一致的格式**（NCHW、6输出、无DFL/sigmoid）是最安全的做法。
3. **训练要充分**，10个epoch可能让你以为是量化或代码的问题，实际上只是模型精度不够。
4. **每一步都要验证**：训练→导出→校准→量化→传输→推理，任何一步出错都会导致最终无输出。
5. **文件传输后一定用md5sum验证**。

## 参考资料

- [D-Robotics RDK Model Zoo](https://github.com/D-Robotics/rdk_model_zoo)
- [地平线OpenExplorer文档](https://developer.d-robotics.cc/)
- [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics)

## 许可证

MIT License

---

**如果这个项目对你有帮助，请给个 ⭐ Star！**
