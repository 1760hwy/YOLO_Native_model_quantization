# Step 4 — Ubuntu 量化打包说明

## 目录结构

```
step4/
├── fire_detect.onnx          # 导出的 FP32 ONNX 模型 (7.19 MB)
├── calibration_f32/          # 校准数据 (100 张, float32 NCHW RGB 0-255)
│   ├── 000000.f32 ~ 000099.f32
│   └── calibration_meta.json
├── 02_modify_head_ubuntu.py  # Ubuntu 内修改 head.py (Docker 内执行)
├── 03_export_onnx_ubuntu.py  # Ubuntu 内重新导出 ONNX (可选，若已有 fire_detect.onnx 则跳过)
├── 04_quantization_config.yaml  # hb_mapper 量化配置
├── 05_run_quantization.sh    # 一键量化脚本
├── 06_run_on_rdkx5.py        # RDK X5 板端推理
└── README.md                 # 详细说明
```

## 传输到 Ubuntu

```bash
# 在 Windows 上，用 scp 传整个 step4 目录
scp -r step/step4 user@<ubuntu_ip>:/data/

# 或用 rsync
rsync -avz step/step4/ user@<ubuntu_ip>:/data/
```

传输后 Ubuntu /data/ 结构：
```
/data/
├── fire_detect.onnx
├── calibration_f32/
│   └── *.f32  (100个)
├── 04_quantization_config.yaml
├── 05_run_quantization.sh
└── ...
```

## Ubuntu 执行顺序

### 1. 启动 OpenExplorer Docker 容器

```bash
docker run -it --rm \
  -v /data:/data \
  openexplorer/ai_toolchain_ubuntu_22_x86:v1.2.8 \
  bash
```

### 2. 容器内执行量化

```bash
# 在容器内
cd /data
hb_mapper makertbin \
  --model-type onnx \
  --config 04_quantization_config.yaml
```

### 3. 产物

量化完成后在 `/data/output_fire_detect/` 生成：
- `fire_detect_bayese_640x640_nv12.bin`  ← 部署到 RDK X5

### 4. 部署到 RDK X5

```bash
scp /data/output_fire_detect/fire_detect_bayese_640x640_nv12.bin \
    root@<rdkx5_ip>:/root/

# 在 RDK X5 上
python3 06_run_on_rdkx5.py
```

## 关键配置说明

| 项目 | 值 |
|------|-----|
| 目标架构 | bayes-e (RDK X5) |
| 输入格式 | NV12 (摄像头原生) |
| 归一化 | scale=1/255，hb_mapper 内部完成 |
| Softmax节点 | `/model.11/m/m.0/attn/Softmax` → BPU int16 |
| 校准算法 | default |
| 优化等级 | O3 |
