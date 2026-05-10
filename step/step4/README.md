# Step 3 — Ubuntu 量化编译 (RDK X5 部署)

**执行环境**: Ubuntu 22.04 + RTX 5080 + Docker (OpenExplorer v1.2.8 GPU 容器)  
**前置条件**: Step 2 完成，`step1/best.onnx` 验证通过

---

## 工作流概览

```
Windows (Step 1/2 完成)
    │
    ▼  scp/rsync 传输文件到 Ubuntu
    │   - step1/best.onnx
    │   - step3/ 目录 (本目录所有脚本)
    │   - BDF-18K 的部分图片 (约 100 张用于校准)
    │
Ubuntu 22.04
    │
    ▼  在 Docker 容器外
    ├── 准备校准数据: 01_prepare_calibration.py
    │
    ▼  进入 OE Docker 容器
    ├── 修改 head.py: 02_modify_head_ubuntu.py
    ├── 导出 ONNX (可选): 03_export_onnx_ubuntu.py
    ├── 执行量化: 05_run_quantization.sh
    │
    ▼  传输 .bin 到 RDK X5
    └── 板端推理: 06_run_on_rdkx5.py
```

---

## 文件清单

| 文件 | 运行环境 | 说明 |
|------|---------|------|
| `01_prepare_calibration.py` | Ubuntu 主机 | 校准数据预处理 |
| `02_modify_head_ubuntu.py` | Docker 容器 | 修改 ultralytics head.py |
| `03_export_onnx_ubuntu.py` | Docker 容器 | 重新导出 ONNX (可选) |
| `04_quantization_config.yaml` | Docker 容器 | hb_mapper 量化配置 |
| `05_run_quantization.sh` | Docker 容器 | 执行量化 + 验证 |
| `06_run_on_rdkx5.py` | RDK X5 | 板端推理脚本 |

---

## 详细步骤

### 1. 文件传输 (Windows → Ubuntu)

```powershell
# Windows PowerShell
$UBUNTU = "user@192.168.x.x"
$REMOTE = "~/rdkx5_workspace"

# 传输模型
scp "C:\e\workspace\experiment\EI_yolo\step\step1\best.onnx" "${UBUNTU}:${REMOTE}/fire_detect.onnx"

# 传输脚本
scp -r "C:\e\workspace\experiment\EI_yolo\step\step3" "${UBUNTU}:${REMOTE}/"

# 传输 100 张校准图片 (从 BDF-18K 任意子集)
scp -r "C:\e\workspace\experiment\EI_yolo\step\step2\sample_images\*.jpg" \
    "${UBUNTU}:${REMOTE}/calibration_raw/"
scp -r "C:\e\workspace\experiment\EI_yolo\step\step2\sample_images\*.png" \
    "${UBUNTU}:${REMOTE}/calibration_raw/"
```

### 2. Ubuntu 主机: 准备校准数据

```bash
# 创建目录
mkdir -p ~/rdkx5_workspace/calibration_raw
mkdir -p ~/rdkx5_workspace/calibration_f32

# 如果还需要更多校准图，可从 DFS 数据集复制
# (先将 BDF-18K 传到 Ubuntu)

# 运行预处理脚本
python3 ~/rdkx5_workspace/step3/01_prepare_calibration.py \
    --src  ~/rdkx5_workspace/calibration_raw \
    --dst  ~/rdkx5_workspace/calibration_f32 \
    --size 640 \
    --num  100
```

### 3. 启动 Docker 容器

```bash
export OE_PKG=~/rdkx5_workspace/horizon_x5_open_explorer_v1.2.8-py310_20240926
export DATA_DIR=~/rdkx5_workspace

# GPU 版容器 (RTX 5080 加速量化)
docker run -it --rm \
    --gpus all \
    --shm-size=16g \
    -v "${OE_PKG}":/open_explorer \
    -v "${DATA_DIR}":/data \
    -v "${DATA_DIR}/fire_detect.onnx":/data/fire_detect.onnx \
    openexplorer/ai_toolchain_ubuntu_20_x5_gpu:v1.2.8-py310

# CPU 版容器 (如无 GPU 或 GPU 版不可用)
# docker run -it --rm \
#     --shm-size=16g \
#     -v "${OE_PKG}":/open_explorer \
#     -v "${DATA_DIR}":/data \
#     openexplorer/ai_toolchain_ubuntu_20_x5_cpu:v1.2.8-py310
```

### 4. Docker 容器内: 安装 ultralytics + 修改 head.py

```bash
# 安装 ultralytics (仅首次)
pip install ultralytics==8.3.28 -i https://pypi.tuna.tsinghua.edu.cn/simple

# 修改 head.py
python3 /data/step3/02_modify_head_ubuntu.py

# (可选) 重新导出 ONNX
# python3 /data/step3/03_export_onnx_ubuntu.py \
#     --model /data/best.pt \
#     --out   /data/fire_detect.onnx
```

### 5. Docker 容器内: 确认 Softmax 节点名

```bash
# 用 hb_mapper checker 找出实际的 Softmax 节点名
hb_mapper checker \
    --model-type onnx \
    --proto /data/fire_detect.onnx 2>&1 | grep -i softmax

# 将输出的节点名填入 04_quantization_config.yaml 的 node_info
# 注意: 不同训练配置可能产生不同节点名！
```

### 6. Docker 容器内: 编辑量化配置

```bash
# 复制配置文件到工作目录
cp /data/step3/04_quantization_config.yaml /data/fire_config.yaml

# 编辑关键参数:
# - onnx_model: 确认路径正确
# - node_info: 填入步骤 5 找到的 Softmax 节点名
nano /data/fire_config.yaml
```

### 7. Docker 容器内: 执行量化

```bash
chmod +x /data/step3/05_run_quantization.sh
/data/step3/05_run_quantization.sh
```

量化成功后，`.bin` 文件位于 `/data/output_fire_detect/` 目录。

### 8. 传输 .bin 到 RDK X5

```bash
# Ubuntu → RDK X5 (通过 SSH)
scp ~/rdkx5_workspace/output_fire_detect/fire_detect_bayese_640x640_nv12.bin \
    root@<RDKX5_IP>:~/
```

### 9. RDK X5 板端推理

```bash
# RDK X5 上
python3 /path/to/06_run_on_rdkx5.py \
    --model fire_detect_bayese_640x640_nv12.bin \
    --input test.jpg \
    --output ./results
```

---

## 关键配置说明

### Softmax node_info 配置

这是 YOLO11 部署到 RDK X5 最关键的配置！

**问题**: YOLO11 的 C2PSA 注意力机制包含 Softmax，默认 INT8 量化会把它推到 CPU，  
导致模型被切分成 3 个子图，FPS 从 90+ 暴跌到 7。

**解决**: 在 `04_quantization_config.yaml` 的 `node_info` 中指定 Softmax 节点使用 INT16：

```yaml
node_info:
  "/model.10/m/m.0/attn/Softmax":
    ON: BPU
    InputType: int16
    OutputType: int16
```

> 节点名称从 `step1/operators.json` 的 Softmax 条目获取，或用 `hb_mapper checker` 确认。

---

## 预期性能指标 (640×640, bayes-e)

| 指标 | 预期值 |
|------|-------|
| BPU 推理延迟 | ~10-15 ms |
| 端到端 FPS | ~40-60 FPS |
| BPU 子图数 | 1 (Softmax 优化后) |
| INT8 精度损失 | < 2% mAP |
| 模型大小 | ~3-5 MB (.bin) |

---

## 故障排查

| 症状 | 原因 | 解决 |
|------|------|------|
| 输出数量不是 6 | head.py 未修改 | 重新运行 02_modify_head_ubuntu.py |
| BPU 子图 > 1 | Softmax 落 CPU | 检查 node_info 节点名是否匹配 |
| 无检测框 | 后处理出错 | 检查 DFL reg_max=16，nc=1 |
| 量化精度损失 > 5% | 校准数据不够或不具代表性 | 增加校准图到 200 张，改用 kl 算法 |
| hb_mapper 内存不足 | 校准数据太多 | 减少到 50 张 |
