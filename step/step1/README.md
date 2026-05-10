# Step 1 — ONNX 导出 + 算子兼容性分析

**执行环境**: Windows (本机)  
**Python**: `C:\d\Anaconda3\envs\yolo\python.exe`  
**日志**: `step1.log`

---

## 目标

将 `yolo11-HGNetV2-C3k2-DWR-LSDECD-SlideLoss/weights/best.pt` 导出为  
RDK X5 (bayes-e) 兼容的 FP32 ONNX 模型，并分析所有算子的兼容状态。

---

## 产出文件

| 文件 | 说明 |
|------|------|
| `best.onnx` | 导出的 FP32 ONNX 模型 |
| `operators.json` | ONNX 图中所有算子及节点列表 |
| `operator_compatibility.json` | 算子兼容性分析报告 |
| `step1.log` | 完整执行日志 |

---

## 快速运行（推荐）

```bat
C:\d\Anaconda3\envs\yolo\python.exe step/step1/run_step1.py
```

---

## 分步运行（调试）

```bat
# 1. 修改 head.py → 6路 NHWC 输出
C:\d\Anaconda3\envs\yolo\python.exe step/step1/01_modify_head.py

# 2. 导出 ONNX (opset=11, imgsz=640)
C:\d\Anaconda3\envs\yolo\python.exe step/step1/02_export_onnx.py

# 3. 提取所有算子 → operators.json
C:\d\Anaconda3\envs\yolo\python.exe step/step1/03_extract_operators.py

# 4. 算子兼容性分析 → operator_compatibility.json
C:\d\Anaconda3\envs\yolo\python.exe step/step1/04_check_compatibility.py
```

---

## 核心设计决策

### 为什么修改 head.py？

RDK X5 的 BPU 后处理接口要求检测头输出为 **6 个分离 tensor (NHWC 格式)**：
- `output[0,1,2]`: bbox DFL 特征 `[1, H, W, 64]`（三个 stride: 8/16/32）
- `output[3,4,5]`: 类别分数 `[1, H, W, nc]`

Ultralytics 默认导出为 1 个合并输出 `[1, nc+64*3, HW_all]`，
不符合 RDK X5 后处理要求，导致无检测框输出。

### 为什么 opset=11？

地平线 OE v1.2.8 (hb_mapper 1.24.3) 支持 opset 10/11，  
opset=11 是最优选择：覆盖所有所需算子且工具链支持最完善。

### 为什么 simplify=False？

onnx-simplifier 可能升级 IR version 到 hb_mapper 不支持的版本，  
导致转换失败。不简化可保证兼容性。

---

## 关键算子兼容性摘要

| 状态 | 说明 | 处理方式 |
|------|------|---------|
| ✅ BPU | 完全在 BPU 运行，INT8 量化 | 无需处理 |
| ⚠️ BPU(int16) | Softmax — 需配置 node_info | 见 `operator_compatibility.json` |
| 🔴 CPU | NMS/Shape 等 — CPU 执行 | 不影响推理正确性 |

> **重要**: 运行完 Step 1 后，请打开 `operator_compatibility.json`，  
> 查看 `critical_actions` 字段，获取 Softmax 节点名称，  
> 填入 Step 3 的 `04_quantization_config.yaml` 的 `node_info` 中。

---

## 断点速查

| 断点 | 位置 | 含义 |
|------|------|------|
| BP-A | 各脚本开头 | 文件路径验证 |
| BP-B | 01_modify_head | 备份 + 写入 head.py |
| BP-C | 01_modify_head | 找到 Detect.forward 范围 |
| BP-D | 01_modify_head | 写入验证 |
| BP-B | 02_export_onnx | ONNX 导出执行 |
| BP-D | 02_export_onnx | 输出节点数量验证 |
| BP-C | 03_extract_ops | Softmax 节点名称定位 |

---

## 常见问题

**Q: 导出时报 DCNv3 相关错误？**  
A: 该模型架构不含 DCNv3，若报错说明 sys.path 混乱。  
   确保使用 `yolo` conda 环境且在 `EI_yolo/` 根目录执行。

**Q: 输出数量不是 6 个？**  
A: `01_modify_head.py` 未成功执行。  
   检查 `step1.log` 中 BP-D 日志，确认 `best.onnx` 中含 `(*bboxes, *clses)`。

**Q: 如何恢复原始 head.py？**  
```bat
copy ultralytics\nn\modules\head.py.bak_rdkx5 ultralytics\nn\modules\head.py
```
