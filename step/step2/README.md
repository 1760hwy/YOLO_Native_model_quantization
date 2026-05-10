# Step 2 — ONNX 推理验证（50张图）

**执行环境**: Windows (本机)  
**Python**: `C:\d\Anaconda3\envs\yolo\python.exe`  
**前置条件**: Step 1 完成，`step1/best.onnx` 存在  
**日志**: `step2.log`

---

## 目标

从 BDF-18K 数据集随机抽取 50 张图片，用 ONNX Runtime 运行推理验证，  
生成可视化结果和验证报告，确认导出模型效果正常再进入量化流程。

---

## 产出文件

| 文件/目录 | 说明 |
|-----------|------|
| `sample_images/` | 50 张抽样图片 (含对应标注文件) |
| `sample_list.json` | 采样清单 (来源、路径、是否有标注) |
| `results/` | 检测可视化图 (红框标注火焰) |
| `validation_report.json` | 完整推理报告 (每张图检测数/耗时) |
| `step2.log` | 完整执行日志 |

---

## 快速运行

```bat
C:\d\Anaconda3\envs\yolo\python.exe step/step2/run_step2.py
```

---

## 分步运行

```bat
# 1. 从 BDF-18K 抽取 50 张样本图
C:\d\Anaconda3\envs\yolo\python.exe step/step2/01_sample_images.py

# 2. ONNX 推理 + 可视化
C:\d\Anaconda3\envs\yolo\python.exe step/step2/02_validate_onnx.py
```

---

## 采样策略

| 来源 | 图片数量 | 是否有标注 | 说明 |
|------|---------|---------|------|
| BoWFire_test | 20 张 | ✅ 有 | 公开火灾数据集测试集 |
| DFS_valid | 20 张 | ✅ 有 | 数字火灾分割数据集验证集 |
| DJI_drone | 10 张 | ❌ 无 | 自拍无人机航拍图 |
| **合计** | **50 张** | | |

随机种子 `seed=42`，结果可复现。

---

## 推理参数

| 参数 | 值 | 说明 |
|------|-----|------|
| 输入尺寸 | 640×640 | 与训练一致 |
| 置信度阈值 | 0.25 | |
| IOU 阈值 | 0.45 | NMS |
| 执行引擎 | CUDA (有GPU时) / CPU | OnnxRuntime |
| 类别 | fire | 单类检测 |

---

## 后处理流程

```
ONNX 6路输出 (NHWC)
    ↓
DFL 解码: [HW, 64] → anchor 到 xyxy 坐标
    ↓
Sigmoid 激活 + 置信度过滤
    ↓
NMS (IoU=0.45)
    ↓
还原到原图坐标 (去 letterbox padding)
    ↓
可视化绘制
```

---

## 验证通过标准

- ✅ 50 张图全部推理成功 (无崩溃)
- ✅ 有火焰的图像能检测到框 (检出率 > 50%)
- ✅ 无火焰图像无误报或误报率低 (< 10%)
- ✅ 单张推理时间 < 100 ms (CPU) / < 20 ms (GPU)

如验证效果不理想，检查:
1. `step1/best.onnx` 输出节点是否为 6 个 NHWC
2. `02_validate_onnx.py` 的 DFL 解码参数 `REG_MAX=16` 是否正确
3. 置信度阈值是否过高 (调低 `CONF_THRES`)

---

## 断点速查

| 断点 | 位置 | 含义 |
|------|------|------|
| BP-A | 02_validate | ONNX 会话加载 |
| BP-B | 02_validate | 单张图推理执行 |
| BP-C | 02_validate | DFL 解码 + NMS |
| BP-D | 02_validate | 可视化保存 |
| BP-E | 02_validate | 汇总报告生成 |
