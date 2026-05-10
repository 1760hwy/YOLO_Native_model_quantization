# 创新点一实验报告：三层校准失配的逐层量化诊断框架

## 一、实验背景与动机

YOLO11 火焰检测模型经 hb_mapper 量化部署至 RDK X5 BPU 后，出现"推理正常但检测框为零"的静默失败。该故障无报错、无异常，仅表现为目标漏检，难以从表象定位根因。

**问题的复杂性在于：** 量化部署涉及三个独立环节，任意一个环节的配置错误均可导致相同的外部表现（检测失败），传统调试方法难以区分各层失配的贡献：

| 层级 | 失配类型 | 潜在影响 |
|------|---------|---------|
| ① | 输入语义失配（双重归一化） | 量化器见到的输入分布与推理时完全不同 |
| ② | 样本分布失配（校准集无目标类别） | KL 散度优化从未覆盖正类 logit 区间 |
| ③ | 量化精度失配（cls 分支 INT8） | 255 个量化级别不足以表达极不对称的 logit 范围 |

本实验设计了**单变量控制的四组对照**，通过 cls 分支 logit 分布作为统一诊断指标，逐层量化各失配类型的实际影响。

---

## 二、诊断指标选择

**为什么选 cls logit 分布？**

YOLO11 cls 分支输出原始 logit（未经 sigmoid），阈值 0 是决策边界：
- `logit > 0` → sigmoid > 0.5 → 产生检测框
- `logit < 0` → sigmoid < 0.5 → 被过滤

因此，cls logit 分布的均值、最大值和正值占比，直接反映模型是否具备产生检测的能力，与任何后处理参数无关，是最接近量化根因的可观测量。

---

## 三、实验设计（单变量控制，四组对照）

### 变量控制矩阵

| 组 | 归一化 | 校准集 | cls 精度 | 测试失配类型 |
|----|--------|--------|---------|------------|
| **G1** | ❌ 双重 /255 | ✓ 含火焰 | INT16 | ① 输入语义失配 |
| **G2** | ✓ 正确 | ❌ 无火焰图片 | INT16 | ② 样本分布失配 |
| **G3** | ✓ 正确 | ✓ 含火焰 | ❌ INT8 | ③ 量化精度失配 |
| **G4** | ✓ 正确 | ✓ 含火焰 | ✓ INT16 | 完整修复（基线） |

**设计原则：** 每组仅改变一个变量，其余两个变量均置于正确配置，确保观测到的差异单独归因于该层失配。

### G1：双重归一化的构造

```python
# prepare_calib_g1_wrong_norm.py
img_f32 = img.astype(np.float32) / 255.0   # 校准数据已归一化到 [0,1]
img_f32.tofile(out_path)
# YAML 中同时设置 scale_value: 0.003921568627451 (= 1/255)
# 等效：量化器输入 = 原始像素 / 255 / 255 ≈ 原始像素 / 65025
```

推理时 hb_mapper 仍按 `/255` 处理 NV12 输入，而量化器在校准时已按 `/255²` 建立量化范围，导致推理输入在量化器的尺度映射下始终落在极窄区间。

### G2：无火焰校准集的构造

```python
# prepare_calib_g2_no_fire.py
# 仅选取标注文件为空（无目标）或无标注文件的图片
# 归一化正确（写入 [0,255] 原始值），YAML 做 /255
```

KL 散度校准时，从未见过火焰图像，cls 分支的正值 logit 区间从未参与优化，导致量化 clip 范围对正类无覆盖。

### G4：完整修复基线

直接复用 step5 已量化的 `fire_detect_bayese_640x640_nv12.bin`，无需重新量化。

---

## 四、实验结果

### 主结果表

| 组 | logit 均值 | logit 最大值 | 正值 logit 占比 | 检测框数（40张） |
|---|---|---|---|---|
| **G1: 双重归一化** | **-18.28** | **-8.14** | **0.00%** | **0** |
| G2: 无火焰校准集 | -15.18 | +1.21 | 0.04% | 647 |
| G3: INT8 cls | -15.07 | +1.26 | 0.06% | 917 |
| **G4: 完整修复（基线）** | -15.09 | +1.30 | 0.06% | **924** |

### 逐层分析

#### 层①：输入语义失配（G1 对比 G4）

```
G4: logit_mean = -15.09,  logit_max = +1.30,  pos_ratio = 0.06%,  det = 924
G1: logit_mean = -18.28,  logit_max = -8.14,  pos_ratio = 0.00%,  det =   0
差值 Δmean = -3.19,  Δmax = -9.44
```

**影响：灾难性**。双重归一化将 logit 均值压低约 3 个单位，最大值由 +1.30 降至 -8.14，整个分布落在决策边界 0 以下，正值占比归零，检测完全失败。

根本原因：量化器在校准时将输入范围理解为 `[0, 1/255] ≈ [0, 0.004]`，而推理时 NV12 原始像素经 `/255` 后范围为 `[0, 1]`，两者相差 255 倍，量化 scale 完全错配，所有激活值在推理阶段产生严重溢出或截断。

#### 层②：样本分布失配（G2 对比 G4）

```
G4: logit_mean = -15.09,  logit_max = +1.30,  det = 924
G2: logit_mean = -15.18,  logit_max = +1.21,  det = 647
差值 Δmean = -0.09,  Δmax = -0.09
```

**影响：中等**。logit 分布几乎与基线一致（均值差 0.09），但检测框数从 924 降至 647（减少 **30%**）。校准集不含火焰图像虽未导致完全失败，但造成正类敏感区间量化精度下降，部分低置信度火焰目标被漏检。

#### 层③：量化精度失配（G3 对比 G4）

```
G4: logit_mean = -15.09,  logit_max = +1.30,  det = 924
G3: logit_mean = -15.07,  logit_max = +1.26,  det = 917
差值 Δmean = +0.02,  Δmax = -0.04
```

**影响：极小**。INT8 量化（255 级）与 INT16 量化结果几乎等价，检测框数差异仅 7 框（< 1%）。原因在于本模型 cls logit 实际有效范围约 [-20, +1.3]，宽度约 21 个单位，即使 INT8 在此范围内的分辨率约为 0.082，对 sigmoid 决策边界附近的影响可忽略。

---

## 五、关键发现

> **三层失配的危害程度差异显著，输入语义失配（层①）是唯一导致检测完全失败的根因。**

1. **层①（双重归一化）是灾难性失配**：logit 均值额外下移 3.2 个单位，最大值低于决策边界 8.1 个单位，检测框从 924 降至 0，**失去 100% 的检测能力**。

2. **层②（无火焰校准集）造成中等程度退化**：logit 分布仅轻微偏移（Δmean = -0.09），但检测召回率下降 30%，说明 KL 校准对未见过的正类存在系统性偏差。

3. **层③（INT8 精度）在本场景下可忽略**：实际有效 logit 范围适中，INT8 分辨率足够，无需强制使用 INT16。

### 实践意义

上述分层结论为量化部署提供了明确的优先级排序：

```
调试优先级：① 归一化一致性 >> ② 校准集多样性 >> ③ 量化位宽
```

开发者在遭遇"推理成功但无检测"时，应首先核查输入预处理流程与 YAML 中 `scale_value` / `mean_value` 是否存在双重处理，而非优先怀疑量化精度。

---

## 六、统一诊断方法论

本实验确立了基于 **cls logit 分布** 的量化诊断方法：

```python
# 在 BPU 推理后，无需任何后处理，直接提取 cls 分支原始输出
cls_logits = extract_cls_outputs(model_output)   # shape: (N_anchors,)

diagnostics = {
    "logit_mean":         cls_logits.mean(),
    "logit_max":          cls_logits.max(),
    "logit_pct_positive": (cls_logits > 0).mean() * 100,
}
# logit_max < 0 → 量化存在严重根因，无需进入后处理调试
```

该指标不依赖置信度阈值、NMS 参数或图像内容，可作为量化质量的第一道快速验证关卡。

---

## 七、论文结论模板

> 本文提出了一种面向端侧量化部署的**三层校准失配诊断框架**，将量化故障分解为输入语义（归一化一致性）、样本覆盖（校准集分布）和量化精度三个独立可控的层次，并以 cls 分支 logit 分布作为统一诊断指标，在 RDK X5 BPU 上开展了严格的单变量控制实验。
>
> 实验结果表明，输入语义失配（双重归一化，层①）是三类失配中唯一导致检测完全失败的根因：其将 cls logit 最大值由基线的 +1.30 压低至 -8.14（检测框从 924 降至 **0**），而样本分布失配（层②）和量化精度失配（层③）造成的 logit 偏移均小于 0.1 个单位，对检测的实质影响分别为 30% 召回率下降和 < 1% 的微小差异。该框架将模糊的"部署失败"转化为可量化的逐层指标，为后续量化流程的规范化提供了实证依据。

---

## 八、附：实验复现命令

```bash
# Step 1 — PC：生成校准数据
python step/EI_I1/prepare_calib_g1_wrong_norm.py
python step/EI_I1/prepare_calib_g2_no_fire.py

# Step 2 — PC→Ubuntu：上传数据与配置
scp -r step/EI_I1/calibration_g1_wrong_norm_f32  user@ubuntu:/data/
scp -r step/EI_I1/calibration_g2_no_fire_f32     user@ubuntu:/data/
scp -r step/EI_I1/                               user@ubuntu:/data/EI_I1/

# Step 3 — Ubuntu Docker：三组量化
docker run -it --rm -v /data:/data \
  openexplorer/ai_toolchain_ubuntu_20_x5_gpu:v1.2.8-py310 \
  bash /data/EI_I1/run_quantization_EI_I1.sh

# Step 4 — 板端：运行推理实验
ssh sunrise@192.168.0.106
cd /home/sunrise/EI_I1
cp /home/sunrise/step5/fire_detect_bayese_640x640_nv12.bin ./fire_detect_g4_full_fix.bin
./run_board_experiment.sh /home/sunrise/step5/test_images

# Step 5 — PC：生成论文图表
scp -r sunrise@192.168.0.106:/home/sunrise/EI_I1/results  ./step/EI_I1/
python step/EI_I1/plot_diagnosis.py --results_dir ./step/EI_I1/results --out_dir ./step/EI_I1/figures
```

输出文件：
- `figures/fig1_logit_distribution.png` — 四组 cls logit 密度分布图（论文核心图）
- `figures/fig2_metrics_comparison.png` — 三项关键指标对比柱状图

---

*实验平台：Horizon RDK X5 / BPU bayes-e / hb_mapper 1.24.3 / YOLO11 火焰检测，输入 640×640 NV12 / 测试集 40 张（BoWFireDataset + 自采集）*
