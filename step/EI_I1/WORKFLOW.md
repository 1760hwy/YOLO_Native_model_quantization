# EI_I1 实验操作流程

## 实验设计：三层校准失配 × 独立变量控制

| 组 | 归一化 | 校准集 | cls精度 | 测试的失配类型 |
|----|--------|--------|---------|--------------|
| **G1** | ❌ 双重 /255 | ✓ 含火焰 | INT16 | ①输入语义失配 |
| **G2** | ✓ 正确 | ❌ 无火焰 | INT16 | ②样本分布失配 |
| **G3** | ✓ 正确 | ✓ 含火焰 | ❌ INT8 | ③量化精度失配 |
| **G4** | ✓ 正确 | ✓ 含火焰 | ✓ INT16 | 完整修复（step5已有）|

---

## Step 1 — PC：生成校准数据（Windows，yolo 环境）

```bash
# G1: 双重归一化校准数据
python step/EI_I1/prepare_calib_g1_wrong_norm.py
# → step/EI_I1/calibration_g1_wrong_norm_f32/

# G2: 无火焰校准数据
python step/EI_I1/prepare_calib_g2_no_fire.py
# → step/EI_I1/calibration_g2_no_fire_f32/

# G3 直接复用 step3 已有的 calibration_fire_f32，无需重新生成
```

---

## Step 2 — PC：上传到 Ubuntu 量化服务器

```bash
UBUNTU="user@ubuntu-server"

scp -r step/EI_I1/calibration_g1_wrong_norm_f32  $UBUNTU:/data/
scp -r step/EI_I1/calibration_g2_no_fire_f32     $UBUNTU:/data/
scp -r step/EI_I1/                               $UBUNTU:/data/EI_I1/
# calibration_fire_f32 应已在 /data/（step4 时已上传）
```

---

## Step 3 — Ubuntu Docker：三组量化

```bash
docker run -it --rm \
  -v /path/to/data:/data \
  openexplorer/ai_toolchain_ubuntu_20_x86:v1.2.8 \
  bash /data/EI_I1/run_quantization_EI_I1.sh

# 输出:
#   /data/output_g1_wrong_norm/fire_detect_g1_wrong_norm.bin
#   /data/output_g2_no_fire/fire_detect_g2_no_fire.bin
#   /data/output_g3_int8_cls/fire_detect_g3_int8_cls.bin
```

---

## Step 4 — PC：下载 .bin，上传到 RDK X5

```powershell
# 下载
scp user@ubuntu:/data/output_g1_wrong_norm/fire_detect_g1_wrong_norm.bin .\step\EI_I1\
scp user@ubuntu:/data/output_g2_no_fire/fire_detect_g2_no_fire.bin       .\step\EI_I1\
scp user@ubuntu:/data/output_g3_int8_cls/fire_detect_g3_int8_cls.bin     .\step\EI_I1\

# 上传到板端
scp .\step\EI_I1\fire_detect_g1_wrong_norm.bin  sunrise@192.168.0.106:/home/sunrise/EI_I1/
scp .\step\EI_I1\fire_detect_g2_no_fire.bin      sunrise@192.168.0.106:/home/sunrise/EI_I1/
scp .\step\EI_I1\fire_detect_g3_int8_cls.bin     sunrise@192.168.0.106:/home/sunrise/EI_I1/
scp .\step\EI_I1\extract_logits_bpu.py           sunrise@192.168.0.106:/home/sunrise/EI_I1/
scp .\step\EI_I1\run_board_experiment.sh         sunrise@192.168.0.106:/home/sunrise/EI_I1/
```

---

## Step 5 — 板端：一键运行实验

```bash
ssh sunrise@192.168.0.106
cd /home/sunrise/EI_I1

# 复制 G4（已有模型）
cp /home/sunrise/step5/fire_detect_bayese_640x640_nv12.bin ./fire_detect_g4_full_fix.bin

chmod +x run_board_experiment.sh
./run_board_experiment.sh /home/sunrise/step5/test_images
```

---

## Step 6 — PC：拉取结果，生成论文图表

```powershell
# 拉取
scp -r sunrise@192.168.0.106:/home/sunrise/EI_I1/results  .\step\EI_I1\

# 生成图表（需要 matplotlib）
pip install matplotlib
python step/EI_I1/plot_diagnosis.py --results_dir .\step\EI_I1\results --out_dir .\step\EI_I1\figures
```

生成文件：
- `figures/fig1_logit_distribution.png` — 四组 logits 分布直方图（论文 Figure）
- `figures/fig2_metrics_comparison.png` — 关键指标对比柱状图

---

## 预期实验数据

| 指标 | G1(双重归一化) | G2(无火焰) | G3(INT8) | G4(完整修复) |
|------|--------------|-----------|---------|------------|
| logit 均值 | ≈ −40 | ≈ −35 | ≈ −15 | ≈ −3 |
| logit 最大值 | < −10 | < −5 | ≈ −2 | > 0 |
| 正值 logit 占比 | ≈ 0% | ≈ 0% | ≈ 0% | > 0% |
| 检测框数（40张）| 0 | 0 | 0 | ~63 |
