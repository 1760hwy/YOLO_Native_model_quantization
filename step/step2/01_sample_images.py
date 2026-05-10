#!/usr/bin/env python3
"""
Step 2-1: 从 BDF-18K 数据集随机抽取 50 张图片
目的: 为 ONNX 验证准备测试集，覆盖不同子数据集

采样策略:
  - PublicDataset1_BoWFireDataset/test/images/ (最多 25 张 PNG，含标注)
  - PublicDataset5_DFS/valid/images/ (最多 15 张，含标注)
  - 自拍数据集推理/ (最多 10 张 DJI JPG，无标注)
  总计 50 张，随机 seed=42 保证可复现

断点说明:
  BP-A: 扫描各子数据集图片
  BP-B: 按比例采样
  BP-C: 复制图片 + 收集标注 (若有)
  BP-D: 保存采样清单 sample_list.json

运行方式:
  C:\d\Anaconda3\envs\yolo\python.exe step/step2/01_sample_images.py
"""

import sys
import json
import shutil
import random
import logging
from pathlib import Path
from typing import List

# ── 日志 ─────────────────────────────────────────────────────────────────────
LOG_FILE = Path(__file__).parent / "step2.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8", mode="a"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("step2_01")

# ── 路径常量 ─────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parents[2]
BDF18K     = ROOT / "BDF-18K"
STEP2_DIR  = Path(__file__).parent
SAMPLES_DIR = STEP2_DIR / "sample_images"
SAMPLE_JSON = STEP2_DIR / "sample_list.json"

RANDOM_SEED = 42
TOTAL_SAMPLES = 50

# 各来源采样配置
SOURCES = [
    {
        "name": "BoWFire_test",
        "img_dir": BDF18K / "公开数据集" / "PublicDataset1_BoWFireDataset" / "test" / "images",
        "lbl_dir": BDF18K / "公开数据集" / "PublicDataset1_BoWFireDataset" / "test" / "labels",
        "n_sample": 20,
        "has_label": True,
        "extensions": [".png", ".jpg", ".jpeg"],
    },
    {
        "name": "DFS_valid",
        "img_dir": BDF18K / "公开数据集" / "PublicDataset5_DFS" / "valid" / "images",
        "lbl_dir": BDF18K / "公开数据集" / "PublicDataset5_DFS" / "valid" / "labels",
        "n_sample": 20,
        "has_label": True,
        "extensions": [".png", ".jpg", ".jpeg"],
    },
    {
        "name": "DJI_drone",
        "img_dir": BDF18K / "自拍数据集推理",
        "lbl_dir": None,
        "n_sample": 10,
        "has_label": False,
        "extensions": [".jpg", ".jpeg", ".png"],
    },
]


def collect_images(src: dict) -> List[Path]:
    """收集某来源的所有图片路径"""
    img_dir = src["img_dir"]
    if not img_dir.exists():
        log.warning(f"    目录不存在: {img_dir}")
        return []
    files = []
    for ext in src["extensions"]:
        files.extend(img_dir.glob(f"*{ext}"))
        files.extend(img_dir.glob(f"*{ext.upper()}"))
    return sorted(set(files))


def main():
    log.info("=" * 60)
    log.info("STEP 2-1: 从 BDF-18K 抽取 50 张验证图片")
    log.info("=" * 60)

    random.seed(RANDOM_SEED)
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    all_samples = []

    for src in SOURCES:
        name = src["name"]
        log.info(f"[BP-A] 扫描 {name}: {src['img_dir']}")
        files = collect_images(src)
        log.info(f"       找到 {len(files)} 张图片")

        # BP-B: 采样
        n = min(src["n_sample"], len(files))
        if n == 0:
            log.warning(f"       {name} 没有可用图片，跳过")
            continue
        sampled = random.sample(files, n)
        log.info(f"[BP-B] 从 {name} 采样 {n} 张")

        # BP-C: 复制到 sample_images/
        for img_path in sampled:
            dest_name = f"{name}_{img_path.name}"
            dest_img  = SAMPLES_DIR / dest_name
            shutil.copy2(img_path, dest_img)

            entry = {
                "source": name,
                "original_path": str(img_path),
                "sample_path": str(dest_img),
                "has_label": src["has_label"],
                "label_path": None,
            }

            # 收集对应标注文件
            if src["has_label"] and src["lbl_dir"]:
                lbl_path = src["lbl_dir"] / (img_path.stem + ".txt")
                if lbl_path.exists():
                    dest_lbl = SAMPLES_DIR / f"{name}_{lbl_path.name}"
                    shutil.copy2(lbl_path, dest_lbl)
                    entry["label_path"] = str(dest_lbl)
                else:
                    log.debug(f"       无标注文件: {lbl_path.name}")

            all_samples.append(entry)

    log.info(f"\n[BP-C] 共复制 {len(all_samples)} 张图片到: {SAMPLES_DIR}")

    # BP-D: 保存采样清单
    result = {
        "total": len(all_samples),
        "seed": RANDOM_SEED,
        "sources": {s["name"]: sum(1 for x in all_samples if x["source"] == s["name"])
                    for s in SOURCES},
        "samples": all_samples,
    }
    SAMPLE_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info(f"[BP-D] 采样清单已保存: {SAMPLE_JSON} ✓")
    log.info(f"       分布: {result['sources']}")
    log.info("=" * 60)
    log.info("STEP 2-1 完成 → 继续运行 02_validate_onnx.py")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
