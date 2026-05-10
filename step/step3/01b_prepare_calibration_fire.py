#!/usr/bin/env python3
"""
Step 3-1b: 准备量化校准数据 —— 仅选取有火焰标注的图片
目的: 修复 cls 分支 INT8 量化失效问题

根因:
  01_prepare_calibration.py 随机采样 100 张，其中无火图片过多，
  导致 cls 激活范围仅覆盖负值区间，INT8 scale 截断所有正样本 logit，
  量化后 cls 输出全为约 -7（fire 置信度 ≈ 0.07%），无法检测。

解决:
  只选有 YOLO 标注文件（非空 .txt）的图片作为校准数据，
  保证 cls 分支在校准时被充分激活到正值区间。

产物: step/step3/calibration_fire_f32/   (替换原 calibration_f32/)
量化时 YAML 改用:
  cal_data_dir:  '/data/calibration_fire_f32'
  calibration_type: 'kl'   ← 已在 YAML 中修改

运行:
  C:\\d\\Anaconda3\\envs\\yolo\\python.exe step/step3/01b_prepare_calibration_fire.py
"""

import sys
import json
import random
import logging
import argparse
import numpy as np
import cv2
from pathlib import Path
from typing import List

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

ROOT      = Path(__file__).resolve().parents[2]
STEP3_DIR = Path(__file__).parent
BDF18K    = ROOT / "BDF-18K"

DEFAULT_SRC = str(BDF18K)
DEFAULT_DST = str(STEP3_DIR / "calibration_fire_f32")

LOG_FILE = STEP3_DIR / "step3_01b.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(str(LOG_FILE), encoding="utf-8", mode="w"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("step3_01b")

EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def scan_fire_images(src_dir: Path) -> List[Path]:
    """递归扫描，只返回有非空标注的图片（有火焰）。
    支持两种布局:
      1. 同目录: img.jpg + img.txt 在同一文件夹
      2. YOLO 分离: .../images/xxx.jpg → .../labels/xxx.txt
    """
    fire_imgs = []
    for p in src_dir.rglob("*"):
        if p.suffix.lower() not in EXTS or not p.is_file():
            continue
        # 优先同目录 .txt
        label = p.with_suffix(".txt")
        if not label.exists():
            # YOLO 分离布局: images/ → labels/
            parts = list(p.parts)
            lower = [pt.lower() for pt in parts]
            if "images" in lower:
                idx = lower.index("images")
                parts[idx] = "labels"
                label = Path(*parts).with_suffix(".txt")
        if label.exists() and label.stat().st_size > 0:
            fire_imgs.append(p)
    return sorted(fire_imgs)


def letterbox(img: np.ndarray, size: int = 640, pad_val: int = 114) -> np.ndarray:
    h, w = img.shape[:2]
    scale = min(size / h, size / w)
    nh, nw = int(h * scale), int(w * scale)
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), pad_val, dtype=np.uint8)
    top  = (size - nh) // 2
    left = (size - nw) // 2
    canvas[top:top + nh, left:left + nw] = img
    return canvas


def main():
    parser = argparse.ArgumentParser(description="准备含火焰的 hb_mapper 校准数据")
    parser.add_argument("--src",  default=DEFAULT_SRC)
    parser.add_argument("--dst",  default=DEFAULT_DST)
    parser.add_argument("--size", type=int, default=640)
    parser.add_argument("--num",  type=int, default=100)
    parser.add_argument("--seed", type=int, default=99)
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("STEP 3-1b: 准备含火焰校准数据 (修复 cls 量化失效)")
    log.info(f"  src : {args.src}")
    log.info(f"  dst : {args.dst}")
    log.info(f"  size={args.size}  num={args.num}  seed={args.seed}")
    log.info("=" * 60)

    src_dir = Path(args.src)
    dst_dir = Path(args.dst)

    if not src_dir.exists():
        log.error(f"源目录不存在: {src_dir}"); sys.exit(1)

    dst_dir.mkdir(parents=True, exist_ok=True)

    # 只扫描有标注（有火）的图片
    log.info(f"[BP-A] 扫描有火焰标注的图片: {src_dir}")
    fire_files = scan_fire_images(src_dir)
    log.info(f"[BP-A] 找到 {len(fire_files)} 张有火图片")

    if not fire_files:
        log.error("未找到带标注的图片，请确认 BDF-18K 目录结构正确")
        sys.exit(1)

    # 随机采样
    random.seed(args.seed)
    n = min(args.num, len(fire_files))
    selected = random.sample(fire_files, n)
    log.info(f"[BP-B] 随机采样 {n} 张 (seed={args.seed})")

    # 来源统计
    src_counts: dict = {}
    for p in selected:
        try:
            key = p.relative_to(src_dir).parts[0]
        except Exception:
            key = "other"
        src_counts[key] = src_counts.get(key, 0) + 1
    log.info(f"[BP-B] 来源分布: {src_counts}")

    # 预处理并保存 .f32
    ok, fail = 0, 0
    meta = []

    for img_path in selected:
        img = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            log.warning(f"  [SKIP] 无法读取: {img_path.name}")
            fail += 1
            continue

        img = letterbox(img, args.size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_nchw = img.astype(np.float32).transpose(2, 0, 1)

        out_name = f"{ok:06d}.f32"
        img_nchw.tofile(str(dst_dir / out_name))
        meta.append({"idx": ok, "src": str(img_path), "f32": out_name})
        ok += 1

        if ok % 20 == 0 or ok == n:
            log.info(f"  [{ok}/{n}] 已处理")

    meta_path = STEP3_DIR / "calibration_fire_meta.json"
    meta_path.write_text(
        json.dumps({"total": ok, "size": args.size, "seed": args.seed,
                    "fire_only": True, "src_counts": src_counts, "files": meta},
                   ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    log.info("=" * 60)
    log.info(f"[BP-D] 完成: 成功={ok}, 失败={fail}")
    log.info(f"  输出: {dst_dir}")
    log.info(f"  元数据: {meta_path}")
    log.info("=" * 60)
    log.info("下一步:")
    log.info("  1. scp -r step/step3/calibration_fire_f32 user@ubuntu:/data/")
    log.info("  2. scp step/step4/04_quantization_config.yaml user@ubuntu:/data/")
    log.info("  3. 在 Ubuntu Docker 内修改 YAML:")
    log.info("       cal_data_dir: '/data/calibration_fire_f32'")
    log.info("       calibration_type: 'kl'   ← 已修改")
    log.info("  4. hb_mapper makertbin --model-type onnx --config /data/04_quantization_config.yaml")


if __name__ == "__main__":
    main()
