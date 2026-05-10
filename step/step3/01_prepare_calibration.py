#!/usr/bin/env python3
"""
Step 3-1: 准备量化校准数据 (Windows 本地执行，产物传 Ubuntu)
目的: 将 BDF-18K 图片预处理为 FP32 NCHW 二进制格式，供 hb_mapper 使用

校准数据规格:
  - 数量: 100 张 (50~200 均可，越多越精确)
  - 格式: float32, NCHW, RGB, 像素值 0-255
          (hb_mapper 内部按 scale_value=1/255 归一化，此处不归一化)
  - 来源: BDF-18K 各子集递归扫描

产物:
  step/step3/calibration_f32/000000.f32 ~ 000099.f32
  → 传到 Ubuntu /data/calibration_f32/ 供 hb_mapper 使用

断点说明:
  BP-A: 递归扫描 BDF-18K 图片
  BP-B: 随机采样
  BP-C: Letterbox + BGR→RGB + float32 + NCHW
  BP-D: 保存 .f32 + 汇总

运行方式:
  C:\d\Anaconda3\envs\yolo\python.exe step/step3/01_prepare_calibration.py
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

# ── 路径 ─────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parents[2]
STEP3_DIR = Path(__file__).parent
BDF18K    = ROOT / "BDF-18K"

DEFAULT_SRC = str(BDF18K)
DEFAULT_DST = str(STEP3_DIR / "calibration_f32")

# ── 日志 ─────────────────────────────────────────────────────────────────────
LOG_FILE = STEP3_DIR / "step3_01.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(str(LOG_FILE), encoding="utf-8", mode="w"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("step3_01")

EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def scan_images(src_dir: Path) -> List[Path]:
    """递归扫描目录下所有图片"""
    files = []
    for p in src_dir.rglob("*"):
        if p.suffix.lower() in EXTS and p.is_file():
            files.append(p)
    return sorted(files)


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
    parser = argparse.ArgumentParser(description="准备 hb_mapper 校准数据")
    parser.add_argument("--src",  default=DEFAULT_SRC, help="图片源目录 (递归扫描)")
    parser.add_argument("--dst",  default=DEFAULT_DST, help="输出 .f32 目录")
    parser.add_argument("--size", type=int, default=640, help="模型输入尺寸")
    parser.add_argument("--num",  type=int, default=100, help="校准样本数量")
    parser.add_argument("--seed", type=int, default=42,  help="随机种子")
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("STEP 3-1: 准备量化校准数据 (Windows -> Ubuntu)")
    log.info(f"  src : {args.src}")
    log.info(f"  dst : {args.dst}")
    log.info(f"  size: {args.size}  num: {args.num}  seed: {args.seed}")
    log.info("=" * 60)

    src_dir = Path(args.src)
    dst_dir = Path(args.dst)

    if not src_dir.exists():
        log.error(f"[BP-A] 源目录不存在: {src_dir}")
        sys.exit(1)

    dst_dir.mkdir(parents=True, exist_ok=True)

    # BP-A: 扫描图片
    log.info(f"[BP-A] 递归扫描: {src_dir}")
    all_files = scan_images(src_dir)
    log.info(f"[BP-A] 找到 {len(all_files)} 张图片")

    if not all_files:
        log.error("[BP-A] 未找到图片，请检查路径")
        sys.exit(1)

    # BP-B: 随机采样
    random.seed(args.seed)
    n = min(args.num, len(all_files))
    selected = random.sample(all_files, n)
    log.info(f"[BP-B] 随机采样 {n} 张 (seed={args.seed})")

    # 统计来源分布
    src_counts: dict = {}
    for p in selected:
        # 取相对 BDF18K 的第一级子目录名
        try:
            rel = p.relative_to(src_dir)
            key = rel.parts[0] if len(rel.parts) > 1 else "root"
        except ValueError:
            key = "other"
        src_counts[key] = src_counts.get(key, 0) + 1
    log.info(f"[BP-B] 来源分布: {src_counts}")

    # BP-C/D: 处理并保存
    file_size_mb = args.size * args.size * 3 * 4 / 1024 / 1024
    ok, fail = 0, 0
    meta = []

    for img_path in selected:
        img = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            log.warning(f"  [SKIP] 无法读取: {img_path.name}")
            fail += 1
            continue

        # BP-C: Letterbox + BGR->RGB + float32 + NCHW
        img = letterbox(img, args.size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_nchw = img.astype(np.float32).transpose(2, 0, 1)  # (C,H,W)

        # BP-D: 保存 .f32
        out_name = f"{ok:06d}.f32"
        img_nchw.tofile(str(dst_dir / out_name))
        meta.append({"idx": ok, "src": str(img_path), "f32": out_name})
        ok += 1

        if ok % 20 == 0 or ok == n:
            log.info(f"  [{ok}/{n}] 已处理")

    # 保存元数据 JSON (放在 step3/ 下，不放进 calibration_f32/ 避免 hb_mapper 误读)
    meta_path = STEP3_DIR / "calibration_meta.json"
    meta_path.write_text(
        json.dumps({"total": ok, "size": args.size, "seed": args.seed,
                    "src_counts": src_counts, "files": meta},
                   ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    log.info("=" * 60)
    log.info(f"[BP-D] 完成: 成功={ok}, 失败={fail}")
    log.info(f"  单文件大小: {file_size_mb:.2f} MB")
    log.info(f"  总计: {file_size_mb * ok:.0f} MB")
    log.info(f"  输出目录: {dst_dir}")
    log.info(f"  元数据:   {meta_path}")
    log.info("=" * 60)
    log.info("STEP 3-1 完成")
    log.info("下一步: 将 calibration_f32/ 传到 Ubuntu /data/calibration_f32/")
    log.info("  scp -r step/step3/calibration_f32 user@ubuntu:/data/")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
