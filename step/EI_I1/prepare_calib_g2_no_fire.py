#!/usr/bin/env python3
"""
EI_I1 — G2: 校准数据（无火焰样本 — 样本分布失配）
=================================================
失配类型②: 样本分布失配

随机采样不经火焰过滤，校准集中几乎不含正样本（fire logit > 0 的激活）。
KL 校准算法从未见过正值 logit 区间，将 INT8 的 255 个量化级全部分配给
密集负值区间 [-40, 0]，正值区间无量化级 → fire 激活量化后映射为大负数。

对照设计:
  G1: wrong_norm  + fire_cal   + INT16  → 仅测试归一化失配
  G2 (本脚本): correct_norm + no_fire_cal + INT16  → 仅测试样本分布失配
  G3: correct_norm + fire_cal  + INT8   → 仅测试量化精度失配
  G4 (step5): 完整修复

运行:
  python prepare_calib_g2_no_fire.py
  # 输出: calibration_g2_no_fire_f32/
"""

import sys, json, random, logging, argparse
import numpy as np
import cv2
from pathlib import Path

ROOT      = Path(__file__).resolve().parents[2]
BDF18K    = ROOT / "BDF-18K"

DEFAULT_SRC = str(BDF18K)
DEFAULT_DST = str(Path(__file__).parent / "calibration_g2_no_fire_f32")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(str(Path(__file__).parent / "g2_prepare.log"), encoding="utf-8", mode="w"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("g2_prepare")

EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def scan_no_fire_images(src_dir: Path):
    """返回没有 YOLO 标注（或标注为空）的图片 — 无火焰图片。"""
    no_fire = []
    for p in src_dir.rglob("*"):
        if p.suffix.lower() not in EXTS or not p.is_file():
            continue
        label = p.with_suffix(".txt")
        if not label.exists():
            parts = list(p.parts)
            lower = [pt.lower() for pt in parts]
            if "images" in lower:
                parts[lower.index("images")] = "labels"
                label = Path(*parts).with_suffix(".txt")
        # 无标注文件 或 标注文件为空 → 无火焰
        if not label.exists() or label.stat().st_size == 0:
            no_fire.append(p)
    return sorted(no_fire)


def letterbox(img, size=640, pad_val=114):
    h, w = img.shape[:2]
    scale = min(size / h, size / w)
    nh, nw = int(h * scale), int(w * scale)
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), pad_val, dtype=np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas[top:top+nh, left:left+nw] = img
    return canvas


def main():
    parser = argparse.ArgumentParser(description="G2 校准数据：无火焰样本（分布失配）")
    parser.add_argument("--src",  default=DEFAULT_SRC)
    parser.add_argument("--dst",  default=DEFAULT_DST)
    parser.add_argument("--size", type=int, default=640)
    parser.add_argument("--num",  type=int, default=100)
    parser.add_argument("--seed", type=int, default=99)
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("EI_I1 G2: 无火焰校准数据（失配类型②）")
    log.info(f"  归一化: 正确（写 [0,255] raw，YAML 做 /255）")
    log.info(f"  校准集: 仅无火焰图片 → cls 正值区间从未被激活")
    log.info(f"  src={args.src}  num={args.num}")
    log.info("=" * 60)

    dst_dir = Path(args.dst)
    dst_dir.mkdir(parents=True, exist_ok=True)

    no_fire_files = scan_no_fire_images(Path(args.src))
    log.info(f"找到 {len(no_fire_files)} 张无火焰图片")

    if len(no_fire_files) < args.num:
        log.warning(f"无火图片不足 {args.num} 张，使用全部 {len(no_fire_files)} 张")

    random.seed(args.seed)
    selected = random.sample(no_fire_files, min(args.num, len(no_fire_files)))
    log.info(f"采样 {len(selected)} 张无火图片")

    ok = 0
    for img_path in selected:
        img = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        img = letterbox(img, args.size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # ── 正确归一化：写 [0,255] raw，YAML 的 scale_value 完成 /255 ─
        img_f32 = img.astype(np.float32).transpose(2, 0, 1)
        # 不额外除以 255.0（与 G1 的区别）
        # ─────────────────────────────────────────────────────────────────

        img_f32.tofile(str(dst_dir / f"{ok:06d}.f32"))
        ok += 1

    log.info(f"完成: {ok} 张写入 {dst_dir}")
    log.info("预期现象: cls logits 最大值 < 0，正值区间量化级为 0，few/no detections")
    (dst_dir.parent / "g2_meta.json").write_text(
        json.dumps({"group": "G2", "normalization": "correct",
                    "fire_only": False, "no_fire_only": True,
                    "num": ok, "seed": args.seed}, indent=2),
        encoding="utf-8"
    )


if __name__ == "__main__":
    main()
