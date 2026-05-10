#!/usr/bin/env python3
"""
EI_I1 — G1: 校准数据（双重归一化 — 错误配置）
==============================================
失配类型①: 输入语义失配

本脚本在写入 .f32 文件前额外执行 img / 255.0，产生值域 [0,1] 的校准数据。
YAML 中同时配置 scale_value: 0.003921568（再除一次255），等效于对像素值
除以 255² ≈ 65025，激活值被压缩约 2.4 个数量级。

对照设计（三组校准数据各控制一个变量）:
  G1 (本脚本): wrong_norm  + fire_cal  + INT16  → 仅测试归一化失配
  G2          : correct_norm + no_fire_cal + INT16  → 仅测试样本分布失配
  G3          : correct_norm + fire_cal  + INT8   → 仅测试量化精度失配
  G4 (step5)  : correct_norm + fire_cal  + INT16  → 完整修复（已完成）

运行:
  python prepare_calib_g1_wrong_norm.py
  # 输出: calibration_g1_wrong_norm_f32/
"""

import sys, json, random, logging, argparse
import numpy as np
import cv2
from pathlib import Path

ROOT      = Path(__file__).resolve().parents[2]
STEP3_DIR = ROOT / "step" / "step3"
BDF18K    = ROOT / "BDF-18K"

DEFAULT_SRC = str(BDF18K)
DEFAULT_DST = str(Path(__file__).parent / "calibration_g1_wrong_norm_f32")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(str(Path(__file__).parent / "g1_prepare.log"), encoding="utf-8", mode="w"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("g1_prepare")

EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def scan_fire_images(src_dir: Path):
    """只返回有非空 YOLO 标注的图片（有火焰）。"""
    fire_imgs = []
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
        if label.exists() and label.stat().st_size > 0:
            fire_imgs.append(p)
    return sorted(fire_imgs)


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
    parser = argparse.ArgumentParser(description="G1 校准数据：双重归一化（错误配置）")
    parser.add_argument("--src",  default=DEFAULT_SRC)
    parser.add_argument("--dst",  default=DEFAULT_DST)
    parser.add_argument("--size", type=int, default=640)
    parser.add_argument("--num",  type=int, default=100)
    parser.add_argument("--seed", type=int, default=99)
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("EI_I1 G1: 双重归一化校准数据（失配类型①）")
    log.info(f"  写入值域 [0,1] float32  +  YAML scale_value=1/255  →  等效 /255²")
    log.info(f"  src={args.src}  dst={args.dst}  num={args.num}")
    log.info("=" * 60)

    dst_dir = Path(args.dst)
    dst_dir.mkdir(parents=True, exist_ok=True)

    fire_files = scan_fire_images(Path(args.src))
    log.info(f"找到 {len(fire_files)} 张有火图片")
    if not fire_files:
        log.error("未找到带标注图片"); sys.exit(1)

    random.seed(args.seed)
    selected = random.sample(fire_files, min(args.num, len(fire_files)))
    log.info(f"采样 {len(selected)} 张")

    ok = 0
    for img_path in selected:
        img = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        img = letterbox(img, args.size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_f32 = img.astype(np.float32).transpose(2, 0, 1)

        # ── 关键: 额外除以 255 ─────────────────────────────────
        # 正确做法是写 [0,255] raw，YAML 的 scale_value=1/255 完成归一化
        # 此处提前除以 255，加上 YAML 再除一次，等效 /255²
        img_f32 = img_f32 / 255.0   # 值域变为 [0, 1]
        # ─────────────────────────────────────────────────────────

        img_f32.tofile(str(dst_dir / f"{ok:06d}.f32"))
        ok += 1

    log.info(f"完成: {ok} 张写入 {dst_dir}")
    log.info("预期现象: cls logits 均值 < -30，sigmoid 后置信度 < 1e-6，0 检测框")
    (dst_dir.parent / "g1_meta.json").write_text(
        json.dumps({"group": "G1", "normalization": "double (/255 in code + /255 in YAML)",
                    "fire_only": True, "num": ok, "seed": args.seed}, indent=2),
        encoding="utf-8"
    )


if __name__ == "__main__":
    main()
