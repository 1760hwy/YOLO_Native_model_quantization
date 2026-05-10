#!/usr/bin/env python3
"""
论文展示图像随机抽取脚本
- 自拍数据集推理：随机抽 300 张（满足 min_dim>=640 的图）
- 公开数据集：随机抽 300 张（BoW 全部 + DFS large/middle 大图）
- 统一中心裁剪到 640×640，不做拉伸
- 输出到 step/EI_sel/selfshot/ 和 step/EI_sel/public/
"""

import random
import shutil
from pathlib import Path
from PIL import Image

ROOT     = Path(__file__).resolve().parents[2]
BDF      = ROOT / "BDF-18K"
SELF_DIR = BDF / "自拍数据集推理"
DFS_DIR  = BDF / "公开数据集" / "PublicDataset5_DFS"
BOW_DIR  = BDF / "公开数据集" / "PublicDataset1_BoWFireDataset"
OUT_SELF = Path(__file__).parent / "selfshot"
OUT_PUB  = Path(__file__).parent / "public"

TARGET_W = 640
TARGET_H = 640
SEED     = 2025
N_SELF   = 300
N_PUBLIC = 300


def center_crop(img: Image.Image) -> Image.Image:
    img = img.convert("RGB")
    iw, ih = img.size
    left = (iw - TARGET_W) // 2
    top  = (ih - TARGET_H) // 2
    return img.crop((left, top, left + TARGET_W, top + TARGET_H))


def qualifies(p: Path) -> bool:
    try:
        with Image.open(p) as im:
            w, h = im.size
        return w >= TARGET_W and h >= TARGET_H
    except Exception:
        return False


def save_images(paths: list, out_dir: Path, prefix: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    # 清空旧文件
    for f in out_dir.iterdir():
        if f.is_file():
            f.unlink()
    saved = 0
    for i, p in enumerate(paths, 1):
        try:
            with Image.open(p) as im:
                cell = center_crop(im)
            out_path = out_dir / f"{prefix}_{i:03d}.jpg"
            cell.save(str(out_path), quality=95)
            saved += 1
        except Exception as e:
            print(f"  [WARN] {p.name}: {e}")
    print(f"  已保存 {saved} 张 → {out_dir}")
    return saved


def main():
    random.seed(SEED)
    print("=" * 60)
    print(f" 随机抽图  目标分辨率: {TARGET_W}×{TARGET_H}（中心裁剪，不拉伸）")
    print("=" * 60)

    # ── 自拍：全部合格图 ───────────────────────────────────────
    print(f"\n[1/2] 自拍数据集推理 → 全部合格图 ...")
    self_pool = [p for p in sorted(SELF_DIR.glob("*.jpg")) if qualifies(p)]
    print(f"  满足 min_dim>={TARGET_W} 的图: {len(self_pool)} 张，全部输出")
    self_sel = self_pool
    save_images(self_sel, OUT_SELF, "selfshot")

    # ── 公开：BoW 全量 + DFS 大图，合并后随机 300 ─────────────
    print(f"\n[2/2] 公开数据集 → {N_PUBLIC} 张 ...")
    pub_pool = []

    # BoW（全部三个 split）
    for split in ("train", "valid", "test"):
        for p in sorted((BOW_DIR / split / "images").glob("*.png")):
            if qualifies(p):
                pub_pool.append(p)
    print(f"  BoW 可用: {len(pub_pool)} 张")

    # DFS root（large + middle 火焰类别）
    dfs_count = 0
    for p in sorted(DFS_DIR.glob("*.jpg")):
        if p.stem.startswith(("large_", "middle_")) and qualifies(p):
            pub_pool.append(p)
            dfs_count += 1
    print(f"  DFS large+middle 大图可用: {dfs_count} 张")
    print(f"  公开总池: {len(pub_pool)} 张")

    pub_sel = random.sample(pub_pool, min(N_PUBLIC, len(pub_pool)))
    save_images(pub_sel, OUT_PUB, "public")

    print(f"\n{'='*60}")
    print(f" 完成！")
    print(f"  自拍: {len(self_sel)} 张（全部合格）→ {OUT_SELF}")
    print(f"  公开: {len(pub_sel)} 张 → {OUT_PUB}")
    print(f"  分辨率: {TARGET_W}×{TARGET_H}，中心裁剪，无拉伸")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
