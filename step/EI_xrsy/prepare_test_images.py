#!/usr/bin/env python3
"""
PC 端：从数据集选取测试图片，统一尺寸后输出到 test_images/ 目录，
然后 scp 到板端使用。

选取规则：small/middle/large 各 3 张，共 9 张，固定种子保证可复现。
输出：test_images/small_001.jpg ... large_003.jpg（已中心裁剪到 640×640）
"""
import random
import shutil
from pathlib import Path
from PIL import Image

ROOT     = Path(__file__).parent.parent.parent          # EI_yolo/
DFS_DIR  = ROOT / "BDF-18K/公开数据集/PublicDataset5_DFS"
OUT_DIR  = Path(__file__).parent / "test_images"
SEED     = 250
N        = 3    # 每类张数

CATEGORIES = {
    "small":  "small_",
    "middle": "middle_",
    "large":  "large_",
}

def center_crop_640(img_path: Path) -> Image.Image:
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    s = min(w, h)
    left = (w - s) // 2
    top  = (h - s) // 2
    img  = img.crop((left, top, left + s, top + s))
    return img.resize((640, 640), Image.LANCZOS)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    random.seed(SEED)

    all_names = []
    for cat, prefix in CATEGORIES.items():
        imgs = sorted(DFS_DIR.glob(f"{prefix}*.jpg"))
        if len(imgs) < N:
            imgs = sorted(DFS_DIR.glob(f"{prefix}*.png"))
        chosen = random.sample(imgs, min(N, len(imgs)))
        for i, p in enumerate(chosen, 1):
            out_name = f"{cat}_{i:03d}.jpg"
            out_path = OUT_DIR / out_name
            img = center_crop_640(p)
            img.save(str(out_path), quality=95)
            print(f"  [{cat}] {p.name}  →  {out_name}")
            all_names.append(out_name)

    print(f"\n[OK] {len(all_names)} 张图片已保存至 {OUT_DIR}")
    print("\n下一步：将 test_images/ 整个目录 scp 到板端，例如：")
    print("  scp -r step/EI_xrsy/test_images  sunrise@192.168.0.106:/home/sunrise/EI_xrsy/")


if __name__ == "__main__":
    main()
