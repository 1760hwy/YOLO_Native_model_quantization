# prepare_calibration.py
"""
Calibration Data Preparation for RDK X5 Quantization
=====================================================
Generates calibration data in the EXACT format required by hb_mapper.

Output format: RGB, float32, NCHW, **0~255 range**

⚠️⚠️⚠️ THE MOST CRITICAL FILE IN THIS ENTIRE PROJECT ⚠️⚠️⚠️

The quantization YAML has:
    norm_type: 'data_scale'
    scale_value: '0.003921568627451'   # = 1/255

This means the BPU will AUTOMATICALLY divide by 255.
So calibration data MUST be in 0~255 range.

If you normalize to 0~1 here, you get DOUBLE normalization:
    pixel / 255 / 255 = pixel / 65025
This causes ALL cls outputs to collapse to strong negatives (-10 to -21),
sigmoid ≈ 0, and ZERO detections. The quantization process will NOT error out!

Usage:
    python prepare_calibration.py <src_image_dir> <dst_output_dir> [input_size] [num_samples]

Examples:
    python prepare_calibration.py ./dataset/images/train ./calibration_f32 640 100
    python prepare_calibration.py /data/fire_images /data/calib 640 50
"""
import cv2
import numpy as np
import os
from pathlib import Path
import random
import sys


def prepare_calibration_data(
    src_dir: str,
    dst_dir: str,
    input_size: int = 640,
    num_samples: int = 100,
    seed: int = 42
):
    """
    Generate calibration data files from images.

    Args:
        src_dir: Directory containing training images
        dst_dir: Output directory for .f32 calibration files
        input_size: Model input size (default 640)
        num_samples: Number of calibration samples (50-100 recommended)
        seed: Random seed for reproducibility
    """
    os.makedirs(dst_dir, exist_ok=True)

    # Find all images
    exts = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
    image_files = []
    for e in exts:
        image_files += list(Path(src_dir).glob(f"*{e}"))
        image_files += list(Path(src_dir).glob(f"*{e.upper()}"))

    print(f"📁 Found {len(image_files)} images in: {src_dir}")

    if len(image_files) == 0:
        print("❌ No images found! Check the source directory.")
        return 0

    if len(image_files) < num_samples:
        print(f"⚠️  Only {len(image_files)} images available, using all")
        num_samples = len(image_files)

    random.seed(seed)
    random.shuffle(image_files)
    image_files = image_files[:num_samples]

    print(f"🔄 Generating {num_samples} calibration files")
    print(f"   Input size:  {input_size}x{input_size}")
    print(f"   Format:      RGB float32 NCHW")
    print(f"   Value range: 0~255 (NOT normalized!)")
    print(f"   Output dir:  {dst_dir}")
    print()

    ok = 0
    for p in image_files:
        img = cv2.imread(str(p))
        if img is None:
            print(f"   ⚠️  Failed to read: {p}")
            continue

        h, w = img.shape[:2]

        # Step 1: Letterbox resize (same as training)
        scale = min(input_size / h, input_size / w)
        nh, nw = int(h * scale), int(w * scale)
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)

        # Pad with 114 (standard YOLO letterbox value)
        canvas = np.full((input_size, input_size, 3), 114, dtype=np.uint8)
        top = (input_size - nh) // 2
        left = (input_size - nw) // 2
        canvas[top:top + nh, left:left + nw] = resized

        # Step 2: BGR → RGB (CRITICAL: training uses RGB)
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

        # Step 3: Convert to float32, KEEP 0~255 range
        # ★★★ DO NOT divide by 255! The BPU's scale_value handles this! ★★★
        rgb_f32 = rgb.astype(np.float32)

        # Step 4: HWC → CHW (NCHW format)
        nchw = rgb_f32.transpose(2, 0, 1)   # (3, H, W)

        # Step 5: Save as binary
        out_path = os.path.join(dst_dir, f"{ok:06d}.f32")
        nchw.tofile(out_path)
        ok += 1

        if ok % 20 == 0 or ok == num_samples:
            print(f"   ✅ Processed {ok}/{num_samples}")

    file_size_mb = input_size * input_size * 3 * 4 / 1024 / 1024
    print(f"\n{'=' * 50}")
    print(f"✅ Done! {ok} calibration files generated")
    print(f"   Per file size: {file_size_mb:.2f} MB")
    print(f"   Total size:    {file_size_mb * ok:.1f} MB")
    print(f"   Output dir:    {dst_dir}")
    print(f"{'=' * 50}")

    return ok


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "./dataset/images/train"
    dst = sys.argv[2] if len(sys.argv) > 2 else "./calibration_f32"
    size = int(sys.argv[3]) if len(sys.argv) > 3 else 640
    num = int(sys.argv[4]) if len(sys.argv) > 4 else 100

    cnt = prepare_calibration_data(src, dst, size, num)
    if cnt == 0:
        print("\n❌ No calibration files generated!")
        print("   Check that your source directory contains images.")
        sys.exit(1)
