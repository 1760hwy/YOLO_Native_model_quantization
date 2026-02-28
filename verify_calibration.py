# verify_calibration.py
"""
Calibration Data Verification Script
=====================================
Validates that calibration data is in the correct format.

Checks:
    - File count
    - File size (should be 3*640*640*4 = 4,915,200 bytes)
    - Value range (should be 0~255, NOT 0~1)
    - Data shape (3, 640, 640)

Usage:
    python verify_calibration.py <calibration_dir> [input_size]
    python verify_calibration.py ./calibration_f32 640
"""
import numpy as np
import os
import sys


def verify_calibration(cal_dir, input_size=640):
    print("=" * 60)
    print("Calibration Data Verification")
    print("=" * 60)

    if not os.path.exists(cal_dir):
        print(f"❌ Directory not found: {cal_dir}")
        return False

    files = sorted([f for f in os.listdir(cal_dir) if f.endswith('.f32')])
    print(f"\nDirectory: {cal_dir}")
    print(f"Total .f32 files: {len(files)}")

    if len(files) == 0:
        print("❌ No .f32 files found!")
        return False

    if len(files) < 50:
        print(f"⚠️  Only {len(files)} files. Recommend 50~100 for good quantization.")

    expected_elements = 3 * input_size * input_size
    expected_bytes = expected_elements * 4  # float32

    all_ok = True
    checked = 0
    range_issues = 0

    for f in files[:5]:  # Check first 5 files in detail
        filepath = os.path.join(cal_dir, f)
        file_size = os.path.getsize(filepath)

        print(f"\n--- {f} ---")
        print(f"  File size: {file_size} bytes (expected: {expected_bytes})")

        if file_size != expected_bytes:
            print(f"  ❌ Size mismatch! Expected {expected_bytes}, got {file_size}")
            all_ok = False
            continue

        data = np.fromfile(filepath, dtype=np.float32)
        print(f"  Elements: {data.shape[0]} (expected: {expected_elements})")

        data_3d = data.reshape(3, input_size, input_size)
        print(f"  Shape: {data_3d.shape}")
        print(f"  Min:   {data.min():.2f}")
        print(f"  Max:   {data.max():.2f}")
        print(f"  Mean:  {data.mean():.2f}")
        print(f"  Dtype: {data.dtype}")

        # ★ The critical check: values should be in 0~255 range
        if data.max() <= 1.1:
            print(f"  🔴 ERROR: Max value is {data.max():.4f}")
            print(f"     Data appears to be normalized to 0~1!")
            print(f"     This will cause DOUBLE NORMALIZATION!")
            print(f"     BPU's scale_value (1/255) will normalize AGAIN!")
            print(f"     Result: pixel / 255 / 255 → all cls outputs collapse")
            print(f"     FIX: Remove /255 from calibration script")
            all_ok = False
            range_issues += 1
        elif data.max() > 256:
            print(f"  ⚠️  Max value > 256, unusual but not necessarily wrong")
        elif data.min() < -1:
            print(f"  ⚠️  Negative values detected, check preprocessing")
        else:
            print(f"  ✅ Value range OK (0~255)")

        checked += 1

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Files checked: {checked}")
    print(f"  Total files:   {len(files)}")

    if range_issues > 0:
        print(f"\n  🔴🔴🔴 CRITICAL ERROR 🔴🔴🔴")
        print(f"  {range_issues} files have values in 0~1 range!")
        print(f"  This WILL cause zero detections after quantization!")
        print(f"  ")
        print(f"  FIX: In prepare_calibration.py, change:")
        print(f"    ❌ rgb_f32 = rgb.astype(np.float32) / 255.0")
        print(f"    ✅ rgb_f32 = rgb.astype(np.float32)")
        print(f"  ")
        print(f"  Then regenerate calibration data.")
        all_ok = False
    elif all_ok:
        print(f"\n  ✅ All checks passed!")
        print(f"  Calibration data is ready for quantization.")
    else:
        print(f"\n  ⚠️  Some issues found, see above.")

    return all_ok


if __name__ == "__main__":
    cal_dir = sys.argv[1] if len(sys.argv) > 1 else "./calibration_f32"
    size = int(sys.argv[2]) if len(sys.argv) > 2 else 640
    verify_calibration(cal_dir, size)
