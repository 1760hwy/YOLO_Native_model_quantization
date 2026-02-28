# diagnose_model.py
"""
RDK X5 Model Diagnostics
==========================
Run this on the RDK X5 board to check if your quantized model
produces correct output ranges.

Usage (on board):
    python3 diagnose_model.py /home/sunrise/fire_detect.bin
    python3 diagnose_model.py /home/sunrise/fire_detect.bin --image test.jpg
"""
import numpy as np
import cv2
import sys

try:
    from hobot_dnn import pyeasy_dnn as dnn
except ImportError:
    print("ERROR: hobot_dnn not available. Run this on the RDK X5 board.")
    sys.exit(1)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def diagnose(model_path, image_path=None):
    print("=" * 60)
    print("RDK X5 Model Diagnostics")
    print("=" * 60)

    # Load model
    print(f"\nLoading: {model_path}")
    models = dnn.load(model_path)
    model = models[0]
    print("✅ Model loaded")

    # Print output info
    print(f"\nOutputs: {len(model.outputs)}")
    for i, out in enumerate(model.outputs):
        p = out.properties
        print(f"  Output[{i}]: shape={p.shape}, layout={p.layout}")

    # Expected format check
    expected_shapes = {
        "NCHW_bbox": [(1, 64, 80, 80), (1, 64, 40, 40), (1, 64, 20, 20)],
        "NCHW_cls":  [(1, 1, 80, 80), (1, 1, 40, 40), (1, 1, 20, 20)],
    }

    shapes = [tuple(out.properties.shape) for out in model.outputs]
    bbox_shapes = [s for s in shapes if s[1] == 64 or s[-1] == 64]
    cls_shapes = [s for s in shapes if s not in bbox_shapes]

    print(f"\n  Bbox outputs: {len(bbox_shapes)}")
    print(f"  Cls outputs:  {len(cls_shapes)}")

    if len(bbox_shapes) != 3 or len(cls_shapes) != 3:
        print("  ⚠️  Expected 3 bbox + 3 cls outputs")

    # Check layout
    layout = model.outputs[0].properties.layout
    if "NCHW" in str(layout):
        print(f"  ✅ Layout: NCHW (matches rdk_model_zoo)")
    else:
        print(f"  ⚠️  Layout: {layout} (expected NCHW)")

    # Prepare test input
    print("\nPreparing test input...")
    SZ = 640
    if image_path:
        img = cv2.imread(image_path)
        if img is None:
            print(f"  ⚠️  Cannot read {image_path}, using blank image")
            img = np.full((480, 640, 3), 114, dtype=np.uint8)
        else:
            print(f"  Using image: {image_path}")
    else:
        img = np.full((480, 640, 3), 114, dtype=np.uint8)
        print("  Using blank gray image (no --image provided)")

    # Letterbox + NV12
    h, w = img.shape[:2]
    scale = min(SZ / h, SZ / w)
    nh, nw = int(h * scale), int(w * scale)
    resized = cv2.resize(img, (nw, nh))
    canvas = np.full((SZ, SZ, 3), 114, dtype=np.uint8)
    top, left = (SZ - nh) // 2, (SZ - nw) // 2
    canvas[top:top+nh, left:left+nw] = resized

    yuv = cv2.cvtColor(canvas, cv2.COLOR_BGR2YUV_I420)
    y = yuv[:SZ, :]
    u = yuv[SZ:SZ+SZ//4, :].reshape(SZ//2, SZ//2)
    v = yuv[SZ+SZ//4:, :].reshape(SZ//2, SZ//2)
    uv = np.stack([u, v], axis=-1).reshape(SZ//2, SZ)
    nv12 = np.concatenate([y, uv], axis=0)

    # Forward
    print("\nRunning inference...")
    outputs = model.forward(nv12)

    print(f"\nOutput analysis:")
    print("-" * 60)

    cls_max_sigmoid = 0.0

    for i, out in enumerate(outputs):
        buf = np.array(out.buffer, copy=False)
        is_cls = (buf.shape[1] == 1) if len(buf.shape) == 4 else False

        tag = "CLS " if is_cls else "BBOX"

        print(f"  [{i}] {tag}: shape={buf.shape}, dtype={buf.dtype}")
        print(f"         min={buf.min():.4f}, max={buf.max():.4f}, mean={buf.mean():.4f}")

        if is_cls:
            sig_max = sigmoid(buf.max())
            print(f"         sigmoid(max) = {sig_max:.6f}")
            cls_max_sigmoid = max(cls_max_sigmoid, sig_max)

    # Diagnosis
    print(f"\n{'=' * 60}")
    print("DIAGNOSIS")
    print(f"{'=' * 60}")

    if cls_max_sigmoid < 0.001:
        print(f"\n  🔴 CRITICAL: cls sigmoid max = {cls_max_sigmoid:.6f}")
        print(f"     All classification scores are near zero!")
        print(f"")
        print(f"     Possible causes (in order of likelihood):")
        print(f"     1. Calibration data was normalized to 0~1 (DOUBLE normalization)")
        print(f"        → Fix: Regenerate with 0~255 range")
        print(f"     2. Model undertrained (< 50 epochs)")
        print(f"        → Fix: Train for 100+ epochs")
        print(f"     3. Test image doesn't contain the target object")
        print(f"        → Fix: Use --image with an image containing the target")
        print(f"     4. Calibration data used BGR instead of RGB")
        print(f"        → Fix: Add cv2.cvtColor(img, COLOR_BGR2RGB)")
    elif cls_max_sigmoid < 0.1 and image_path:
        print(f"\n  ⚠️  cls sigmoid max = {cls_max_sigmoid:.4f}")
        print(f"     Low confidence. Model may need more training,")
        print(f"     or the test image doesn't contain the target.")
    elif cls_max_sigmoid >= 0.1:
        print(f"\n  ✅ cls sigmoid max = {cls_max_sigmoid:.4f}")
        print(f"     Model is producing non-zero classification scores.")
        print(f"     Detection should work with appropriate conf_thresh.")
    else:
        print(f"\n  ℹ️  cls sigmoid max = {cls_max_sigmoid:.6f}")
        print(f"     Using blank image - low scores are expected.")
        print(f"     Re-run with --image <path_to_image_with_target>")

    print()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="RDK X5 Model Diagnostics")
    parser.add_argument("model", help="Path to .bin model file")
    parser.add_argument("--image", help="Test image path (optional)")
    args = parser.parse_args()
    diagnose(args.model, args.image)
