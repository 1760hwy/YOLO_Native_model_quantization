# docker_verify.py
"""
Float Model Verification in Docker
====================================
Before quantizing, verify that the ONNX model can detect targets
using float32 inference. If float model can't detect, quantized
model definitely won't either.

Usage (inside Docker):
    python3 docker_verify.py fire_detect.onnx test_fire.jpg

Requirements:
    pip install onnxruntime opencv-python numpy
"""
import numpy as np
import cv2
import sys
import os


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def verify_float_model(onnx_path, image_path=None, input_size=640):
    try:
        import onnxruntime as ort
    except ImportError:
        print("ERROR: pip install onnxruntime")
        return

    print("=" * 60)
    print("Float Model Verification")
    print("=" * 60)

    # Load model
    print(f"\nLoading: {onnx_path}")
    sess = ort.InferenceSession(onnx_path)

    # Print IO info
    print(f"\nInputs:")
    for inp in sess.get_inputs():
        print(f"  {inp.name}: {inp.shape} ({inp.type})")

    print(f"\nOutputs:")
    for out in sess.get_outputs():
        print(f"  {out.name}: {out.shape}")

    # Prepare input
    if image_path and os.path.exists(image_path):
        img = cv2.imread(image_path)
        print(f"\nUsing image: {image_path} ({img.shape[1]}x{img.shape[0]})")
    else:
        img = np.full((480, 640, 3), 114, dtype=np.uint8)
        print("\nUsing blank gray image (provide image path for real test)")

    h, w = img.shape[:2]
    scale = min(input_size / h, input_size / w)
    nh, nw = int(h * scale), int(w * scale)
    resized = cv2.resize(img, (nw, nh))
    canvas = np.full((input_size, input_size, 3), 114, dtype=np.uint8)
    top, left = (input_size - nh) // 2, (input_size - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized

    # BGR → RGB → float32 → /255 → NCHW
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    data = rgb.astype(np.float32).transpose(2, 0, 1)[np.newaxis] / 255.0

    print(f"\nInput tensor: shape={data.shape}, range=[{data.min():.3f}, {data.max():.3f}]")

    # Run inference
    print("\nRunning float32 inference...")
    input_name = sess.get_inputs()[0].name
    outputs = sess.run(None, {input_name: data})

    print(f"\nOutput analysis:")
    print("-" * 60)

    max_cls_sigmoid = 0.0

    for i, out in enumerate(outputs):
        name = sess.get_outputs()[i].name
        is_cls = "cls" in name.lower() or (out.shape[1] != 64 and out.shape[1] < 10)

        tag = "CLS " if is_cls else "BBOX"
        print(f"  [{i}] {tag} {name}: shape={out.shape}")
        print(f"       min={out.min():.4f}, max={out.max():.4f}, mean={out.mean():.4f}")

        if is_cls:
            sig_max = sigmoid(out.max())
            print(f"       sigmoid(max) = {sig_max:.6f}")
            max_cls_sigmoid = max(max_cls_sigmoid, sig_max)

    # Diagnosis
    print(f"\n{'=' * 60}")
    if max_cls_sigmoid > 0.5:
        print(f"✅ Float model CAN detect (sigmoid max = {max_cls_sigmoid:.4f})")
        print(f"   Safe to proceed with quantization.")
    elif max_cls_sigmoid > 0.1:
        print(f"⚠️  Float model has weak detection (sigmoid max = {max_cls_sigmoid:.4f})")
        print(f"   Consider training for more epochs.")
    else:
        print(f"🔴 Float model CANNOT detect (sigmoid max = {max_cls_sigmoid:.6f})")
        if image_path:
            print(f"   Model may need more training (100+ epochs)")
        else:
            print(f"   Try again with an image containing the target object:")
            print(f"   python3 docker_verify.py {onnx_path} path/to/fire_image.jpg")
    print("=" * 60)


if __name__ == "__main__":
    onnx_path = sys.argv[1] if len(sys.argv) > 1 else "fire_detect.onnx"
    image_path = sys.argv[2] if len(sys.argv) > 2 else None
    verify_float_model(onnx_path, image_path)
