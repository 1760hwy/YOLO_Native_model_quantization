# export_for_rdkx5.py
"""
ONNX Export Script for RDK X5 Deployment
=========================================
Exports YOLO11 model with 6 raw outputs in NCHW format.
- No DFL decoding in graph (done in post-processing)
- No sigmoid in graph (done in post-processing)
- Outputs: bbox_P3, cls_P3, bbox_P4, cls_P4, bbox_P5, cls_P5

This matches the exact format used by rdk_model_zoo official models.

Usage:
    python export_for_rdkx5.py

    # Custom model path:
    python export_for_rdkx5.py --model path/to/best.pt --output fire_detect.onnx

CRITICAL NOTES:
    1. Do NOT add .permute() - keep NCHW format
    2. Do NOT include DFL or sigmoid in the graph
    3. Use opset_version=11 for Bayes-e BPU compatibility
"""
import os
os.environ["PYTORCH_ONNX_USE_LEGACY_EXPORTER"] = "1"
import warnings
warnings.filterwarnings('ignore')
import argparse
import torch
import types
from ultralytics import YOLO


def export_for_rdkx5(model_path, onnx_path):
    """Export YOLO11 model to ONNX with 6 raw NCHW outputs."""

    print("=" * 60)
    print("YOLO11 → ONNX Export for RDK X5")
    print("=" * 60)

    # Load model
    model = YOLO(model_path)
    head = model.model.model[-1]

    print(f"\nModel: {model_path}")
    print(f"Head: {head.__class__.__name__}")
    print(f"  nc (num classes): {head.nc}")
    print(f"  nl (num levels):  {head.nl}")
    print(f"  cv2 type: {type(head.cv2).__name__}")
    print(f"  cv3 type: {type(head.cv3).__name__}")

    # Verify it's standard Detect with ModuleList
    assert head.__class__.__name__ == 'Detect', \
        f"Expected 'Detect' head, got '{head.__class__.__name__}'"
    assert 'ModuleList' in type(head.cv2).__name__, \
        f"Expected ModuleList for cv2, got '{type(head.cv2).__name__}'"

    # ★★★ Custom forward: 6 outputs, NCHW, no permute ★★★
    # This is the KEY to matching rdk_model_zoo format
    def new_forward(self, x):
        result = []
        for i in range(self.nl):
            bbox = self.cv2[i](x[i])   # (B, 64, H, W) - NCHW
            cls  = self.cv3[i](x[i])   # (B, nc, H, W) - NCHW
            result.append(bbox)         # Even indices: bbox
            result.append(cls)          # Odd indices: cls
        return result

    head.forward = types.MethodType(new_forward, head)

    # Test forward pass
    model.model.eval()
    dummy = torch.randn(1, 3, 640, 640)
    with torch.no_grad():
        out = model.model(dummy)

    print(f"\nOutputs: {len(out)}")
    for i, o in enumerate(out):
        tag = "bbox" if i % 2 == 0 else "cls"
        level = ["P3(80x80)", "P4(40x40)", "P5(20x20)"][i // 2]
        print(f"  [{i}] {tag}_{level}: {tuple(o.shape)}")

    # Export to ONNX
    print(f"\nExporting to: {onnx_path}")
    nc = head.nc
    torch.onnx.export(
        model.model,
        dummy,
        onnx_path,
        input_names=['images'],
        output_names=[
            'bbox_P3', 'cls_P3',
            'bbox_P4', 'cls_P4',
            'bbox_P5', 'cls_P5',
        ],
        opset_version=11,
        do_constant_folding=True,
    )

    file_size = os.path.getsize(onnx_path) / 1024 / 1024
    print(f"\n✅ Export successful!")
    print(f"   File: {onnx_path}")
    print(f"   Size: {file_size:.1f} MB")

    # Verify ONNX
    print("\nVerifying ONNX...")
    import onnx
    m = onnx.load(onnx_path)

    # Check outputs
    print(f"  ONNX outputs: {len(m.graph.output)}")
    for o in m.graph.output:
        dims = [d.dim_value for d in o.type.tensor_type.shape.dim]
        print(f"    {o.name}: {dims}")

    # Check for Softmax nodes (should only be attention, not DFL)
    softmax_nodes = [n.name for n in m.graph.node if 'Softmax' in n.op_type]
    print(f"\n  Softmax nodes: {len(softmax_nodes)}")
    for name in softmax_nodes:
        print(f"    {name}")

    if any('dfl' in name.lower() for name in softmax_nodes):
        print("\n  ⚠️  WARNING: DFL Softmax found in graph!")
        print("     This will cause BPU to split into many subgraphs.")
        print("     Check your export code.")
    else:
        print("\n  ✅ No DFL Softmax in graph (correct)")

    # Check for shared conv (sign of LSDECD)
    share_conv = [n.name for n in m.graph.node if 'share_conv' in n.name]
    if share_conv:
        print(f"\n  ⚠️  WARNING: Shared conv nodes found ({len(share_conv)})")
        print("     This may cause quantization issues.")
    else:
        print("  ✅ No shared conv nodes (correct)")

    print("\n" + "=" * 60)
    print("Next step: Prepare calibration data")
    print("  python prepare_calibration.py <image_dir> <output_dir>")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export YOLO11 for RDK X5")
    parser.add_argument("--model", type=str,
                        default="results/fire-detect/weights/best.pt",
                        help="Path to trained model weights")
    parser.add_argument("--output", type=str,
                        default="results/fire-detect/weights/fire_detect.onnx",
                        help="Output ONNX path")
    args = parser.parse_args()

    export_for_rdkx5(args.model, args.output)
