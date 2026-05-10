#!/usr/bin/env python3
"""
Step 3-3: Ubuntu Docker 容器内导出 ONNX (可选，如使用 Windows best.onnx 则跳过)
目的: 在 Docker 容器内使用相同配置重新导出，确保与 hb_mapper 1.24.3 完全兼容

运行环境: OpenExplorer Docker 容器
运行命令:
  python3 /data/step3/03_export_onnx_ubuntu.py \
    --model /models/best.pt \
    --out   /data/fire_detect.onnx

断点说明:
  BP-A: 检查 .pt 文件
  BP-B: 导出 ONNX
  BP-C: 验证 (onnx.checker + 输出数量)
"""

import sys
import logging
import argparse
from pathlib import Path

LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(str(LOG_DIR / "step3_03.log"), encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("step3_03")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/models/best.pt")
    parser.add_argument("--out",   default="/data/fire_detect.onnx")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--opset", type=int, default=11)
    args = parser.parse_args()

    log.info("=" * 60)
    log.info(f"STEP 3-3: 导出 ONNX (Ubuntu/Docker) | opset={args.opset}, imgsz={args.imgsz}")
    log.info("=" * 60)

    # BP-A: 检查模型
    model_path = Path(args.model)
    log.info(f"[BP-A] 模型: {model_path}")
    if not model_path.exists():
        log.error(f"[BP-A] 找不到 {model_path}")
        sys.exit(1)
    log.info(f"[BP-A] 大小: {model_path.stat().st_size/1024/1024:.2f} MB ✓")

    # BP-B: 导出
    from ultralytics import YOLO
    model = YOLO(str(model_path))
    log.info("[BP-B] 开始导出 ...")
    exported = model.export(
        format="onnx",
        imgsz=args.imgsz,
        opset=args.opset,
        simplify=False,
        dynamic=False,
        half=False,
        verbose=False,
    )
    log.info(f"[BP-B] 导出完成: {exported} ✓")

    out_path = Path(args.out)
    if Path(str(exported)) != out_path:
        import shutil
        shutil.copy2(str(exported), str(out_path))
        log.info(f"[BP-B] 已复制到: {out_path}")

    # BP-C: 验证
    import onnx
    onnx_model = onnx.load(str(out_path))
    onnx.checker.check_model(onnx_model)
    n_out = len(onnx_model.graph.output)
    log.info(f"[BP-C] onnx.checker 通过 ✓ | 输出数量: {n_out}")

    if n_out != 6:
        log.error(f"[BP-C] 期望 6 个输出，实际 {n_out} — 检查 head.py 修改")
        sys.exit(1)

    for i, out in enumerate(onnx_model.graph.output):
        shape = [d.dim_value for d in out.type.tensor_type.shape.dim]
        log.info(f"       输出{i}: {out.name} → {shape}")

    log.info("=" * 60)
    log.info("STEP 3-3 完成 → 继续 04_quantization_config.yaml")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
