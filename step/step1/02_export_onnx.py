#!/usr/bin/env python3
"""
Step 1-2: 导出 ONNX 模型
目的: 将 best.pt 导出为 opset=11 的 FP32 ONNX，供 hb_mapper 量化使用

配置:
  - imgsz=640 (与训练一致)
  - opset=11 (RDK X5 兼容最优 opset)
  - simplify=False (避免 IR version 问题)
  - dynamic=False (BPU 不支持动态 shape)
  - half=False (FP32，量化时再转 INT8)

断点说明:
  BP-A: 加载 .pt 模型
  BP-B: 执行 ONNX 导出
  BP-C: 验证输出文件
  BP-D: 验证输出节点数量及形状

运行方式:
  C:\d\Anaconda3\envs\yolo\python.exe step/step1/02_export_onnx.py
"""

import sys
import logging
from pathlib import Path

# ── 日志配置 ────────────────────────────────────────────────────────────────
LOG_FILE = Path(__file__).parent / "step1.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8", mode="a"),
        logging.StreamHandler(sys.stdout),
    ],
)
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

log = logging.getLogger("step1_02")

# ── 路径常量 ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
MODEL_PT  = ROOT / "yolo11-HGNetV2-C3k2-DWR-LSDECD-SlideLoss" / "weights" / "best.pt"
ONNX_OUT  = Path(__file__).parent / "best.onnx"

# 导出参数
IMGSZ   = 640     # 必须与训练尺寸一致
OPSET   = 11      # RDK X5 推荐 opset 10/11，选 11
DEVICE  = "cpu"   # 导出时用 cpu，避免 cuda graph 残留算子


def main():
    log.info("=" * 60)
    log.info("STEP 1-2: 导出 ONNX (opset=%d, imgsz=%d)", OPSET, IMGSZ)
    log.info("=" * 60)

    # BP-A: 加载模型
    log.info(f"[BP-A] 加载 .pt 模型: {MODEL_PT}")
    if not MODEL_PT.exists():
        log.error(f"[BP-A] 找不到模型文件: {MODEL_PT}")
        sys.exit(1)

    # 必须在 EI_yolo 根目录下导入本地 ultralytics
    sys.path.insert(0, str(ROOT))
    from ultralytics import YOLO

    model = YOLO(str(MODEL_PT))
    log.info("[BP-A] 模型加载成功 ✓")
    log.info(f"       模型信息: {model.info(verbose=False)}")

    # BP-B: 执行导出
    log.info(f"[BP-B] Start ONNX export ...")
    log.info(f"       imgsz={IMGSZ}, opset={OPSET}, simplify=False, dynamic=False")
    try:
        exported = model.export(
            format="onnx",
            imgsz=IMGSZ,
            opset=OPSET,
            simplify=False,   # 不做图化简，避免 IR version 冲突
            dynamic=False,    # 静态 shape (BPU 必须)
            half=False,       # FP32 导出
            device=DEVICE,
            verbose=False,
        )
        log.info(f"[BP-B] Export done: {exported}")
    except Exception as e:
        log.error(f"[BP-B] Export FAILED: {e}", exc_info=True)
        sys.exit(1)

    # 将导出的 onnx 复制到 step1/best.onnx
    src = Path(str(exported))
    if src.exists() and src.resolve() != ONNX_OUT.resolve():
        import shutil
        shutil.copy2(src, ONNX_OUT)
        log.info(f"[BP-B] Copied to: {ONNX_OUT}")
    elif src.resolve() == ONNX_OUT.resolve():
        log.info(f"[BP-B] ONNX already at target path")
    else:
        log.error(f"[BP-B] Exported file not found: {src}")
        sys.exit(1)

    # BP-C: 验证文件存在和大小
    size_mb = ONNX_OUT.stat().st_size / 1024 / 1024
    log.info(f"[BP-C] ONNX 文件: {ONNX_OUT}")
    log.info(f"[BP-C] 文件大小: {size_mb:.2f} MB ✓")

    # BP-D: 验证 ONNX 结构
    import onnx
    log.info(f"[BP-D] ONNX 版本: {onnx.__version__}")
    onnx_model = onnx.load(str(ONNX_OUT))

    try:
        onnx.checker.check_model(onnx_model)
        log.info("[BP-D] onnx.checker 通过 ✓")
    except Exception as e:
        log.warning(f"[BP-D] onnx.checker 警告: {e}")

    ir_ver    = onnx_model.ir_version
    opset_ver = onnx_model.opset_import[0].version
    n_inputs  = len(onnx_model.graph.input)
    n_outputs = len(onnx_model.graph.output)
    log.info(f"[BP-D] IR Version: {ir_ver}")
    log.info(f"[BP-D] Opset Version: {opset_ver}")
    log.info(f"[BP-D] 输入数量: {n_inputs}")
    log.info(f"[BP-D] 输出数量: {n_outputs}")

    # 输入形状
    for inp in onnx_model.graph.input:
        shape = [d.dim_value for d in inp.type.tensor_type.shape.dim]
        log.info(f"[BP-D]   输入: {inp.name} → {shape}")

    # 输出形状
    expected_outputs = 6  # 3 bbox + 3 cls (NHWC)
    for i, out in enumerate(onnx_model.graph.output):
        shape = [d.dim_value for d in out.type.tensor_type.shape.dim]
        log.info(f"[BP-D]   输出{i}: {out.name} → {shape}")

    if n_outputs == expected_outputs:
        log.info(f"[BP-D] 输出数量正确: {n_outputs} 个 (3 bbox + 3 cls) ✓")
    else:
        log.error(f"[BP-D] 输出数量异常! 期望 {expected_outputs}，实际 {n_outputs}")
        log.error("       请检查 01_modify_head.py 是否已成功执行")
        sys.exit(1)

    log.info("=" * 60)
    log.info("STEP 1-2 完成 → 继续运行 03_extract_operators.py")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
