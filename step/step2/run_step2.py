#!/usr/bin/env python3
"""
Step 2 总入口: 01_sample_images → 02_validate_onnx
日志写入 step2.log

运行方式:
  C:\d\Anaconda3\envs\yolo\python.exe step/step2/run_step2.py
"""

import subprocess
import sys
import logging
from pathlib import Path

LOG_FILE = Path(__file__).parent / "step2.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8", mode="w"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("step2_runner")

PYTHON = sys.executable
STEP2  = Path(__file__).parent

SCRIPTS = [
    ("01_sample_images.py",  "从 BDF-18K 抽取 50 张图"),
    ("02_validate_onnx.py",  "ONNX 推理验证 + 可视化"),
]


def run(script_name: str, desc: str) -> bool:
    script = STEP2 / script_name
    log.info(f"\n{'='*60}")
    log.info(f"运行: {script_name} — {desc}")
    log.info(f"{'='*60}")
    result = subprocess.run([PYTHON, str(script)], capture_output=False)
    if result.returncode != 0:
        log.error(f"❌ {script_name} 失败 (exit={result.returncode})")
        return False
    log.info(f"✅ {script_name} 完成")
    return True


def main():
    log.info("=" * 60)
    log.info("STEP 2 总入口 — ONNX 推理验证")
    log.info(f"Python: {PYTHON}")
    log.info("=" * 60)

    for script, desc in SCRIPTS:
        if not run(script, desc):
            log.error(f"Step 2 在 {script} 处中止")
            sys.exit(1)

    log.info("\n" + "=" * 60)
    log.info("✅ STEP 2 全部完成!")
    log.info("产出文件:")
    for f in ["sample_list.json", "validation_report.json", "step2.log"]:
        p = STEP2 / f
        size = f"{p.stat().st_size/1024:.1f} KB" if p.exists() else "未生成"
        log.info(f"  {f}: {size}")
    vis_dir = STEP2 / "results"
    if vis_dir.exists():
        n_vis = len(list(vis_dir.glob("*.jpg")))
        log.info(f"  results/: {n_vis} 张可视化图")
    log.info("\n下一步: 将 step1/best.onnx 传输到 Ubuntu 执行 Step 3 量化")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
