#!/usr/bin/env python3
"""
Step 1 总入口: 依次执行 01 → 02 → 03 → 04
每步失败则终止，日志写入 step1.log

运行方式:
  C:\d\Anaconda3\envs\yolo\python.exe step/step1/run_step1.py
"""

import subprocess
import sys
import logging
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

LOG_FILE = Path(__file__).parent / "step1.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8", mode="w"),  # 新建日志
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("step1_runner")

PYTHON = sys.executable
STEP1  = Path(__file__).parent

SCRIPTS = [
    ("01_modify_head.py",       "修改 head.py → 6路 NHWC 输出"),
    ("01b_gn_to_bn.py",         "GroupNorm→BatchNorm + 导出 ONNX (全 BPU)"),
    ("03_extract_operators.py", "提取所有算子 → operators.json"),
    ("04_check_compatibility.py","兼容性分析 → operator_compatibility.json"),
]


def run(script_name: str, desc: str) -> bool:
    script = STEP1 / script_name
    log.info(f"\n{'='*60}")
    log.info(f"运行: {script_name} — {desc}")
    log.info(f"{'='*60}")
    result = subprocess.run(
        [PYTHON, str(script)],
        capture_output=False,
    )
    if result.returncode != 0:
        log.error(f"❌ {script_name} 失败 (exit={result.returncode})")
        return False
    log.info(f"✅ {script_name} 完成")
    return True


def main():
    log.info("=" * 60)
    log.info("STEP 1 总入口 — ONNX 导出 + 算子兼容性分析")
    log.info(f"Python: {PYTHON}")
    log.info("=" * 60)

    for script, desc in SCRIPTS:
        if not run(script, desc):
            log.error(f"Step 1 在 {script} 处中止，请查看上方错误信息")
            sys.exit(1)

    log.info("\n" + "=" * 60)
    log.info("✅ STEP 1 全部完成!")
    log.info("产出文件:")
    for f in ["best.onnx", "operators.json", "operator_compatibility.json", "step1.log"]:
        p = STEP1 / f
        size = f"{p.stat().st_size/1024:.1f} KB" if p.exists() else "未生成"
        log.info(f"  {f}: {size}")
    log.info("\n下一步: cd step/step2 && python run_step2.py")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
