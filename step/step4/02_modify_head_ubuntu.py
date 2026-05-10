#!/usr/bin/env python3
"""
Step 3-2: Ubuntu 环境修改 Ultralytics head.py (Docker 容器内执行)
目的: 在 Ubuntu Docker 容器内安装的 ultralytics 中做相同的 6路 NHWC 输出修改

运行环境: OpenExplorer Docker 容器
运行命令:
  pip install ultralytics==8.3.28 -i https://pypi.tuna.tsinghua.edu.cn/simple
  python3 /data/step3/02_modify_head_ubuntu.py

注意: Ubuntu 端导出 ONNX 是为了确保与容器内工具链完全兼容
      如果 Windows 端 step1/best.onnx 已通过 checker 验证，可跳过此步直接用 best.onnx

断点说明:
  BP-A: 定位 head.py 路径
  BP-B: 备份 + 修改
  BP-C: 验证修改
"""

import os
import sys
import shutil
import logging
from pathlib import Path

LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(str(LOG_DIR / "step3_02.log"), encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("step3_02")

NEW_FORWARD = '''\
    def forward(self, x):
        """RDK X5 BPU: 6路 NHWC 分离输出 (3 bbox + 3 cls)"""
        if self.end2end:
            return self.forward_end2end(x)
        bboxes = [self.cv2[i](x[i]).permute(0, 2, 3, 1).contiguous() for i in range(self.nl)]
        clses  = [self.cv3[i](x[i]).permute(0, 2, 3, 1).contiguous() for i in range(self.nl)]
        return (*bboxes, *clses)

'''


def main():
    log.info("=" * 60)
    log.info("STEP 3-2: 修改 Ultralytics head.py (Ubuntu/Docker)")
    log.info("=" * 60)

    # BP-A: 定位 head.py
    try:
        import ultralytics
        ul_path = Path(ultralytics.__file__).parent
        head_path = ul_path / "nn" / "modules" / "head.py"
    except ImportError:
        log.error("[BP-A] ultralytics 未安装，请先: pip install ultralytics==8.3.28")
        sys.exit(1)

    log.info(f"[BP-A] ultralytics: {ul_path}")
    log.info(f"[BP-A] head.py: {head_path}")
    if not head_path.exists():
        log.error("[BP-A] head.py 不存在")
        sys.exit(1)

    content = head_path.read_text(encoding="utf-8")
    if "RDK X5 BPU" in content:
        log.warning("[BP-A] 已修改，跳过")
        return

    # BP-B: 备份 + 修改
    bak = head_path.with_suffix(".py.bak")
    if not bak.exists():
        shutil.copy2(head_path, bak)
        log.info(f"[BP-B] 备份: {bak}")

    lines = content.splitlines(keepends=True)
    in_detect = False
    fw_start = fw_end = -1
    method_indent = 0
    detect_indent = 0

    for i, line in enumerate(lines):
        s = line.strip()
        cur_indent = len(line) - len(line.lstrip()) if s else 999

        if s.startswith("class Detect(") and not in_detect:
            in_detect = True
            detect_indent = cur_indent
            continue

        if in_detect:
            if s.startswith("class ") and cur_indent <= detect_indent:
                in_detect = False
                if fw_start != -1 and fw_end == -1:
                    fw_end = i
                break
            if s.startswith("def forward(self, x):") and fw_start == -1:
                fw_start = i
                method_indent = cur_indent
                continue
            if fw_start != -1 and fw_end == -1:
                if s and cur_indent <= method_indent and s.startswith("def "):
                    fw_end = i
                    break

    if fw_start == -1:
        log.error("[BP-B] 未找到 Detect.forward")
        sys.exit(1)
    if fw_end == -1:
        fw_end = len(lines)

    log.info(f"[BP-B] Detect.forward: 行 {fw_start+1}~{fw_end}")
    base_indent = " " * method_indent
    new_fw = "\n".join(base_indent + l if l.strip() else l
                       for l in NEW_FORWARD.splitlines()) + "\n"
    new_lines = lines[:fw_start] + [new_fw] + lines[fw_end:]
    head_path.write_text("".join(new_lines), encoding="utf-8")

    # BP-C: 验证
    check = head_path.read_text(encoding="utf-8")
    assert "RDK X5 BPU" in check
    log.info("[BP-C] 验证通过 ✓")
    log.info("=" * 60)
    log.info("STEP 3-2 完成 → 继续 03_export_onnx_ubuntu.py")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
