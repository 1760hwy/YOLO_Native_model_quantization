#!/usr/bin/env python3
"""
Step 1-1: 修改 Detect_LSDECD.forward
目的: 将 Detect_LSDECD.forward 改为6路分离输出(NHWC格式)，适配 RDK X5 BPU 后处理

重要说明:
  本项目模型使用 Detect_LSDECD (Lightweight Shared Detail Enhanced Convolutional Detection Head)
  位于 ultralytics/nn/extra_modules/head.py (非标准 modules/head.py)

  Detect_LSDECD 架构特点:
    - cv2, cv3: 共享单个 nn.Conv2d (非 ModuleList)
    - share_conv: 各 scale 共用的深度增强卷积块
    - scale[i]: 每个 stride 独立的可学习缩放因子

  原 forward:
    x[i] = share_conv(conv[i](x[i]))                    # 共享主干
    x[i] = cat(scale[i](cv2(x[i])), cv3(x[i]))          # bbox+cls 拼接
    → 返回 (1, nc+64, 8400) 合并输出

  新 forward (6路 NHWC):
    feat[i] = share_conv(conv[i](x[i]))                  # 保留共享主干
    bbox[i] = scale[i](cv2(feat[i])).permute(NHWC)       # [1, H, W, 64]
    cls[i]  = cv3(feat[i]).permute(NHWC)                 # [1, H, W, nc]
    → 返回 (bbox[0], bbox[1], bbox[2], cls[0], cls[1], cls[2])

断点说明:
  BP-A: 找到并验证 extra_modules/head.py
  BP-B: 备份原文件
  BP-C: 定位 Detect_LSDECD.forward 精确行范围
  BP-D: 替换并验证

运行方式:
  C:\d\Anaconda3\envs\yolo\python.exe step/step1/01_modify_head.py
"""

import sys
import shutil
import logging
from pathlib import Path

# ── 日志 ─────────────────────────────────────────────────────────────────────
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
log = logging.getLogger("step1_01")

# ── 路径 ─────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parents[2]
HEAD_PATH   = ROOT / "ultralytics" / "nn" / "extra_modules" / "head.py"
BACKUP_PATH = HEAD_PATH.with_suffix(".py.bak_rdkx5")

# ── 原始 forward 特征字符串（验证是否为未修改版本）───────────────────────────
ORIG_MARKER = "Concatenates and returns predicted bounding boxes and class probabilities."

# ── 新的 6路 NHWC forward（Detect_LSDECD 专用）──────────────────────────────
# 注意: Detect_LSDECD 的 cv2/cv3 是单个共享 nn.Conv2d，需分离调用
NEW_FORWARD = """\
    def forward(self, x):
        \"\"\"
        RDK X5 BPU 6-output format for Detect_LSDECD (3 bbox + 3 cls, NHWC).

        Architecture note:
          - share_conv: shared depthwise-enhanced conv block across scales
          - cv2: shared single Conv2d for bbox DFL (output dim = reg_max*4 = 64)
          - cv3: shared single Conv2d for cls scores (output dim = nc)
          - scale[i]: per-stride learnable scale factor

        Output shapes (imgsz=640):
          bboxes[0]: [1, 80, 80, 64]   stride=8
          bboxes[1]: [1, 40, 40, 64]   stride=16
          bboxes[2]: [1, 20, 20, 64]   stride=32
          clses[0]:  [1, 80, 80, nc]   stride=8
          clses[1]:  [1, 40, 40, nc]   stride=16
          clses[2]:  [1, 20, 20, nc]   stride=32
        \"\"\"
        if self.training:
            # Training path: keep original per-scale concat output
            for i in range(self.nl):
                x[i] = self.conv[i](x[i])
                x[i] = self.share_conv(x[i])
                x[i] = torch.cat((self.scale[i](self.cv2(x[i])), self.cv3(x[i])), 1)
            return x

        # Inference / Export path: 6 separate NHWC outputs for RDK X5
        feats = [self.share_conv(self.conv[i](x[i])) for i in range(self.nl)]
        bboxes = [
            self.scale[i](self.cv2(feats[i])).permute(0, 2, 3, 1).contiguous()
            for i in range(self.nl)
        ]
        clses = [
            self.cv3(feats[i]).permute(0, 2, 3, 1).contiguous()
            for i in range(self.nl)
        ]
        return (*bboxes, *clses)

"""


def find_class_method_range(lines: list, class_name: str, method_name: str):
    """定位 class_name.method_name 的行范围（0-indexed）"""
    in_target = False
    class_indent = -1
    method_start = -1
    method_indent = -1

    for i, line in enumerate(lines):
        s = line.strip()
        if not s:
            continue
        indent = len(line) - len(line.lstrip())

        if not in_target:
            if s.startswith(f"class {class_name}(") or s.startswith(f"class {class_name}:"):
                in_target = True
                class_indent = indent
            continue

        # 类结束
        if indent <= class_indent and (s.startswith("class ") or
                                        (s.startswith("def ") and indent == class_indent)):
            if method_start != -1:
                return method_start, i
            in_target = False
            continue

        # 找到目标方法
        if s.startswith(f"def {method_name}(") and method_start == -1:
            method_start = i
            method_indent = indent
            continue

        # 方法结束
        if method_start != -1:
            if indent == method_indent and s.startswith("def "):
                return method_start, i

    if method_start != -1:
        return method_start, len(lines)
    return -1, -1


def main():
    log.info("=" * 60)
    log.info("STEP 1-1: Modify Detect_LSDECD.forward -> 6-output NHWC")
    log.info("          Target: extra_modules/head.py")
    log.info("=" * 60)

    # BP-A: 验证目标文件
    log.info(f"[BP-A] Target: {HEAD_PATH}")
    if not HEAD_PATH.exists():
        log.error(f"[BP-A] File not found: {HEAD_PATH}")
        sys.exit(1)

    content = HEAD_PATH.read_text(encoding="utf-8")

    if "RDK X5 BPU 6-output format for Detect_LSDECD" in content:
        log.warning("[BP-A] Detect_LSDECD.forward already modified. Skip.")
        log.info("       To reapply: restore from .bak_rdkx5 first")
        return

    if "class Detect_LSDECD" not in content:
        log.error("[BP-A] Detect_LSDECD class not found in file!")
        sys.exit(1)
    log.info("[BP-A] Detect_LSDECD class confirmed. OK")

    # BP-B: 备份
    if not BACKUP_PATH.exists():
        shutil.copy2(HEAD_PATH, BACKUP_PATH)
        log.info(f"[BP-B] Backup: {BACKUP_PATH}")
    else:
        log.warning(f"[BP-B] Backup exists, skip: {BACKUP_PATH}")

    lines = content.splitlines(keepends=True)

    # BP-C: 定位 Detect_LSDECD.forward
    fw_start, fw_end = find_class_method_range(lines, "Detect_LSDECD", "forward")
    if fw_start == -1:
        log.error("[BP-C] Cannot locate Detect_LSDECD.forward!")
        sys.exit(1)

    log.info(f"[BP-C] Detect_LSDECD.forward: lines {fw_start+1}..{fw_end} (1-indexed)")
    log.info(f"       First line: {lines[fw_start].rstrip()!r}")
    if fw_end < len(lines):
        log.info(f"       Next method: {lines[fw_end].rstrip()!r}")

    orig_indent = len(lines[fw_start]) - len(lines[fw_start].lstrip())
    log.info(f"[BP-C] Method indent: {orig_indent} spaces")

    if orig_indent != 4:
        log.warning(f"[BP-C] Unexpected indent {orig_indent}, adjusting NEW_FORWARD ...")
        delta = orig_indent - 4
        adjusted = []
        for line in NEW_FORWARD.splitlines(keepends=True):
            if line.strip():
                adjusted.append(" " * delta + line if delta > 0 else line[(-delta):])
            else:
                adjusted.append(line)
        new_fw = "".join(adjusted)
    else:
        new_fw = NEW_FORWARD

    # BP-D: 替换并写回
    new_lines = lines[:fw_start] + [new_fw] + lines[fw_end:]
    HEAD_PATH.write_text("".join(new_lines), encoding="utf-8")
    log.info("[BP-D] Written.")

    # 验证
    verify = HEAD_PATH.read_text(encoding="utf-8")
    assert "RDK X5 BPU 6-output format for Detect_LSDECD" in verify
    assert "(*bboxes, *clses)" in verify

    # 检查缩进
    vlines = verify.splitlines()
    for i, vl in enumerate(vlines):
        if "RDK X5 BPU 6-output format for Detect_LSDECD" in vl:
            log.info(f"[BP-D] Marker found at line {i+1}")
            for j in range(i, -1, -1):
                if "def forward(self, x):" in vlines[j]:
                    actual_indent = len(vlines[j]) - len(vlines[j].lstrip())
                    log.info(f"[BP-D] forward indent = {actual_indent} (expected {orig_indent})")
                    if actual_indent != orig_indent:
                        log.error("[BP-D] Indent mismatch!")
                        sys.exit(1)
                    break

    log.info("[BP-D] Verification passed.")
    log.info("=" * 60)
    log.info("STEP 1-1 complete -> run 02_export_onnx.py")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
