#!/usr/bin/env python3
"""
Step 2-2: ONNX 模型推理验证
目的:
  1. 用 onnxruntime 对 50 张样本图片做检测推理
  2. 对比 PyTorch 原始模型输出 (仅检查 bbox 数量一致性)
  3. 绘制检测框并保存可视化结果
  4. 生成验证报告 validation_report.json

后处理说明 (6路 NHWC 输出解码):
  - 输出 [0,1,2]: bbox DFL特征，shape [1,H,W,64]  (4 * reg_max=16)
  - 输出 [3,4,5]: cls 分数，   shape [1,H,W,nc]
  - DFL解码 → xyxy → NMS

断点说明:
  BP-A: 加载 ONNX 会话
  BP-B: 逐图推理
  BP-C: 解码 + NMS
  BP-D: 绘制可视化并保存
  BP-E: 生成汇总报告

运行方式:
  C:\d\Anaconda3\envs\yolo\python.exe step/step2/02_validate_onnx.py
"""

import sys
import json
import time
import logging
import numpy as np
import cv2
from pathlib import Path
from typing import Dict

# ── 日志 ─────────────────────────────────────────────────────────────────────
LOG_FILE = Path(__file__).parent / "step2.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8", mode="a"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("step2_02")

# ── 路径 ─────────────────────────────────────────────────────────────────────
ROOT         = Path(__file__).resolve().parents[2]
STEP2_DIR    = Path(__file__).parent
ONNX_PATH    = ROOT / "step" / "step1" / "best.onnx"
SAMPLE_JSON  = STEP2_DIR / "sample_list.json"
VIS_DIR      = STEP2_DIR / "results"
REPORT_JSON  = STEP2_DIR / "validation_report.json"

# ── 推理参数 ──────────────────────────────────────────────────────────────────
IMGSZ     = 640
CONF_THRES = 0.25
IOU_THRES  = 0.45
CLASS_NAMES = ["fire"]
REG_MAX   = 16   # DFL reg_max


def letterbox(img: np.ndarray, new_shape=(640, 640), color=(114, 114, 114)):
    """保持长宽比 Letterbox resize"""
    h, w = img.shape[:2]
    scale = min(new_shape[0] / h, new_shape[1] / w)
    new_h, new_w = int(round(h * scale)), int(round(w * scale))
    dh = (new_shape[0] - new_h) / 2
    dw = (new_shape[1] - new_w) / 2
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right  = int(round(dw - 0.1)), int(round(dw + 0.1))

    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, scale, (dw, dh)


def preprocess(img_bgr: np.ndarray):
    """BGR → RGB → letterbox → NCHW float32 / 255"""
    img, scale, pad = letterbox(img_bgr, (IMGSZ, IMGSZ))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # HWC → CHW
    img = np.expand_dims(img, 0)        # CHW → NCHW
    return img, scale, pad


def dfl_decode(pred: np.ndarray) -> np.ndarray:
    """
    DFL (Distribution Focal Loss) 解码:
    pred: [N, reg_max*4]  →  xywh: [N, 4]
    """
    n, _ = pred.shape
    pred = pred.reshape(n, 4, REG_MAX)
    # softmax along last dim
    exp = np.exp(pred - pred.max(axis=-1, keepdims=True))
    prob = exp / exp.sum(axis=-1, keepdims=True)
    # 期望值: Σ(prob * i)
    bins = np.arange(REG_MAX, dtype=np.float32)
    dist = (prob * bins).sum(axis=-1)  # [N, 4]
    return dist  # [lt_x, lt_y, rb_x, rb_y] 相对于 anchor


def make_anchors(h: int, w: int, stride: int):
    """生成 anchor 中心点 (grid 坐标 + 0.5)"""
    ys = np.arange(h, dtype=np.float32) + 0.5
    xs = np.arange(w, dtype=np.float32) + 0.5
    grid_y, grid_x = np.meshgrid(ys, xs, indexing="ij")
    anchors = np.stack([grid_x, grid_y], axis=-1).reshape(-1, 2)
    return anchors


def decode_outputs(onnx_outputs: list) -> np.ndarray:
    """
    解码 6 路 NHWC 输出为 xyxy 格式 bbox
    outputs[0..2]: bbox [1,H,W,64]  (NHWC)
    outputs[3..5]: cls  [1,H,W,nc]  (NHWC)
    返回: [N_total, 6] = [x1, y1, x2, y2, conf, cls_id]  (pixel in imgsz)
    """
    strides = [8, 16, 32]
    all_boxes = []

    for i, stride in enumerate(strides):
        # bbox feature: [1,H,W,64] → [H*W, 64]
        bbox_feat = onnx_outputs[i][0]        # [H, W, 64]
        h_feat, w_feat = bbox_feat.shape[:2]
        bbox_feat = bbox_feat.reshape(-1, REG_MAX * 4)   # [HW, 64]

        # cls scores: [1,H,W,nc] → [H*W, nc]
        cls_feat = onnx_outputs[i + 3][0]     # [H, W, nc]
        cls_feat = cls_feat.reshape(-1, len(CLASS_NAMES))  # [HW, nc]

        # sigmoid 激活
        cls_prob = 1.0 / (1.0 + np.exp(-cls_feat))   # [HW, nc]

        # 最大类别得分和 id
        max_score = cls_prob.max(axis=-1)             # [HW]
        max_cls   = cls_prob.argmax(axis=-1)           # [HW]

        # 过滤低置信度
        mask = max_score > CONF_THRES
        if not mask.any():
            continue

        bbox_f  = bbox_feat[mask]   # [M, 64]
        scores  = max_score[mask]   # [M]
        cls_ids = max_cls[mask]     # [M]

        # anchor 中心点
        anchors = make_anchors(h_feat, w_feat, stride)[mask]  # [M, 2]

        # DFL 解码: [M, 4] → lt_x, lt_y, rb_x, rb_y (单位: cell)
        dist = dfl_decode(bbox_f)   # [M, 4]

        # 转为 pixel 坐标
        x1 = (anchors[:, 0] - dist[:, 0]) * stride
        y1 = (anchors[:, 1] - dist[:, 1]) * stride
        x2 = (anchors[:, 0] + dist[:, 2]) * stride
        y2 = (anchors[:, 1] + dist[:, 3]) * stride

        boxes = np.stack([x1, y1, x2, y2, scores, cls_ids.astype(np.float32)], axis=1)
        all_boxes.append(boxes)

    if not all_boxes:
        return np.zeros((0, 6), dtype=np.float32)

    return np.concatenate(all_boxes, axis=0)


def nms(boxes: np.ndarray, iou_thres: float) -> np.ndarray:
    """简单 NMS (按类别分别执行)"""
    if len(boxes) == 0:
        return np.zeros((0, 6), dtype=np.float32)

    keep_all = []
    for cls_id in np.unique(boxes[:, 5]):
        mask  = boxes[:, 5] == cls_id
        bboxes = boxes[mask]
        x1, y1, x2, y2 = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
        scores = bboxes[:, 4]
        order  = scores.argsort()[::-1]
        keep   = []

        while order.size > 0:
            i = order[0]
            keep.append(i)
            if order.size == 1:
                break
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            inter = w * h
            area_i = (x2[i] - x1[i]) * (y2[i] - y1[i])
            area_j = (x2[order[1:]] - x1[order[1:]]) * (y2[order[1:]] - y1[order[1:]])
            iou = inter / (area_i + area_j - inter + 1e-6)
            order = order[np.concatenate(([0], np.where(iou <= iou_thres)[0] + 1))]
            order = order[1:]

        keep_all.append(bboxes[keep])

    return np.concatenate(keep_all, axis=0) if keep_all else np.zeros((0, 6), dtype=np.float32)


def draw_boxes(img_bgr: np.ndarray, boxes: np.ndarray, scale: float, pad: tuple,
               orig_h: int, orig_w: int) -> np.ndarray:
    """在原图上绘制检测框 (从 letterbox 坐标转回原图坐标)"""
    img_vis = img_bgr.copy()
    pad_x, pad_y = pad

    for box in boxes:
        x1, y1, x2, y2, conf, cls_id = box
        # 去除 letterbox padding，还原到原图尺寸
        x1 = (x1 - pad_x) / scale
        y1 = (y1 - pad_y) / scale
        x2 = (x2 - pad_x) / scale
        y2 = (y2 - pad_y) / scale
        # 裁剪到图像边界
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(orig_w, x2), min(orig_h, y2)

        x1i, y1i, x2i, y2i = int(x1), int(y1), int(x2), int(y2)
        cls_name = CLASS_NAMES[int(cls_id)] if int(cls_id) < len(CLASS_NAMES) else str(int(cls_id))
        label = f"{cls_name} {conf:.2f}"

        cv2.rectangle(img_vis, (x1i, y1i), (x2i, y2i), (0, 0, 255), 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img_vis, (x1i, y1i - th - 4), (x1i + tw, y1i), (0, 0, 255), -1)
        cv2.putText(img_vis, label, (x1i, y1i - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return img_vis


def main():
    log.info("=" * 60)
    log.info("STEP 2-2: ONNX 推理验证 (50 张图)")
    log.info("=" * 60)

    # BP-A: 加载 ONNX 会话
    log.info(f"[BP-A] 加载 ONNX: {ONNX_PATH}")
    if not ONNX_PATH.exists():
        log.error("[BP-A] 找不到 best.onnx，请先运行 Step 1")
        sys.exit(1)

    import onnxruntime as ort
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    try:
        sess = ort.InferenceSession(str(ONNX_PATH), providers=providers)
        ep_used = sess.get_providers()[0]
    except Exception:
        sess = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
        ep_used = "CPUExecutionProvider"

    log.info(f"[BP-A] ONNX Runtime {ort.__version__} | Provider: {ep_used} ✓")

    input_name = sess.get_inputs()[0].name
    log.info(f"[BP-A] 输入名称: {input_name}, shape: {sess.get_inputs()[0].shape}")
    log.info(f"[BP-A] 输出数量: {len(sess.get_outputs())}")

    # 加载采样清单
    if not SAMPLE_JSON.exists():
        log.error("[BP-A] 找不到 sample_list.json，请先运行 01_sample_images.py")
        sys.exit(1)
    with SAMPLE_JSON.open(encoding="utf-8") as f:
        sample_data = json.load(f)
    samples = sample_data["samples"]
    log.info(f"[BP-A] 待验证图片: {len(samples)} 张")

    VIS_DIR.mkdir(parents=True, exist_ok=True)

    # BP-B/C/D: 逐图推理
    log.info("[BP-B] 开始逐图推理 ...")
    per_image_results = []
    total_det = 0
    total_time_ms = 0.0
    failed = 0

    for idx, sample in enumerate(samples, 1):
        img_path = Path(sample["sample_path"])
        if not img_path.exists():
            log.warning(f"[{idx:02d}] 图片不存在: {img_path.name}")
            failed += 1
            continue

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            log.warning(f"[{idx:02d}] 无法读取: {img_path.name}")
            failed += 1
            continue

        orig_h, orig_w = img_bgr.shape[:2]

        # 预处理
        inp, scale, pad = preprocess(img_bgr)

        # BP-B: ONNX 推理
        t0 = time.perf_counter()
        onnx_outs = sess.run(None, {input_name: inp})
        t1 = time.perf_counter()
        infer_ms = (t1 - t0) * 1000

        # BP-C: 解码 + NMS
        raw_boxes = decode_outputs(onnx_outs)
        final_boxes = nms(raw_boxes, IOU_THRES)
        n_det = len(final_boxes)

        total_det += n_det
        total_time_ms += infer_ms

        log.info(
            f"[{idx:02d}/{len(samples)}] {img_path.name[:40]:40s} "
            f"| {n_det:2d} det | {infer_ms:.1f} ms | {orig_w}x{orig_h}"
        )

        # BP-D: 绘制可视化
        vis = draw_boxes(img_bgr, final_boxes, scale, pad, orig_h, orig_w)
        vis_name = f"{idx:03d}_{sample['source']}_{img_path.stem}_vis.jpg"
        cv2.imwrite(str(VIS_DIR / vis_name), vis)

        per_image_results.append({
            "idx": idx,
            "source": sample["source"],
            "file": img_path.name,
            "orig_size": [orig_w, orig_h],
            "n_detections": n_det,
            "infer_ms": round(infer_ms, 2),
            "detections": [
                {
                    "x1": float(b[0]), "y1": float(b[1]),
                    "x2": float(b[2]), "y2": float(b[3]),
                    "conf": float(b[4]),
                    "class": CLASS_NAMES[int(b[5])] if int(b[5]) < len(CLASS_NAMES) else str(int(b[5])),
                }
                for b in final_boxes
            ],
            "vis_file": vis_name,
        })

    # BP-E: 生成汇总报告
    n_valid = len(per_image_results)
    avg_ms  = total_time_ms / n_valid if n_valid else 0
    log.info("\n" + "=" * 60)
    log.info(f"[BP-E] 验证完成: {n_valid} 张成功 | {failed} 张失败")
    log.info(f"       总检测框: {total_det} 个")
    log.info(f"       平均推理: {avg_ms:.1f} ms/张")
    log.info(f"       估计 FPS (纯推理): {1000/avg_ms:.1f}" if avg_ms > 0 else "")

    # 按来源统计
    source_stats: Dict[str, Dict] = {}
    for r in per_image_results:
        s = r["source"]
        if s not in source_stats:
            source_stats[s] = {"count": 0, "detections": 0, "images_with_det": 0}
        source_stats[s]["count"] += 1
        source_stats[s]["detections"] += r["n_detections"]
        if r["n_detections"] > 0:
            source_stats[s]["images_with_det"] += 1

    for s, st in source_stats.items():
        det_rate = st["images_with_det"] / st["count"] * 100
        log.info(f"       [{s}] {st['count']} 张 | {st['detections']} 框 | 检出率 {det_rate:.0f}%")

    report = {
        "model": str(ONNX_PATH),
        "onnxruntime_version": ort.__version__,
        "execution_provider": ep_used,
        "imgsz": IMGSZ,
        "conf_thres": CONF_THRES,
        "iou_thres": IOU_THRES,
        "classes": CLASS_NAMES,
        "total_images": len(samples),
        "successful": n_valid,
        "failed": failed,
        "total_detections": total_det,
        "avg_infer_ms": round(avg_ms, 2),
        "est_fps": round(1000 / avg_ms, 1) if avg_ms > 0 else 0,
        "source_stats": source_stats,
        "per_image": per_image_results,
    }
    REPORT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info(f"\n[BP-E] 验证报告: {REPORT_JSON} ✓")
    log.info(f"[BP-E] 可视化结果: {VIS_DIR}/ ✓")
    log.info("=" * 60)
    log.info("STEP 2-2 完成 ✅ — 请查看 results/ 目录中的检测可视化")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
