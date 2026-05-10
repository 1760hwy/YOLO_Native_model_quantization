#!/usr/bin/env python3
"""
Step 3-6: RDK X5 板端推理脚本
运行环境: RDK X5 开发板 (AArch64, Python 3.10)

功能:
  - 加载 .bin BPU 模型
  - 读取视频流或图片进行推理
  - 后处理: DFL 解码 + NMS
  - 可视化显示检测结果

运行命令:
  # 单张图片测试
  python3 run_on_rdkx5.py --input test.jpg --model fire_detect_bayese_640x640_nv12.bin

  # 视频文件
  python3 run_on_rdkx5.py --input video.mp4 --model fire_detect.bin

  # USB 摄像头 (实时)
  python3 run_on_rdkx5.py --input 0 --model fire_detect.bin --show

依赖安装 (RDK X5):
  sudo apt install python3-bpu-pydev
  pip3 install numpy opencv-python

断点说明:
  BP-A: 加载 .bin 模型
  BP-B: 读取输入 + 预处理 (BGR→NV12)
  BP-C: BPU 推理
  BP-D: DFL 解码 + NMS
  BP-E: 可视化保存
"""

import sys
import time
import argparse
import logging
import numpy as np
import cv2
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("rdkx5_infer")

# ── 推理参数 ──────────────────────────────────────────────────
IMGSZ      = 640
CONF_THRES = 0.25
IOU_THRES  = 0.45
CLASS_NAMES = ["fire"]
REG_MAX    = 16


def bgr_to_nv12(img_bgr: np.ndarray, size: int) -> np.ndarray:
    """BGR → Letterbox resize → NV12 (YUV420SP)"""
    h, w = img_bgr.shape[:2]
    scale = min(size / h, size / w)
    nh, nw = int(h * scale), int(w * scale)
    img = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    top  = (size - nh) // 2
    left = (size - nw) // 2
    canvas[top:top+nh, left:left+nw] = img

    yuv = cv2.cvtColor(canvas, cv2.COLOR_BGR2YUV_I420)
    nv12 = np.zeros_like(yuv)
    nv12[:size, :] = yuv[:size, :]          # Y plane
    u = yuv[size:size + size//4, :].reshape(-1)
    v = yuv[size + size//4:, :].reshape(-1)
    uv = np.stack([u, v], axis=1).reshape(-1)
    nv12[size:, :] = uv.reshape(size//2, size)
    return nv12, scale, (left, top)


def dfl_decode(pred: np.ndarray) -> np.ndarray:
    n = pred.shape[0]
    pred = pred.reshape(n, 4, REG_MAX)
    exp = np.exp(pred - pred.max(axis=-1, keepdims=True))
    prob = exp / exp.sum(axis=-1, keepdims=True)
    bins = np.arange(REG_MAX, dtype=np.float32)
    return (prob * bins).sum(axis=-1)


def make_anchors(h, w):
    ys = np.arange(h, dtype=np.float32) + 0.5
    xs = np.arange(w, dtype=np.float32) + 0.5
    gy, gx = np.meshgrid(ys, xs, indexing="ij")
    return np.stack([gx, gy], axis=-1).reshape(-1, 2)


def decode_6outputs(outputs: list) -> np.ndarray:
    strides = [8, 16, 32]
    all_boxes = []
    for i, stride in enumerate(strides):
        # bbox: NHWC → [HW, 64]
        bbox = outputs[i]
        if bbox.ndim == 4:
            bbox = bbox[0]          # [H,W,64]
        h, w = bbox.shape[:2]
        bbox = bbox.reshape(-1, REG_MAX * 4)

        # cls: NHWC → [HW, nc]
        cls = outputs[i + 3]
        if cls.ndim == 4:
            cls = cls[0]
        cls = cls.reshape(-1, len(CLASS_NAMES))
        cls_prob = 1.0 / (1.0 + np.exp(-cls))

        max_score = cls_prob.max(axis=-1)
        max_cls   = cls_prob.argmax(axis=-1)
        mask = max_score > CONF_THRES
        if not mask.any():
            continue

        anchors = make_anchors(h, w)[mask]
        dist = dfl_decode(bbox[mask])
        scores = max_score[mask]
        cls_ids = max_cls[mask]

        x1 = (anchors[:, 0] - dist[:, 0]) * stride
        y1 = (anchors[:, 1] - dist[:, 1]) * stride
        x2 = (anchors[:, 0] + dist[:, 2]) * stride
        y2 = (anchors[:, 1] + dist[:, 3]) * stride
        boxes = np.stack([x1, y1, x2, y2, scores, cls_ids.astype(np.float32)], axis=1)
        all_boxes.append(boxes)

    if not all_boxes:
        return np.zeros((0, 6))
    return np.concatenate(all_boxes, axis=0)


def nms(boxes: np.ndarray, iou_thres: float) -> np.ndarray:
    if len(boxes) == 0:
        return boxes
    keep_all = []
    for cls_id in np.unique(boxes[:, 5]):
        b = boxes[boxes[:, 5] == cls_id]
        x1, y1, x2, y2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
        order = b[:, 4].argsort()[::-1]
        keep = []
        while order.size:
            i = order[0]
            keep.append(i)
            if order.size == 1: break
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            iou = (w * h) / ((x2[i]-x1[i])*(y2[i]-y1[i]) +
                              (x2[order[1:]]-x1[order[1:]])*(y2[order[1:]]-y1[order[1:]]) -
                              w*h + 1e-6)
            order = order[np.concatenate(([0], np.where(iou <= iou_thres)[0] + 1))][1:]
        keep_all.append(b[keep])
    return np.concatenate(keep_all) if keep_all else np.zeros((0, 6))


def draw(img_bgr, boxes, scale, pad, orig_h, orig_w):
    vis = img_bgr.copy()
    pad_x, pad_y = pad
    for box in boxes:
        x1 = max(0, (box[0] - pad_x) / scale)
        y1 = max(0, (box[1] - pad_y) / scale)
        x2 = min(orig_w, (box[2] - pad_x) / scale)
        y2 = min(orig_h, (box[3] - pad_y) / scale)
        conf, cls_id = float(box[4]), int(box[5])
        name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else str(cls_id)
        label = f"{name} {conf:.2f}"
        cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
        cv2.putText(vis, label, (int(x1), int(y1) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return vis


def main():
    parser = argparse.ArgumentParser(description="RDK X5 Fire Detection")
    parser.add_argument("--model",  required=True, help=".bin 模型路径")
    parser.add_argument("--input",  required=True, help="图片/视频路径或摄像头 ID")
    parser.add_argument("--output", default="output", help="输出目录")
    parser.add_argument("--show",   action="store_true", help="实时显示")
    parser.add_argument("--conf",   type=float, default=CONF_THRES)
    args = parser.parse_args()

    global CONF_THRES
    CONF_THRES = args.conf

    log.info("=" * 50)
    log.info(f"STEP 3-6: RDK X5 推理")
    log.info(f"  模型: {args.model}")
    log.info(f"  输入: {args.input}")
    log.info("=" * 50)

    # BP-A: 加载 BPU 模型
    log.info("[BP-A] 加载 BPU 模型 ...")
    try:
        from hobot_dnn import pyeasy_dnn as dnn
    except ImportError:
        log.error("[BP-A] 未找到 hobot_dnn，请在 RDK X5 上运行")
        log.error("       安装: sudo apt install python3-bpu-pydev")
        sys.exit(1)

    models = dnn.load(args.model)
    model = models[0]
    log.info(f"[BP-A] 模型加载成功 ✓")
    log.info(f"       输入: {model.inputs[0].properties.shape}")
    log.info(f"       输出数量: {len(model.outputs)}")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 打开输入源
    try:
        src_id = int(args.input)
        cap = cv2.VideoCapture(src_id)
        is_video = True
    except ValueError:
        inp = Path(args.input)
        if inp.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
            is_video = False
        else:
            cap = cv2.VideoCapture(str(inp))
            is_video = True

    frame_idx = 0
    total_time = 0.0

    def process_frame(img_bgr):
        nonlocal frame_idx, total_time
        orig_h, orig_w = img_bgr.shape[:2]

        # BP-B: 预处理 BGR → NV12
        nv12_data, scale, pad = bgr_to_nv12(img_bgr, IMGSZ)

        # 构建输入 tensor
        input_tensor = dnn.pyDNNTensor(
            nv12_data,
            model.inputs[0].properties
        )

        # BP-C: BPU 推理
        t0 = time.perf_counter()
        outputs = model.forward([input_tensor])
        t1 = time.perf_counter()
        infer_ms = (t1 - t0) * 1000
        total_time += infer_ms

        # 取出 numpy 结果
        out_arrays = [o.buffer for o in outputs]

        # BP-D: 解码 + NMS
        raw_boxes = decode_6outputs(out_arrays)
        final_boxes = nms(raw_boxes, IOU_THRES)
        n_det = len(final_boxes)

        log.info(f"[{frame_idx:04d}] {n_det} det | {infer_ms:.1f} ms")

        # BP-E: 可视化
        vis = draw(img_bgr, final_boxes, scale, pad, orig_h, orig_w)
        if not is_video:
            out_path = out_dir / f"result_{inp.stem}.jpg"
            cv2.imwrite(str(out_path), vis)
            log.info(f"[BP-E] 结果保存: {out_path}")
        else:
            out_path = out_dir / f"frame_{frame_idx:06d}.jpg"
            cv2.imwrite(str(out_path), vis)

        if args.show:
            cv2.imshow("Fire Detection - RDK X5", vis)

        frame_idx += 1
        return final_boxes

    if not is_video:
        img = cv2.imread(args.input)
        if img is None:
            log.error(f"无法读取图片: {args.input}")
            sys.exit(1)
        process_frame(img)
    else:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            process_frame(frame)
            if args.show and cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()

    if frame_idx > 0:
        avg_ms = total_time / frame_idx
        log.info(f"\n平均推理: {avg_ms:.1f} ms/帧 | FPS: {1000/avg_ms:.1f}")

    if args.show:
        cv2.destroyAllWindows()

    log.info("=" * 50)
    log.info("✅ 推理完成!")
    log.info("=" * 50)


if __name__ == "__main__":
    main()
