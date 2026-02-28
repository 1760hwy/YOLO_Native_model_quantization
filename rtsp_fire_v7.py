#!/usr/bin/env python3
# rtsp_fire_v7.py
"""
Fire Detection V7 - RDK X5 Real-time Inference
================================================
Supports USB camera and RTSP stream input.
Auto-adapts to NCHW/NHWC output formats.
Red bounding boxes with white text labels.

Usage:
    # USB camera
    python3 rtsp_fire_v7.py --source 0 --conf 0.3

    # RTSP camera
    python3 rtsp_fire_v7.py --source "rtsp://admin:pass@192.168.1.100:554/stream1"

    # Adjust threshold to reduce false positives
    python3 rtsp_fire_v7.py --source 0 --conf 0.5

Controls:
    q     - Quit
    +/=   - Increase confidence threshold
    -     - Decrease confidence threshold
"""
import os
import cv2
import numpy as np
import time
import argparse
from hobot_dnn import pyeasy_dnn as dnn

os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|max_delay;0"
)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


class FireDetectorV7:
    """
    YOLO11 Fire Detector for RDK X5.

    Features:
        - Auto-adaptive NCHW/NHWC output handling
        - DFL bbox decoding
        - Area filtering to reduce false positives
        - Red bounding boxes
    """

    def __init__(self, model_path, conf_thresh=0.30, nms_thresh=0.50,
                 class_names=("fire",)):
        self.conf_thresh = float(conf_thresh)
        self.nms_thresh = float(nms_thresh)
        self.reg_max = 16
        self.class_names = list(class_names)
        self.num_classes = len(self.class_names)

        # Load model
        models = dnn.load(model_path)
        self.model = models[0]

        # Parse input shape
        in_shape = list(self.model.inputs[0].properties.shape)
        if in_shape[1] == 3:  # NCHW
            self.input_h = int(in_shape[2])
            self.input_w = int(in_shape[3])
        else:  # NHWC
            self.input_h = int(in_shape[1])
            self.input_w = int(in_shape[2])

        print(f"Model loaded: {self.input_w}x{self.input_h}")
        print(f"Classes: {self.class_names}")

        # Parse output heads
        self.heads = self._parse_heads()

        # Pre-compute grid for each head
        for h in self.heads:
            H, W = h["hw"]
            gy, gx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
            h["grid"] = np.stack([gx, gy], axis=-1).reshape(-1, 2).astype(np.float32)

    def _parse_heads(self):
        """Parse model outputs into bbox/cls pairs by shape matching."""
        bbox_map, cls_map = {}, {}

        for idx, out in enumerate(self.model.outputs):
            shp = list(out.properties.shape)
            if len(shp) != 4:
                continue

            # Identify bbox outputs (channel dim = 64 = 4 * reg_max)
            if shp[1] == 64:       # NCHW: (1, 64, H, W)
                bbox_map[(shp[2], shp[3])] = (idx, "NCHW")
            elif shp[-1] == 64:    # NHWC: (1, H, W, 64)
                bbox_map[(shp[1], shp[2])] = (idx, "NHWC")
            # Identify cls outputs
            elif shp[1] in (1, self.num_classes) and shp[2] > 1:  # NCHW cls
                cls_map[(shp[2], shp[3])] = (idx, "NCHW")
            elif shp[-1] in (1, self.num_classes):  # NHWC cls
                cls_map[(shp[1], shp[2])] = (idx, "NHWC")

        heads = []
        for hw in sorted(bbox_map.keys(), key=lambda x: -x[0]):
            if hw not in cls_map:
                print(f"  ⚠️  No cls match for bbox {hw}")
                continue
            b_idx, b_fmt = bbox_map[hw]
            c_idx, c_fmt = cls_map[hw]
            stride = self.input_h // hw[0]
            heads.append({
                "bbox_idx": b_idx, "cls_idx": c_idx,
                "hw": hw, "stride": stride,
                "bbox_fmt": b_fmt, "cls_fmt": c_fmt,
            })

        print(f"Heads: {[(h['stride'], h['hw'], h['bbox_fmt']) for h in heads]}")
        return heads

    def bgr_to_nv12(self, img):
        """Convert BGR image to NV12 with letterbox padding."""
        h, w = img.shape[:2]
        scale = min(self.input_h / h, self.input_w / w)
        new_h, new_w = int(h * scale), int(w * scale)
        resized = cv2.resize(img, (new_w, new_h))

        canvas = np.full((self.input_h, self.input_w, 3), 114, dtype=np.uint8)
        top = (self.input_h - new_h) // 2
        left = (self.input_w - new_w) // 2
        canvas[top:top + new_h, left:left + new_w] = resized

        # BGR → YUV_I420 → NV12
        yuv = cv2.cvtColor(canvas, cv2.COLOR_BGR2YUV_I420)
        y = yuv[:self.input_h, :]
        u = yuv[self.input_h:self.input_h + self.input_h // 4, :].reshape(
            self.input_h // 2, self.input_w // 2)
        v = yuv[self.input_h + self.input_h // 4:, :].reshape(
            self.input_h // 2, self.input_w // 2)
        uv = np.stack([u, v], axis=-1).reshape(self.input_h // 2, self.input_w)
        nv12 = np.concatenate([y, uv], axis=0)
        return nv12, scale, left, top

    def dfl_decode(self, bbox_raw):
        """DFL decoding: 64 channels → 4 LTRB offsets."""
        bbox = bbox_raw.reshape(-1, 4, self.reg_max)
        bbox_exp = np.exp(bbox - np.max(bbox, axis=-1, keepdims=True))
        bbox_sm = bbox_exp / np.sum(bbox_exp, axis=-1, keepdims=True)
        weights = np.arange(self.reg_max, dtype=np.float32).reshape(1, 1, -1)
        return np.sum(bbox_sm * weights, axis=-1)

    def _extract_feat(self, out, idx, fmt, H, W, C):
        """Extract feature map, handling both NCHW and NHWC."""
        buf = np.array(out[idx].buffer, copy=False).astype(np.float32)
        if fmt == "NCHW":
            return buf.reshape(1, C, H, W).transpose(0, 2, 3, 1).reshape(-1, C)
        else:
            return buf.reshape(-1, C)

    def detect(self, frame):
        """Run detection on a single frame."""
        orig_h, orig_w = frame.shape[:2]
        nv12, scale, pad_left, pad_top = self.bgr_to_nv12(frame)
        outs = self.model.forward(nv12)

        all_boxes, all_scores, all_cls = [], [], []

        for h in self.heads:
            H, W = h["hw"]
            stride = h["stride"]
            grid = h["grid"]

            bbox_feat = self._extract_feat(
                outs, h["bbox_idx"], h["bbox_fmt"], H, W, 64)
            cls_feat = self._extract_feat(
                outs, h["cls_idx"], h["cls_fmt"], H, W, self.num_classes)

            # Sigmoid + fast threshold filtering
            scores = sigmoid(cls_feat)
            max_scores = np.max(scores, axis=1)
            keep = max_scores >= self.conf_thresh
            if not np.any(keep):
                continue

            max_cls = np.argmax(scores, axis=1)
            bbox_keep = bbox_feat[keep]
            score_keep = max_scores[keep]
            cls_keep = max_cls[keep]
            grid_keep = grid[keep]

            # DFL decode
            ltrb = self.dfl_decode(bbox_keep)

            # Convert to pixel coordinates
            xc = (grid_keep[:, 0] + 0.5) * stride
            yc = (grid_keep[:, 1] + 0.5) * stride

            x1 = np.clip((xc - ltrb[:, 0] * stride - pad_left) / scale, 0, orig_w)
            y1 = np.clip((yc - ltrb[:, 1] * stride - pad_top) / scale, 0, orig_h)
            x2 = np.clip((xc + ltrb[:, 2] * stride - pad_left) / scale, 0, orig_w)
            y2 = np.clip((yc + ltrb[:, 3] * stride - pad_top) / scale, 0, orig_h)

            # Filter tiny boxes (likely false positives)
            areas = (x2 - x1) * (y2 - y1)
            valid = areas > 100
            if not np.any(valid):
                continue

            boxes = np.stack([x1[valid], y1[valid], x2[valid], y2[valid]], axis=1)
            all_boxes.append(boxes)
            all_scores.append(score_keep[valid])
            all_cls.append(cls_keep[valid])

        if not all_boxes:
            return np.empty((0, 4)), np.empty((0,)), np.empty((0,), dtype=int)

        boxes = np.concatenate(all_boxes)
        scores = np.concatenate(all_scores)
        classes = np.concatenate(all_cls)

        # NMS
        idxs = cv2.dnn.NMSBoxes(
            boxes.tolist(), scores.tolist(),
            self.conf_thresh, self.nms_thresh
        )
        if len(idxs) > 0:
            idxs = np.array(idxs).flatten()
            return boxes[idxs], scores[idxs], classes[idxs]

        return np.empty((0, 4)), np.empty((0,)), np.empty((0,), dtype=int)

    def draw(self, frame, boxes, scores, classes):
        """Draw red bounding boxes with white text labels."""
        RED = (0, 0, 255)
        WHITE = (255, 255, 255)

        for box, score, cls in zip(boxes, scores, classes):
            x1, y1, x2, y2 = map(int, box)
            c = int(cls)
            name = self.class_names[c] if c < len(self.class_names) else str(c)
            label = f"{name}:{float(score):.2f}"

            # Red rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), RED, 2)

            # Red label background + white text
            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), RED, -1)
            cv2.putText(frame, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2)
        return frame


def main():
    parser = argparse.ArgumentParser(
        description="Fire Detection V7 - RDK X5",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  USB camera:     python3 rtsp_fire_v7.py --source 0
  RTSP camera:    python3 rtsp_fire_v7.py --source "rtsp://user:pass@ip:554/stream1"
  Higher thresh:  python3 rtsp_fire_v7.py --source 0 --conf 0.5
  Custom model:   python3 rtsp_fire_v7.py --model /path/to/model.bin --source 0
        """
    )
    parser.add_argument("--model", default="/home/sunrise/fire_detect.bin",
                        help="Path to quantized .bin model")
    parser.add_argument("--source", default="0",
                        help="Video source: 0=USB, rtsp://...=RTSP")
    parser.add_argument("--conf", type=float, default=0.30,
                        help="Confidence threshold (default: 0.30)")
    parser.add_argument("--nms", type=float, default=0.50,
                        help="NMS IoU threshold (default: 0.50)")
    parser.add_argument("--classes", nargs="+", default=["fire"],
                        help="Class names (default: fire)")
    args = parser.parse_args()

    print("=" * 60)
    print("  Fire Detection V7 - RDK X5")
    print("=" * 60)

    det = FireDetectorV7(
        args.model,
        conf_thresh=args.conf,
        nms_thresh=args.nms,
        class_names=args.classes,
    )

    # Open video source
    if args.source.startswith("rtsp"):
        cap = cv2.VideoCapture(args.source, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    else:
        cap = cv2.VideoCapture(int(args.source))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("ERROR: Cannot open video source")
        return

    print(f"\nSource: {args.source}")
    print(f"Conf: {args.conf}, NMS: {args.nms}")
    print(f"Classes: {args.classes}")
    print("Press 'q' to quit, '+'/'-' adjust threshold")
    print("-" * 60)

    fps_list = []
    frame_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            t0 = time.time()
            boxes, scores, classes = det.detect(frame)
            result = det.draw(frame, boxes, scores, classes)
            dt = time.time() - t0

            fps_list.append(1.0 / max(dt, 1e-6))
            if len(fps_list) > 30:
                fps_list.pop(0)
            fps = np.mean(fps_list)

            # HUD
            info = f"FPS:{fps:.1f} Fire:{len(boxes)} T:{det.conf_thresh:.2f}"
            cv2.putText(result, info, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.imshow("Fire Detection V7", result)

            frame_count += 1
            if frame_count % 100 == 0:
                print(f"Frame {frame_count}: FPS={fps:.1f}, Det={len(boxes)}")

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key in (ord('+'), ord('=')):
                det.conf_thresh = min(0.95, det.conf_thresh + 0.05)
                print(f"Threshold: {det.conf_thresh:.2f}")
            elif key == ord('-'):
                det.conf_thresh = max(0.05, det.conf_thresh - 0.05)
                print(f"Threshold: {det.conf_thresh:.2f}")

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if fps_list:
            print(f"\nAverage FPS: {np.mean(fps_list):.1f}")


if __name__ == "__main__":
    main()
