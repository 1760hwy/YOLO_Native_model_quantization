#!/usr/bin/env python3
"""
ONNX Float32 validation - run fire_detect.onnx on BoWFireDataset test images,
save result images with boxes to step4/onnx_validation/
"""
import sys, json
import numpy as np
import cv2
import onnxruntime as ort
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
ONNX_PATH  = SCRIPT_DIR / "fire_detect.onnx"
IMG_DIR    = SCRIPT_DIR.parents[1] / "BDF-18K/公开数据集/PublicDataset1_BoWFireDataset/test/images"
OUT_DIR    = SCRIPT_DIR / "onnx_validation"
IMGSZ      = 640
REG_MAX    = 16
CONF_THRES = 0.25
IOU_THRES  = 0.45
CLASS_NAMES = ["fire"]


def letterbox(img, size=640, pad_val=114):
    h, w = img.shape[:2]
    scale = min(size / h, size / w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), pad_val, dtype=np.uint8)
    top  = (size - nh) // 2
    left = (size - nw) // 2
    canvas[top:top + nh, left:left + nw] = img
    return canvas, scale, (left, top)


def preprocess(img_bgr):
    lb, scale, pad = letterbox(img_bgr)
    rgb = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    nchw = rgb.transpose(2, 0, 1)[np.newaxis]  # [1,3,640,640]
    return nchw, scale, pad


def dfl_decode(pred):
    n, _ = pred.shape
    pred = pred.reshape(n, 4, REG_MAX)
    exp  = np.exp(pred - pred.max(axis=-1, keepdims=True))
    prob = exp / exp.sum(axis=-1, keepdims=True)
    return (prob * np.arange(REG_MAX, dtype=np.float32)).sum(axis=-1)


def make_anchors(h, w):
    ys = np.arange(h, dtype=np.float32) + 0.5
    xs = np.arange(w, dtype=np.float32) + 0.5
    gy, gx = np.meshgrid(ys, xs, indexing="ij")
    return np.stack([gx, gy], axis=-1).reshape(-1, 2)


def decode_outputs(outs, conf_thres=0.25):
    """
    outs: list of 6 ONNX outputs
    Sorts by shape: C=64 → bbox, C=1 → cls
    Returns [N, 6] boxes: x1,y1,x2,y2,conf,cls_id
    """
    strides    = [8, 16, 32]
    expected_h = {IMGSZ // s: s for s in strides}
    bbox_map, cls_map = {}, {}

    for o in outs:
        # ONNX outputs may be NCHW or NHWC - detect by checking dims
        arr = o  # numpy array already
        if arr.ndim == 4:
            # heuristic: if shape[1] in {64,1} → NCHW, else NHWC
            if arr.shape[1] in (REG_MAX * 4, len(CLASS_NAMES)):
                # NCHW [N,C,H,W] → convert to [N,H,W,C]
                arr = arr.transpose(0, 2, 3, 1)
        h = arr.shape[1]
        c = arr.shape[3]
        if h not in expected_h:
            continue
        if c == REG_MAX * 4:
            bbox_map[h] = arr
        elif c == len(CLASS_NAMES):
            cls_map[h] = arr

    if len(bbox_map) != 3 or len(cls_map) != 3:
        print(f"[ERROR] output parse failed: bbox keys={list(bbox_map.keys())} cls keys={list(cls_map.keys())}")
        print("  Shapes of raw outputs:")
        for i, o in enumerate(outs):
            print(f"    out[{i}]: {o.shape}")
        sys.exit(1)

    all_boxes = []
    for s in strides:
        h_key = IMGSZ // s
        bf = bbox_map[h_key][0].reshape(-1, REG_MAX * 4)   # [HW, 64]
        cf = cls_map[h_key][0].reshape(-1, len(CLASS_NAMES))  # [HW, 1]

        cf_min, cf_max = float(cf.min()), float(cf.max())
        if cf_min >= 0.0 and cf_max <= 1.0:
            cp = cf
        else:
            cp = 1.0 / (1.0 + np.exp(-cf))

        print(f"  stride={s}: cls logit range [{cf_min:.3f}, {cf_max:.3f}]  "
              f"max_sigmoid={float(cp.max()):.4f}")

        ms, mc = cp.max(axis=-1), cp.argmax(axis=-1)
        mask = ms > conf_thres
        print(f"  stride={s}: candidates(conf>{conf_thres})={mask.sum()}")

        if not mask.any():
            continue

        h_f, w_f = bbox_map[h_key].shape[1], bbox_map[h_key].shape[2]
        anc  = make_anchors(h_f, w_f)[mask]
        dist = dfl_decode(bf[mask])
        x1   = (anc[:, 0] - dist[:, 0]) * s
        y1   = (anc[:, 1] - dist[:, 1]) * s
        x2   = (anc[:, 0] + dist[:, 2]) * s
        y2   = (anc[:, 1] + dist[:, 3]) * s
        all_boxes.append(
            np.stack([x1, y1, x2, y2, ms[mask], mc[mask].astype(np.float32)], 1)
        )

    if not all_boxes:
        return np.zeros((0, 6), np.float32)
    return np.concatenate(all_boxes, 0)


def nms(boxes, iou_thres=0.45):
    if len(boxes) == 0:
        return boxes
    keep_all = []
    for cls_id in np.unique(boxes[:, 5]):
        b = boxes[boxes[:, 5] == cls_id]
        x1, y1, x2, y2, s = b[:,0],b[:,1],b[:,2],b[:,3],b[:,4]
        order = s.argsort()[::-1]
        keep = []
        while order.size:
            i = order[0]; keep.append(i)
            if order.size == 1: break
            iw = (np.minimum(x2[i], x2[order[1:]]) - np.maximum(x1[i], x1[order[1:]])).clip(0)
            ih = (np.minimum(y2[i], y2[order[1:]]) - np.maximum(y1[i], y1[order[1:]])).clip(0)
            inter = iw * ih
            union = ((x2[i]-x1[i])*(y2[i]-y1[i])
                     + (x2[order[1:]]-x1[order[1:]])*(y2[order[1:]]-y1[order[1:]])
                     - inter + 1e-6)
            order = order[np.concatenate(([0], np.where(inter/union <= iou_thres)[0]+1))][1:]
        keep_all.append(b[keep])
    return np.concatenate(keep_all)


def draw_boxes(img, boxes, scale, pad):
    vis = img.copy()
    oh, ow = img.shape[:2]
    for b in boxes:
        x1 = max(0,  (b[0] - pad[0]) / scale)
        y1 = max(0,  (b[1] - pad[1]) / scale)
        x2 = min(ow, (b[2] - pad[0]) / scale)
        y2 = min(oh, (b[3] - pad[1]) / scale)
        label = f"{CLASS_NAMES[int(b[5])]} {b[4]:.2f}"
        cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(vis, (int(x1), int(y1)-th-6), (int(x1)+tw+2, int(y1)), (0, 0, 255), -1)
        cv2.putText(vis, label, (int(x1)+1, int(y1)-3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
    return vis


def main():
    if sys.platform == "win32":
        try: sys.stdout.reconfigure(encoding="utf-8")
        except: pass

    print(f"ONNX: {ONNX_PATH}")
    print(f"Images: {IMG_DIR}")
    print(f"Output: {OUT_DIR}")
    print(f"conf={CONF_THRES}  iou={IOU_THRES}")
    print("=" * 60)

    if not ONNX_PATH.exists():
        print(f"[ERROR] ONNX not found: {ONNX_PATH}"); sys.exit(1)

    sess = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    print(f"Input name: {input_name}")
    print(f"Outputs: {[o.name for o in sess.get_outputs()]}")
    print(f"Output shapes: {[o.shape for o in sess.get_outputs()]}")
    print()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    img_paths = sorted(IMG_DIR.glob("*.png"))[:20]  # first 20 images
    if not img_paths:
        img_paths = sorted(list(IMG_DIR.glob("*.jpg")) + list(IMG_DIR.glob("*.png")))[:20]
    print(f"Found {len(img_paths)} images\n")

    records = []
    total_det = 0

    for idx, p in enumerate(img_paths):
        img = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            print(f"  [SKIP] {p.name}"); continue

        inp, scale, pad = preprocess(img)
        outs = sess.run(None, {input_name: inp})

        print(f"[{idx+1:02d}] {p.name}")
        boxes = decode_outputs(outs, CONF_THRES)
        boxes = nms(boxes, IOU_THRES)
        total_det += len(boxes)
        print(f"  → {len(boxes)} detections after NMS")

        vis = draw_boxes(img, boxes, scale, pad)
        out_name = f"{idx+1:03d}_{p.stem}_onnx.jpg"
        cv2.imencode(".jpg", vis)[1].tofile(str(OUT_DIR / out_name))

        det_list = [{
            "class": CLASS_NAMES[int(b[5])],
            "conf":  round(float(b[4]), 4),
            "x1": round(float((b[0]-pad[0])/scale), 1),
            "y1": round(float((b[1]-pad[1])/scale), 1),
            "x2": round(float((b[2]-pad[0])/scale), 1),
            "y2": round(float((b[3]-pad[1])/scale), 1),
        } for b in boxes]
        records.append({"img": p.name, "n_det": len(boxes), "detections": det_list})
        print()

    print("=" * 60)
    print(f"Total: {len(records)} images,  {total_det} detections")
    print(f"Results saved to: {OUT_DIR}")

    report = {"onnx": str(ONNX_PATH.name), "conf": CONF_THRES, "iou": IOU_THRES,
              "total_images": len(records), "total_detections": total_det, "results": records}
    (OUT_DIR / "onnx_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
