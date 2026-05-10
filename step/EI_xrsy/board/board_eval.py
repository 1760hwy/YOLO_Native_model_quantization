#!/usr/bin/env python3
"""
板端评估脚本 — 输出预测 txt 供 PC 端计算 mAP
用法：python3 board_eval.py --config E --model /path/to/model.bin --images /path/valid_images --out ./preds
输出：preds/config_E/stem.txt  每行 "conf x1 y1 x2 y2"（原图像素坐标）
"""
import argparse, cv2, numpy as np
from pathlib import Path

REG_MAX     = 16
NC          = 1
CONF_THRESH = 0.25
IOU_THRESH  = 0.45
INPUT_SIZE  = 640


def bgr_to_nv12(bgr, size=INPUT_SIZE):
    ih, iw = bgr.shape[:2]
    scale  = min(size / ih, size / iw)
    nh, nw = int(ih * scale), int(iw * scale)
    resized = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    padded  = np.full((size, size, 3), 114, dtype=np.uint8)
    pt, pl  = (size - nh) // 2, (size - nw) // 2
    padded[pt:pt+nh, pl:pl+nw] = resized
    yuv  = cv2.cvtColor(padded, cv2.COLOR_BGR2YUV_I420)
    y_pl = yuv[:size, :]
    u_pl = yuv[size:size + size//4, :].reshape(size//2, size//2)
    v_pl = yuv[size + size//4:,    :].reshape(size//2, size//2)
    uv_pl = np.empty((size // 2, size), dtype=np.uint8)
    uv_pl[:, 0::2] = u_pl
    uv_pl[:, 1::2] = v_pl
    return np.vstack([y_pl, uv_pl]), scale, pt, pl


def _sigmoid(x): return 1.0 / (1.0 + np.exp(-np.clip(x, -88, 88)))
def _softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True); e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)

def _dfl(raw):
    n = raw.shape[0]; raw = raw.reshape(n, 4, REG_MAX)
    raw = _softmax(raw, axis=-1)
    return (raw * np.arange(REG_MAX, dtype=np.float32)).sum(-1)

def _make_anchors(fh, fw, stride):
    gy, gx = np.mgrid[0:fh, 0:fw]
    pts = np.stack([gx + 0.5, gy + 0.5], -1).reshape(-1, 2).astype(np.float32)
    return pts, np.full((fh * fw, 1), stride, dtype=np.float32)

def _dist2bbox(dist, anchors):
    lt, rb = dist[:, :2], dist[:, 2:]
    return np.concatenate([anchors + (rb - lt) / 2, lt + rb], axis=-1)

def _nms(boxes, scores, iou_thr):
    if not len(boxes): return []
    x1,y1,x2,y2 = boxes[:,0],boxes[:,1],boxes[:,2],boxes[:,3]
    areas = (x2-x1)*(y2-y1); order = scores.argsort()[::-1]; keep = []
    while len(order):
        i = order[0]; keep.append(i)
        if len(order) == 1: break
        xx1 = np.maximum(x1[i], x1[order[1:]]); yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]]); yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2-xx1) * np.maximum(0, yy2-yy1)
        iou   = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[np.where(iou <= iou_thr)[0] + 1]
    return keep

def _decode_branch(bbox_data, cls_data, stride, scale, pt, pl, orig_h, orig_w):
    fh, fw = bbox_data.shape[:2]
    anchors, strides_t = _make_anchors(fh, fw, stride)
    dist  = _dfl(bbox_data.reshape(-1, REG_MAX * 4))
    dbox  = _dist2bbox(dist, anchors) * strides_t
    confs = _sigmoid(cls_data.reshape(-1, NC)[:, 0])
    mask  = confs > CONF_THRESH
    dbox, confs = dbox[mask], confs[mask]
    if not len(dbox): return np.empty((0,4)), np.empty(0)
    cx, cy, bw, bh = dbox[:,0], dbox[:,1], dbox[:,2], dbox[:,3]
    x1 = ((cx - bw/2 - pl) / scale).clip(0, orig_w)
    y1 = ((cy - bh/2 - pt) / scale).clip(0, orig_h)
    x2 = ((cx + bw/2 - pl) / scale).clip(0, orig_w)
    y2 = ((cy + bh/2 - pt) / scale).clip(0, orig_h)
    return np.stack([x1, y1, x2, y2], -1), confs

def decode_shape(outputs, scale, pt, pl, orig_h, orig_w):
    bbox_map, cls_map = {}, {}
    for out in outputs:
        buf = np.array(out.buffer[0]); h, w, c = buf.shape
        stride = INPUT_SIZE // max(h, w)
        if   c == REG_MAX * 4: bbox_map[stride] = buf
        elif c == NC:          cls_map[stride]  = buf
    ab, ac = [], []
    for s in [8, 16, 32]:
        if s not in bbox_map or s not in cls_map: continue
        b, c = _decode_branch(bbox_map[s], cls_map[s], s, scale, pt, pl, orig_h, orig_w)
        if len(b): ab.append(b); ac.append(c)
    if not ab: return np.empty((0,4)), np.empty(0)
    boxes = np.concatenate(ab); confs = np.concatenate(ac)
    keep  = _nms(boxes, confs, IOU_THRESH)
    return boxes[keep], confs[keep]

def decode_fixed(outputs, scale, pt, pl, orig_h, orig_w):
    ab, ac = [], []
    try:
        for i, s in enumerate([8, 16, 32]):
            b, c = _decode_branch(np.array(outputs[i].buffer[0]),
                                  np.array(outputs[i+3].buffer[0]),
                                  s, scale, pt, pl, orig_h, orig_w)
            if len(b): ab.append(b); ac.append(c)
    except (IndexError, ValueError) as e:
        print(f"  [WARN] fixed-index: {e}")
        return np.empty((0,4)), np.empty(0)
    if not ab: return np.empty((0,4)), np.empty(0)
    boxes = np.concatenate(ab); confs = np.concatenate(ac)
    keep  = _nms(boxes, confs, IOU_THRESH)
    return boxes[keep], confs[keep]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",  required=True, choices=list("ABCDE"))
    ap.add_argument("--model",   required=True)
    ap.add_argument("--images",  default="./valid_images")
    ap.add_argument("--out",     default="./preds")
    args = ap.parse_args()

    use_shape = args.config in ("C", "E")

    from hobot_dnn import pyeasy_dnn as dnn
    print(f"[Config {args.config}] loading {args.model} …")
    model = dnn.load([args.model])[0]

    img_paths = sorted(Path(args.images).glob("*.jpg"))
    out_dir   = Path(args.out) / f"config_{args.config}"
    out_dir.mkdir(parents=True, exist_ok=True)

    n_det = 0
    for p in img_paths:
        bgr = cv2.imread(str(p))
        if bgr is None: continue
        orig_h, orig_w = bgr.shape[:2]
        nv12, scale, pt, pl = bgr_to_nv12(bgr)
        outputs = model.forward(nv12.astype(np.uint8))

        if use_shape:
            boxes, confs = decode_shape(outputs, scale, pt, pl, orig_h, orig_w)
        else:
            boxes, confs = decode_fixed(outputs, scale, pt, pl, orig_h, orig_w)

        txt = out_dir / (p.stem + ".txt")
        with open(txt, "w") as f:
            for box, conf in zip(boxes, confs):
                f.write(f"{conf:.6f} {box[0]:.2f} {box[1]:.2f} {box[2]:.2f} {box[3]:.2f}\n")
        n_det += len(boxes)

    print(f"[Config {args.config}] done — {len(img_paths)} imgs, {n_det} dets → {out_dir}")


if __name__ == "__main__":
    main()
