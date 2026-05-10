#!/usr/bin/env python3
"""
板端推理脚本（RDK X5 / Bayes-e BPU）
用法：
  python3 run_board_inference.py --config A --model /path/to/model.bin --images /path/to/test_images
  python3 run_board_inference.py --config E --model /path/to/model.bin --images /path/to/test_images

支持的 config：
  A  — 固定索引解码（基线，通常解码失败）
  B  — 固定索引解码（INT16 Softmax 模型）
  C  — 形状驱动解码 + 错误校准模型
  D  — 固定索引解码 + 正确校准模型
  E  — 形状驱动解码 + 正确校准模型（完整方案）

输出：results/<config>/xxx.jpg（含检测框标注）
"""
import argparse, cv2, numpy as np
from pathlib import Path

REG_MAX    = 16
NC         = 1        # 单类：火焰
CONF_THRESH = 0.25
IOU_THRESH  = 0.45
INPUT_SIZE  = 640


# ── NV12 转换 ─────────────────────────────────────────────────
def bgr_to_nv12(bgr: np.ndarray, size: int = INPUT_SIZE):
    """letterbox → BGR → NV12，返回 nv12 数组及 padding 参数"""
    ih, iw = bgr.shape[:2]
    scale  = min(size / ih, size / iw)
    nh, nw = int(ih * scale), int(iw * scale)
    resized = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    padded  = np.full((size, size, 3), 114, dtype=np.uint8)
    pt, pl  = (size - nh) // 2, (size - nw) // 2
    padded[pt:pt+nh, pl:pl+nw] = resized

    yuv   = cv2.cvtColor(padded, cv2.COLOR_BGR2YUV_I420)
    y_pl  = yuv[:size, :]                                           # [640, 640]
    # I420: U rows are (size//4 rows × size cols) = size//2 × size//2 pixels
    u_pl  = yuv[size:size + size//4, :].reshape(size//2, size//2)  # [320, 320]
    v_pl  = yuv[size + size//4:,    :].reshape(size//2, size//2)   # [320, 320]
    uv_pl = np.empty((size // 2, size), dtype=np.uint8)
    uv_pl[:, 0::2] = u_pl
    uv_pl[:, 1::2] = v_pl
    nv12  = np.vstack([y_pl, uv_pl])                               # [960, 640]
    return nv12, scale, pt, pl


# ── 纯 numpy 后处理 ───────────────────────────────────────────
def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -88, 88)))

def _softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x); return e / e.sum(axis=axis, keepdims=True)

def _dfl(raw, reg_max=REG_MAX):
    """raw: [N, 4*reg_max] → [N, 4]（DFL decode）"""
    n = raw.shape[0]
    raw = raw.reshape(n, 4, reg_max)
    raw = _softmax(raw, axis=-1)
    proj = np.arange(reg_max, dtype=np.float32)
    return (raw * proj).sum(-1)

def _make_anchors(fh, fw, stride, offset=0.5):
    gy, gx = np.mgrid[0:fh, 0:fw]
    pts = np.stack([gx + offset, gy + offset], -1).reshape(-1, 2).astype(np.float32)
    return pts, np.full((fh * fw, 1), stride, dtype=np.float32)

def _dist2bbox(dist, anchors):
    """dist: [N,4] lt/rb → xywh"""
    lt, rb = dist[:, :2], dist[:, 2:]
    c  = anchors + (rb - lt) / 2
    wh = lt + rb
    return np.concatenate([c, wh], axis=-1)

def _nms(boxes, scores, iou_thr):
    if len(boxes) == 0: return []
    x1,y1,x2,y2 = boxes[:,0],boxes[:,1],boxes[:,2],boxes[:,3]
    areas = (x2-x1)*(y2-y1)
    order = scores.argsort()[::-1]
    keep  = []
    while len(order):
        i = order[0]; keep.append(i)
        if len(order) == 1: break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2-xx1) * np.maximum(0, yy2-yy1)
        iou   = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[np.where(iou <= iou_thr)[0] + 1]
    return keep

def _decode_branch(bbox_data, cls_data, stride, scale, pt, pl, orig_h, orig_w):
    """解码单个 stride 分支，返回 (boxes_xyxy, confs)"""
    fh, fw = bbox_data.shape[:2]
    anchors, strides_t = _make_anchors(fh, fw, stride)
    bbox_flat = bbox_data.reshape(-1, REG_MAX * 4)
    cls_flat  = cls_data.reshape(-1, NC)
    dist  = _dfl(bbox_flat)
    dbox  = _dist2bbox(dist, anchors) * strides_t      # xywh in padded 640 space
    confs = _sigmoid(cls_flat[:, 0])

    mask = confs > CONF_THRESH
    dbox, confs = dbox[mask], confs[mask]
    if len(dbox) == 0: return np.empty((0,4)), np.empty(0)

    cx, cy, bw, bh = dbox[:,0], dbox[:,1], dbox[:,2], dbox[:,3]
    x1 = (cx - bw/2 - pl) / scale; y1 = (cy - bh/2 - pt) / scale
    x2 = (cx + bw/2 - pl) / scale; y2 = (cy + bh/2 - pt) / scale
    boxes = np.stack([x1, y1, x2, y2], -1).clip(0)
    boxes[:, [0,2]] = boxes[:, [0,2]].clip(0, orig_w)
    boxes[:, [1,3]] = boxes[:, [1,3]].clip(0, orig_h)
    return boxes, confs


# ── 形状驱动解码 ──────────────────────────────────────────────
def decode_shape_driven(outputs, scale, pt, pl, orig_h, orig_w):
    """根据通道数识别 bbox/cls 分支，与输出顺序无关"""
    bbox_map, cls_map = {}, {}
    for out in outputs:
        buf = np.array(out.buffer[0])          # [H, W, C]
        h, w, c = buf.shape
        stride = INPUT_SIZE // max(h, w)
        if   c == REG_MAX * 4:  bbox_map[stride] = buf
        elif c == NC:           cls_map[stride]  = buf

    all_boxes, all_confs = [], []
    for stride in [8, 16, 32]:
        if stride not in bbox_map or stride not in cls_map:
            continue
        boxes, confs = _decode_branch(
            bbox_map[stride], cls_map[stride], stride,
            scale, pt, pl, orig_h, orig_w)
        if len(boxes): all_boxes.append(boxes); all_confs.append(confs)

    if not all_boxes:
        return []
    boxes  = np.concatenate(all_boxes)
    confs  = np.concatenate(all_confs)
    keep   = _nms(boxes, confs, IOU_THRESH)
    return [(boxes[i], float(confs[i])) for i in keep]


# ── 固定索引解码（基线对照，Config A/B/D）────────────────────
def decode_fixed_index(outputs, scale, pt, pl, orig_h, orig_w):
    """假定 outputs[0,1,2]=bbox，outputs[3,4,5]=cls（可能失败）"""
    strides = [8, 16, 32]
    all_boxes, all_confs = [], []
    try:
        for i, stride in enumerate(strides):
            bbox_buf = np.array(outputs[i].buffer[0])       # [H,W,64]
            cls_buf  = np.array(outputs[i + 3].buffer[0])   # [H,W,1]
            boxes, confs = _decode_branch(
                bbox_buf, cls_buf, stride,
                scale, pt, pl, orig_h, orig_w)
            if len(boxes): all_boxes.append(boxes); all_confs.append(confs)
    except (IndexError, ValueError) as e:
        print(f"  [WARN] Fixed-index decode error: {e}  → 0 detections")
        return []

    if not all_boxes: return []
    boxes = np.concatenate(all_boxes)
    confs = np.concatenate(all_confs)
    keep  = _nms(boxes, confs, IOU_THRESH)
    return [(boxes[i], float(confs[i])) for i in keep]


# ── 画框 ──────────────────────────────────────────────────────
def draw_detections(bgr: np.ndarray, dets: list, color=(0, 0, 220)) -> np.ndarray:
    img = bgr.copy()
    for box, conf in dets:
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = f"fire {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, max(y1-th-4, 0)), (x1+tw+2, y1), color, -1)
        cv2.putText(img, label, (x1+1, max(y1-3, th)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
    return img


# ── 主函数 ────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, choices=["A","B","C","D","E"],
                    help="消融配置（A=基线, E=完整方案）")
    ap.add_argument("--model",  required=True, help=".bin 模型路径")
    ap.add_argument("--images", default="./test_images", help="测试图片目录")
    ap.add_argument("--out",    default="./results",     help="输出根目录")
    ap.add_argument("--conf",   type=float, default=0.25)
    args = ap.parse_args()

    CONF_THRESH = args.conf
    use_shape = args.config in ("C", "E")

    from hobot_dnn import pyeasy_dnn as dnn
    print(f"[Config {args.config}] 加载模型 {args.model} …")
    models = dnn.load([args.model])
    model  = models[0]

    img_paths = sorted(Path(args.images).glob("*.jpg"))
    out_dir   = Path(args.out) / f"config_{args.config}"
    out_dir.mkdir(parents=True, exist_ok=True)

    total_det = 0
    for p in img_paths:
        bgr = cv2.imread(str(p))
        orig_h, orig_w = bgr.shape[:2]

        nv12, scale, pt, pl = bgr_to_nv12(bgr)

        # 构造输入 tensor
        input_array = nv12.astype(np.uint8)
        outputs = model.forward(input_array)

        if use_shape:
            dets = decode_shape_driven(outputs, scale, pt, pl, orig_h, orig_w)
        else:
            dets = decode_fixed_index(outputs, scale, pt, pl, orig_h, orig_w)

        result = draw_detections(bgr, dets)
        out_path = out_dir / p.name
        cv2.imwrite(str(out_path), result)
        total_det += len(dets)
        print(f"  {p.name:30s}  {len(dets):3d} det  → {out_path.name}")

    print(f"\n[Config {args.config}] 完成：{len(img_paths)} 张，共 {total_det} 个检测框")
    print(f"结果保存至 {out_dir}")


if __name__ == "__main__":
    main()
