#!/usr/bin/env python3
"""
PC端浮点基线 — 表7补充数据
1. 在40张测试图上提取 cls logit 统计 + 检测框数
2. 在946张valid图上计算 mAP@0.5
使用 PyTorch PT 模型直接推理，与训练一致的解码路径
"""
import sys, cv2, numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

CONF_THRESH = 0.25
IOU_THRESH  = 0.45
INPUT_SIZE  = 640

MODEL_PATH  = Path(__file__).parents[2] / "model/yolo11-HGNetV2-C3k2-DWR-LSDECD-SlideLoss.pt"
IMAGES_40   = Path(__file__).parent / "test40_images"
IMAGES_946  = Path(__file__).parents[2] / "BDF-18K/公开数据集/PublicDataset5_DFS/valid/images"
LABELS_946  = Path(__file__).parents[2] / "BDF-18K/公开数据集/PublicDataset5_DFS/valid/labels"


def letterbox(bgr, size=INPUT_SIZE):
    ih, iw = bgr.shape[:2]
    scale  = min(size / ih, size / iw)
    nh, nw = int(ih * scale), int(iw * scale)
    resized = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    padded  = np.full((size, size, 3), 114, dtype=np.uint8)
    pt, pl  = (size - nh) // 2, (size - nw) // 2
    padded[pt:pt+nh, pl:pl+nw] = resized
    return padded, scale, pt, pl


def load_model():
    import torch
    from ultralytics import YOLO
    model = YOLO(str(MODEL_PATH))
    model.model.eval()
    return model


def infer_raw(model, bgr):
    """返回 (cls_logits_flat, boxes_xyxy, confs) — boxes/confs 已映射回原图坐标"""
    import torch
    from ultralytics.utils import ops
    from ultralytics.utils.tal import make_anchors, dist2bbox

    orig_h, orig_w = bgr.shape[:2]
    padded, scale, pt, pl = letterbox(bgr)
    rgb = padded[:, :, ::-1].copy()
    inp = torch.from_numpy(rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.0

    with torch.no_grad():
        raw = model.model(inp)

    if isinstance(raw, tuple) and len(raw) == 2:
        raw = raw[0]

    nl = len(raw) // 2
    bboxes_nhwc = raw[:nl]
    clses_nhwc  = raw[nl:]

    detect_head = model.model.model[-1]
    bbox_nchw = [b.permute(0, 3, 1, 2).contiguous() for b in bboxes_nhwc]
    cls_nchw  = [c.permute(0, 3, 1, 2).contiguous() for c in clses_nhwc]

    anchors, strides_t = (t.transpose(0, 1) for t in
                          make_anchors(bbox_nchw, detect_head.stride, 0.5))

    # raw cls logits (before sigmoid), shape [1, nc, total_anchors]
    cls_flat  = torch.cat([c.view(1, detect_head.nc, -1) for c in cls_nchw], dim=2)
    cls_logits = cls_flat[0, 0].cpu().numpy()  # [total_anchors]

    bbox_flat = torch.cat([b.view(1, detect_head.reg_max * 4, -1) for b in bbox_nchw], dim=2)
    dfl_out   = detect_head.dfl(bbox_flat)
    dbox      = dist2bbox(dfl_out, anchors.unsqueeze(0), xywh=True, dim=1) * strides_t
    pred      = torch.cat((dbox, cls_flat.sigmoid()), dim=1)

    results = ops.non_max_suppression(pred, CONF_THRESH, IOU_THRESH, max_det=300)
    det = results[0].cpu().numpy()  # [N, 6] xyxy conf cls

    boxes, confs = np.empty((0, 4)), np.empty(0)
    if len(det):
        det[:, [0, 2]] = (det[:, [0, 2]] - pl) / scale
        det[:, [1, 3]] = (det[:, [1, 3]] - pt) / scale
        det[:, :4] = np.clip(det[:, :4], 0, None)
        det[:, [0, 2]] = np.clip(det[:, [0, 2]], 0, orig_w)
        det[:, [1, 3]] = np.clip(det[:, [1, 3]], 0, orig_h)
        boxes = det[:, :4]
        confs = det[:, 4]

    return cls_logits, boxes, confs


# ── Part 1: logit stats + det count on 40 images ───────────────────────────
def part1_logit_40(model):
    imgs = sorted(IMAGES_40.glob("*.jpg"))
    print(f"[Part1] {len(imgs)} 张测试图 -> PC浮点基线 logit统计")
    all_logits, total_det = [], 0
    for p in imgs:
        bgr = cv2.imread(str(p))
        if bgr is None: continue
        logits, boxes, _ = infer_raw(model, bgr)
        all_logits.append(logits); total_det += len(boxes)
    v = np.concatenate(all_logits)
    r = {"mean": float(v.mean()), "max": float(v.max()),
         "pos%": float((v > 0).mean() * 100), "boxes": total_det}
    print(f"  logit均值    : {r['mean']:.2f}")
    print(f"  logit最大值   : {r['max']:.2f}")
    print(f"  正值占比      : {r['pos%']:.2f}%")
    print(f"  检测框数(40张): {r['boxes']}")
    return r


# ── Part 2: mAP@0.5 on 946 images ──────────────────────────────────────────
def box_iou(a, b):
    x1,y1=max(a[0],b[0]),max(a[1],b[1])
    x2,y2=min(a[2],b[2]),min(a[3],b[3])
    inter=max(0,x2-x1)*max(0,y2-y1)
    ua=(a[2]-a[0])*(a[3]-a[1]); ub=(b[2]-b[0])*(b[3]-b[1])
    return inter/(ua+ub-inter+1e-6)

def load_gt(lbl_path, iw, ih):
    if not lbl_path.exists(): return []
    boxes = []
    for line in lbl_path.read_text().strip().splitlines():
        p = line.split()
        if len(p) < 5: continue
        cx,cy,w,h = float(p[1]),float(p[2]),float(p[3]),float(p[4])
        boxes.append([(cx-w/2)*iw,(cy-h/2)*ih,(cx+w/2)*iw,(cy+h/2)*ih])
    return boxes

def calc_ap(tp_arr, n_gt):
    if n_gt == 0: return 0.0
    tp=np.array(tp_arr); fp=1-tp
    tpc=np.cumsum(tp); fpc=np.cumsum(fp)
    rec=tpc/n_gt; prec=tpc/(tpc+fpc+1e-6)
    return sum(prec[rec>=t].max() if (rec>=t).any() else 0.0
               for t in np.linspace(0,1,11)) / 11

def part2_map_946(model):
    imgs = sorted(IMAGES_946.glob("*.jpg"))
    print(f"\n[Part2] {len(imgs)} 张 -> mAP@0.5")
    all_dets, n_gt = [], 0
    for i, p in enumerate(imgs):
        bgr = cv2.imread(str(p))
        if bgr is None: continue
        orig_h, orig_w = bgr.shape[:2]
        gt = load_gt(LABELS_946 / (p.stem + ".txt"), orig_w, orig_h)
        n_gt += len(gt)
        _, boxes, confs = infer_raw(model, bgr)
        matched = set()
        for box, conf in sorted(zip(boxes.tolist(), confs.tolist()), key=lambda x: -x[1]):
            best_iou, best_gi = 0.0, -1
            for gi, gb in enumerate(gt):
                if gi in matched: continue
                v = box_iou(box, gb)
                if v > best_iou: best_iou, best_gi = v, gi
            if best_iou >= 0.5 and best_gi >= 0:
                matched.add(best_gi); all_dets.append((conf, 1))
            else:
                all_dets.append((conf, 0))
        if (i+1) % 100 == 0:
            print(f"  {i+1}/{len(imgs)}...", flush=True)

    all_dets.sort(key=lambda x: -x[0])
    tp_arr = [d[1] for d in all_dets]
    ap = calc_ap(tp_arr, n_gt)
    n_tp = sum(tp_arr)
    print(f"  mAP@0.5: {ap*100:.1f}%  GT={n_gt}  pred={len(all_dets)}  TP={n_tp}")
    return ap


def main():
    print("加载模型:", MODEL_PATH)
    model = load_model()

    r1 = part1_logit_40(model)
    ap = part2_map_946(model)

    print("\n" + "="*60)
    print("PC端浮点基线 完整数据：")
    print(f"  logit均值    : {r1['mean']:.2f}")
    print(f"  logit最大值   : {r1['max']:.2f}")
    print(f"  正值占比      : {r1['pos%']:.2f}%")
    print(f"  检测框数(40张): {r1['boxes']}")
    print(f"  mAP@0.5(946张): {ap*100:.1f}%")
    print("="*60)


if __name__ == "__main__":
    main()
