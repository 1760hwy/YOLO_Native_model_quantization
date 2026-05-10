#!/usr/bin/env python3
"""
PC 端 mAP@0.5 计算
输入：step/EI_xrsy/preds/config_X/  （板端推理输出）
      BDF-18K/公开数据集/PublicDataset5_DFS/valid/labels/
输出：各配置 mAP@0.5 对比表
"""
import numpy as np
from pathlib import Path
from PIL import Image

ROOT      = Path(__file__).resolve().parents[2]
PREDS_DIR = Path(__file__).parent / "preds"
LABELS_DIR = ROOT / "BDF-18K/公开数据集/PublicDataset5_DFS/valid/labels"
IMAGES_DIR = ROOT / "BDF-18K/公开数据集/PublicDataset5_DFS/valid/images"
IOU_THR   = 0.5


def load_gt(label_path, img_w, img_h):
    if not label_path.exists():
        return []
    boxes = []
    for line in label_path.read_text().strip().splitlines():
        p = line.strip().split()
        if len(p) < 5:
            continue
        cx, cy, w, h = float(p[1]), float(p[2]), float(p[3]), float(p[4])
        boxes.append([(cx - w/2)*img_w, (cy - h/2)*img_h,
                       (cx + w/2)*img_w, (cy + h/2)*img_h])
    return boxes


def load_preds(pred_path):
    if not pred_path.exists():
        return []
    preds = []
    for line in pred_path.read_text().strip().splitlines():
        p = line.strip().split()
        if len(p) < 5:
            continue
        preds.append([float(x) for x in p])   # conf x1 y1 x2 y2
    return sorted(preds, key=lambda x: -x[0])


def box_iou(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    ua = (a[2]-a[0])*(a[3]-a[1]); ub = (b[2]-b[0])*(b[3]-b[1])
    return inter / (ua + ub - inter + 1e-6)


def calc_ap(tp_arr, n_gt):
    if n_gt == 0:
        return 0.0
    tp  = np.array(tp_arr, dtype=float)
    fp  = 1 - tp
    tpc = np.cumsum(tp); fpc = np.cumsum(fp)
    rec  = tpc / n_gt
    prec = tpc / (tpc + fpc + 1e-6)
    # 11-point interpolation
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        mask = rec >= t
        ap += (prec[mask].max() if mask.any() else 0.0)
    return ap / 11


def eval_config(cfg):
    pred_dir = PREDS_DIR / f"config_{cfg}"
    if not pred_dir.exists():
        return None

    img_files = sorted(IMAGES_DIR.glob("*.jpg"))
    all_dets  = []    # (conf, is_tp)
    n_gt_total = 0

    for img_path in img_files:
        with Image.open(img_path) as im:
            img_w, img_h = im.size

        gt   = load_gt(LABELS_DIR / (img_path.stem + ".txt"), img_w, img_h)
        pred = load_preds(pred_dir / (img_path.stem + ".txt"))
        n_gt_total += len(gt)

        matched = set()
        for det in pred:
            conf = det[0]; det_box = det[1:]
            best_iou, best_gi = 0.0, -1
            for gi, gt_box in enumerate(gt):
                if gi in matched:
                    continue
                v = box_iou(det_box, gt_box)
                if v > best_iou:
                    best_iou, best_gi = v, gi
            if best_iou >= IOU_THR and best_gi >= 0:
                matched.add(best_gi)
                all_dets.append((conf, 1))
            else:
                all_dets.append((conf, 0))

    all_dets.sort(key=lambda x: -x[0])
    tp_arr = [d[1] for d in all_dets]
    ap = calc_ap(tp_arr, n_gt_total)
    n_det = len(all_dets)
    n_tp  = sum(tp_arr)
    return ap, n_gt_total, n_det, int(n_tp)


def main():
    print("=" * 58)
    print(f"{'配置':<10} {'mAP@0.5':>10} {'GT框':>8} {'预测框':>8} {'TP':>6}")
    print("-" * 58)
    for cfg in list("ABCDE"):
        r = eval_config(cfg)
        if r is None:
            print(f"Config {cfg}   — 未找到预测文件（先跑板端）")
        else:
            ap, n_gt, n_det, n_tp = r
            print(f"Config {cfg}   {ap*100:>9.1f}%  {n_gt:>8}  {n_det:>8}  {n_tp:>6}")
    print("=" * 58)


if __name__ == "__main__":
    main()
