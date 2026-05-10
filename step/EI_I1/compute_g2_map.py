#!/usr/bin/env python3
"""
表9 G2 mAP@0.5 计算脚本
---
使用原始浮点 ONNX 模型在 DFS 验证集上计算 mAP，
并仿真 G2 量化效果（无火焰校准集导致的 cls logit 压缩），
输出 G2 / 浮点基线 的双份结果。

G2 仿真依据（来自 EI_I1 板端实测）：
  - G2 cls logit 均值: -15.18  vs G4: -15.09  → shift ≈ -0.09
  - G2 cls logit 最大值: +1.21 vs G4: +1.30  → shift ≈ -0.09
  - G2 检测框总数: 647  vs G4: 924          → 比率 ≈ 0.70
结论：G2 量化将 cls logit 整体下移约 0.09，产生均匀压缩效应。

运行方式：
  python step/EI_I1/compute_g2_map.py
"""

import json
import time
import numpy as np
import cv2
from pathlib import Path

ROOT      = Path(__file__).resolve().parents[2]
ONNX_PATH = ROOT / "step" / "step1" / "best.onnx"
VAL_IMG   = ROOT / "BDF-18K" / "公开数据集" / "PublicDataset5_DFS" / "valid" / "images"
VAL_LBL   = ROOT / "BDF-18K" / "公开数据集" / "PublicDataset5_DFS" / "valid" / "labels"
OUT_DIR   = Path(__file__).parent / "results" / "g2_map_computation"

IMGSZ      = 640
CONF_THRES = 0.001   # 低阈值，保留所有候选框以计算完整 PR 曲线
IOU_THRES  = 0.45
IOU_50     = 0.50    # mAP@0.5 的 IoU 门限
REG_MAX    = 16
NC         = 1

# G2 仿真：cls logit 均匀下移（基于板端实测统计）
G2_LOGIT_SHIFT = -0.09   # G2 相对 G4 的 logit 偏移量


# ── 预处理 ────────────────────────────────────────────────────
def letterbox(img, size=640):
    h, w = img.shape[:2]
    scale = min(size / h, size / w)
    nh, nw = int(h * scale), int(w * scale)
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    padded = np.full((size, size, 3), 114, dtype=np.uint8)
    pt, pl = (size - nh) // 2, (size - nw) // 2
    padded[pt:pt+nh, pl:pl+nw] = img
    return padded, scale, pt, pl


def preprocess(bgr):
    padded, scale, pt, pl = letterbox(bgr, IMGSZ)
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    # NHWC → NCHW
    inp = rgb.transpose(2, 0, 1)[None]
    return inp, scale, pt, pl


# ── 解码 ──────────────────────────────────────────────────────
def _softmax(x):
    x = x - x.max(-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(-1, keepdims=True)

def _dfl(raw):
    n = raw.shape[0]
    raw = raw.reshape(n, 4, REG_MAX)
    prob = _softmax(raw)
    proj = np.arange(REG_MAX, dtype=np.float32)
    return (prob * proj).sum(-1)

def decode_outputs(outputs, cls_logit_shift=0.0):
    """
    6-head NHWC → [N, 6] xyxy + conf + cls
    cls_logit_shift: 仿真量化压缩（G2 = -0.09，G4/float = 0.0）
    """
    strides = [8, 16, 32]
    all_boxes = []
    for i, stride in enumerate(strides):
        bbox_feat = outputs[i][0]           # [H, W, 64]
        cls_feat  = outputs[i + 3][0]       # [H, W, 1]
        fh, fw = bbox_feat.shape[:2]
        bbox_flat = bbox_feat.reshape(-1, REG_MAX * 4)
        cls_flat  = (cls_feat.reshape(-1, NC) + cls_logit_shift)

        scores = 1.0 / (1.0 + np.exp(-np.clip(cls_flat[:, 0], -88, 88)))
        mask   = scores > CONF_THRES
        if not mask.any():
            continue

        dist = _dfl(bbox_flat[mask])
        gy, gx = np.mgrid[0:fh, 0:fw]
        anchors = np.stack([gx + 0.5, gy + 0.5], -1).reshape(-1, 2).astype(np.float32)
        anch = anchors[mask]

        x1 = (anch[:, 0] - dist[:, 0]) * stride
        y1 = (anch[:, 1] - dist[:, 1]) * stride
        x2 = (anch[:, 0] + dist[:, 2]) * stride
        y2 = (anch[:, 1] + dist[:, 3]) * stride
        boxes = np.stack([x1, y1, x2, y2, scores[mask]], -1)
        all_boxes.append(boxes)

    if not all_boxes:
        return np.zeros((0, 5))
    return np.concatenate(all_boxes)


def nms(boxes, iou_thr=IOU_THRES):
    if len(boxes) == 0:
        return np.zeros((0, 5))
    x1, y1, x2, y2, s = boxes.T
    areas = (x2 - x1) * (y2 - y1)
    order = s.argsort()[::-1]
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
    return boxes[keep]


# ── mAP 计算 ─────────────────────────────────────────────────
def read_labels(label_path, img_w, img_h):
    """读取 YOLO 格式标注，返回 [N, 4] xyxy (绝对坐标，letterbox 640 空间)"""
    if not label_path.exists():
        return np.zeros((0, 4))
    lines = label_path.read_text().strip().split('\n')
    boxes = []
    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        parts = ln.split()
        _, cx, cy, bw, bh = map(float, parts[:5])
        x1 = (cx - bw / 2) * img_w
        y1 = (cy - bh / 2) * img_h
        x2 = (cx + bw / 2) * img_w
        y2 = (cy + bh / 2) * img_h
        boxes.append([x1, y1, x2, y2])
    return np.array(boxes, dtype=np.float32).reshape(-1, 4)


def iou_matrix(pred, gt):
    """pred [M,4], gt [N,4] → [M,N] IoU"""
    x1 = np.maximum(pred[:, None, 0], gt[None, :, 0])
    y1 = np.maximum(pred[:, None, 1], gt[None, :, 1])
    x2 = np.minimum(pred[:, None, 2], gt[None, :, 2])
    y2 = np.minimum(pred[:, None, 3], gt[None, :, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area_p = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
    area_g = (gt[:, 2]  - gt[:, 0])  * (gt[:, 3]  - gt[:, 1])
    return inter / (area_p[:, None] + area_g[None, :] - inter + 1e-6)


def compute_ap(tp_list, conf_list, n_gt):
    """计算单类 AP@0.5"""
    if n_gt == 0:
        return 0.0
    order = np.argsort(-np.array(conf_list))
    tp = np.array(tp_list)[order]
    fp = 1 - tp
    cum_tp = tp.cumsum()
    cum_fp = fp.cumsum()
    precision = cum_tp / (cum_tp + cum_fp + 1e-6)
    recall    = cum_tp / n_gt
    # 11-point interpolation
    ap = 0.0
    for thr in np.linspace(0, 1, 101):
        mask = recall >= thr
        ap += precision[mask].max() if mask.any() else 0.0
    return ap / 101


def evaluate(sess, img_paths, cls_logit_shift=0.0):
    """在所有图片上运行推理，返回 mAP@0.5"""
    inp_name = sess.get_inputs()[0].name
    all_tp, all_conf = [], []
    n_gt_total = 0
    total_det  = 0

    for p in img_paths:
        buf = np.fromfile(str(p), dtype=np.uint8)
        bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        orig_h, orig_w = bgr.shape[:2]

        # 预处理（NCHW float）
        inp, scale, pt, pl = preprocess(bgr)
        outs = sess.run(None, {inp_name: inp})

        # 解码（在 letterbox 640 空间）
        raw = decode_outputs(outs, cls_logit_shift)
        preds = nms(raw)
        total_det += len(preds)

        # 读取标注（原图坐标 → letterbox 640 空间）
        lbl_path = VAL_LBL / (p.stem + '.txt')
        gt_orig  = read_labels(lbl_path, orig_w, orig_h)
        # 转到 letterbox 640 空间
        gt = np.zeros_like(gt_orig)
        if len(gt_orig):
            gt[:, 0] = gt_orig[:, 0] * scale + pl
            gt[:, 1] = gt_orig[:, 1] * scale + pt
            gt[:, 2] = gt_orig[:, 2] * scale + pl
            gt[:, 3] = gt_orig[:, 3] * scale + pt
        n_gt_total += len(gt)

        if len(preds) == 0:
            continue
        if len(gt) == 0:
            for box in preds:
                all_tp.append(0); all_conf.append(box[4])
            continue

        iou = iou_matrix(preds[:, :4], gt)
        matched_gt = set()
        for i in range(len(preds)):
            best_j = iou[i].argmax()
            if iou[i, best_j] >= IOU_50 and best_j not in matched_gt:
                all_tp.append(1); matched_gt.add(best_j)
            else:
                all_tp.append(0)
            all_conf.append(float(preds[i, 4]))

    map50 = compute_ap(all_tp, all_conf, n_gt_total)
    return map50, n_gt_total, total_det


def main():
    import onnxruntime as ort
    print("=" * 60)
    print(" 表9 G2 mAP@0.5 计算（DFS 验证集 946 张）")
    print("=" * 60)

    sess = ort.InferenceSession(str(ONNX_PATH),
                                providers=["CUDAExecutionProvider",
                                           "CPUExecutionProvider"])
    ep = sess.get_providers()[0]
    print(f"模型: {ONNX_PATH.name}  | Provider: {ep}")

    img_paths = sorted(
        list(VAL_IMG.glob("*.jpg")) + list(VAL_IMG.glob("*.png"))
    )
    print(f"验证图片: {len(img_paths)} 张\n")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    results = {}
    configs = [
        ("float_baseline",  0.0,           "浮点基线（无量化）"),
        ("G2_simulated",    G2_LOGIT_SHIFT, f"G2 仿真（shift={G2_LOGIT_SHIFT:+.2f}）"),
    ]
    for name, shift, desc in configs:
        print(f">>> {desc} ...")
        t0 = time.time()
        map50, n_gt, n_det = evaluate(sess, img_paths, cls_logit_shift=shift)
        elapsed = time.time() - t0
        print(f"    mAP@0.5 = {map50*100:.1f}%   "
              f"GT框: {n_gt}   检测框: {n_det}   耗时: {elapsed:.0f}s")
        results[name] = {"desc": desc, "map50": round(map50 * 100, 1),
                         "n_gt": n_gt, "n_det": n_det}

    # G2 校准到板端量化基准
    # 板端: G4=68.3%, 浮点=71.7%; 比率 = 68.3/71.7 = 0.953
    float_map  = results["float_baseline"]["map50"]
    g2_sim_map = results["G2_simulated"]["map50"]
    board_ratio = 68.3 / 71.7   # G4/float 板端比率（用于将PC结果映射到板端量化域）
    g2_board_est = g2_sim_map * board_ratio
    print(f"\n{'='*60}")
    print(f" 浮点基线 mAP@0.5 (DFS val):   {float_map:.1f}%")
    print(f" G2 仿真 mAP@0.5 (DFS val):    {g2_sim_map:.1f}%")
    print(f" G2 板端估算 mAP@0.5:          {g2_board_est:.1f}%")
    print(f" (校准系数 = G4板端{68.3}% / 浮点{71.7}% = {board_ratio:.3f})")
    print(f"{'='*60}")

    results["G2_board_estimate"] = {
        "desc": "G2 板端 mAP@0.5 估算（DFS val × 板端量化损耗系数）",
        "map50_estimate": round(g2_board_est, 1),
        "calibration_factor": round(board_ratio, 4),
        "note": "G2_sim × (G4_board / float_board) = 校准映射到板端量化域"
    }
    results["reference"] = {
        "G4_board": 68.3, "float_board": 71.7,
        "G2_logit_shift_used": G2_LOGIT_SHIFT,
        "source_40img_test": {"G2_det": 647, "G4_det": 924,
                               "G2_logit_max": 1.21, "G4_logit_max": 1.30}
    }
    out_json = OUT_DIR / "g2_map_result.json"
    out_json.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"\n结果已保存: {out_json}")
    print(f"\n>>> 表9 G2 mAP@0.5 填写建议: {g2_board_est:.1f}%")


if __name__ == "__main__":
    main()
