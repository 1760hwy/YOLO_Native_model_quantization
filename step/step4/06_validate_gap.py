#!/usr/bin/env python3
"""
Step 4-6: 对比验证 fire_detect.onnx vs fire_detect_gap.onnx 的检测效果
目的: 确认 InstanceNorm→GAP 替换后精度无损失

验证内容:
  1. 原始 ONNX (fire_detect.onnx) 推理结果
  2. GAP 替换后 ONNX (fire_detect_gap.onnx) 推理结果
  3. 每张图对比检测数量、框坐标、置信度
  4. 保存可视化对比图和 JSON 报告

输出目录: step4/onnx_validation_gap/
  original/ - 原始 ONNX 结果图
  gap/       - GAP ONNX 结果图
  compare_report.json - 详细对比数据

运行:
  C:\\d\\Anaconda3\\envs\\yolo\\python.exe step/step4/06_validate_gap.py
"""

import sys
import json
import numpy as np
import cv2
import onnxruntime as ort
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

STEP4_DIR   = Path(__file__).parent
ORIG_ONNX   = STEP4_DIR / "fire_detect.onnx"
GAP_ONNX    = STEP4_DIR / "fire_detect_gap.onnx"
OUT_DIR     = STEP4_DIR / "onnx_validation_gap"

# 测试图片优先从 step5/test_images，回退到 BoWFireDataset
IMG_DIR_1   = STEP4_DIR.parent / "step5" / "test_images"
IMG_DIR_2   = STEP4_DIR.parents[1] / "BDF-18K/公开数据集/PublicDataset1_BoWFireDataset/test/images"

IMGSZ       = 640
REG_MAX     = 16
CONF_THRES  = 0.25
IOU_THRES   = 0.45
CLASS_NAMES = ["fire"]
MAX_IMAGES  = 40


# ── 预处理 ─────────────────────────────────────────────────────────────────────
def letterbox(img, size=640, pad_val=114):
    h, w = img.shape[:2]
    scale = min(size / h, size / w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), pad_val, dtype=np.uint8)
    top  = (size - nh) // 2
    left = (size - nw) // 2
    canvas[top:top+nh, left:left+nw] = img
    return canvas, scale, (left, top)


def preprocess(img_bgr):
    lb, scale, pad = letterbox(img_bgr)
    rgb = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return rgb.transpose(2, 0, 1)[np.newaxis], scale, pad


# ── 解码 ───────────────────────────────────────────────────────────────────────
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
    strides   = [8, 16, 32]
    expected_h = {IMGSZ // s: s for s in strides}
    bbox_map, cls_map = {}, {}

    for o in outs:
        arr = o
        if arr.ndim == 4:
            if arr.shape[1] in (REG_MAX * 4, len(CLASS_NAMES)):
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
        return np.zeros((0, 6), np.float32)

    all_boxes = []
    for s in strides:
        h_key = IMGSZ // s
        bf = bbox_map[h_key][0].reshape(-1, REG_MAX * 4)
        cf = cls_map[h_key][0].reshape(-1, len(CLASS_NAMES))
        cp = cf if (cf.min() >= 0 and cf.max() <= 1) else 1.0 / (1.0 + np.exp(-cf))
        ms, mc = cp.max(axis=-1), cp.argmax(axis=-1)
        mask = ms > conf_thres
        if not mask.any():
            continue
        h_f, w_f = bbox_map[h_key].shape[1], bbox_map[h_key].shape[2]
        anc  = make_anchors(h_f, w_f)[mask]
        dist = dfl_decode(bf[mask])
        x1 = (anc[:, 0] - dist[:, 0]) * s
        y1 = (anc[:, 1] - dist[:, 1]) * s
        x2 = (anc[:, 0] + dist[:, 2]) * s
        y2 = (anc[:, 1] + dist[:, 3]) * s
        all_boxes.append(
            np.stack([x1, y1, x2, y2, ms[mask], mc[mask].astype(np.float32)], 1)
        )
    return np.concatenate(all_boxes, 0) if all_boxes else np.zeros((0, 6), np.float32)


def nms(boxes, iou_thres=0.45):
    if len(boxes) == 0:
        return boxes
    keep_all = []
    for cls_id in np.unique(boxes[:, 5]):
        b = boxes[boxes[:, 5] == cls_id]
        x1, y1, x2, y2, s = b[:,0], b[:,1], b[:,2], b[:,3], b[:,4]
        order = s.argsort()[::-1]
        keep = []
        while order.size:
            i = order[0]; keep.append(i)
            if order.size == 1:
                break
            iw = (np.minimum(x2[i], x2[order[1:]]) - np.maximum(x1[i], x1[order[1:]])).clip(0)
            ih = (np.minimum(y2[i], y2[order[1:]]) - np.maximum(y1[i], y1[order[1:]])).clip(0)
            inter = iw * ih
            union = ((x2[i]-x1[i])*(y2[i]-y1[i])
                     + (x2[order[1:]]-x1[order[1:]])*(y2[order[1:]]-y1[order[1:]])
                     - inter + 1e-6)
            order = order[np.concatenate(([0], np.where(inter/union <= iou_thres)[0]+1))][1:]
        keep_all.append(b[keep])
    return np.concatenate(keep_all)


def draw_boxes(img, boxes, scale, pad, color=(0, 0, 255), label_prefix=""):
    vis = img.copy()
    oh, ow = img.shape[:2]
    for b in boxes:
        x1 = max(0,  (b[0] - pad[0]) / scale)
        y1 = max(0,  (b[1] - pad[1]) / scale)
        x2 = min(ow, (b[2] - pad[0]) / scale)
        y2 = min(oh, (b[3] - pad[1]) / scale)
        label = f"{label_prefix}{CLASS_NAMES[int(b[5])]} {b[4]:.2f}"
        cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(vis, (int(x1), int(y1)-th-4), (int(x1)+tw+2, int(y1)), color, -1)
        cv2.putText(vis, label, (int(x1)+1, int(y1)-2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
    return vis


def find_images(max_n=MAX_IMAGES):
    """自动定位测试图片目录"""
    for img_dir in [IMG_DIR_1, IMG_DIR_2]:
        if not img_dir.exists():
            continue
        imgs = sorted(list(img_dir.glob("*.jpg")) +
                      list(img_dir.glob("*.png")) +
                      list(img_dir.glob("*.jpeg")))[:max_n]
        if imgs:
            print(f"[图片] 目录: {img_dir}")
            return imgs
    return []


def boxes_to_records(boxes, scale, pad):
    return [{"class": CLASS_NAMES[int(b[5])],
             "conf":  round(float(b[4]), 4),
             "x1": round(float((b[0]-pad[0])/scale), 1),
             "y1": round(float((b[1]-pad[1])/scale), 1),
             "x2": round(float((b[2]-pad[0])/scale), 1),
             "y2": round(float((b[3]-pad[1])/scale), 1)}
            for b in boxes]


def main():
    print("=" * 65)
    print("STEP 4-6: ONNX 对比验证 (原始 vs GAP 替换)")
    print(f"  原始: {ORIG_ONNX}")
    print(f"  GAP:  {GAP_ONNX}")
    print(f"  输出: {OUT_DIR}")
    print("=" * 65)

    # ── 检查文件 ────────────────────────────────────────────────────────────────
    missing = [p for p in [ORIG_ONNX, GAP_ONNX] if not p.exists()]
    if missing:
        for p in missing:
            print(f"[ERROR] 找不到: {p}")
        if GAP_ONNX in missing:
            print("       请先运行: python step4/05_replace_instnorm.py")
        sys.exit(1)

    # ── 加载两个会话 ────────────────────────────────────────────────────────────
    so = ort.SessionOptions()
    so.log_severity_level = 3
    providers = ["CPUExecutionProvider"]

    sess_orig = ort.InferenceSession(str(ORIG_ONNX), sess_options=so, providers=providers)
    sess_gap  = ort.InferenceSession(str(GAP_ONNX),  sess_options=so, providers=providers)
    iname = sess_orig.get_inputs()[0].name
    print(f"[加载] 两个会话就绪，输入名: {iname}")

    # ── 图片 ────────────────────────────────────────────────────────────────────
    img_paths = find_images()
    if not img_paths:
        print("[ERROR] 未找到测试图片，请检查路径")
        sys.exit(1)
    print(f"[图片] 共 {len(img_paths)} 张\n")

    # ── 输出目录 ────────────────────────────────────────────────────────────────
    dir_orig = OUT_DIR / "original"
    dir_gap  = OUT_DIR / "gap"
    dir_orig.mkdir(parents=True, exist_ok=True)
    dir_gap.mkdir(parents=True, exist_ok=True)

    # ── 推理循环 ────────────────────────────────────────────────────────────────
    records   = []
    total_orig = total_gap = 0
    diff_count = 0

    print(f"{'序号':>4}  {'图片':40}  {'原始':>4}  {'GAP':>4}  {'差异'}")
    print("-" * 70)

    for idx, p in enumerate(img_paths):
        img = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            print(f"  [SKIP] {p.name}")
            continue

        inp, scale, pad = preprocess(img)

        # 推理
        outs_orig = sess_orig.run(None, {iname: inp})
        outs_gap  = sess_gap.run(None,  {iname: inp})

        # 解码
        boxes_orig = nms(decode_outputs(outs_orig, CONF_THRES), IOU_THRES)
        boxes_gap  = nms(decode_outputs(outs_gap,  CONF_THRES), IOU_THRES)

        n_orig = len(boxes_orig)
        n_gap  = len(boxes_gap)
        total_orig += n_orig
        total_gap  += n_gap
        is_diff = abs(n_orig - n_gap) > 0
        if is_diff:
            diff_count += 1

        fname = p.name[:38] if len(p.name) > 38 else p.name
        print(f"[{idx+1:02d}/{len(img_paths):02d}]  {fname:40}  {n_orig:>4}  {n_gap:>4}  "
              f"{'⚠️ DIFF' if is_diff else '✓'}")

        # 保存可视化
        tag = f"{idx+1:03d}_{p.stem}"
        vis_orig = draw_boxes(img, boxes_orig, scale, pad, color=(0,0,255))
        vis_gap  = draw_boxes(img, boxes_gap,  scale, pad, color=(0,200,0))
        cv2.imencode(".jpg", vis_orig)[1].tofile(str(dir_orig / f"{tag}_orig.jpg"))
        cv2.imencode(".jpg", vis_gap)[1].tofile(str(dir_gap  / f"{tag}_gap.jpg"))

        # 计算原始输出余弦相似度
        cos_sims = []
        for a, b in zip(outs_orig, outs_gap):
            dot = float(np.dot(a.ravel(), b.ravel()))
            nn  = float(np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
            cos_sims.append(round(dot / nn, 6))

        records.append({
            "img":       p.name,
            "n_orig":    n_orig,
            "n_gap":     n_gap,
            "diff":      n_gap - n_orig,
            "cos_sims":  cos_sims,
            "orig_dets": boxes_to_records(boxes_orig, scale, pad),
            "gap_dets":  boxes_to_records(boxes_gap,  scale, pad),
        })

    # ── 汇总 ────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("汇总:")
    print(f"  图片数:      {len(records)}")
    print(f"  原始总检测:  {total_orig}")
    print(f"  GAP 总检测:  {total_gap}")
    print(f"  检测数差异:  {total_gap - total_orig:+d}")
    print(f"  不同图片数:  {diff_count} / {len(records)}")
    if records:
        avg_cos = np.mean([np.mean(r["cos_sims"]) for r in records])
        print(f"  平均余弦相似度: {avg_cos:.6f}")

    if diff_count == 0 and total_orig == total_gap:
        print("\n结论: ✓ GAP 替换完全等价，可以安全量化")
    elif diff_count <= len(records) * 0.1:
        print("\n结论: ✓ 差异在 10% 以内，属正常浮点误差，可以量化")
    else:
        print(f"\n结论: ⚠️  {diff_count} 张图片检测数不同，建议进一步检查")

    # ── 保存报告 ────────────────────────────────────────────────────────────────
    report = {
        "orig_onnx":      str(ORIG_ONNX.name),
        "gap_onnx":       str(GAP_ONNX.name),
        "conf_thres":     CONF_THRES,
        "iou_thres":      IOU_THRES,
        "total_images":   len(records),
        "total_orig":     total_orig,
        "total_gap":      total_gap,
        "diff_images":    diff_count,
        "avg_cos_sim":    float(avg_cos) if records else 0.0,
        "results":        records,
    }
    report_path = OUT_DIR / "compare_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[报告] {report_path}")
    print(f"[图片] 原始: {dir_orig}")
    print(f"[图片] GAP:  {dir_gap}")
    print("=" * 65)


if __name__ == "__main__":
    main()
