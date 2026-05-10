#!/usr/bin/env python3
"""
Step 1-4: 对比 best.onnx vs best_gap.onnx 检测结果
目的: 确认 InstanceNorm→GlobalAveragePool 替换前后，检测完全一致

验证内容:
  1. 所有 45 张测试图片的检测框数量一致
  2. 检测框坐标差异 < 0.01 pixel
  3. 置信度差异 < 0.001
  4. 保存对比可视化 (前 8 张)

运行:
  C:\\d\\Anaconda3\\envs\\yolo\\python.exe step/step1/04_compare_gap_onnx.py
"""

import sys, json, time
import numpy as np
import cv2
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

STEP1_DIR   = Path(__file__).parent
STEP2_DIR   = STEP1_DIR.parent / "step2"
ORIG_ONNX   = STEP1_DIR / "best.onnx"
GAP_ONNX    = STEP1_DIR / "best_gap.onnx"
SAMPLE_JSON = STEP2_DIR / "sample_list.json"
VIS_DIR     = STEP1_DIR / "compare_vis"

IMGSZ      = 640
CONF_THRES = 0.25
IOU_THRES  = 0.45
REG_MAX    = 16
CLASS_NAMES = ["fire"]


# ── 推理工具函数 (复用 step2/02_validate_onnx.py 的逻辑) ──────────────────────

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    h, w = img.shape[:2]
    scale = min(new_shape[0] / h, new_shape[1] / w)
    new_h, new_w = int(round(h * scale)), int(round(w * scale))
    dh = (new_shape[0] - new_h) / 2
    dw = (new_shape[1] - new_w) / 2
    top,    bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left,   right  = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, scale, (dw, dh)


def preprocess(img_bgr):
    img, scale, pad = letterbox(img_bgr, (IMGSZ, IMGSZ))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return np.expand_dims(np.transpose(img, (2, 0, 1)), 0), scale, pad


def dfl_decode(pred):
    n, _ = pred.shape
    pred = pred.reshape(n, 4, REG_MAX)
    exp  = np.exp(pred - pred.max(axis=-1, keepdims=True))
    prob = exp / exp.sum(axis=-1, keepdims=True)
    return (prob * np.arange(REG_MAX, dtype=np.float32)).sum(axis=-1)


def make_anchors(h, w, stride):
    ys = np.arange(h, dtype=np.float32) + 0.5
    xs = np.arange(w, dtype=np.float32) + 0.5
    gy, gx = np.meshgrid(ys, xs, indexing="ij")
    return np.stack([gx, gy], axis=-1).reshape(-1, 2)


def decode_outputs(outs):
    strides, all_boxes = [8, 16, 32], []
    for i, stride in enumerate(strides):
        bf = outs[i][0].reshape(-1, REG_MAX * 4)
        cf = outs[i + 3][0].reshape(-1, len(CLASS_NAMES))
        cp = 1.0 / (1.0 + np.exp(-cf))
        ms, mc = cp.max(axis=-1), cp.argmax(axis=-1)
        mask = ms > CONF_THRES
        if not mask.any():
            continue
        h_f, w_f = outs[i][0].shape[:2]
        anc  = make_anchors(h_f, w_f, stride)[mask]
        dist = dfl_decode(bf[mask])
        x1 = (anc[:, 0] - dist[:, 0]) * stride
        y1 = (anc[:, 1] - dist[:, 1]) * stride
        x2 = (anc[:, 0] + dist[:, 2]) * stride
        y2 = (anc[:, 1] + dist[:, 3]) * stride
        all_boxes.append(np.stack([x1, y1, x2, y2, ms[mask], mc[mask].astype(np.float32)], 1))
    return np.concatenate(all_boxes, 0) if all_boxes else np.zeros((0, 6), dtype=np.float32)


def nms(boxes, iou_thres=IOU_THRES):
    if len(boxes) == 0:
        return boxes
    keep_all = []
    for cls_id in np.unique(boxes[:, 5]):
        b = boxes[boxes[:, 5] == cls_id]
        x1, y1, x2, y2, s = b[:,0], b[:,1], b[:,2], b[:,3], b[:,4]
        order = s.argsort()[::-1]
        keep  = []
        while order.size:
            i = order[0]; keep.append(i)
            if order.size == 1: break
            iou = (np.minimum(x2[i], x2[order[1:]]) - np.maximum(x1[i], x1[order[1:]])).clip(0) * \
                  (np.minimum(y2[i], y2[order[1:]]) - np.maximum(y1[i], y1[order[1:]])).clip(0) / (
                  (x2[i]-x1[i])*(y2[i]-y1[i]) + (x2[order[1:]]-x1[order[1:]])*(y2[order[1:]]-y1[order[1:]]) -
                  (np.minimum(x2[i], x2[order[1:]]) - np.maximum(x1[i], x1[order[1:]])).clip(0) *
                  (np.minimum(y2[i], y2[order[1:]]) - np.maximum(y1[i], y1[order[1:]])).clip(0) + 1e-6)
            order = order[np.concatenate(([0], np.where(iou <= iou_thres)[0] + 1))][1:]
        keep_all.append(b[keep])
    return np.concatenate(keep_all) if keep_all else np.zeros((0, 6), dtype=np.float32)


def infer(sess, img_bgr):
    inp, scale, pad = preprocess(img_bgr)
    outs = sess.run(None, {sess.get_inputs()[0].name: inp})
    boxes = decode_outputs(outs)
    boxes = nms(boxes)
    return boxes, scale, pad


def draw_boxes(img, boxes, scale, pad, color=(0, 0, 255)):
    vis = img.copy()
    oh, ow = img.shape[:2]
    for b in boxes:
        x1 = max(0, (b[0] - pad[0]) / scale)
        y1 = max(0, (b[1] - pad[1]) / scale)
        x2 = min(ow, (b[2] - pad[0]) / scale)
        y2 = min(oh, (b[3] - pad[1]) / scale)
        label = f"fire {b[4]:.2f}"
        cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(vis, (int(x1), int(y1)-th-4), (int(x1)+tw, int(y1)), color, -1)
        cv2.putText(vis, label, (int(x1), int(y1)-2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
    return vis


# ── 主程序 ────────────────────────────────────────────────────────────────────

def main():
    import onnxruntime as ort

    print("=" * 65)
    print("STEP 1-4: best.onnx vs best_gap.onnx 检测对比")
    print("=" * 65)

    for p in [ORIG_ONNX, GAP_ONNX, SAMPLE_JSON]:
        if not p.exists():
            print(f"[ERROR] 找不到: {p}")
            sys.exit(1)

    so = ort.SessionOptions(); so.log_severity_level = 3
    providers = ["CPUExecutionProvider"]
    sess_orig = ort.InferenceSession(str(ORIG_ONNX), sess_options=so, providers=providers)
    sess_gap  = ort.InferenceSession(str(GAP_ONNX),  sess_options=so, providers=providers)
    print(f"[加载] best.onnx     ✓")
    print(f"[加载] best_gap.onnx ✓")

    with open(SAMPLE_JSON, encoding="utf-8") as f:
        data = json.load(f)
    samples = data["samples"] if isinstance(data, dict) else data
    img_paths = [Path(s["sample_path"]) for s in samples]
    print(f"[图片] 共 {len(img_paths)} 张\n")

    VIS_DIR.mkdir(exist_ok=True)
    VIS_LIMIT = 8   # 保存前 N 张对比图

    results = []
    count_match = 0
    box_diffs   = []
    conf_diffs  = []

    for idx, img_path in enumerate(img_paths):
        img = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            print(f"  [SKIP] 无法读取: {img_path.name}")
            continue

        boxes_orig, sc_o, pd_o = infer(sess_orig, img)
        boxes_gap,  sc_g, pd_g = infer(sess_gap,  img)

        n_orig = len(boxes_orig)
        n_gap  = len(boxes_gap)
        matched = (n_orig == n_gap)
        if matched:
            count_match += 1

        # 逐框对比 (框数相同时)
        max_bd, max_cd = 0.0, 0.0
        if matched and n_orig > 0:
            # 按置信度排序后对齐
            bo = boxes_orig[boxes_orig[:, 4].argsort()[::-1]]
            bg = boxes_gap [boxes_gap [:, 4].argsort()[::-1]]
            bd = float(np.abs(bo[:, :4] - bg[:, :4]).max())
            cd = float(np.abs(bo[:, 4]  - bg[:, 4] ).max())
            box_diffs.append(bd); conf_diffs.append(cd)
            max_bd, max_cd = bd, cd

        status = "OK" if matched else f"MISMATCH(orig={n_orig} gap={n_gap})"
        print(f"  [{idx+1:02d}/{len(img_paths)}] {img_path.name[:50]:<52} "
              f"det={n_orig}  {status}  box_diff={max_bd:.2e}  conf_diff={max_cd:.2e}")

        results.append({"img": img_path.name, "n_orig": n_orig, "n_gap": n_gap,
                        "matched": matched, "box_diff": max_bd, "conf_diff": max_cd})

        # 保存对比可视化
        if idx < VIS_LIMIT:
            vis_o = draw_boxes(img, boxes_orig, sc_o, pd_o, color=(0, 0, 255))
            vis_g = draw_boxes(img, boxes_gap,  sc_g, pd_g, color=(0, 200, 0))
            # 标题区
            h, w = vis_o.shape[:2]
            bar = np.zeros((28, w * 2, 3), dtype=np.uint8)
            cv2.putText(bar, "best.onnx (red)",     (8,   20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255),   1)
            cv2.putText(bar, "best_gap.onnx (green)",(w+8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,200,0),  1)
            side = np.concatenate([vis_o, vis_g], axis=1)
            out  = np.concatenate([bar, side], axis=0)
            cv2.imencode(".jpg", out)[1].tofile(str(VIS_DIR / f"{idx+1:02d}_{img_path.stem}.jpg"))

    # ── 汇总 ──────────────────────────────────────────────────────────────────
    total = len(results)
    print()
    print("=" * 65)
    print(f"汇总 ({total} 张图片)")
    print(f"  检测数量一致: {count_match}/{total}  {'✓ 全部一致' if count_match == total else '✗ 存在差异'}")
    if box_diffs:
        print(f"  框坐标最大差异: {max(box_diffs):.2e} px  (均值: {np.mean(box_diffs):.2e})")
        print(f"  置信度最大差异: {max(conf_diffs):.2e}    (均值: {np.mean(conf_diffs):.2e})")
    print(f"  对比可视化: {VIS_DIR}/")
    print("=" * 65)

    mismatches = [r for r in results if not r["matched"]]
    if mismatches:
        print("\n[不一致图片]")
        for r in mismatches:
            print(f"  {r['img']}  orig={r['n_orig']}  gap={r['n_gap']}")


if __name__ == "__main__":
    main()
