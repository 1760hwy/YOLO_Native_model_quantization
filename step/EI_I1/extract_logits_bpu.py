#!/usr/bin/env python3
"""
EI_I1 — 板端 cls logits 提取脚本
==================================
对指定 .bin 模型运行推理，提取 cls 分支原始 logits（sigmoid 之前），
保存为 .npy 用于 PC 端绘制诊断直方图。

每个模型输出:
  results/{model_stem}/cls_logits_all.npy   shape: (N_images * H*W_sum, 1)
  results/{model_stem}/detection_report.json

Run on RDK X5:
  python3 extract_logits_bpu.py --model fire_detect_g1_wrong_norm.bin \\
      --img_dir ./test_images --out_dir ./results
"""

import sys, time, argparse, json
import numpy as np
import cv2
from pathlib import Path

IMGSZ       = 640
REG_MAX     = 16
CLASS_NAMES = ["fire"]
CONF_THRES  = 0.10   # 低阈值，保留更多 logit 样本用于分布分析


def letterbox(img, size=640, pad_val=114):
    h, w = img.shape[:2]
    scale = min(size / h, size / w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), pad_val, dtype=np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas[top:top+nh, left:left+nw] = resized
    return canvas, scale, (left, top)


def bgr_to_nv12(img_bgr):
    canvas, scale, pad = letterbox(img_bgr, IMGSZ)
    yuv = cv2.cvtColor(canvas, cv2.COLOR_BGR2YUV_I420)
    y = yuv[:IMGSZ, :]
    u = yuv[IMGSZ: IMGSZ+IMGSZ//4, :].reshape(IMGSZ//2, IMGSZ//2)
    v = yuv[IMGSZ+IMGSZ//4:,       :].reshape(IMGSZ//2, IMGSZ//2)
    uv = np.stack([u, v], axis=-1).reshape(IMGSZ//2, IMGSZ)
    return np.concatenate([y, uv], axis=0).astype(np.uint8), scale, pad


def parse_outputs_shape_driven(raw_outs):
    """形状驱动解析，与 detect_shape_driven.py 完全一致。"""
    strides    = [8, 16, 32]
    expected_h = {IMGSZ // s: s for s in strides}
    bbox_ch    = REG_MAX * 4
    num_cls    = len(CLASS_NAMES)
    bbox_map, cls_map = {}, {}

    for o in raw_outs:
        shape = list(o.properties.shape)
        arr   = np.array(o.buffer, dtype=np.float32).reshape(shape)
        if len(shape) != 4:
            continue
        if shape[1] in (bbox_ch, num_cls, 1):
            arr = arr.transpose(0, 2, 3, 1); h = shape[2]
        elif shape[3] in (bbox_ch, num_cls, 1):
            h = shape[1]
        else:
            continue
        if h not in expected_h:
            continue
        c = arr.shape[3]
        if c == bbox_ch:
            bbox_map[h] = arr
        elif c in (num_cls, 1):
            cls_map[h] = arr

    ordered = []
    for s in strides:
        ordered.append(bbox_map[IMGSZ // s])
    for s in strides:
        ordered.append(cls_map[IMGSZ // s])
    return ordered


def dfl_decode(pred):
    pred = pred.reshape(-1, 4, REG_MAX)
    exp  = np.exp(pred - pred.max(axis=-1, keepdims=True))
    prob = exp / exp.sum(axis=-1, keepdims=True)
    return (prob * np.arange(REG_MAX, dtype=np.float32)).sum(axis=-1)


def make_anchors(h, w):
    ys = np.arange(h, dtype=np.float32) + 0.5
    xs = np.arange(w, dtype=np.float32) + 0.5
    gy, gx = np.meshgrid(ys, xs, indexing="ij")
    return np.stack([gx, gy], axis=-1).reshape(-1, 2)


def decode_and_count(outs, conf_thres):
    strides, all_boxes = [8, 16, 32], []
    for i, stride in enumerate(strides):
        bf = outs[i][0].reshape(-1, REG_MAX * 4)
        cf = outs[i+3][0].reshape(-1, len(CLASS_NAMES))
        lo, hi = float(cf.min()), float(cf.max())
        cp = cf if (lo >= 0 and hi <= 1) else 1.0 / (1.0 + np.exp(-cf.clip(-88, 88)))
        ms, mc = cp.max(axis=-1), cp.argmax(axis=-1)
        mask = ms > conf_thres
        if not mask.any():
            continue
        h_f, w_f = outs[i].shape[1], outs[i].shape[2]
        anc  = make_anchors(h_f, w_f)[mask]
        dist = dfl_decode(bf[mask])
        x1 = (anc[:,0] - dist[:,0]) * stride
        y1 = (anc[:,1] - dist[:,1]) * stride
        x2 = (anc[:,0] + dist[:,2]) * stride
        y2 = (anc[:,1] + dist[:,3]) * stride
        all_boxes.append(
            np.stack([x1, y1, x2, y2, ms[mask], mc[mask].astype(np.float32)], axis=1)
        )
    return np.concatenate(all_boxes) if all_boxes else np.zeros((0, 6), np.float32)


def main():
    parser = argparse.ArgumentParser(description="EI_I1 板端 cls logits 提取")
    parser.add_argument("--model",   required=True, help=".bin 模型路径")
    parser.add_argument("--img_dir", default="./test_images")
    parser.add_argument("--out_dir", default="./results")
    parser.add_argument("--conf",    type=float, default=CONF_THRES)
    args = parser.parse_args()

    try:
        from hobot_dnn import pyeasy_dnn as dnn
    except ImportError:
        print("[ERROR] hobot_dnn 未找到，请在 RDK X5 上运行"); sys.exit(1)

    model_path = Path(args.model)
    img_dir    = Path(args.img_dir)
    model_stem = model_path.stem
    out_dir    = Path(args.out_dir) / model_stem
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"EI_I1 logits 提取: {model_stem}")
    print(f"  图片: {img_dir}   conf={args.conf}")
    print("=" * 60)

    models = dnn.load(str(model_path))
    model  = models[0]

    dummy = np.zeros((IMGSZ*3//2, IMGSZ), dtype=np.uint8)
    for _ in range(3):
        model.forward([dummy])

    exts      = {".jpg", ".jpeg", ".png", ".bmp"}
    img_paths = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in exts])
    if not img_paths:
        print(f"[ERROR] {img_dir} 中无图片"); sys.exit(1)

    all_logits  = []   # 收集所有图片、所有尺度的 cls logits
    records     = []
    infer_times = []

    for idx, img_path in enumerate(img_paths):
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        nv12, scale, pad = bgr_to_nv12(img)
        t0 = time.time()
        raw_outs = model.forward([nv12])
        t1 = time.time()

        outs = parse_outputs_shape_driven(raw_outs)

        # ── 收集三个尺度的 cls logits ─────────────────────────
        for s_idx in range(3):
            cf = outs[s_idx + 3][0].reshape(-1, len(CLASS_NAMES))
            all_logits.append(cf)   # shape: (H*W, 1)

        boxes    = decode_and_count(outs, args.conf)
        infer_ms = (t1 - t0) * 1e3
        infer_times.append(infer_ms)

        lo = float(np.concatenate([outs[i+3][0].reshape(-1) for i in range(3)]).min())
        hi = float(np.concatenate([outs[i+3][0].reshape(-1) for i in range(3)]).max())

        print(f"  [{idx+1:02d}/{len(img_paths):02d}] {img_path.name[:40]:<42} "
              f"det={len(boxes):2d}  logit=[{lo:.2f},{hi:.2f}]  {infer_ms:.1f}ms")

        records.append({
            "img": img_path.name,
            "n_det": len(boxes),
            "logit_min": round(lo, 4),
            "logit_max": round(hi, 4),
            "infer_ms": round(infer_ms, 2),
        })

    # ── 保存 logits ───────────────────────────────────────────
    logits_all = np.concatenate(all_logits, axis=0)   # (total_pixels, 1)
    npy_path   = out_dir / "cls_logits_all.npy"
    np.save(str(npy_path), logits_all)

    # ── 统计摘要 ──────────────────────────────────────────────
    lv = logits_all.ravel()
    summary = {
        "model":        model_stem,
        "num_images":   len(records),
        "total_det":    sum(r["n_det"] for r in records),
        "logit_mean":   round(float(lv.mean()), 4),
        "logit_std":    round(float(lv.std()),  4),
        "logit_min":    round(float(lv.min()),  4),
        "logit_max":    round(float(lv.max()),  4),
        "logit_pct_positive": round(float((lv > 0).mean()) * 100, 2),
        "avg_infer_ms": round(sum(infer_times) / len(infer_times), 2),
        "logits_npy":   str(npy_path),
        "results":      records,
    }

    rp = out_dir / "logit_report.json"
    rp.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print()
    print("=" * 60)
    print(f"  logit 均值: {summary['logit_mean']:.4f}")
    print(f"  logit 最大: {summary['logit_max']:.4f}")
    print(f"  正值占比:   {summary['logit_pct_positive']:.2f}%")
    print(f"  总检测框:   {summary['total_det']}")
    print(f"  logits → {npy_path}")
    print(f"  报告  → {rp}")
    print("=" * 60)


if __name__ == "__main__":
    main()
