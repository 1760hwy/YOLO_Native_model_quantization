#!/usr/bin/env python3
"""
Innovation II — Control Group: Fixed-Index Output Head Decoding
===============================================================

Decodes YOLO11 BPU outputs by assuming a FIXED positional order:
    index 0, 1, 2  → bbox branches (strides 8, 16, 32)
    index 3, 4, 5  → cls  branches (strides 8, 16, 32)

This is the "naive" approach a developer would write without knowing that
the compiler may reorder output tensors at compile time.

  --simulate-reorder   Apply the same permutation used by detect_shape_driven.py
                       to expose the fixed-index decoder's failure.

Expected result matrix (2 × 2):

                    | natural order | reordered |
    ────────────────┼───────────────┼───────────┤
    fixed-index     |     ✅ OK     |  ❌ FAIL  |
    shape-driven    |     ✅ OK     |  ✅ OK    |

Run on RDK X5:
    python3 detect_fixed_index.py \\
        --model fire_detect_bayese_640x640_nv12.bin \\
        --img_dir ./test_images --out_dir ./results/fixed_index_natural

    python3 detect_fixed_index.py --simulate-reorder \\
        --model fire_detect_bayese_640x640_nv12.bin \\
        --img_dir ./test_images --out_dir ./results/fixed_index_reordered
"""

import sys, time, argparse, json
import numpy as np
import cv2
from pathlib import Path

IMGSZ       = 640
REG_MAX     = 16
CLASS_NAMES = ["fire"]

REORDER_PERM = [3, 0, 4, 1, 5, 2]


# ── Preprocessing (identical to detect_shape_driven.py) ────────────────────────

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
    u = yuv[IMGSZ: IMGSZ + IMGSZ//4, :].reshape(IMGSZ//2, IMGSZ//2)
    v = yuv[IMGSZ + IMGSZ//4:,       :].reshape(IMGSZ//2, IMGSZ//2)
    uv = np.stack([u, v], axis=-1).reshape(IMGSZ//2, IMGSZ)
    return np.concatenate([y, uv], axis=0).astype(np.uint8), scale, pad


# ── Fixed-index output parsing (CONTROL — naive approach) ─────────────────────

def parse_outputs_fixed(raw_outs, simulate_reorder=False, debug=False):
    """
    Assume positional order: [bbox_s8, bbox_s16, bbox_s32, cls_s8, cls_s16, cls_s32].
    No channel-based validation; tensors are consumed by position.

    Failure mechanism when order is wrong:
      - output[0] may be cls (channel=1) instead of bbox (channel=64).
      - dfl_decode expects reshape to (-1, 4, 16) = (-1, 64).
      - Actual shape is (-1, 1) → reshape raises ValueError.
    """
    if simulate_reorder:
        raw_outs = [raw_outs[i] for i in REORDER_PERM]
        if debug:
            print(f"  [SIM] Outputs permuted: {REORDER_PERM}  (cls/bbox interleaved)")

    ordered = []
    for i, o in enumerate(raw_outs):
        shape = list(o.properties.shape)
        arr   = np.array(o.buffer, dtype=np.float32).reshape(shape)
        if debug:
            print(f"  raw[{i}] shape={shape}")

        # Heuristic NCHW→NHWC based on shape position only — no semantic check.
        if len(shape) == 4 and shape[1] < shape[2]:
            arr = arr.transpose(0, 2, 3, 1)

        ordered.append(arr)
    return ordered   # blindly returns in whatever order they arrived


# ── Post-processing ────────────────────────────────────────────────────────────

def dfl_decode(pred):
    # Raises ValueError when pred has wrong channel count (e.g., 1 instead of 64).
    pred = pred.reshape(-1, 4, REG_MAX)
    exp  = np.exp(pred - pred.max(axis=-1, keepdims=True))
    prob = exp / exp.sum(axis=-1, keepdims=True)
    return (prob * np.arange(REG_MAX, dtype=np.float32)).sum(axis=-1)


def make_anchors(h, w):
    ys = np.arange(h, dtype=np.float32) + 0.5
    xs = np.arange(w, dtype=np.float32) + 0.5
    gy, gx = np.meshgrid(ys, xs, indexing="ij")
    return np.stack([gx, gy], axis=-1).reshape(-1, 2)


def decode_outputs_fixed(outs, conf_thres, debug=False):
    """
    Decode assuming outs[0:3]=bbox, outs[3:6]=cls at strides [8,16,32].
    No validation: if the order is wrong, this raises an exception.
    """
    strides, all_boxes = [8, 16, 32], []
    for i, stride in enumerate(strides):
        bf = outs[i][0]               # expected shape: (H*W, 64)
        cf = outs[i + 3][0]           # expected shape: (H*W, 1)

        # Flatten assuming bbox at index i, cls at index i+3.
        n_px = bf.shape[0] * bf.shape[1]
        bf   = bf.reshape(n_px, -1)   # (-1, 64) — raises if channel ≠ 64
        cf   = cf.reshape(n_px, -1)

        # dfl_decode raises ValueError if bf.shape[-1] != 64
        lo, hi = float(cf.min()), float(cf.max())
        cp = cf if (lo >= 0 and hi <= 1) else 1.0 / (1.0 + np.exp(-cf.clip(-88, 88)))
        if debug:
            print(f"  stride={stride}: logit=[{lo:.3f},{hi:.3f}]  max_conf={cp.max():.4f}")

        ms, mc = cp.max(axis=-1), cp.argmax(axis=-1)
        mask = ms > conf_thres
        if not mask.any():
            continue

        h_f, w_f = outs[i].shape[0], outs[i].shape[1]
        anc  = make_anchors(h_f, w_f)[mask]
        dist = dfl_decode(bf[mask])   # ← crashes here if bf is actually cls (ch=1)

        x1 = (anc[:, 0] - dist[:, 0]) * stride
        y1 = (anc[:, 1] - dist[:, 1]) * stride
        x2 = (anc[:, 0] + dist[:, 2]) * stride
        y2 = (anc[:, 1] + dist[:, 3]) * stride
        all_boxes.append(
            np.stack([x1, y1, x2, y2, ms[mask], mc[mask].astype(np.float32)], axis=1)
        )
    return np.concatenate(all_boxes) if all_boxes else np.zeros((0, 6), np.float32)


def nms(boxes, iou_thres):
    if len(boxes) == 0:
        return boxes
    keep_all = []
    for cls_id in np.unique(boxes[:, 5]):
        b  = boxes[boxes[:, 5] == cls_id]
        x1, y1, x2, y2, s = b[:,0], b[:,1], b[:,2], b[:,3], b[:,4]
        order = s.argsort()[::-1]
        keep  = []
        while order.size:
            i = order[0]; keep.append(i)
            if order.size == 1: break
            iw = (np.minimum(x2[i], x2[order[1:]]) - np.maximum(x1[i], x1[order[1:]])).clip(0)
            ih = (np.minimum(y2[i], y2[order[1:]]) - np.maximum(y1[i], y1[order[1:]])).clip(0)
            inter = iw * ih
            union = (x2[i]-x1[i])*(y2[i]-y1[i]) + \
                    (x2[order[1:]]-x1[order[1:]])*(y2[order[1:]]-y1[order[1:]]) - inter + 1e-6
            order = order[np.concatenate(([0], np.where(inter/union <= iou_thres)[0]+1))][1:]
        keep_all.append(b[keep])
    return np.concatenate(keep_all) if keep_all else np.zeros((0, 6), np.float32)


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
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(vis, (int(x1), int(y1)-th-4), (int(x1)+tw, int(y1)), (0, 0, 255), -1)
        cv2.putText(vis, label, (int(x1), int(y1)-2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return vis


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Fixed-index BPU decoder (Innovation II — control group)"
    )
    parser.add_argument("--model",            default="fire_detect_bayese_640x640_nv12.bin")
    parser.add_argument("--img_dir",          default="./test_images")
    parser.add_argument("--out_dir",          default="./results/fixed_index_natural")
    parser.add_argument("--conf",    type=float, default=0.25)
    parser.add_argument("--iou",     type=float, default=0.45)
    parser.add_argument("--simulate-reorder", action="store_true",
                        help="Permute raw outputs before decoding to expose index failure")
    parser.add_argument("--debug",   action="store_true")
    args = parser.parse_args()

    try:
        from hobot_dnn import pyeasy_dnn as dnn
    except ImportError:
        print("[ERROR] hobot_dnn not found — run this script on the RDK X5 board.")
        sys.exit(1)

    model_path = Path(args.model)
    img_dir    = Path(args.img_dir)
    result_dir = Path(args.out_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    mode_tag = "REORDERED" if args.simulate_reorder else "NATURAL"
    print("=" * 65)
    print(f"Fixed-Index Decoder  [{mode_tag} output order]")
    print(f"  Model:   {model_path}")
    print(f"  Images:  {img_dir}   conf={args.conf}  iou={args.iou}")
    if args.simulate_reorder:
        print(f"  Permutation: {REORDER_PERM}  (cls/bbox interleaved)")
        print(f"  *** Expects outputs in [bbox×3, cls×3] order — will fail ***")
    print("=" * 65)

    models = dnn.load(str(model_path))
    model  = models[0]

    dummy = np.zeros((IMGSZ * 3 // 2, IMGSZ), dtype=np.uint8)
    for _ in range(3):
        model.forward([dummy])

    exts      = {".jpg", ".jpeg", ".png", ".bmp"}
    img_paths = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in exts])
    if not img_paths:
        print(f"[ERROR] No images in {img_dir}"); sys.exit(1)

    print(f"[INFO] {len(img_paths)} image(s)\n")
    pre_ms_list, infer_ms_list, post_ms_list = [], [], []
    records = []
    success = 0
    errors  = []
    first   = True

    for idx, img_path in enumerate(img_paths):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  [SKIP] {img_path.name}"); continue

        t1 = time.time()
        nv12, scale, pad = bgr_to_nv12(img)
        t2 = time.time()
        raw_outs = model.forward([nv12])
        t3 = time.time()

        decode_ok  = True
        error_msg  = ""
        boxes      = np.zeros((0, 6), np.float32)

        try:
            outs  = parse_outputs_fixed(raw_outs,
                                        simulate_reorder=args.simulate_reorder,
                                        debug=(args.debug and first))
            boxes = decode_outputs_fixed(outs, args.conf, debug=(args.debug and first))
            boxes = nms(boxes, args.iou)
            success += 1
        except Exception as e:
            decode_ok = False
            error_msg = type(e).__name__ + ": " + str(e)[:80]
            errors.append({"img": img_path.name, "error": error_msg})
            print(f"  [{idx+1:02d}] DECODE FAIL — {error_msg}")

        t4 = time.time()
        first = False

        pre  = (t2 - t1) * 1e3
        infer= (t3 - t2) * 1e3
        post = (t4 - t3) * 1e3
        pre_ms_list.append(pre); infer_ms_list.append(infer); post_ms_list.append(post)

        n_det  = len(boxes)
        status = "OK" if decode_ok else "FAIL"
        if decode_ok:
            print(f"  [{idx+1:02d}/{len(img_paths):02d}] {img_path.name[:45]:<47} "
                  f"det={n_det:2d}  [{status}]  {infer:.1f}ms")

        vis = draw_boxes(img, boxes, scale, pad)
        cv2.imwrite(str(result_dir / f"{idx+1:03d}_{img_path.stem}.jpg"), vis)

        oh, ow = img.shape[:2]
        records.append({
            "img":       img_path.name,
            "decode_ok": decode_ok,
            "error":     error_msg,
            "n_det":     n_det,
            "pre_ms":    round(pre, 2),
            "infer_ms":  round(infer, 2),
            "post_ms":   round(post, 2),
        })

    n = len(pre_ms_list)
    avg_pre   = sum(pre_ms_list)   / n
    avg_infer = sum(infer_ms_list) / n
    avg_post  = sum(post_ms_list)  / n
    total_det = sum(r["n_det"] for r in records)

    print()
    print("=" * 65)
    print(f"Fixed-Index [{mode_tag}]  —  {success}/{len(img_paths)} decoded successfully")
    if errors:
        print(f"  First error: {errors[0]['error']}")
    print(f"  Total detections: {total_det}")
    print(f"  Avg infer: {avg_infer:.1f} ms  |  end-to-end: {avg_pre+avg_infer+avg_post:.1f} ms")
    print("=" * 65)

    report = {
        "method":           "fixed_index",
        "simulate_reorder": args.simulate_reorder,
        "reorder_perm":     REORDER_PERM if args.simulate_reorder else None,
        "assumption":       "outputs[0:3]=bbox, outputs[3:6]=cls at strides [8,16,32]",
        "num_images":       len(img_paths),
        "decode_success":   success,
        "decode_fail":      len(img_paths) - success,
        "decode_success_rate": f"{success/len(img_paths)*100:.1f}%",
        "total_detections": total_det,
        "errors_sample":    errors[:3],
        "avg_pre_ms":   round(avg_pre,   2),
        "avg_infer_ms": round(avg_infer, 2),
        "avg_post_ms":  round(avg_post,  2),
        "results":      records,
    }
    rp = result_dir / "detection_report.json"
    rp.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[REPORT] {rp}")


if __name__ == "__main__":
    main()
