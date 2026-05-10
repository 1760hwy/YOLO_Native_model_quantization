#!/usr/bin/env python3
"""
Innovation II — Experiment Group: Shape-Driven Output Head Adaptive Matching
=============================================================================

Decodes YOLO11 BPU outputs by inspecting tensor geometry (channel count and
spatial size) rather than relying on a fixed output index.

  bbox branch:  channel == REG_MAX * 4 == 64
  cls  branch:  channel == num_classes  == 1

This makes decoding robust to any output permutation the compiler produces.

Flags:
  --simulate-reorder   Shuffle raw outputs to mimic compiler reordering.
                       Shape-driven method recovers correctly; fixed-index fails.

Run on RDK X5:
    python3 detect_shape_driven.py \\
        --model fire_detect_bayese_640x640_nv12.bin \\
        --img_dir ./test_images --out_dir ./results/shape_driven_natural

    python3 detect_shape_driven.py --simulate-reorder \\
        --model fire_detect_bayese_640x640_nv12.bin \\
        --img_dir ./test_images --out_dir ./results/shape_driven_reordered
"""

import sys, time, argparse, json
import numpy as np
import cv2
from pathlib import Path

IMGSZ       = 640
REG_MAX     = 16
CLASS_NAMES = ["fire"]

# Fixed permutation that mimics a realistic compiler reordering:
#   natural:   [bbox_s8, bbox_s16, bbox_s32, cls_s8, cls_s16, cls_s32]
#   reordered: [cls_s8, bbox_s8, cls_s16, bbox_s16, cls_s32, bbox_s32]
REORDER_PERM = [3, 0, 4, 1, 5, 2]


# ── Preprocessing ──────────────────────────────────────────────────────────────

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


# ── Shape-driven output parsing (INNOVATION) ───────────────────────────────────

def parse_outputs(raw_outs, simulate_reorder=False, debug=False):
    """
    Identify bbox / cls branches by inspecting each tensor's channel dimension.
    Correct regardless of the order the compiler places outputs in raw_outs.

    Key invariants:
        bbox channel == REG_MAX * 4 == 64
        cls  channel == num_classes  == 1
    """
    if simulate_reorder:
        raw_outs = [raw_outs[i] for i in REORDER_PERM]
        if debug:
            print("  [SIM] Outputs permuted:", REORDER_PERM)

    strides   = [8, 16, 32]
    expected_h = {IMGSZ // s: s for s in strides}
    bbox_ch   = REG_MAX * 4   # 64
    num_cls   = len(CLASS_NAMES)

    bbox_map, cls_map = {}, {}

    for i, o in enumerate(raw_outs):
        shape = list(o.properties.shape)
        arr   = np.array(o.buffer, dtype=np.float32).reshape(shape)

        if debug:
            print(f"  raw[{i}] shape={shape}  min={arr.min():.3f} max={arr.max():.3f}")

        if len(shape) != 4:
            continue

        # ── Step 1: normalise to NHWC ──────────────────────────────────────
        # Check channel dimension by value, not by position.
        if shape[1] in (bbox_ch, num_cls, 1):        # NCHW
            arr = arr.transpose(0, 2, 3, 1)
            h = shape[2]
        elif shape[3] in (bbox_ch, num_cls, 1):       # NHWC
            h = shape[1]
        else:
            if debug:
                print(f"    [WARN] unrecognised shape, skip")
            continue

        if h not in expected_h:
            continue

        # ── Step 2: route by channel count ─────────────────────────────────
        c = arr.shape[3]
        if c == bbox_ch:
            bbox_map[h] = arr
            if debug: print(f"    → bbox  H={h}")
        elif c in (num_cls, 1):
            cls_map[h] = arr
            if debug: print(f"    → cls   H={h}")

    if len(bbox_map) != 3 or len(cls_map) != 3:
        raise RuntimeError(
            f"Output parsing failed: "
            f"bbox_keys={list(bbox_map.keys())}  cls_keys={list(cls_map.keys())}"
        )

    ordered = []
    for s in strides:
        ordered.append(bbox_map[IMGSZ // s])
    for s in strides:
        ordered.append(cls_map[IMGSZ // s])
    return ordered   # always [b8, b16, b32, c8, c16, c32] in NHWC


# ── Post-processing ────────────────────────────────────────────────────────────

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


def decode_outputs(outs, conf_thres, debug=False):
    strides, all_boxes = [8, 16, 32], []
    for i, stride in enumerate(strides):
        bf = outs[i][0].reshape(-1, REG_MAX * 4)
        cf = outs[i + 3][0].reshape(-1, len(CLASS_NAMES))
        lo, hi = float(cf.min()), float(cf.max())
        cp = cf if (lo >= 0 and hi <= 1) else 1.0 / (1.0 + np.exp(-cf.clip(-88, 88)))
        if debug:
            print(f"  stride={stride}: logit=[{lo:.3f},{hi:.3f}]  max_conf={cp.max():.4f}")
        ms, mc = cp.max(axis=-1), cp.argmax(axis=-1)
        mask = ms > conf_thres
        if not mask.any():
            continue
        h_f, w_f = outs[i].shape[1], outs[i].shape[2]
        anc  = make_anchors(h_f, w_f)[mask]
        dist = dfl_decode(bf[mask])
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
        cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 200, 0), 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(vis, (int(x1), int(y1)-th-4), (int(x1)+tw, int(y1)), (0, 200, 0), -1)
        cv2.putText(vis, label, (int(x1), int(y1)-2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return vis


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Shape-driven BPU decoder (Innovation II — experiment group)"
    )
    parser.add_argument("--model",            default="fire_detect_bayese_640x640_nv12.bin")
    parser.add_argument("--img_dir",          default="./test_images")
    parser.add_argument("--out_dir",          default="./results/shape_driven_natural")
    parser.add_argument("--conf",    type=float, default=0.25)
    parser.add_argument("--iou",     type=float, default=0.45)
    parser.add_argument("--simulate-reorder", action="store_true",
                        help="Apply fixed permutation to raw outputs before decoding "
                             "(simulates compiler reordering)")
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
    print(f"Shape-Driven Decoder  [{mode_tag} output order]")
    print(f"  Model:   {model_path}")
    print(f"  Images:  {img_dir}   conf={args.conf}  iou={args.iou}")
    if args.simulate_reorder:
        print(f"  Permutation: {REORDER_PERM}  (cls/bbox interleaved)")
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
    records    = []
    success    = 0
    first      = True

    for idx, img_path in enumerate(img_paths):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  [SKIP] {img_path.name}"); continue

        t1 = time.time()
        nv12, scale, pad = bgr_to_nv12(img)
        t2 = time.time()
        raw_outs = model.forward([nv12])
        t3 = time.time()

        decode_ok = True
        try:
            outs  = parse_outputs(raw_outs,
                                  simulate_reorder=args.simulate_reorder,
                                  debug=(args.debug and first))
            boxes = decode_outputs(outs, args.conf, debug=(args.debug and first))
            boxes = nms(boxes, args.iou)
            success += 1
        except Exception as e:
            print(f"  [{idx+1:02d}] DECODE ERROR: {e}")
            decode_ok = False
            boxes = np.zeros((0, 6), np.float32)

        t4 = time.time()
        first = False

        pre  = (t2 - t1) * 1e3
        infer= (t3 - t2) * 1e3
        post = (t4 - t3) * 1e3
        pre_ms_list.append(pre); infer_ms_list.append(infer); post_ms_list.append(post)

        n_det = len(boxes)
        status = "OK" if decode_ok else "FAIL"
        print(f"  [{idx+1:02d}/{len(img_paths):02d}] {img_path.name[:45]:<47} "
              f"det={n_det:2d}  [{status}]  {infer:.1f}ms")

        vis = draw_boxes(img, boxes, scale, pad)
        cv2.imwrite(str(result_dir / f"{idx+1:03d}_{img_path.stem}.jpg"), vis)

        oh, ow = img.shape[:2]
        records.append({
            "img":      img_path.name,
            "decode_ok": decode_ok,
            "n_det":    n_det,
            "pre_ms":   round(pre, 2),
            "infer_ms": round(infer, 2),
            "post_ms":  round(post, 2),
        })

    n = len(pre_ms_list)
    avg_pre   = sum(pre_ms_list)   / n
    avg_infer = sum(infer_ms_list) / n
    avg_post  = sum(post_ms_list)  / n
    total_det = sum(r["n_det"] for r in records)

    print()
    print("=" * 65)
    print(f"Shape-Driven [{mode_tag}]  —  {success}/{len(img_paths)} decoded successfully")
    print(f"  Total detections: {total_det}")
    print(f"  Avg infer: {avg_infer:.1f} ms  |  end-to-end: {avg_pre+avg_infer+avg_post:.1f} ms")
    print("=" * 65)

    report = {
        "method":          "shape_driven",
        "simulate_reorder": args.simulate_reorder,
        "reorder_perm":     REORDER_PERM if args.simulate_reorder else None,
        "num_images":       len(img_paths),
        "decode_success":   success,
        "decode_fail":      len(img_paths) - success,
        "decode_success_rate": f"{success/len(img_paths)*100:.1f}%",
        "total_detections": total_det,
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
