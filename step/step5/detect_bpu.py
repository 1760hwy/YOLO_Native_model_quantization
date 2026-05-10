#!/usr/bin/env python3
"""
RDK X5 BPU 火焰检测推理脚本 v3 - NV12 输入（完整流程）
量化配置关键信息:
  input_type_rt: 'nv12'   → 运行时喂 NV12 uint8 [H*3//2, W]
  BPU 内部自动: NV12→RGB→×(1/255) 归一化
  cls Conv: ON=CPU, float32 输出（解决 INT8 量化失效）
  输出: 6 路 float32, NCHW or NHWC 自适应 (3 bbox + 3 cls)

NV12 格式说明:
  shape = [H*3//2, W]，例如 640x640 → [960, 640]
  前 H 行: Y 亮度平面
  后 H//2 行: UV 色度平面（U/V 交错排列）

运行方式 (在 RDK X5 上):
  python3 detect_bpu.py [--model fire_detect_bayese_640x640_nv12.bin] [--conf 0.25] [--debug]
"""

import sys
import time
import argparse
import json
import numpy as np
import cv2
from pathlib import Path

SCRIPT_DIR  = Path(__file__).parent
MODEL_PATH  = SCRIPT_DIR / "fire_detect_bayese_640x640_nv12.bin"
IMG_DIR     = SCRIPT_DIR / "test_images"
RESULT_DIR  = SCRIPT_DIR / "results"

IMGSZ       = 640
REG_MAX     = 16
CLASS_NAMES = ["fire"]


# ── 预处理 ───────────────────────────────────────────────────────────────────

def letterbox(img, size=640, pad_val=114):
    h, w = img.shape[:2]
    scale = min(size / h, size / w)
    new_h, new_w = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), pad_val, dtype=np.uint8)
    top  = (size - new_h) // 2
    left = (size - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = resized
    return canvas, scale, (left, top)


def bgr_to_nv12(img_bgr):
    """
    BGR uint8 → Letterbox → NV12 uint8 [H*3//2, W]

    NV12 格式 (以 640×640 为例):
      Y 平面:  [640, 640]  (亮度)
      UV 平面: [320, 640]  (U/V 交错, 2×2 降采样)
      拼接后:  [960, 640]

    BPU 内部自动完成 NV12→RGB→×(1/255)，板端无需手动归一化。
    返回: (nv12 [H*3//2, W] uint8, scale float, pad (left, top))
    """
    canvas, scale, pad = letterbox(img_bgr, IMGSZ)

    # BGR → YUV I420 (OpenCV 内置，I420: YYYY...UU..VV..)
    yuv_i420 = cv2.cvtColor(canvas, cv2.COLOR_BGR2YUV_I420)

    # 拆分 Y / U / V 平面
    y   = yuv_i420[:IMGSZ, :]                                        # [H, W]
    u   = yuv_i420[IMGSZ  : IMGSZ + IMGSZ // 4, :].reshape(IMGSZ // 2, IMGSZ // 2)
    v   = yuv_i420[IMGSZ + IMGSZ // 4 :        , :].reshape(IMGSZ // 2, IMGSZ // 2)

    # 交错 U/V → NV12 的 UV 平面 [H//2, W]
    uv  = np.stack([u, v], axis=-1).reshape(IMGSZ // 2, IMGSZ)

    # 拼接 Y + UV → NV12 [H*3//2, W]
    nv12 = np.concatenate([y, uv], axis=0).astype(np.uint8)
    return nv12, scale, pad


# ── 输出解析 ─────────────────────────────────────────────────────────────────

def parse_outputs(raw_outs, debug=False):
    """
    将 hobot_dnn forward() 的 6 路输出按形状自适应分类。

    支持 NCHW [N,C,H,W] 和 NHWC [N,H,W,C] 两种输出格式:
      bbox: C = REG_MAX * 4 = 64
      cls:  C = len(CLASS_NAMES) = 1
    返回 [bbox_s8, bbox_s16, bbox_s32, cls_s8, cls_s16, cls_s32]
    """
    strides    = [8, 16, 32]
    expected_h = {IMGSZ // s: s for s in strides}   # {80:8, 40:16, 20:32}
    num_cls    = len(CLASS_NAMES)
    bbox_ch    = REG_MAX * 4   # 64

    bbox_map, cls_map = {}, {}   # key: H (spatial height)

    for i, o in enumerate(raw_outs):
        shape = list(o.properties.shape)
        arr   = np.array(o.buffer, dtype=np.float32).reshape(shape)

        if debug:
            print(f"  raw_out[{i}] shape={shape}  "
                  f"min={arr.min():.3f} max={arr.max():.3f} mean={arr.mean():.3f}")

        if len(shape) != 4:
            print(f"  [WARN] output[{i}] shape={shape} 非 4D，跳过")
            continue

        # 自适应 NCHW / NHWC
        if shape[1] == bbox_ch:          # NCHW bbox [N,64,H,W]
            h, fmt = shape[2], "NCHW"
            arr = arr.transpose(0, 2, 3, 1)   # → NHWC
        elif shape[1] in (num_cls, 1):   # NCHW cls  [N,1,H,W]
            h, fmt = shape[2], "NCHW"
            arr = arr.transpose(0, 2, 3, 1)   # → NHWC
        elif shape[3] == bbox_ch:        # NHWC bbox [N,H,W,64]
            h, fmt = shape[1], "NHWC"
        elif shape[3] in (num_cls, 1):   # NHWC cls  [N,H,W,1]
            h, fmt = shape[1], "NHWC"
        else:
            print(f"  [WARN] output[{i}] shape={shape} 无法识别，跳过")
            continue

        if h not in expected_h:
            print(f"  [WARN] output[{i}] H={h} 不在期望 {list(expected_h.keys())} 中")
            continue

        c = arr.shape[3]   # 转换后统一是 NHWC
        if c == bbox_ch:
            bbox_map[h] = arr
        else:
            cls_map[h] = arr

        if debug:
            print(f"    → {fmt} H={h} C={c} ({'bbox' if c==bbox_ch else 'cls'})")

    if len(bbox_map) != 3 or len(cls_map) != 3:
        print(f"[ERROR] 输出解析失败: bbox_keys={list(bbox_map.keys())} cls_keys={list(cls_map.keys())}")
        sys.exit(1)

    ordered = []
    for s in strides:
        ordered.append(bbox_map[IMGSZ // s])
    for s in strides:
        ordered.append(cls_map[IMGSZ // s])
    return ordered   # [bbox_80, bbox_40, bbox_20, cls_80, cls_40, cls_20]  NHWC


# ── 后处理 ───────────────────────────────────────────────────────────────────

def dfl_decode(pred):
    """pred: [N, REG_MAX*4] → [N, 4]"""
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
    """outs: [bbox_s8, bbox_s16, bbox_s32, cls_s8, cls_s16, cls_s32] NHWC float32"""
    strides   = [8, 16, 32]
    all_boxes = []

    for i, stride in enumerate(strides):
        bf = outs[i][0].reshape(-1, REG_MAX * 4)           # [HW, 64]
        cf = outs[i + 3][0].reshape(-1, len(CLASS_NAMES))  # [HW, 1]

        cf_min, cf_max = float(cf.min()), float(cf.max())
        # cls Conv ON=CPU → float32 logit；若 sigmoid 已融合则值在 (0,1)
        if cf_min >= 0.0 and cf_max <= 1.0:
            cp = cf
        else:
            cp = 1.0 / (1.0 + np.exp(-cf.clip(-88, 88)))

        if debug:
            print(f"  stride={stride}: cls logit [{cf_min:.3f}, {cf_max:.3f}]  "
                  f"sigmoid_max={float(cp.max()):.4f}")

        ms, mc = cp.max(axis=-1), cp.argmax(axis=-1)
        mask   = ms > conf_thres

        if debug:
            print(f"  stride={stride}: conf>{conf_thres} 候选: {mask.sum()}")

        if not mask.any():
            continue

        h_f = outs[i].shape[1]
        w_f = outs[i].shape[2]
        anc  = make_anchors(h_f, w_f)[mask]
        dist = dfl_decode(bf[mask])
        x1   = (anc[:, 0] - dist[:, 0]) * stride
        y1   = (anc[:, 1] - dist[:, 1]) * stride
        x2   = (anc[:, 0] + dist[:, 2]) * stride
        y2   = (anc[:, 1] + dist[:, 3]) * stride
        all_boxes.append(
            np.stack([x1, y1, x2, y2, ms[mask], mc[mask].astype(np.float32)], 1)
        )

    if not all_boxes:
        return np.zeros((0, 6), np.float32)
    return np.concatenate(all_boxes, 0)


def nms(boxes, iou_thres):
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
            iw = (np.minimum(x2[i], x2[order[1:]]) - np.maximum(x1[i], x1[order[1:]])).clip(0)
            ih = (np.minimum(y2[i], y2[order[1:]]) - np.maximum(y1[i], y1[order[1:]])).clip(0)
            inter = iw * ih
            union = ((x2[i]-x1[i])*(y2[i]-y1[i])
                     + (x2[order[1:]]-x1[order[1:]])*(y2[order[1:]]-y1[order[1:]])
                     - inter + 1e-6)
            order = order[np.concatenate(([0], np.where(inter / union <= iou_thres)[0] + 1))][1:]
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


# ── 主程序 ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="RDK X5 BPU 火焰检测 (NV12 输入)")
    parser.add_argument("--model",   default=str(MODEL_PATH))
    parser.add_argument("--img_dir", default=str(IMG_DIR))
    parser.add_argument("--out_dir", default=str(RESULT_DIR))
    parser.add_argument("--conf",    type=float, default=0.25)
    parser.add_argument("--iou",     type=float, default=0.45)
    parser.add_argument("--debug",   action="store_true",
                        help="打印首张图每路输出 shape、cls logit 范围、候选框数量")
    args = parser.parse_args()

    try:
        from hobot_dnn import pyeasy_dnn as dnn
    except ImportError:
        print("[ERROR] 未找到 hobot_dnn，请在 RDK X5 开发板上运行")
        sys.exit(1)

    model_path = Path(args.model)
    img_dir    = Path(args.img_dir)
    result_dir = Path(args.out_dir)

    print("=" * 65)
    print("RDK X5 BPU 火焰检测 v2 (NV12 输入)")
    print(f"  模型:     {model_path}")
    print(f"  图片目录: {img_dir}")
    print(f"  conf={args.conf}  iou={args.iou}  debug={args.debug}")
    print("=" * 65)

    if not model_path.exists():
        print(f"[ERROR] 找不到模型: {model_path}"); sys.exit(1)
    if not img_dir.exists():
        print(f"[ERROR] 找不到图片目录: {img_dir}"); sys.exit(1)

    # ── 加载模型 ──────────────────────────────────────────────────────────────
    print("[加载] 正在加载 BPU 模型 ...")
    models = dnn.load(str(model_path))
    model  = models[0]

    print(f"[模型] 输入数量: {len(model.inputs)}")
    for i, inp in enumerate(model.inputs):
        p = inp.properties
        print(f"  input [{i}]: shape={p.shape}  dtype={p.dtype}")

    print(f"[模型] 输出数量: {len(model.outputs)}")
    for i, out in enumerate(model.outputs):
        p = out.properties
        print(f"  output[{i}]: shape={p.shape}  dtype={p.dtype}")

    n_out = len(model.outputs)
    if n_out != 6:
        print(f"[ERROR] 期望 6 路输出，实际 {n_out} 路")
        sys.exit(1)

    # ── 预热 ──────────────────────────────────────────────────────────────────
    # NV12 shape: [H*3//2, W]，例如 640×640 → [960, 640]
    dummy = np.zeros((IMGSZ * 3 // 2, IMGSZ), dtype=np.uint8)
    for _ in range(3):
        model.forward([dummy])
    print("[预热] 完成\n")

    # ── 扫描图片 ──────────────────────────────────────────────────────────────
    result_dir.mkdir(parents=True, exist_ok=True)
    exts      = {".jpg", ".jpeg", ".png", ".bmp"}
    img_paths = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in exts])
    if not img_paths:
        print(f"[ERROR] {img_dir} 中没有图片"); sys.exit(1)
    print(f"[图片] 共 {len(img_paths)} 张\n")

    # ── 推理循环 ──────────────────────────────────────────────────────────────
    pre_times, infer_times, post_times = [], [], []
    records   = []
    first_img = True

    for idx, img_path in enumerate(img_paths):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  [SKIP] 无法读取: {img_path.name}"); continue

        t1 = time.time()
        nv12, scale, pad = bgr_to_nv12(img)
        t2 = time.time()

        raw_outs = model.forward([nv12])
        t3 = time.time()

        do_debug = args.debug and first_img
        if do_debug:
            print(f"\n[DEBUG] 第一张图 {img_path.name}:")
        outs  = parse_outputs(raw_outs, debug=do_debug)
        boxes = decode_outputs(outs, args.conf, debug=do_debug)
        boxes = nms(boxes, args.iou)
        t4 = time.time()
        first_img = False

        pre_ms, infer_ms, post_ms = (t2-t1)*1e3, (t3-t2)*1e3, (t4-t3)*1e3
        pre_times.append(pre_ms); infer_times.append(infer_ms); post_times.append(post_ms)

        n_det = len(boxes)
        print(f"  [{idx+1:02d}/{len(img_paths)}] {img_path.name[:46]:<48} "
              f"det={n_det}  infer={infer_ms:.1f}ms")

        vis = draw_boxes(img, boxes, scale, pad)
        cv2.imwrite(str(result_dir / f"{idx+1:03d}_{img_path.stem}_result.jpg"), vis)

        oh, ow = img.shape[:2]
        det_list = [{
            "class": CLASS_NAMES[int(b[5])],
            "conf":  round(float(b[4]), 4),
            "x1": round(float(max(0,  (b[0]-pad[0])/scale)), 1),
            "y1": round(float(max(0,  (b[1]-pad[1])/scale)), 1),
            "x2": round(float(min(ow, (b[2]-pad[0])/scale)), 1),
            "y2": round(float(min(oh, (b[3]-pad[1])/scale)), 1),
        } for b in boxes]
        records.append({"img": img_path.name, "n_det": n_det, "detections": det_list,
                        "pre_ms": round(pre_ms, 2), "infer_ms": round(infer_ms, 2),
                        "post_ms": round(post_ms, 2)})

    # ── 汇总 ──────────────────────────────────────────────────────────────────
    n = len(pre_times)
    if n == 0:
        print("[ERROR] 没有处理任何图片"); sys.exit(1)

    avg_pre   = sum(pre_times)   / n
    avg_infer = sum(infer_times) / n
    avg_post  = sum(post_times)  / n
    avg_total = avg_pre + avg_infer + avg_post
    total_det = sum(r["n_det"] for r in records)

    print()
    print("=" * 65)
    print(f"汇总 ({n} 张图片  总检测框: {total_det})")
    print(f"  预处理:   avg={avg_pre:.1f}ms")
    print(f"  BPU推理:  avg={avg_infer:.1f}ms")
    print(f"  后处理:   avg={avg_post:.1f}ms")
    print(f"  端到端:   avg={avg_total:.1f}ms  理论FPS≈{1000/avg_total:.1f}")
    print(f"  结果保存: {result_dir}/")
    print("=" * 65)

    report = {
        "model": model_path.name, "imgsz": IMGSZ, "input_type": "nv12",
        "conf_thres": args.conf, "iou_thres": args.iou,
        "num_images": n, "total_detections": total_det,
        "avg_pre_ms": round(avg_pre, 2), "avg_infer_ms": round(avg_infer, 2),
        "avg_post_ms": round(avg_post, 2), "avg_total_ms": round(avg_total, 2),
        "fps_estimate": round(1000 / avg_total, 1),
        "results": records,
    }
    rp = result_dir / "detection_report.json"
    rp.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[报告] {rp}")


if __name__ == "__main__":
    main()
