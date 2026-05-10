#!/usr/bin/env python3
"""
全量实验复现 — 板端脚本 (分步模式)
--step logit     : 4组 logit统计 (40张, ~30s)
--step map_g1    : G1 mAP@0.5 (946张, ~2min)
--step map_g2    : G2 mAP@0.5
--step map_g3    : G3 mAP@0.5
--step map_g4    : G4 mAP@0.5
--step fps       : 5配置 FPS基准 (~2min)
每步结果追加写入 /home/sunrise/board_results.json
"""
import cv2, numpy as np, json, time, argparse
from pathlib import Path

REG_MAX     = 16
NC          = 1
CONF_THRESH = 0.25
IOU_THRESH  = 0.45
INPUT_SIZE  = 640

MODELS = {
    "G1": "/home/sunrise/EI_I1/fire_detect_g1_wrong_norm.bin",
    "G2": "/home/sunrise/EI_I1/fire_detect_g2_no_fire.bin",
    "G3": "/home/sunrise/EI_I1/fire_detect_g3_int8_cls.bin",
    "G4": "/home/sunrise/EI_I1/fire_detect_g4_full_fix.bin",
}
FPS_CONFIGS = [
    ("A", "/home/sunrise/EI_I1/fire_detect_g1_wrong_norm.bin"),
    ("B", "/home/sunrise/EI_I1/fire_detect_g2_no_fire.bin"),
    ("C", "/home/sunrise/EI_I1/fire_detect_g2_no_fire.bin"),
    ("D", "/home/sunrise/step5/fire_detect_bayese_640x640_nv12.bin"),
    ("E", "/home/sunrise/step5/fire_detect_bayese_640x640_nv12.bin"),
]

TEST40    = "/home/sunrise/EI_I1/test40"
VALID_IMG = "/home/sunrise/EI_xrsy/valid_images"
VALID_LBL = "/home/sunrise/EI_xrsy/valid_labels"
OUT_JSON  = "/home/sunrise/board_results.json"


def log(msg): print(msg, flush=True)

def load_results():
    try:
        return json.loads(Path(OUT_JSON).read_text())
    except Exception:
        return {}

def save_results(d):
    r = load_results()
    r.update(d)
    Path(OUT_JSON).write_text(json.dumps(r, indent=2, ensure_ascii=False))


# ── preprocessing ─────────────────────────────────────────────────────────────
def bgr_to_nv12(bgr, size=INPUT_SIZE):
    ih, iw = bgr.shape[:2]
    scale  = min(size / ih, size / iw)
    nh, nw = int(ih * scale), int(iw * scale)
    resized = cv2.resize(bgr, (nw, nh))
    padded  = np.full((size, size, 3), 114, dtype=np.uint8)
    pt, pl  = (size - nh) // 2, (size - nw) // 2
    padded[pt:pt+nh, pl:pl+nw] = resized
    yuv  = cv2.cvtColor(padded, cv2.COLOR_BGR2YUV_I420)
    y_pl = yuv[:size, :]
    u_pl = yuv[size:size + size//4, :].reshape(size//2, size//2)
    v_pl = yuv[size + size//4:,    :].reshape(size//2, size//2)
    uv_pl = np.empty((size // 2, size), dtype=np.uint8)
    uv_pl[:, 0::2] = u_pl; uv_pl[:, 1::2] = v_pl
    return np.vstack([y_pl, uv_pl]), scale, pt, pl


# ── decode ────────────────────────────────────────────────────────────────────
def _sigmoid(x): return 1.0 / (1.0 + np.exp(-np.clip(x, -88, 88)))
def _softmax(x):
    x = x - x.max(-1, keepdims=True); e = np.exp(x)
    return e / e.sum(-1, keepdims=True)
def _dfl(raw):
    raw = raw.reshape(raw.shape[0], 4, REG_MAX)
    return (_softmax(raw) * np.arange(REG_MAX, dtype=np.float32)).sum(-1)
def _nms(boxes, scores, thr):
    if not len(boxes): return []
    x1,y1,x2,y2 = boxes[:,0],boxes[:,1],boxes[:,2],boxes[:,3]
    areas = (x2-x1)*(y2-y1); idx = scores.argsort()[::-1]; keep = []
    while len(idx):
        i = idx[0]; keep.append(i)
        if len(idx) == 1: break
        ix1=np.maximum(x1[i],x1[idx[1:]]); iy1=np.maximum(y1[i],y1[idx[1:]])
        ix2=np.minimum(x2[i],x2[idx[1:]]); iy2=np.minimum(y2[i],y2[idx[1:]])
        inter=np.maximum(0,ix2-ix1)*np.maximum(0,iy2-iy1)
        iou=inter/(areas[i]+areas[idx[1:]]-inter+1e-6)
        idx=idx[np.where(iou<=thr)[0]+1]
    return keep

def decode(outputs, scale, pt, pl, orig_h, orig_w):
    bbox_map, cls_map = {}, {}
    for out in outputs:
        buf = np.array(out.buffer[0]); h, w, c = buf.shape
        s = INPUT_SIZE // max(h, w)
        if   c == REG_MAX * 4: bbox_map[s] = buf
        elif c == NC:          cls_map[s]  = buf
    all_logits, ab, ac = [], [], []
    for s in [8, 16, 32]:
        if s not in bbox_map or s not in cls_map: continue
        cls_raw = cls_map[s].reshape(-1).astype(np.float32)
        all_logits.append(cls_raw)
        fh, fw = bbox_map[s].shape[:2]
        gy, gx = np.mgrid[0:fh, 0:fw]
        anch = np.stack([gx+0.5, gy+0.5], -1).reshape(-1, 2).astype(np.float32)
        dist = _dfl(bbox_map[s].reshape(-1, REG_MAX*4))
        lt, rb = dist[:, :2], dist[:, 2:]
        dbox = np.concatenate([anch + (rb-lt)/2, lt+rb], -1) * s
        confs = _sigmoid(cls_raw)
        m = confs > CONF_THRESH
        if not m.any(): continue
        cx,cy,bw,bh = dbox[m,0],dbox[m,1],dbox[m,2],dbox[m,3]
        boxes = np.stack([
            ((cx-bw/2-pl)/scale).clip(0, orig_w),
            ((cy-bh/2-pt)/scale).clip(0, orig_h),
            ((cx+bw/2-pl)/scale).clip(0, orig_w),
            ((cy+bh/2-pt)/scale).clip(0, orig_h),
        ], -1)
        ab.append(boxes); ac.append(confs[m])
    logits_all = np.concatenate(all_logits) if all_logits else np.zeros(0)
    if not ab:
        return logits_all, np.empty((0,4)), np.empty(0)
    boxes_all = np.concatenate(ab); confs_all = np.concatenate(ac)
    keep = _nms(boxes_all, confs_all, IOU_THRESH)
    return logits_all, boxes_all[keep], confs_all[keep]


# ── mAP helpers ───────────────────────────────────────────────────────────────
def load_gt(lbl_path, iw, ih):
    p = Path(lbl_path)
    if not p.exists(): return []
    boxes = []
    for line in p.read_text().strip().splitlines():
        parts = line.split()
        if len(parts) < 5: continue
        cx,cy,w,h = float(parts[1]),float(parts[2]),float(parts[3]),float(parts[4])
        boxes.append([(cx-w/2)*iw,(cy-h/2)*ih,(cx+w/2)*iw,(cy+h/2)*ih])
    return boxes

def box_iou(a, b):
    x1,y1=max(a[0],b[0]),max(a[1],b[1])
    x2,y2=min(a[2],b[2]),min(a[3],b[3])
    inter=max(0,x2-x1)*max(0,y2-y1)
    ua=(a[2]-a[0])*(a[3]-a[1]); ub=(b[2]-b[0])*(b[3]-b[1])
    return inter/(ua+ub-inter+1e-6)

def calc_ap(tp_arr, n_gt):
    if n_gt == 0: return 0.0
    tp=np.array(tp_arr); tpc=np.cumsum(tp); fpc=np.cumsum(1-tp)
    rec=tpc/n_gt; prec=tpc/(tpc+fpc+1e-6)
    return sum(prec[rec>=t].max() if (rec>=t).any() else 0.0
               for t in np.linspace(0,1,11)) / 11


# ── step: logit stats ─────────────────────────────────────────────────────────
def step_logit():
    from hobot_dnn import pyeasy_dnn as dnn
    imgs = sorted(Path(TEST40).glob("*.jpg"))
    log(f"logit stats: {len(imgs)} images")
    out = {}
    for name, model_path in MODELS.items():
        if not Path(model_path).exists():
            log(f"  [{name}] not found: {model_path}"); out[name] = None; continue
        log(f"  [{name}] loading...")
        model = dnn.load([model_path])[0]
        all_logits, total_det = [], 0
        for p in imgs:
            bgr = cv2.imread(str(p))
            if bgr is None: continue
            oh, ow = bgr.shape[:2]
            nv12, scale, pt, pl = bgr_to_nv12(bgr)
            outputs = model.forward(nv12.astype(np.uint8))
            logits, boxes, _ = decode(outputs, scale, pt, pl, oh, ow)
            all_logits.append(logits); total_det += len(boxes)
        v = np.concatenate(all_logits) if all_logits else np.zeros(1)
        r = {"mean": float(v.mean()), "max": float(v.max()),
             "pos_pct": float((v > 0).mean() * 100), "boxes": total_det}
        out[name] = r
        log(f"    mean={r['mean']:.2f} max={r['max']:.2f} pos%={r['pos_pct']:.2f}% boxes={r['boxes']}")
    save_results({"logit": out})
    log("step_logit DONE")


# ── step: mAP for one group ───────────────────────────────────────────────────
def step_map(group):
    from hobot_dnn import pyeasy_dnn as dnn
    model_path = MODELS[group]
    if not Path(model_path).exists():
        log(f"[{group}] model not found"); save_results({f"mAP_{group}": None}); return
    imgs = sorted(Path(VALID_IMG).glob("*.jpg"))
    lbl_dir = Path(VALID_LBL)
    log(f"[{group}] mAP on {len(imgs)} images, labels in {lbl_dir}")
    model = dnn.load([model_path])[0]
    all_dets, n_gt = [], 0
    for i, p in enumerate(imgs):
        bgr = cv2.imread(str(p))
        if bgr is None: continue
        oh, ow = bgr.shape[:2]
        gt = load_gt(lbl_dir / (p.stem + ".txt"), ow, oh)
        n_gt += len(gt)
        nv12, scale, pt, pl = bgr_to_nv12(bgr)
        outputs = model.forward(nv12.astype(np.uint8))
        _, boxes, confs = decode(outputs, scale, pt, pl, oh, ow)
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
        if (i+1) % 200 == 0:
            log(f"  {i+1}/{len(imgs)}...")
    all_dets.sort(key=lambda x: -x[0])
    ap = calc_ap([d[1] for d in all_dets], n_gt)
    log(f"  [{group}] mAP@0.5={ap*100:.1f}% GT={n_gt} preds={len(all_dets)}")
    save_results({f"mAP_{group}": float(ap)})
    log(f"step_map_{group} DONE")


# ── step: FPS benchmark ───────────────────────────────────────────────────────
def step_fps():
    from hobot_dnn import pyeasy_dnn as dnn
    fps_imgs = sorted(Path(VALID_IMG).glob("*.jpg"))[:50]
    out = {}
    loaded = {}
    for cfg, model_path in FPS_CONFIGS:
        if not Path(model_path).exists():
            log(f"  [Config {cfg}] not found"); continue
        if model_path not in loaded:
            log(f"  [Config {cfg}] loading {model_path}")
            loaded[model_path] = dnn.load([model_path])[0]
        model = loaded[model_path]
        for p in fps_imgs[:5]:
            bgr = cv2.imread(str(p))
            if bgr is None: continue
            nv12, _, _, _ = bgr_to_nv12(bgr)
            model.forward(nv12.astype(np.uint8))
        latencies = []
        for i in range(200):
            p = fps_imgs[i % len(fps_imgs)]
            bgr = cv2.imread(str(p))
            if bgr is None: continue
            nv12, _, _, _ = bgr_to_nv12(bgr)
            t0 = time.perf_counter()
            model.forward(nv12.astype(np.uint8))
            latencies.append((time.perf_counter() - t0) * 1000)
        lat = np.array(latencies)
        r = {"avg_ms": float(lat.mean()), "p95_ms": float(np.percentile(lat, 95)),
             "fps": float(1000.0 / lat.mean())}
        out[cfg] = r
        log(f"  [Config {cfg}] avg={r['avg_ms']:.1f}ms P95={r['p95_ms']:.1f}ms FPS={r['fps']:.1f}")
    save_results({"fps": out})
    log("step_fps DONE")


# ── main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", required=True,
                    choices=["logit", "map_g1", "map_g2", "map_g3", "map_g4", "fps"])
    args = ap.parse_args()

    if   args.step == "logit":  step_logit()
    elif args.step == "map_g1": step_map("G1")
    elif args.step == "map_g2": step_map("G2")
    elif args.step == "map_g3": step_map("G3")
    elif args.step == "map_g4": step_map("G4")
    elif args.step == "fps":    step_fps()
