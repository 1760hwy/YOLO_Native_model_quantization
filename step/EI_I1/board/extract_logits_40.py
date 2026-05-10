#!/usr/bin/env python3
"""
表7复现：G1/G2/G3/G4在40张图上的 cls logit 统计 + 检测框数
用法: python3 extract_logits_40.py --images /home/sunrise/EI_I1/test40
"""
import argparse, cv2, numpy as np
from pathlib import Path

REG_MAX     = 16
NC          = 1
CONF_THRESH = 0.25
IOU_THRESH  = 0.45
INPUT_SIZE  = 640


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


def _sigmoid(x): return 1.0 / (1.0 + np.exp(-np.clip(x, -88, 88)))
def _softmax(x):
    x = x - x.max(-1, keepdims=True); e = np.exp(x)
    return e / e.sum(-1, keepdims=True)
def _dfl(raw):
    n = raw.shape[0]
    raw = raw.reshape(n, 4, REG_MAX)
    return (_softmax(raw) * np.arange(REG_MAX, dtype=np.float32)).sum(-1)
def _nms(boxes, scores, thr):
    if not len(boxes): return []
    x1,y1,x2,y2 = boxes[:,0],boxes[:,1],boxes[:,2],boxes[:,3]
    a=(x2-x1)*(y2-y1); idx=scores.argsort()[::-1]; keep=[]
    while len(idx):
        i=idx[0]; keep.append(i)
        if len(idx)==1: break
        ix1=np.maximum(x1[i],x1[idx[1:]]); iy1=np.maximum(y1[i],y1[idx[1:]])
        ix2=np.minimum(x2[i],x2[idx[1:]]); iy2=np.minimum(y2[i],y2[idx[1:]])
        inter=np.maximum(0,ix2-ix1)*np.maximum(0,iy2-iy1)
        iou=inter/(a[i]+a[idx[1:]]-inter+1e-6)
        idx=idx[np.where(iou<=thr)[0]+1]
    return keep


def infer_one(outputs, scale, pt, pl, orig_h, orig_w):
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
        dbox = np.concatenate([anch+(rb-lt)/2, lt+rb], -1) * s
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
        return logits_all, 0
    boxes = np.concatenate(ab); confs = np.concatenate(ac)
    keep = _nms(boxes, confs, IOU_THRESH)
    return logits_all, len(keep)


def run_group(name, model_path, images_dir):
    from hobot_dnn import pyeasy_dnn as dnn
    if not Path(model_path).exists():
        print("  [" + name + "] 模型不存在，跳过: " + model_path, flush=True)
        return None

    print("  [" + name + "] 加载 " + model_path, flush=True)
    model = dnn.load([model_path])[0]
    imgs  = sorted(Path(images_dir).glob("*.jpg"))
    print("  图片数: " + str(len(imgs)), flush=True)

    all_logits, total_det = [], 0
    for p in imgs:
        bgr = cv2.imread(str(p))
        if bgr is None: continue
        orig_h, orig_w = bgr.shape[:2]
        nv12, scale, pt, pl = bgr_to_nv12(bgr)
        outputs = model.forward(nv12.astype(np.uint8))
        logits, n = infer_one(outputs, scale, pt, pl, orig_h, orig_w)
        all_logits.append(logits); total_det += n

    if not all_logits: return None
    v = np.concatenate(all_logits)
    return {"mean": float(v.mean()), "max": float(v.max()),
            "pos%": float((v > 0).mean() * 100), "boxes": total_det, "n": len(imgs)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="/home/sunrise/EI_I1/test40")
    args = ap.parse_args()

    groups = [
        ("G1 双重归一化",     "/home/sunrise/EI_I1/fire_detect_g1_wrong_norm.bin"),
        ("G2 无火焰校准集",   "/home/sunrise/EI_I1/fire_detect_g2_no_fire.bin"),
        ("G3 INT8分类分支",   "/home/sunrise/EI_I1/fire_detect_g3_int8_cls.bin"),
        ("G4 完整修复(本文)", "/home/sunrise/EI_I1/fire_detect_g4_full_fix.bin"),
    ]

    print("\n" + "=" * 74)
    print("组别                     logit均值   logit最大值   正值占比   检测框数(40张)")
    print("-" * 74)
    for name, path in groups:
        r = run_group(name, path, args.images)
        if r is None:
            print(name + "  N/A")
        else:
            print(name.ljust(24) +
                  "  " + str(round(r["mean"], 2)).rjust(8) +
                  "  " + str(round(r["max"], 2)).rjust(12) +
                  "  " + (str(round(r["pos%"], 2)) + "%").rjust(9) +
                  "  " + str(r["boxes"]).rjust(14))
    print("=" * 74)


if __name__ == "__main__":
    main()
