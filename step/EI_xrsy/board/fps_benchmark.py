#!/usr/bin/env python3
"""
板端FPS基准测试 — 对应论文表5和表9的FPS列
测量每个配置在valid图上的端到端延迟和FPS
"""
import argparse, time, cv2, numpy as np
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
    resized = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    padded  = np.full((size, size, 3), 114, dtype=np.uint8)
    pt, pl  = (size - nh) // 2, (size - nw) // 2
    padded[pt:pt+nh, pl:pl+nw] = resized
    yuv  = cv2.cvtColor(padded, cv2.COLOR_BGR2YUV_I420)
    y_pl = yuv[:size, :]
    u_pl = yuv[size:size + size//4, :].reshape(size//2, size//2)
    v_pl = yuv[size + size//4:,    :].reshape(size//2, size//2)
    uv_pl = np.empty((size // 2, size), dtype=np.uint8)
    uv_pl[:, 0::2] = u_pl
    uv_pl[:, 1::2] = v_pl
    return np.vstack([y_pl, uv_pl])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",  required=True, choices=list("ABCDE"))
    ap.add_argument("--model",   required=True)
    ap.add_argument("--images",  default="./valid_images")
    ap.add_argument("--n",       type=int, default=200, help="测试帧数")
    args = ap.parse_args()

    from hobot_dnn import pyeasy_dnn as dnn
    print(f"[Config {args.config}] 加载模型 {args.model} ...")
    model = dnn.load([args.model])[0]

    img_paths = sorted(Path(args.images).glob("*.jpg"))
    if not img_paths:
        print("未找到图片"); return

    # 预热 5 帧
    print("  预热中...")
    for p in img_paths[:5]:
        bgr = cv2.imread(str(p))
        if bgr is None: continue
        nv12 = bgr_to_nv12(bgr)
        model.forward(nv12.astype(np.uint8))

    # 正式测试 N 帧（循环复用图片）
    print(f"  开始计时，共 {args.n} 帧...")
    latencies = []
    for i in range(args.n):
        p = img_paths[i % len(img_paths)]
        bgr = cv2.imread(str(p))
        if bgr is None: continue
        nv12 = bgr_to_nv12(bgr)

        t0 = time.perf_counter()
        model.forward(nv12.astype(np.uint8))
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)  # ms

    latencies = np.array(latencies)
    avg_ms  = latencies.mean()
    p95_ms  = np.percentile(latencies, 95)
    fps     = 1000.0 / avg_ms

    print(f"\n[Config {args.config}] 结果（{len(latencies)} 帧）")
    print(f"  平均延迟  : {avg_ms:.1f} ms")
    print(f"  P95 延迟  : {p95_ms:.1f} ms")
    print(f"  推理 FPS  : {fps:.1f}")

if __name__ == "__main__":
    main()
