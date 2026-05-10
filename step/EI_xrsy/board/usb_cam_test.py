#!/usr/bin/env python3
"""
USB摄像头推理性能测试 — 对应论文表8 USB摄像头行
640x480@30fps，持续300秒，统计延迟/FPS/掉帧率
"""
import cv2, numpy as np, time, sys
from pathlib import Path

REG_MAX    = 16
NC         = 1
CONF_THRESH= 0.25
IOU_THRESH = 0.45
INPUT_SIZE = 640
MODEL_PATH = "/home/sunrise/step5/fire_detect_bayese_640x640_nv12.bin"
CAM_IDX    = 0    # /dev/video0
TEST_SECS  = 300

def bgr_to_nv12(bgr, size=INPUT_SIZE):
    ih, iw = bgr.shape[:2]
    scale  = min(size / ih, size / iw)
    nh, nw = int(ih * scale), int(iw * scale)
    resized = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    padded  = np.full((size, size, 3), 114, dtype=np.uint8)
    pt, pl  = (size - nh) // 2, (size - nw) // 2
    padded[pt:pt+nh, pl:pl+nw] = resized
    yuv   = cv2.cvtColor(padded, cv2.COLOR_BGR2YUV_I420)
    y_pl  = yuv[:size, :]
    u_pl  = yuv[size:size + size//4, :].reshape(size//2, size//2)
    v_pl  = yuv[size + size//4:,    :].reshape(size//2, size//2)
    uv_pl = np.empty((size // 2, size), dtype=np.uint8)
    uv_pl[:, 0::2] = u_pl; uv_pl[:, 1::2] = v_pl
    return np.vstack([y_pl, uv_pl]), scale, (size-nh)//2, (size-nw)//2

def main():
    from hobot_dnn import pyeasy_dnn as dnn
    print(f"加载模型 {MODEL_PATH} ...")
    model = dnn.load([MODEL_PATH])[0]

    cap = cv2.VideoCapture(CAM_IDX)
    if not cap.isOpened():
        print(f"无法打开摄像头 /dev/video{CAM_IDX}"); sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    actual_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"摄像头: {actual_w}x{actual_h}@{actual_fps:.0f}fps")
    print(f"测试时长: {TEST_SECS}s")

    latencies     = []
    total_frames  = 0
    drop_frames   = 0
    start_wall    = time.perf_counter()

    while time.perf_counter() - start_wall < TEST_SECS:
        ret, frame = cap.read()
        if not ret:
            drop_frames += 1
            continue
        total_frames += 1

        nv12, scale, pt, pl = bgr_to_nv12(frame)
        t0 = time.perf_counter()
        model.forward(nv12.astype(np.uint8))
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)

    cap.release()
    elapsed = time.perf_counter() - start_wall

    lat   = np.array(latencies)
    avg   = lat.mean()
    p95   = np.percentile(lat, 95)
    infer_fps  = 1000.0 / avg
    video_fps  = total_frames / elapsed
    drop_rate  = drop_frames / max(1, total_frames + drop_frames) * 100

    print(f"\n=== USB摄像头测试结果 ===")
    print(f"输入规格      : {actual_w}x{actual_h}@{actual_fps:.0f}fps")
    print(f"测试帧数      : {total_frames}")
    print(f"测试时长      : {elapsed:.1f}s")
    print(f"平均延迟      : {avg:.1f} ms")
    print(f"P95延迟       : {p95:.1f} ms")
    print(f"推理吞吐FPS   : {infer_fps:.1f}")
    print(f"实际视频FPS   : {video_fps:.1f}")
    print(f"掉帧率        : {drop_rate:.1f}%")
    print(f"  → 论文表8期望: avg=25.9ms, P95=33.2ms, 推理FPS=38.7, 视频FPS=25.7, 掉帧=14.3%")

if __name__ == "__main__":
    main()
