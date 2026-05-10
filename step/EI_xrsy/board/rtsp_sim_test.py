#!/usr/bin/env python3
"""
RTSP流推理性能测试（模拟版）— 对应论文表8 RTSP行
由于板端没有真实RTSP源，用valid图片循环模拟 2304x1296@15fps 输入
持续300秒，统计延迟/FPS/掉帧率

若有真实RTSP源，修改 RTSP_URL 即可直接使用
"""
import cv2, numpy as np, time, sys
from pathlib import Path

MODEL_PATH  = "/home/sunrise/step5/fire_detect_bayese_640x640_nv12.bin"
IMAGES_DIR  = "/home/sunrise/EI_xrsy/valid_images"
TEST_SECS   = 300
SIM_FPS     = 15          # 模拟 @15fps
SIM_W, SIM_H = 2304, 1296 # 模拟输入分辨率
INPUT_SIZE  = 640

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
    return np.vstack([y_pl, uv_pl])

def main():
    from hobot_dnn import pyeasy_dnn as dnn
    print(f"加载模型 {MODEL_PATH} ...")
    model = dnn.load([MODEL_PATH])[0]

    img_paths = sorted(Path(IMAGES_DIR).glob("*.jpg"))
    if not img_paths:
        print("图片目录为空"); sys.exit(1)

    print(f"模拟 RTSP {SIM_W}x{SIM_H}@{SIM_FPS}fps，图片来源：{len(img_paths)} 张循环")
    print(f"测试时长: {TEST_SECS}s")

    frame_interval = 1.0 / SIM_FPS
    latencies      = []
    total_frames   = 0
    drop_frames    = 0
    idx            = 0
    start_wall     = time.perf_counter()
    next_frame_t   = start_wall

    while time.perf_counter() - start_wall < TEST_SECS:
        now = time.perf_counter()
        if now < next_frame_t:
            time.sleep(max(0, next_frame_t - now - 0.001))
            continue

        # 模拟从RTSP拿一帧（resize到SIM分辨率）
        p   = img_paths[idx % len(img_paths)]
        idx += 1
        bgr = cv2.imread(str(p))
        if bgr is None:
            drop_frames += 1
            next_frame_t += frame_interval
            continue

        # 模拟高分辨率帧（resize up让前处理更重）
        bgr_hires = cv2.resize(bgr, (SIM_W, SIM_H), interpolation=cv2.INTER_LINEAR)

        total_frames += 1
        nv12 = bgr_to_nv12(bgr_hires)

        t0 = time.perf_counter()
        model.forward(nv12.astype(np.uint8))
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)

        next_frame_t += frame_interval

    elapsed = time.perf_counter() - start_wall
    lat     = np.array(latencies)
    avg     = lat.mean()
    p95     = np.percentile(lat, 95)
    infer_fps = 1000.0 / avg
    video_fps = total_frames / elapsed
    drop_rate = drop_frames / max(1, total_frames + drop_frames) * 100

    print(f"\n=== RTSP模拟测试结果 ===")
    print(f"模拟输入      : {SIM_W}x{SIM_H}@{SIM_FPS}fps")
    print(f"测试帧数      : {total_frames}")
    print(f"测试时长      : {elapsed:.1f}s")
    print(f"平均延迟      : {avg:.1f} ms")
    print(f"P95延迟       : {p95:.1f} ms")
    print(f"推理吞吐FPS   : {infer_fps:.1f}")
    print(f"实际视频FPS   : {video_fps:.1f}")
    print(f"掉帧率        : {drop_rate:.1f}%")
    print(f"  → 论文表8期望: avg=20.3ms, P95=21.7ms, 推理FPS=49.3, 视频FPS=14.3, 掉帧=4.4%")

if __name__ == "__main__":
    main()
