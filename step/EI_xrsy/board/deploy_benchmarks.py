#!/usr/bin/env python3
"""
上传并运行全量基准测试脚本到板端
测试顺序：FPS基准 → USB摄像头 → RTSP模拟
"""
import paramiko, sys
from pathlib import Path

HOST  = "192.168.0.106"
USER  = "sunrise"
PASS  = "sunrise"
REMOTE_BASE = "/home/sunrise/EI_xrsy"
LOCAL_BOARD  = Path(__file__).parent

SCRIPTS = [
    "fps_benchmark.py",
    "usb_cam_test.py",
    "rtsp_sim_test.py",
    "run_all_benchmarks.sh",
]

def ssh_run_stream(client, cmd, timeout=600):
    print(f"\n  $ {cmd}")
    print("-" * 60)
    chan = client.get_transport().open_session()
    chan.get_pty()
    chan.settimeout(timeout)
    chan.exec_command(cmd)
    buf = b""
    while True:
        try:
            chunk = chan.recv(4096)
            if not chunk:
                break
            buf += chunk
            text = buf.decode(errors="replace")
            lines = text.split("\n")
            for line in lines[:-1]:
                print(line)
            buf = lines[-1].encode()
        except Exception:
            break
    if buf:
        print(buf.decode(errors="replace"))
    chan.close()

def main():
    print("=" * 60)
    print(f" 连接板端 {USER}@{HOST} ...")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, username=USER, password=PASS, timeout=15)
    sftp = client.open_sftp()
    print("  已连接")

    # 1. 上传脚本
    print("\n[1] 上传基准测试脚本 ...")
    for f in SCRIPTS:
        local_f = LOCAL_BOARD / f
        if not local_f.exists():
            print(f"  跳过（不存在）: {f}")
            continue
        remote_f = f"{REMOTE_BASE}/{f}"
        sftp.put(str(local_f), remote_f)
        print(f"  -> {f}")

    ssh_run_stream(client, f"chmod +x {REMOTE_BASE}/run_all_benchmarks.sh")

    # 2. 确认valid_images是否还在
    print("\n[2] 确认valid_images存在 ...")
    _, stdout, _ = client.exec_command(f"ls {REMOTE_BASE}/valid_images/*.jpg 2>/dev/null | wc -l")
    n = stdout.read().decode().strip()
    print(f"  valid_images/*.jpg: {n} 张")
    if int(n) == 0:
        print("  [!] 图片不存在，请先运行 deploy_and_run.py 上传图片")
        sftp.close(); client.close(); sys.exit(1)

    # 3. FPS基准测试（~10 min）
    print("\n[3] 运行 FPS 基准测试 (5 configs × 200 frames) ...")
    ssh_run_stream(client,
        f"cd {REMOTE_BASE} && "
        "python3 fps_benchmark.py --config A --model /home/sunrise/step5/fire_detect_g1_wrong_norm.bin --images ./valid_images --n 200 && "
        "python3 fps_benchmark.py --config B --model /home/sunrise/step5/fire_detect_g2_no_fire.bin   --images ./valid_images --n 200 && "
        "python3 fps_benchmark.py --config C --model /home/sunrise/step5/fire_detect_g2_no_fire.bin   --images ./valid_images --n 200 && "
        "python3 fps_benchmark.py --config D --model /home/sunrise/step5/fire_detect_bayese_640x640_nv12.bin --images ./valid_images --n 200 && "
        "python3 fps_benchmark.py --config E --model /home/sunrise/step5/fire_detect_bayese_640x640_nv12.bin --images ./valid_images --n 200",
        timeout=900)

    # 4. USB摄像头测试（300s）
    print("\n[4] 运行 USB 摄像头测试 (300s) ...")
    ssh_run_stream(client,
        f"cd {REMOTE_BASE} && python3 usb_cam_test.py",
        timeout=400)

    # 5. RTSP模拟测试（300s）
    print("\n[5] 运行 RTSP 模拟测试 (300s) ...")
    ssh_run_stream(client,
        f"cd {REMOTE_BASE} && python3 rtsp_sim_test.py",
        timeout=400)

    sftp.close()
    client.close()
    print("\n[OK] 全部测试完成")

if __name__ == "__main__":
    main()
