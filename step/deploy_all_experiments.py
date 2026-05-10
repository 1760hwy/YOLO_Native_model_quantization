#!/usr/bin/env python3
"""
全量实验复现部署脚本
1. 上传 valid labels + 实验脚本到板端
2. 启动板端实验 (nohup，后台运行)
3. 本地运行 PC浮点基线
4. 轮询等待板端完成
5. 板端完成后运行 USB 摄像头测试 (300s)
6. 运行 RTSP 模拟测试 (300s)
7. 打印全部结果汇总
"""
import paramiko, json, time, subprocess, sys, threading
from pathlib import Path

HOST = "192.168.0.106"; USER = "sunrise"; PASS = "sunrise"

LOCAL_LABELS  = Path(__file__).parents[1] / "BDF-18K/公开数据集/PublicDataset5_DFS/valid/labels"
LOCAL_BOARD   = Path(__file__).parent / "EI_I1/board"
LOCAL_XRSY    = Path(__file__).parent / "EI_xrsy/board"
LOCAL_PC_BASE = Path(__file__).parent / "EI_I1/pc_baseline.py"
LOCAL_RESULT  = Path(__file__).parent / "EI_I1/board_results.json"

REMOTE_LABELS = "/home/sunrise/EI_xrsy/valid_labels"
REMOTE_BASE   = "/home/sunrise"
BOARD_JSON    = "/home/sunrise/board_results.json"
BOARD_LOG     = "/home/sunrise/board_run.log"


def ssh_connect():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, username=USER, password=PASS, timeout=20)
    return client


def ssh_stream(client, cmd, timeout=600):
    print(f"\n  $ {cmd[:120]}")
    print("-" * 60)
    chan = client.get_transport().open_session()
    chan.get_pty()
    chan.settimeout(timeout)
    chan.exec_command(cmd)
    buf = b""
    while True:
        try:
            chunk = chan.recv(4096)
            if not chunk: break
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


def upload_labels(sftp, local_dir, remote_dir):
    print(f"\n[Upload] labels {local_dir} -> {remote_dir}")
    try:
        sftp.stat(remote_dir)
    except IOError:
        sftp.mkdir(remote_dir)
    txts = sorted(local_dir.glob("*.txt"))
    print(f"  {len(txts)} label files ...", end="", flush=True)
    for i, f in enumerate(txts):
        sftp.put(str(f), f"{remote_dir}/{f.name}")
        if (i+1) % 200 == 0:
            print(f" {i+1}", end="", flush=True)
    print(f" done ({len(txts)} files)")


def main():
    print("=" * 60)
    print(f"连接板端 {USER}@{HOST} ...")
    client = ssh_connect()
    sftp = client.open_sftp()
    print("  已连接")

    # ── Step 1: 上传 labels ───────────────────────────────────────────────────
    _, out, _ = client.exec_command(f"ls {REMOTE_LABELS}/*.txt 2>/dev/null | wc -l")
    n_existing = int(out.read().decode().strip())
    if n_existing >= 900:
        print(f"\n[1] Labels 已存在 ({n_existing} files)，跳过上传")
    else:
        print(f"\n[1] 上传 valid labels ({n_existing} 现有)")
        upload_labels(sftp, LOCAL_LABELS, REMOTE_LABELS)

    # ── Step 2: 上传板端脚本 ──────────────────────────────────────────────────
    print("\n[2] 上传板端实验脚本 ...")
    scripts = [
        (LOCAL_BOARD / "run_all_tables.py",   f"{REMOTE_BASE}/run_all_tables.py"),
        (LOCAL_BOARD / "extract_logits_40.py", f"{REMOTE_BASE}/EI_I1/board/extract_logits_40.py"),
        (LOCAL_XRSY  / "usb_cam_test.py",      f"{REMOTE_BASE}/EI_xrsy/usb_cam_test.py"),
        (LOCAL_XRSY  / "rtsp_sim_test.py",     f"{REMOTE_BASE}/EI_xrsy/rtsp_sim_test.py"),
    ]
    for local_f, remote_f in scripts:
        if local_f.exists():
            sftp.put(str(local_f), remote_f)
            print(f"  -> {local_f.name}")
        else:
            print(f"  跳过(不存在): {local_f.name}")

    # ── Step 3: 启动板端实验 (nohup 后台) ────────────────────────────────────
    print("\n[3] 启动板端实验 (后台 nohup) ...")
    _, out, _ = client.exec_command(f"test -f {BOARD_JSON} && echo EXISTS")
    if out.read().decode().strip() == "EXISTS":
        client.exec_command(f"rm {BOARD_JSON}")
    ssh_stream(client,
        f"nohup python3 -u {REMOTE_BASE}/run_all_tables.py "
        f"> {BOARD_LOG} 2>&1 & echo PID=$!",
        timeout=10)

    # ── Step 4: 本地 PC 浮点基线 ─────────────────────────────────────────────
    print("\n[4] 本地运行 PC浮点基线 (后台) ...")
    pc_log = str(LOCAL_PC_BASE.parent / "pc_baseline.log")
    pc_proc = subprocess.Popen(
        ["C:/d/Anaconda3/envs/yolo/python.exe", str(LOCAL_PC_BASE)],
        stdout=open(pc_log, "w", encoding="utf-8"),
        stderr=subprocess.STDOUT,
        cwd=str(LOCAL_PC_BASE.parents[2])
    )
    print(f"  PC baseline PID={pc_proc.pid}, log={pc_log}")

    # ── Step 5: 轮询板端完成 ──────────────────────────────────────────────────
    print("\n[5] 等待板端实验完成 (轮询中) ...")
    poll_start = time.time()
    while True:
        time.sleep(30)
        elapsed = int(time.time() - poll_start)
        _, out, _ = client.exec_command(f"test -f {BOARD_JSON} && echo DONE || echo RUNNING")
        status = out.read().decode().strip()
        # 显示最后几行 log
        _, tail_out, _ = client.exec_command(f"tail -3 {BOARD_LOG} 2>/dev/null")
        tail = tail_out.read().decode(errors="replace").strip()
        print(f"  [{elapsed}s] 板端: {status} | {tail[:80]}")
        if status == "DONE":
            break
        if elapsed > 1800:
            print("  [WARN] 超时 30min，继续读取结果")
            break

    # ── Step 6: 下载板端结果 ──────────────────────────────────────────────────
    print("\n[6] 下载板端结果 ...")
    sftp.get(BOARD_JSON, str(LOCAL_RESULT))
    with open(LOCAL_RESULT, encoding="utf-8") as f:
        board_results = json.load(f)
    print(f"  已保存至 {LOCAL_RESULT}")

    # ── Step 7: USB 摄像头测试 (300s) ────────────────────────────────────────
    print("\n[7] USB 摄像头测试 (300s) ...")
    ssh_stream(client,
        f"cd {REMOTE_BASE}/EI_xrsy && python3 -u usb_cam_test.py",
        timeout=400)

    # ── Step 8: RTSP 模拟测试 (300s) ─────────────────────────────────────────
    print("\n[8] RTSP 模拟测试 (300s) ...")
    ssh_stream(client,
        f"cd {REMOTE_BASE}/EI_xrsy && python3 -u rtsp_sim_test.py",
        timeout=400)

    # ── Step 9: 等待 PC 基线完成 ─────────────────────────────────────────────
    print("\n[9] 等待 PC 基线完成 ...")
    pc_proc.wait()
    pc_data = {}
    try:
        with open(pc_log, encoding="utf-8") as f:
            pc_output = f.read()
        print(pc_output[-2000:])
        # 解析关键数值
        import re
        for pat, key in [
            (r"logit均值\s*:\s*([-\d.]+)", "mean"),
            (r"logit最大值\s*:\s*([-\d.]+)", "max"),
            (r"正值占比\s*:\s*([\d.]+)%", "pos_pct"),
            (r"检测框数\(40张\)\s*:\s*(\d+)", "boxes"),
            (r"mAP@0\.5\(946张\)\s*:\s*([\d.]+)%", "mAP_pct"),
        ]:
            m = re.search(pat, pc_output)
            if m: pc_data[key] = float(m.group(1))
    except Exception as e:
        print(f"  解析PC基线结果出错: {e}")

    # ── 打印汇总 ──────────────────────────────────────────────────────────────
    sftp.close()
    client.close()

    print("\n" + "=" * 80)
    print("全部实验结果汇总")
    print("=" * 80)

    print("\n【表7】逐组消融结果")
    print(f"{'组别':<12} {'logit均值':>10} {'logit最大值':>12} {'正值占比':>9} {'检测框数':>10} {'mAP@0.5':>9}")
    print("-" * 65)
    for name, key in [("G1 双重归一化","G1"), ("G2 无火焰校准","G2"),
                       ("G3 INT8分类","G3"), ("G4 完整修复","G4")]:
        r = board_results.get(key)
        if r is None:
            print(f"{name:<12}  N/A")
        else:
            print(f"{name:<12}  {r['mean']:>10.2f}  {r['max']:>12.2f}  "
                  f"{r['pos_pct']:>8.2f}%  {r['boxes']:>10}  {r['mAP']*100:>8.1f}%")
    if pc_data:
        print(f"{'PC浮点基线':<12}  {pc_data.get('mean',0):>10.2f}  {pc_data.get('max',0):>12.2f}  "
              f"{pc_data.get('pos_pct',0):>8.2f}%  {int(pc_data.get('boxes',0)):>10}  "
              f"{pc_data.get('mAP_pct',0):>8.1f}%")
    print("-" * 65)

    print("\n【表9】FPS基准")
    fps = board_results.get("fps", {})
    print(f"{'配置':<6} {'模型':<8} {'平均延迟':>10} {'P95延迟':>10} {'推理FPS':>10}")
    print("-" * 46)
    cfg_names = {"A":"G1", "B":"G2", "C":"G2(shape)", "D":"G4", "E":"G4(shape)"}
    for cfg in "ABCDE":
        r = fps.get(cfg)
        if r:
            print(f"  {cfg:<4} {cfg_names[cfg]:<12} {r['avg_ms']:>10.1f}ms "
                  f"{r['p95_ms']:>10.1f}ms {r['fps']:>10.1f}")

    print("\n[全部实验完成]")
    print("=" * 80)


if __name__ == "__main__":
    main()
