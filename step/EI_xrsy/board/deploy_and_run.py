#!/usr/bin/env python3
"""
上传 valid 图 + 评估脚本到板端，执行推理，拉回预测 txt
"""
import paramiko, stat, sys
from pathlib import Path

HOST  = "192.168.0.106"
USER  = "sunrise"
PASS  = "sunrise"
REMOTE_BASE = "/home/sunrise/EI_xrsy"

LOCAL_IMAGES = Path(r"C:\e\workspace\experiment\EI_yolo\BDF-18K\公开数据集\PublicDataset5_DFS\valid\images")
LOCAL_BOARD  = Path(__file__).parent
LOCAL_PREDS  = Path(__file__).parent.parent / "preds"

REMOTE_IMAGES = f"{REMOTE_BASE}/valid_images"
REMOTE_PREDS  = f"{REMOTE_BASE}/preds"


def progress(n, total, prefix=""):
    pct = n / total * 100
    bar = "#" * int(pct / 2) + "." * (50 - int(pct / 2))
    print(f"\r{prefix} [{bar}] {n}/{total} ({pct:.0f}%)", end="", flush=True)


def sftp_mkdir_p(sftp, remote_path):
    parts = remote_path.replace("\\", "/").split("/")
    cur = ""
    for p in parts:
        if not p:
            cur = "/"; continue
        cur = cur + "/" + p if cur != "/" else "/" + p
        try:
            sftp.stat(cur)
        except FileNotFoundError:
            sftp.mkdir(cur)


def upload_dir(sftp, local_dir, remote_dir):
    sftp_mkdir_p(sftp, remote_dir)
    files = sorted(local_dir.glob("*.jpg"))
    for i, f in enumerate(files, 1):
        sftp.put(str(f), f"{remote_dir}/{f.name}")
        progress(i, len(files), "  上传图片")
    print()
    return len(files)


def download_dir(sftp, remote_dir, local_dir):
    local_dir.mkdir(parents=True, exist_ok=True)
    try:
        entries = sftp.listdir_attr(remote_dir)
    except FileNotFoundError:
        print(f"  远端目录不存在: {remote_dir}")
        return 0
    txts = [e for e in entries if e.filename.endswith(".txt")]
    for i, entry in enumerate(txts, 1):
        sftp.get(f"{remote_dir}/{entry.filename}", str(local_dir / entry.filename))
        progress(i, len(txts), "  下载预测")
    print()
    return len(txts)


def ssh_run(client, cmd, timeout=600):
    print(f"  $ {cmd}")
    _, stdout, stderr = client.exec_command(cmd, timeout=timeout, get_pty=True)
    out = stdout.read().decode(errors="replace")
    err = stderr.read().decode(errors="replace")
    if out.strip(): print(out.strip())
    if err.strip(): print("[STDERR]", err.strip()[:500])
    return out


def main():
    print("=" * 60)
    print(f" 连接板端 {USER}@{HOST} …")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, username=USER, password=PASS, timeout=15)
    sftp = client.open_sftp()
    print("  已连接")

    # 1. 上传脚本
    print("\n[1/4] 上传评估脚本 …")
    sftp_mkdir_p(sftp, REMOTE_BASE)
    for f in ["board_eval.py", "run_eval_all.sh"]:
        sftp.put(str(LOCAL_BOARD / f), f"{REMOTE_BASE}/{f}")
        print(f"  → {f}")
    ssh_run(client, f"chmod +x {REMOTE_BASE}/run_eval_all.sh")

    # 2. 上传 valid 图片（946 张）
    print("\n[2/4] 上传 valid 图片（946 张）…")
    n = upload_dir(sftp, LOCAL_IMAGES, REMOTE_IMAGES)
    print(f"  共上传 {n} 张")

    # 3. 板端推理（五个配置）
    print("\n[3/4] 板端推理（A/B/C/D/E）…")
    ssh_run(client, f"cd {REMOTE_BASE} && bash run_eval_all.sh", timeout=1800)

    # 4. 下载预测 txt
    print("\n[4/4] 下载预测结果 …")
    for cfg in list("ABCDE"):
        rd = f"{REMOTE_PREDS}/config_{cfg}"
        ld = LOCAL_PREDS / f"config_{cfg}"
        n = download_dir(sftp, rd, ld)
        print(f"  Config {cfg}: {n} 个 txt 文件")

    sftp.close()
    client.close()
    print("\n[OK] 全部完成，运行 calc_map.py 计算 mAP")


if __name__ == "__main__":
    main()
