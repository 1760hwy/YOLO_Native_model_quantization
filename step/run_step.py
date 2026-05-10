#!/usr/bin/env python3
"""单步执行板端实验的辅助脚本，每步直接阻塞直到完成"""
import paramiko, sys, time

HOST = "192.168.0.106"; USER = "sunrise"; PASS = "sunrise"
SCRIPT = "/home/sunrise/run_all_tables.py"

def run_step(step, timeout=300):
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, username=USER, password=PASS, timeout=20)
    cmd = f"python3 -u {SCRIPT} --step {step}"
    print(f"\n{'='*60}\n[{step}] {cmd}\n{'-'*60}")
    chan = client.get_transport().open_session()
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
                print(line, flush=True)
            buf = lines[-1].encode()
        except Exception as e:
            print(f"[recv error: {e}]"); break
    if buf:
        print(buf.decode(errors="replace"))
    rc = chan.recv_exit_status()
    chan.close(); client.close()
    print(f"[{step}] exit={rc}")
    return rc

if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else "logit"
    timeout = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    sys.exit(run_step(step, timeout))
