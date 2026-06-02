# musdb 작업 종료를 30초 주기로 감시 → 종료 즉시 Thermal v8 학습 자동 시작
# musdb는 사용자의 다른 프로젝트(measure_sdr_musdb)이므로 절대 건드리지 않고 종료만 대기
from __future__ import annotations

import os
import subprocess
import sys
import time
from datetime import datetime

TRAIN_DIR = os.path.dirname(os.path.abspath(__file__))


def musdb_running() -> bool:
    r = subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         "(Get-CimInstance Win32_Process -Filter \"name='python.exe'\" -ErrorAction SilentlyContinue | "
         "Where-Object { $_.CommandLine -like '*musdb*' }).Count"],
        capture_output=True, text=True, timeout=30,
    )
    out = r.stdout.strip()
    return out not in ("", "0")


def main():
    print(f"[wait] musdb 종료 감시 시작 ({datetime.now():%H:%M:%S})", flush=True)
    waited = 0
    while musdb_running():
        time.sleep(30)
        waited += 30
        if waited % 300 == 0:
            print(f"[wait] musdb 아직 실행 중... ({waited//60}분 경과)", flush=True)

    print(f"[wait] musdb 종료 감지 ({datetime.now():%H:%M:%S}) → GPU 해제 대기 10초", flush=True)
    time.sleep(10)

    print(f"[wait] Thermal v8 학습 시작 ({datetime.now():%H:%M:%S})", flush=True)
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    subprocess.run([sys.executable, "train_thermal_yolo.py"], cwd=TRAIN_DIR, env=env)


if __name__ == "__main__":
    main()
