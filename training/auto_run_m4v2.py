# =============================================
# auto_run_m4v2.py
# M4 Context 학습 종료를 감지하면 자동으로 train_m4v2_local.py 실행
# - 5분마다 M4 Context 프로세스 생존 확인
# - 종료 감지 시 train_m4v2_local.py 자동 시작
# - 로그: auto_run_m4v2.log
#
# 사용법: cd backend/training && python -u auto_run_m4v2.py (백그라운드)
# =============================================

import subprocess
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CHECK_INTERVAL = 300  # 5분마다 체크
LOG_PATH = Path("auto_run_m4v2.log")
RESULTS_CSV = Path("../../runs/detect/runs/m4_context/train/results.csv")  # 진행률 확인
EXPECTED_LAST_EPOCH = 50


def log(msg: str):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def is_m4_training_alive() -> bool:
    """train_m4_context_yolo.py 프로세스 살아있는지 확인 (Windows)."""
    try:
        result = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "Get-CimInstance Win32_Process -Filter \"Name like '%python%'\" | Where-Object { $_.CommandLine -match 'train_m4_context_yolo' } | Select-Object -First 1 -ExpandProperty ProcessId",
            ],
            capture_output=True,
            text=True,
            timeout=20,
        )
        pid = result.stdout.strip()
        return bool(pid)
    except Exception as e:
        log(f"WARN: 프로세스 체크 실패 - {e}")
        return True  # 실패 시 안전하게 살아있다고 가정


def get_current_epoch() -> int:
    """results.csv 마지막 줄에서 epoch 추출."""
    if not RESULTS_CSV.exists():
        return -1
    try:
        with open(RESULTS_CSV, "r", encoding="utf-8") as f:
            lines = f.readlines()
        if len(lines) < 2:
            return -1
        last = lines[-1].strip().split(",")
        return int(last[0])
    except Exception:
        return -1


def run_m4v2():
    """train_m4v2_local.py 실행."""
    log("=" * 60)
    log("M4 Context 종료 감지 -> train_m4v2_local.py 시작")
    log("=" * 60)

    venv_python = Path("../venv/Scripts/python.exe").resolve()
    if not venv_python.exists():
        log(f"ERROR: venv python 못 찾음: {venv_python}")
        return

    script = Path("train_m4v2_local.py").resolve()
    if not script.exists():
        log(f"ERROR: train_m4v2_local.py 못 찾음: {script}")
        return

    log(f"실행: {venv_python} -u {script}")
    start = time.time()
    try:
        subprocess.run(
            [str(venv_python), "-u", str(script)],
            cwd=str(Path(".").resolve()),
            check=False,
        )
    except Exception as e:
        log(f"ERROR: 실행 실패 - {e}")
        return

    elapsed = (time.time() - start) / 3600
    log(f"M4v2 종료 (소요 {elapsed:.1f}h)")


def main():
    log("=" * 60)
    log("auto_run_m4v2 시작 - M4 Context 종료 대기")
    log(f"  체크 간격: {CHECK_INTERVAL}초")
    log(f"  results.csv: {RESULTS_CSV.resolve()}")
    log("=" * 60)

    consecutive_dead = 0
    while True:
        alive = is_m4_training_alive()
        ep = get_current_epoch()

        if alive:
            consecutive_dead = 0
            log(f"M4 진행 중 (epoch {ep}/{EXPECTED_LAST_EPOCH})")
        else:
            consecutive_dead += 1
            log(f"M4 프로세스 미감지 ({consecutive_dead}/2)")
            # 2회 연속 미감지 시 종료로 판단 (잘못된 일시적 감지 방지)
            if consecutive_dead >= 2:
                log(f"M4 종료 확정 (마지막 epoch: {ep})")
                break

        time.sleep(CHECK_INTERVAL)

    # M4 종료 감지 후 30초 대기 (ONNX export 등 마무리)
    log("ONNX export 대기 30초")
    time.sleep(30)

    run_m4v2()
    log("auto_run_m4v2 종료")


if __name__ == "__main__":
    main()
