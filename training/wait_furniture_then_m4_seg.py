# =============================================
# wait_furniture_then_m4_seg.py
# chain v1.2 완료(Furniture 끝) 감지 → M4 seg 재학습
#
# 배경:
#   - chain v1.2 첫 시도에서 M4_Seg는 bbox 라벨 80% 문제로 38초만에 실패
#   - convert_m4_bbox_to_polygon.py로 95,875개 bbox → polygon 변환 완료
#   - 라벨 무결성 검증 PASS
#   - 현재 furniture 학습 중 → 끝나면 GPU 해제됨 → M4 seg 재시도
#
# 동작:
#   - 5분마다 runs/chain_status.txt 확인
#   - "전체 완료" 감지 시 train_m4_context_seg.py 실행
#   - chain v1.2와 동일 백업/모니터링 인프라 활용
# =============================================

from __future__ import annotations

import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

TRAIN = Path(__file__).resolve().parent
RUNS = TRAIN / "runs"
STATUS = RUNS / "chain_status.txt"
ACTIVE_LOG = RUNS / "chain_active.log"
HISTORY = RUNS / "chain_history.log"
POLL = 300  # 5분


def log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[wait {ts}] {msg}", flush=True)
    with open(HISTORY, "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now():%m-%d %H:%M:%S}] (wait) {msg}\n")


def chain_completed() -> bool:
    """chain_status.txt가 '전체 완료'로 끝났는지."""
    try:
        s = STATUS.read_text(encoding="utf-8").strip()
        return "전체 완료" in s or "v1.2 전체 완료" in s
    except FileNotFoundError:
        return False


def main():
    log("M4 seg 재실행 대기 시작 (chain v1.2 완료 감지)")

    waited = 0
    while not chain_completed():
        time.sleep(POLL)
        waited += POLL
        if waited % 1800 == 0:  # 30분마다
            log(f"대기 중... {waited//60}분 경과")

    log("chain v1.2 완료 감지! M4_Seg 재학습 시작")

    # 상태 업데이트
    STATUS.write_text(
        f"{datetime.now():%m-%d %H:%M:%S} | [재시도] M4_Seg 학습 중 (train_m4_context_seg.py)",
        encoding="utf-8",
    )

    with open(ACTIVE_LOG, "w", encoding="utf-8") as f:
        f.write(f"{'='*60}\n[{datetime.now():%m-%d %H:%M:%S}] [재시도] M4_Seg 시작 (라벨 변환 후)\n{'='*60}\n")

    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    with open(ACTIVE_LOG, "a", encoding="utf-8") as logf:
        proc = subprocess.run(
            [sys.executable, "train_m4_context_seg.py"],
            cwd=str(TRAIN),
            stdout=logf,
            stderr=subprocess.STDOUT,
            env=env,
        )

    ok = proc.returncode == 0
    result = "성공" if ok else f"실패(exit {proc.returncode})"
    log(f"M4_Seg 재시도 {result}")

    final_status = f"{datetime.now():%m-%d %H:%M:%S} | v1.2 전체 완료 + M4_Seg 재시도 {result}"
    STATUS.write_text(final_status, encoding="utf-8")


if __name__ == "__main__":
    main()
