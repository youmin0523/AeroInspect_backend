# =============================================
# train_chain_v1_1.py
# v1.1 재학습 체이닝: Thermal → M5 → M4 → furniture 순차 자동 실행
# GPU 한 대(8GB) 순차. 한 모델 실패해도 다음 모델 계속 진행.
#
# 상태 파일 (Monitor가 5분마다 읽음):
#   runs/chain_status.txt   현재 단계 (한 줄)
#   runs/chain_active.log   현재 모델 stdout (모델 시작 시 새로 truncate)
#   runs/chain_history.log  단계 전환 이력 (append)
#
# 사용법:
#   cd backend/training
#   python train_chain_v1_1.py
# =============================================

from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

TRAIN_DIR = Path(__file__).resolve().parent
RUNS = TRAIN_DIR / "runs"
ACTIVE_LOG = RUNS / "chain_active.log"
STATUS = RUNS / "chain_status.txt"
HISTORY = RUNS / "chain_history.log"

# Thermal은 단일 실행 중(별도) → 체이닝은 나머지 3개
STAGES = [
    ("M5_FrameSeg", "train_m5_frame_seg.py"),
    ("M4_Context", "train_m4_context_yolo.py"),
    ("furniture", "train_furniture_aware.py"),
]


def now() -> str:
    return datetime.now().strftime("%m-%d %H:%M:%S")


def set_status(msg: str) -> None:
    STATUS.write_text(f"{now()} | {msg}", encoding="utf-8")


def history(msg: str) -> None:
    with open(HISTORY, "a", encoding="utf-8") as f:
        f.write(f"[{now()}] {msg}\n")


def main() -> None:
    RUNS.mkdir(parents=True, exist_ok=True)
    total = len(STAGES)
    history(f"===== 체이닝 시작 (총 {total}개 모델) =====")

    results = []
    for i, (name, script) in enumerate(STAGES, 1):
        tag = f"[{i}/{total}] {name}"
        set_status(f"{tag} 학습 중 ({script})")
        history(f"{tag} 시작 → {script}")

        # 현재 모델 로그만 보이도록 truncate + 헤더
        with open(ACTIVE_LOG, "w", encoding="utf-8") as f:
            f.write(f"{'='*60}\n[{now()}] {tag} 시작: {script}\n{'='*60}\n")

        env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
        with open(ACTIVE_LOG, "a", encoding="utf-8") as logf:
            proc = subprocess.run(
                [sys.executable, script],
                cwd=str(TRAIN_DIR),
                stdout=logf,
                stderr=subprocess.STDOUT,
                env=env,
            )

        ok = proc.returncode == 0
        result = "성공" if ok else f"실패(exit {proc.returncode})"
        results.append((name, result))
        history(f"{tag} {result}")
        with open(ACTIVE_LOG, "a", encoding="utf-8") as f:
            f.write(f"\n[{now()}] {tag} {result}\n")

    summary = " | ".join(f"{n}:{r}" for n, r in results)
    set_status(f"전체 완료 — {summary}")
    history(f"===== 체이닝 종료 — {summary} =====")
    print(f"[chain] 완료: {summary}", flush=True)


if __name__ == "__main__":
    main()
