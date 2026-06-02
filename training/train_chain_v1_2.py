# =============================================
# train_chain_v1_2.py
# v1.2 재학습 체이닝 (Recall·Precision 균형 재설계)
#
# 변경점 (v1.1 → v1.2):
#   - M4 bbox → seg 전환 (train_m4_context_seg.py)
#   - Thermal Moisture/delam YOLO 포기 → PatchCore anomaly로 전환
#   - furniture는 coco 보강 데이터로 학습
#
# 사전 조건:
#   - prepare_thermal_anomaly.py 먼저 실행 완료 (정상 패치 생성)
#   - chain이 시작될 때 datasets/thermal_anomaly/good/ 존재 확인
#
# 8GB GPU 순차 실행. 한 단계 실패해도 다음 단계 계속.
#
# 상태 파일 (Monitor가 5분마다 읽음):
#   runs/chain_status.txt   현재 단계 (한 줄)
#   runs/chain_active.log   현재 모델 stdout (모델 시작 시 새로 truncate)
#   runs/chain_history.log  단계 전환 이력 (append)
#
# 사용법:
#   cd backend/training
#   python train_chain_v1_2.py
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

# 우선순위: M4 seg(가장 오래걸림) → thermal_anomaly → furniture
# v1.1과 달리 M4 먼저 시작 — 학습 길이 가장 길고, 다른 모델 의존성 없음
STAGES = [
    ("M4_Seg", "train_m4_context_seg.py"),
    ("ThermalAnomaly", "train_thermal_anomaly.py"),
    ("Furniture", "train_furniture_aware.py"),
]


def now() -> str:
    return datetime.now().strftime("%m-%d %H:%M:%S")


def set_status(msg: str) -> None:
    STATUS.write_text(f"{now()} | {msg}", encoding="utf-8")


def history(msg: str) -> None:
    with open(HISTORY, "a", encoding="utf-8") as f:
        f.write(f"[{now()}] {msg}\n")


def precondition_ok(script: str) -> tuple[bool, str]:
    """단계별 사전조건 검증."""
    if script == "train_thermal_anomaly.py":
        anomaly_dir = TRAIN_DIR / "datasets" / "thermal_anomaly" / "good"
        if not anomaly_dir.exists():
            return False, f"정상 패치 디렉토리 없음: {anomaly_dir}"
        n = len(list(anomaly_dir.glob("*.jpg")))
        if n < 100:
            return False, f"정상 패치 부족 ({n} < 100). prepare_thermal_anomaly.py 먼저 실행"
        return True, f"정상 패치 {n}개 확인"
    return True, ""


def main() -> None:
    RUNS.mkdir(parents=True, exist_ok=True)
    total = len(STAGES)
    history(f"===== v1.2 체이닝 시작 (총 {total}개 모델) =====")

    results = []
    for i, (name, script) in enumerate(STAGES, 1):
        tag = f"[{i}/{total}] {name}"

        ok, msg = precondition_ok(script)
        if not ok:
            set_status(f"{tag} 사전조건 미충족 — 스킵 ({msg})")
            history(f"{tag} 스킵: {msg}")
            results.append((name, f"스킵 ({msg})"))
            continue

        set_status(f"{tag} 학습 중 ({script})")
        history(f"{tag} 시작 → {script} | {msg}")

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

        ok_run = proc.returncode == 0
        result = "성공" if ok_run else f"실패(exit {proc.returncode})"
        results.append((name, result))
        history(f"{tag} {result}")
        with open(ACTIVE_LOG, "a", encoding="utf-8") as f:
            f.write(f"\n[{now()}] {tag} {result}\n")

    summary = " | ".join(f"{n}:{r}" for n, r in results)
    set_status(f"v1.2 전체 완료 — {summary}")
    history(f"===== v1.2 체이닝 종료 — {summary} =====")
    print(f"[chain v1.2] 완료: {summary}", flush=True)


if __name__ == "__main__":
    main()
