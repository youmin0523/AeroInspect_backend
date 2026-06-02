# =============================================
# backup_checkpoints.py
# 학습 중 체크포인트 자동 백업 (컴퓨터 뻗을 경우 대비)
#
# 동작:
#   - 10분마다 runs/ 트리에서 best.pt / last.pt 발견되면 backups/로 복사
#   - 파일 mtime 변경 시에만 복사 (불필요 IO 방지)
#   - 백업 디렉토리: backend/training/backups/<run_name>/
#
# 사용법 (백그라운드):
#   python backup_checkpoints.py
# =============================================

from __future__ import annotations

import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

TRAIN = Path(__file__).resolve().parent
RUNS = TRAIN.parent.parent / "runs"           # ultralytics 글로벌 runs (segment/detect)
LOCAL_RUNS = TRAIN / "runs"                   # training/runs (chain status)
BACKUPS = TRAIN / "backups"
INTERVAL = 600  # 10분
PATTERNS = ["best.pt", "last.pt", "best.onnx"]


def log(msg: str) -> None:
    print(f"[backup {datetime.now():%H:%M:%S}] {msg}", flush=True)


def collect_checkpoints() -> list[Path]:
    """ultralytics runs/ 와 training/runs/ 모두 스캔."""
    cks: list[Path] = []
    for root in (RUNS, LOCAL_RUNS):
        if not root.exists():
            continue
        for pattern in PATTERNS:
            cks.extend(root.rglob(pattern))
    return cks


def backup_one(src: Path) -> bool:
    """src를 BACKUPS/<run_name>/<filename>로 복사. mtime 같으면 스킵."""
    # 부모 디렉토리 이름 = 모델 run name (e.g. m4_context_seg/train/weights/best.pt → m4_context_seg_train)
    rel_parts = src.relative_to(src.parents[3] if len(src.parents) >= 4 else src.parents[-1]).parts
    # 너무 깊은 경우 weights 윗 폴더만 사용
    if "weights" in rel_parts:
        idx = rel_parts.index("weights")
        run_id = "_".join(rel_parts[max(0, idx - 2):idx]) or "unknown"
    else:
        run_id = src.parent.name

    dst_dir = BACKUPS / run_id
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name

    if dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime:
        return False  # 더 최신 또는 동일

    try:
        shutil.copy2(src, dst)
        return True
    except (PermissionError, OSError) as e:
        log(f"  skip {src.name} ({e})")
        return False


def main() -> None:
    BACKUPS.mkdir(parents=True, exist_ok=True)
    log(f"체크포인트 백업 데몬 시작 — 10분 주기 → {BACKUPS}")
    while True:
        cks = collect_checkpoints()
        copied = 0
        for ck in cks:
            if backup_one(ck):
                copied += 1
        if copied:
            log(f"백업 {copied}건 (총 발견 {len(cks)})")
        else:
            log(f"변경 없음 (스캔 {len(cks)})")
        time.sleep(INTERVAL)


if __name__ == "__main__":
    main()
