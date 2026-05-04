# =============================================
# monitor_live.py
# 학습 출력 파일을 실시간 감시하여 training_log.txt에 append
# 새 줄이 추가되는 즉시 타임스탬프와 함께 기록
#
# 사용법 (별도 터미널):
#   cd backend/training
#   python monitor_live.py
# =============================================

import io
import os
import sys
import time
from datetime import datetime
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

LOG_FILE = Path("training_log.txt")

# 감시 대상 파일들
TASKS_DIR = os.path.expandvars(
    r"C:\Users\Codelab\AppData\Local\Temp\claude"
    r"\c--Users-Codelab-Desktop-PROJECT-TEAM-PROJECT-2-Drone-project"
    r"\958d11c0-2e1d-469a-afec-1ac2e25d0647\tasks"
)

WATCH_FILES = {
    "M5-YOLO": os.path.join(TASKS_DIR, "bsmkz9p1p.output"),
    "M1+M3-ResNet": os.path.join(TASKS_DIR, "bz20n7f0p.output"),
}


def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def tail_new_lines(path: str, last_pos: int) -> tuple:
    """파일에서 last_pos 이후 새 줄들을 읽어 반환."""
    try:
        size = os.path.getsize(path)
        if size <= last_pos:
            return [], last_pos

        with open(path, "r", encoding="utf-8", errors="replace") as f:
            f.seek(last_pos)
            new_content = f.read()
            new_pos = f.tell()

        lines = [l.strip() for l in new_content.split("\n") if l.strip()]
        return lines, new_pos
    except Exception:
        return [], last_pos


def filter_important(line: str) -> bool:
    """로그에 기록할 중요 줄인지 판별. progress bar 제외."""
    # progress bar 줄 제외 (━, ──, it/s, %, /56 등)
    if "it/s" in line or "━" in line or "──" in line:
        return False
    if "/56" in line or "/1709" in line:
        return False
    keywords = [
        "   all ", "Val Acc", "val_acc",
        "best", "Best", "DONE", "완료", "실패", "FAIL",
        "ONNX", "onnx", "저장", "save", "Device",
        "EarlyStopping", "Stopping", "patience",
        "M1R_DONE", "M3R_DONE", "M5_DONE",
    ]
    return any(kw in line for kw in keywords)


def main():
    log("=== 실시간 학습 모니터 시작 ===")
    log(f"감시 대상: {', '.join(WATCH_FILES.keys())}")

    # 현재 파일 끝부터 감시 시작 (기존 내용 무시)
    positions = {}
    for label, path in WATCH_FILES.items():
        try:
            positions[label] = os.path.getsize(path)
            log(f"  {label}: {path} (시작 위치: {positions[label]} bytes)")
        except Exception:
            positions[label] = 0
            log(f"  {label}: 파일 없음, 0부터 시작")

    log("")

    while True:
        for label, path in WATCH_FILES.items():
            new_lines, new_pos = tail_new_lines(path, positions[label])
            if new_lines:
                important = [l for l in new_lines if filter_important(l)]
                if important:
                    for line in important:
                        # ANSI escape 제거
                        clean = line
                        while "\x1b[" in clean:
                            start = clean.index("\x1b[")
                            end = clean.index("m", start) + 1 if "m" in clean[start:] else start + 4
                            clean = clean[:start] + clean[end:]
                        log(f"[{label}] {clean[:200]}")
                positions[label] = new_pos

        time.sleep(5)  # 5초마다 체크


if __name__ == "__main__":
    main()
