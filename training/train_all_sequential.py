# =============================================
# train_all_sequential.py
# 전체 모델 순차 학습 + 로그 기록
# M5 완료 대기 → M1~M3 YOLO → M1/M3 ResNet → M4 U-Net
# 로그: training/training_log.txt (append, 아침에 확인용)
#
# 사용법:
#   cd backend/training
#   python train_all_sequential.py
# =============================================

import io
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

LOG_FILE = Path("training_log.txt")
VENV_PYTHON = str(Path("../venv/Scripts/python.exe").resolve())


def log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run_script(name: str, script: str, timeout_hours: float = 6) -> bool:
    """학습 스크립트 실행. 성공 여부 반환."""
    log(f"=== {name} 학습 시작 ===")
    start = time.time()
    try:
        result = subprocess.run(
            [VENV_PYTHON, "-u", script],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=int(timeout_hours * 3600),
        )
        elapsed = (time.time() - start) / 60
        if result.returncode == 0:
            # 마지막 10줄만 로그에 기록
            last_lines = result.stdout.strip().split("\n")[-10:]
            for line in last_lines:
                log(f"  {line}")
            log(f"=== {name} 완료 ({elapsed:.1f}분) ===")
            return True
        else:
            log(f"=== {name} 실패 (exit={result.returncode}, {elapsed:.1f}분) ===")
            err_lines = result.stderr.strip().split("\n")[-5:]
            for line in err_lines:
                log(f"  ERR: {line}")
            return False
    except subprocess.TimeoutExpired:
        log(f"=== {name} 타임아웃 ({timeout_hours}시간) ===")
        return False
    except Exception as e:
        log(f"=== {name} 예외: {e} ===")
        return False


def check_onnx(path: str) -> bool:
    exists = os.path.exists(path)
    size_mb = os.path.getsize(path) / (1024 * 1024) if exists else 0
    log(f"  ONNX: {path} — {'OK' if exists else 'MISSING'} ({size_mb:.1f}MB)")
    return exists


def main():
    log("=" * 60)
    log("전체 모델 순차 학습 파이프라인 시작")
    log("=" * 60)

    wd = "../models_weights"
    results = {}

    # ── M1 YOLO 재학습 ──
    scripts = [
        # GPU 순차 실행 (RTX 5070 8GB — 한 번에 하나씩)
        ("M1-YOLO", "train_m1_yolo_structural.py", f"{wd}/m1_yolo_structural.onnx"),
        ("M2-YOLO", "train_m2_yolo_surface.py", f"{wd}/m2_yolo_surface.onnx"),
        ("M3-YOLO", "train_m3_yolo_floor_window.py", f"{wd}/m3_yolo_floor_window.onnx"),
        ("M1-ResNet", "train_m1_resnet_crack.py", f"{wd}/m1_resnet_crack_classifier.onnx"),
        ("M3-ResNet", "train_m3_resnet_floor_window.py", f"{wd}/m3_resnet_floor_window_classifier.onnx"),
        ("M4-UNet", "train_m4_thermal_unet.py", f"{wd}/m4_unet_thermal_insulation.onnx"),
        ("M5-YOLO-v2", "train_m5_frame_seg.py", f"{wd}/m5_yolo_seg_frames.onnx"),  # 배경 축소 재학습
    ]

    for name, script, onnx_path in scripts:
        if not os.path.exists(script):
            log(f"  {name}: 스크립트 없음 ({script}) — 스킵")
            results[name] = "SKIP"
            continue

        success = run_script(name, script)
        results[name] = "OK" if success else "FAIL"

        if success:
            check_onnx(onnx_path)

        log("")

    # ── 최종 요약 ──
    log("=" * 60)
    log("최종 학습 결과 요약")
    log("=" * 60)
    for name, status in results.items():
        icon = "PASS" if status == "OK" else status
        log(f"  {icon:6s} {name}")

    # 평가 실행
    log("")
    log("=== 전체 모델 평가 시작 ===")
    run_script("Evaluation", "eval/evaluate_all.py", timeout_hours=1)

    log("")
    log("=" * 60)
    log("전체 파이프라인 완료")
    log("=" * 60)


if __name__ == "__main__":
    main()
