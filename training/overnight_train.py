# =============================================
# overnight_train.py
# 밤새 자동 학습 파이프라인
# - 현재 진행 중인 모델(M1-ResNet, M3-ResNet, M3-YOLO, M4) 완료 대기
# - GPU 여유 확보되면 다음 모델 자동 시작
# - CPU/GPU 적절 배분
# - 전체 완료 후 evaluate_all.py 자동 실행
# - 미비 모델 식별 + 재학습
# - 모든 로그 training_log.txt에 기록
#
# 사용법:
#   cd backend/training
#   python -u overnight_train.py
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
WEIGHTS_DIR = Path("../models_weights")


def log(msg: str):
    ts = datetime.now().strftime("%m/%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def get_system_load() -> str:
    """GPU + CPU + RAM 부하를 한 줄로 반환."""
    gpu_str = ""
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if r.returncode == 0:
            parts = r.stdout.strip().split(",")
            used = int(parts[0].strip())
            total = int(parts[1].strip())
            util = parts[2].strip()
            gpu_str = f"GPU:{used}/{total}MB({util}%)"
    except Exception:
        gpu_str = "GPU:N/A"

    try:
        import psutil
        cpu_pct = psutil.cpu_percent(interval=0.5)
        mem = psutil.virtual_memory()
        cpu_str = f"CPU:{cpu_pct:.0f}% RAM:{mem.used//1024//1024//1024}G/{mem.total//1024//1024//1024}G({mem.percent:.0f}%)"
    except Exception:
        cpu_str = "CPU:N/A"

    return f"{gpu_str} | {cpu_str}"


def gpu_free_mb() -> int:
    """nvidia-smi로 GPU 여유 메모리 확인."""
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        return int(r.stdout.strip())
    except Exception:
        return 0


def wait_gpu(min_free_mb: int = 4000, poll_sec: int = 300):
    """GPU 여유 메모리가 min_free_mb 이상이 될 때까지 대기."""
    while True:
        free = gpu_free_mb()
        load = get_system_load()
        if free >= min_free_mb:
            log(f"  GPU 여유 확보: {free}MB >= {min_free_mb}MB | {load}")
            return
        log(f"  대기 중... (GPU free:{free}MB, 필요:{min_free_mb}MB) | {load}")
        time.sleep(poll_sec)


def run_train(name: str, script: str, device: str = "gpu", timeout_hours: float = 6) -> bool:
    """학습 스크립트 실행."""
    load = get_system_load()
    log(f"{'='*50}")
    log(f"{name} 학습 시작 ({device.upper()}) | {load}")
    log(f"{'='*50}")

    env = os.environ.copy()
    if device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = "-1"

    start = time.time()
    try:
        result = subprocess.run(
            [VENV_PYTHON, "-u", script],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=int(timeout_hours * 3600), env=env,
        )
        elapsed = (time.time() - start) / 60

        # 마지막 15줄 로그
        lines = result.stdout.strip().split("\n")
        for line in lines[-15:]:
            if any(kw in line for kw in ["Epoch", "all ", "Val Acc", "Dice", "best", "ONNX", "완료", "저장"]):
                log(f"  {line.strip()[:150]}")

        load = get_system_load()
        if result.returncode == 0:
            log(f"{name} 완료 ({elapsed:.1f}분) | {load}")
            return True
        else:
            log(f"{name} 실패 (exit={result.returncode}, {elapsed:.1f}분) | {load}")
            for line in result.stderr.strip().split("\n")[-5:]:
                log(f"  ERR: {line.strip()[:150]}")
            return False
    except subprocess.TimeoutExpired:
        log(f"{name} 타임아웃 ({timeout_hours}시간)")
        return False
    except Exception as e:
        log(f"{name} 예외: {e}")
        return False


def check_onnx(_name: str, path: str):
    if os.path.exists(path):
        size = os.path.getsize(path) / (1024 * 1024)
        log(f"  ONNX 확인: {path} ({size:.1f}MB)")
    else:
        log(f"  ONNX 없음: {path}")


def run_evaluation() -> dict:
    """전체 모델 평가 실행 + 결과 반환."""
    log(f"{'='*50}")
    log("전체 모델 평가 (IoU@0.5)")
    log(f"{'='*50}")

    result = subprocess.run(
        [VENV_PYTHON, "-u", "eval/evaluate_all.py"],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=3600,
    )

    for line in result.stdout.strip().split("\n"):
        if any(kw in line for kw in ["Recall", "Precision", "Accuracy", "PASS", "FAIL", "all ", "F1", "mAP"]):
            log(f"  {line.strip()[:150]}")

    # JSON 결과 읽기
    try:
        import json
        with open("eval/evaluation_results.json", "r") as f:
            return json.load(f)
    except Exception:
        return {}


def identify_weak_models(results: dict) -> list:
    """목표 미달 모델 식별."""
    # 재학습 기준: 90% 미만 → 재학습, 90~95% → PASS, 95%+ → 우수
    targets = {
        "M1-YOLO": ("recall", 0.90),
        "M1-ResNet": ("accuracy", 0.90),
        "M2-YOLO": ("recall", 0.90),
        "M2-ResNet": ("accuracy", 0.90),
        "M3-YOLO": ("recall", 0.90),
        "M3-ResNet": ("accuracy", 0.90),
    }

    weak = []
    for name, (metric, threshold) in targets.items():
        if name not in results:
            weak.append((name, metric, 0.0, threshold, "평가 결과 없음"))
            continue
        actual = results[name].get(metric, 0.0)
        if actual < threshold:
            weak.append((name, metric, actual, threshold, f"{metric}={actual:.4f} < {threshold}"))

    return weak


def main():
    log("")
    log("=" * 60)
    log("밤새 자동 학습 파이프라인 시작")
    log("=" * 60)
    log("")

    results_status = {}

    # ── Phase 1: 현재 진행 중인 모델 완료 대기 ──
    log("Phase 1: 진행 중인 모델 완료 대기 (M1-ResNet, M3-ResNet, M3-YOLO, M4)")
    log("  → 이 모델들은 별도 프로세스로 이미 학습 중")
    log("  → GPU 여유가 확보될 때까지 대기 후 다음 단계 시작")
    log("")

    # ── Phase 2a: M3-YOLO Phase 2 이어서 (GPU, 경로 수정됨) ──
    wait_gpu(4000)
    ok = run_train("M3-YOLO", "train_m3_yolo_floor_window.py", "gpu")
    results_status["M3-YOLO"] = ok
    check_onnx("M3-YOLO", str(WEIGHTS_DIR / "m3_yolo_floor_window.onnx"))
    log("")

    # ── Phase 2b: M1-YOLO 재학습 (GPU 필수) ──
    wait_gpu(4000)  # 4GB 이상 여유 시 시작
    ok = run_train("M1-YOLO", "train_m1_yolo_structural.py", "gpu")
    results_status["M1-YOLO"] = ok
    check_onnx("M1-YOLO", str(WEIGHTS_DIR / "m1_yolo_structural.onnx"))
    log("")

    # ── Phase 3: M2-ResNet (CPU) + M2-YOLO 데이터 확인 ──
    # M2-ResNet은 CPU로 — M2 crop 추출 먼저
    log("M2-ResNet crop 데이터 확인...")
    m2_crop_dir = Path("datasets/surface_crops/train")
    if m2_crop_dir.exists():
        import glob
        count = len(glob.glob(str(m2_crop_dir / "**/*.jpg"), recursive=True))
        log(f"  M2 crop 데이터: {count}장")
        if count < 100:
            log("  M2 crop 부족 — YOLO bbox에서 추출 시도")
            subprocess.run(
                [VENV_PYTHON, "-u", "extract_resnet_crops.py", "--model", "m2"],
                capture_output=True, text=True, encoding="utf-8", errors="replace",
                timeout=600,
            )
    else:
        log("  M2 crop 디렉토리 없음 — 추출 스크립트 실행")
        # extract_resnet_crops.py에 m2 config 추가 필요할 수 있음
    log("")

    # M2-YOLO (GPU)
    wait_gpu(4000)
    ok = run_train("M2-YOLO", "train_m2_yolo_surface.py", "gpu")
    results_status["M2-YOLO"] = ok
    check_onnx("M2-YOLO", str(WEIGHTS_DIR / "m2_yolo_surface.onnx"))
    log("")

    # ── Phase 4: M5 v2 재학습 (GPU, 배경 축소) ──
    wait_gpu(4000)
    ok = run_train("M5-v2", "train_m5_frame_seg.py", "gpu")
    results_status["M5-v2"] = ok
    check_onnx("M5-v2", str(WEIGHTS_DIR / "m5_yolo_seg_frames.onnx"))
    log("")

    # ── Phase 5: 전체 평가 ──
    eval_results = run_evaluation()
    log("")

    # ── Phase 6: 미비 모델 식별 ──
    log("=" * 60)
    log("미비 모델 분석")
    log("=" * 60)

    weak = identify_weak_models(eval_results)
    if weak:
        log(f"목표 미달 모델 {len(weak)}개:")
        for name, metric, actual, target, reason in weak:
            log(f"  {name}: {reason} (목표 {metric}>={target})")

        # ── 원인 분석: 점수가 너무 낮으면(< 0.5) 코드/데이터 문제 가능성 ──
        log("")
        log("미비 모델 원인 분석...")
        retrain_map = {
            "M1-YOLO": ("train_m1_yolo_structural.py", "datasets/structural"),
            "M2-YOLO": ("train_m2_yolo_surface.py", "datasets/surface"),
            "M3-YOLO": ("train_m3_yolo_floor_window.py", "datasets/floor_window"),
            "M1-ResNet": ("train_m1_resnet_crack.py", "datasets/structural_crops"),
            "M3-ResNet": ("train_m3_resnet_floor_window.py", "datasets/floor_window_crops"),
        }

        for name, metric, actual, target, reason in weak:
            log(f"")
            log(f"  [{name}] 분석 중... ({metric}={actual:.4f})")

            # 점수가 극도로 낮으면 (< 0.5) 코드/데이터 버그 가능성
            if actual < 0.5:
                log(f"  [{name}] ⚠ {metric}={actual:.4f} < 0.5 — 데이터/코드 문제 가능성!")

                if name in retrain_map:
                    script, dataset_dir = retrain_map[name]

                    # 데이터셋 존재 확인
                    dataset_path = Path(dataset_dir)
                    if not dataset_path.exists():
                        log(f"  [{name}] 데이터셋 디렉토리 없음: {dataset_dir}")
                        continue

                    # 클래스 분포 확인
                    if "YOLO" in name:
                        from collections import Counter
                        cls_counter = Counter()
                        label_dir = dataset_path / "labels" / "test"
                        if label_dir.exists():
                            for lbl in label_dir.glob("*.txt"):
                                for line in lbl.read_text().strip().split("\n"):
                                    if line.strip() and len(line.strip().split()) >= 5:
                                        cls_counter[int(line.strip().split()[0])] += 1
                            log(f"  [{name}] GT 클래스 분포: {dict(cls_counter)}")

                    # evaluate 코드의 클래스 순서 vs data.yaml 비교
                    data_yaml = dataset_path / "data.yaml"
                    if data_yaml.exists():
                        import yaml
                        with open(data_yaml, encoding="utf-8") as f:
                            cfg = yaml.safe_load(f)
                        log(f"  [{name}] data.yaml classes: {cfg.get('names', 'N/A')}")

                log(f"  [{name}] → 재학습 전 수동 점검 권장")
            else:
                log(f"  [{name}] {metric}={actual:.4f} — 재학습으로 개선 가능")

            # 재학습 시도
            if name in retrain_map:
                script, _ = retrain_map[name]
                device = "gpu" if "YOLO" in name else "cpu"
                if device == "gpu":
                    wait_gpu(4000)
                log(f"  [{name}] 재학습 시작 ({device.upper()})...")
                run_train(f"{name}-retry", script, device)
                log("")

        # 재평가
        log("재학습 후 재평가...")
        eval_results2 = run_evaluation()
        weak2 = identify_weak_models(eval_results2)
        if weak2:
            log(f"여전히 목표 미달: {len(weak2)}개")
            for name, metric, actual, target, reason in weak2:
                log(f"  {name}: {reason}")
            log("")
            log("수동 점검 필요 항목은 training_log.txt 확인")
        else:
            log("전 모델 목표 달성!")
    else:
        log("전 모델 목표 달성!")

    # ── 최종 요약 ──
    log("")
    log("=" * 60)
    log("최종 학습 결과 요약")
    log("=" * 60)

    all_models = ["M1-YOLO", "M1-ResNet", "M2-YOLO", "M2-ResNet", "M3-YOLO", "M3-ResNet", "M4-UNet", "M5-v2"]
    for name in all_models:
        onnx_map = {
            "M1-YOLO": "m1_yolo_structural.onnx",
            "M1-ResNet": "m1_resnet_crack_classifier.onnx",
            "M2-YOLO": "m2_yolo_surface.onnx",
            "M2-ResNet": "m2_resnet_surface_classifier.onnx",
            "M3-YOLO": "m3_yolo_floor_window.onnx",
            "M3-ResNet": "m3_resnet_floor_window_classifier.onnx",
            "M4-UNet": "m4_unet_thermal_insulation.onnx",
            "M5-v2": "m5_yolo_seg_frames.onnx",
        }
        onnx_path = WEIGHTS_DIR / onnx_map.get(name, "")
        exists = onnx_path.exists() if onnx_path.name else False
        size = onnx_path.stat().st_size / (1024 * 1024) if exists else 0

        # 평가 결과
        eval_key = name.replace("-v2", "-YOLO").replace("-UNet", "")
        perf = ""
        if eval_key in eval_results:
            r = eval_results[eval_key]
            if "recall" in r:
                perf = f"Recall={r['recall']:.4f} mAP50={r.get('mAP_50', 0):.4f}"
            elif "accuracy" in r:
                perf = f"Accuracy={r['accuracy']:.4f}"

        status = "OK" if exists else "MISSING"
        log(f"  {status:7s} {name:15s} {size:6.1f}MB  {perf}")

    log("")
    log("=" * 60)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    log(f"전체 파이프라인 완료: {ts}")
    log("=" * 60)
    log("")
    log("아침에 이 파일을 확인하세요: backend/training/training_log.txt")


if __name__ == "__main__":
    main()
