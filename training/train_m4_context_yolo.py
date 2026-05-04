# =============================================
# train_m4_context_yolo.py
# M4 Context: 벽/천장/바닥/창/문 5 클래스 객체 탐지
# - 데이터셋: datasets/m4_context (frames + floor_window 통합, ~10500장)
# - 모델: yolov8m → ONNX
# - 출력: backend/models_weights/m4_yolo_context_elements.onnx
#
# 사용법: cd backend/training && python -u train_m4_context_yolo.py
# =============================================

import sys
import shutil
from pathlib import Path

from ultralytics import YOLO

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DATA_YAML = "datasets/m4_context/data.yaml"
PROJECT = "runs/m4_context"
WEIGHTS_DIR = Path("../models_weights")
OUTPUT_NAME = "m4_yolo_context_elements"


def train():
    print("=" * 60)
    print("[M4-Context] yolov8m 학습 시작 (5 클래스: wall/ceiling/floor/window/door)")
    print("=" * 60)

    model = YOLO("yolov8m.pt")
    model.train(
        data=DATA_YAML,
        epochs=50,
        batch=4,
        imgsz=960,
        cache="disk",
        workers=4,
        optimizer="AdamW",
        lr0=1e-3,            # fresh train
        lrf=0.01,
        patience=10,
        warmup_epochs=3,
        close_mosaic=10,
        hsv_h=0.015, hsv_s=0.5, hsv_v=0.4,
        degrees=5.0, translate=0.1, scale=0.5,
        shear=2.0, perspective=0.001,
        flipud=0.0, fliplr=0.5,
        mosaic=1.0, mixup=0.05,
        erasing=0.0,
        copy_paste=0.3,
        multi_scale=0.2,
        save_period=5,
        plots=True,
        project=PROJECT,
        name="train",
        exist_ok=True,
    )

    # ONNX export
    best_path = Path(f"{PROJECT}/train/weights/best.pt")
    if not best_path.exists():
        print(f"ERROR: best.pt 없음 — {best_path}")
        return

    print(f"\n[M4-Context] best.pt: {best_path} ({best_path.stat().st_size/1024/1024:.1f}MB)")
    best_model = YOLO(str(best_path))
    best_model.export(format="onnx", opset=17, dynamic=True, simplify=True)

    onnx_path = best_path.with_suffix(".onnx")
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    dst = WEIGHTS_DIR / f"{OUTPUT_NAME}.onnx"
    shutil.copy2(onnx_path, dst)
    print(f"[M4-Context] ONNX 저장: {dst} ({dst.stat().st_size/1024/1024:.1f}MB)")

    # 평가
    print("\n[M4-Context] val 평가...")
    metrics = best_model.val(data=DATA_YAML, imgsz=960, batch=4)
    print(f"  mAP50:    {metrics.box.map50:.4f}")
    print(f"  mAP50-95: {metrics.box.map:.4f}")
    print(f"  precision: {metrics.box.mp:.4f}")
    print(f"  recall:    {metrics.box.mr:.4f}")
    print(f"  0.9 도달? {'YES ✅' if metrics.box.map50 >= 0.9 else 'NO'}")


if __name__ == "__main__":
    train()
