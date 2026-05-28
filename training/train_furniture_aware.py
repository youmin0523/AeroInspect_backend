# =============================================
# train_furniture_aware.py
# furniture_aware: 빌트인 가구 인식 (10 classes)
# M1+M2+M3 검출이 가구 위면 false positive 차단용 게이트 모델
# 데이터셋: datasets/furniture_aware (nc=10)
# 출력: models_weights/furniture_aware.onnx
#
# 클래스: wall, ceiling, floor, window, door,
#         cabinet_builtin, kitchen_appliance, countertop_sink, kitchen_island, shelf
# 8GB GPU(RTX 5070 Laptop) 대응: batch=8, imgsz=640, AMP
#
# 사용법:
#   cd backend/training
#   python train_furniture_aware.py
# =============================================

from __future__ import annotations

import shutil
from pathlib import Path

from ultralytics import YOLO

DATA_YAML = "datasets/furniture_aware/data.yaml"
PROJECT = "runs"
NAME = "furniture_aware_v2"
WEIGHTS_DIR = Path("../models_weights")
OUTPUT_NAME = "furniture_aware"

EPOCHS = 80
BATCH = 4          # OOM 방지 (thermal batch8 메모리포화 역효과 교훈)
IMGSZ = 768        # 작은 빌트인 가구(kitchen_island 초소형29% 등) 대응 (Moisture 교훈)
PATIENCE = 20


def train():
    print("=" * 60)
    print("[furniture_aware] 빌트인 가구 인식 학습 (nc=10)")
    print("=" * 60)

    model = YOLO("yolov8m.pt")
    model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        batch=BATCH,
        imgsz=IMGSZ,
        patience=PATIENCE,
        optimizer="AdamW",
        lr0=1e-3,
        cos_lr=True,
        warmup_epochs=3,
        close_mosaic=10,
        amp=True,
        cache=False,
        workers=4,
        project=PROJECT,
        name=NAME,
        exist_ok=True,
        device=0,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=5.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
    )

    best = Path(PROJECT) / NAME / "weights" / "best.pt"
    if not best.exists():
        print(f"[furniture_aware] 경고: best.pt 없음 ({best})")
        return

    m = YOLO(str(best))
    m.export(format="onnx", opset=14, dynamic=False, simplify=True)

    onnx_src = Path(PROJECT) / NAME / "weights" / "best.onnx"
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    if onnx_src.exists():
        dst = WEIGHTS_DIR / f"{OUTPUT_NAME}.onnx"
        if dst.exists():
            shutil.copy2(dst, WEIGHTS_DIR / f"{OUTPUT_NAME}_prev.onnx")
        shutil.copy2(onnx_src, dst)
        print(f"[furniture_aware] ONNX 저장 완료: {dst}")
    else:
        print(f"[furniture_aware] 경고: best.onnx 없음 ({onnx_src})")


if __name__ == "__main__":
    train()
