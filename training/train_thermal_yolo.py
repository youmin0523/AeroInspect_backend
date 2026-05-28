# =============================================
# train_thermal_yolo.py
# Thermal-YOLO: 열화상 결함 검출 (Crack / Moisture / delamination)
# 데이터셋: datasets/thermal_yolo (nc=3)
# 출력: models_weights/thermal_yolo.onnx
#
# baseline v1: mAP50=0.614 / mAP50-95=0.299 (epoch 100) → 재학습으로 개선
# 8GB GPU(RTX 5070 Laptop) 대응: batch=8, AMP, cache=False
#
# 사용법:
#   cd backend/training
#   python train_thermal_yolo.py
# =============================================

from __future__ import annotations

import shutil
from pathlib import Path

from ultralytics import YOLO

DATA_YAML = "datasets/thermal_yolo/data.yaml"   # nc=3 Crack 복원 (Crack 0.38 살림)
PROJECT = "runs"
NAME = "thermal_v11"  # v10 + Crack oversample(30→240) 반영. imgsz960(Moisture 0.369 성공) 유지
WEIGHTS_DIR = Path("../models_weights")
OUTPUT_NAME = "thermal_yolo"

EPOCHS = 120
BATCH = 4          # imgsz960 8GB 최적 (batch8은 메모리포화로 2.5배 느림) + cache=ram
IMGSZ = 960        # Moisture/delam 작은 객체(중앙 0.4~1.5%) 해상도 확보
PATIENCE = 25
LR0 = 1e-3
WARMUP_EPOCHS = 3
CLOSE_MOSAIC = 10   # v1 검증값


def train():
    print("=" * 60)
    print("[Thermal-YOLO] 재학습 (Crack/Moisture/delamination)")
    print("=" * 60)

    model = YOLO("yolov8m.pt")
    model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        batch=BATCH,
        imgsz=IMGSZ,
        patience=PATIENCE,
        optimizer="AdamW",
        lr0=LR0,
        cos_lr=True,
        warmup_epochs=WARMUP_EPOCHS,
        close_mosaic=CLOSE_MOSAIC,
        amp=True,
        cache="ram",        # uv 정리 후 RAM 여유(21.6GB) → IO 제거로 속도 ↑
        workers=8,
        project=PROJECT,
        name=NAME,
        exist_ok=True,
        device=0,
        # 열화상 특화 augmentation: 색·밝기=온도 정보이므로 약하게 (v1 검증값)
        hsv_h=0.015,
        hsv_s=0.2,      # 채도 약하게 (온도 정보 보존)
        hsv_v=0.3,      # 명도 약하게
        degrees=0.0,    # 회전 X
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0,      # 온도맵 혼합 부적절
    )

    # best.pt → ONNX export
    best = Path(PROJECT) / NAME / "weights" / "best.pt"
    if not best.exists():
        print(f"[Thermal-YOLO] 경고: best.pt 없음 ({best})")
        return

    m = YOLO(str(best))
    m.export(format="onnx", opset=14, dynamic=False, simplify=True)

    onnx_src = Path(PROJECT) / NAME / "weights" / "best.onnx"
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    if onnx_src.exists():
        # 기존 운영 파일 백업 후 교체
        dst = WEIGHTS_DIR / f"{OUTPUT_NAME}.onnx"
        if dst.exists():
            shutil.copy2(dst, WEIGHTS_DIR / f"{OUTPUT_NAME}_prev.onnx")
        shutil.copy2(onnx_src, dst)
        print(f"[Thermal-YOLO] ONNX 저장 완료: {dst}")
    else:
        print(f"[Thermal-YOLO] 경고: best.onnx 없음 ({onnx_src})")


if __name__ == "__main__":
    train()
