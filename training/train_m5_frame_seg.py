# =============================================
# train_m5_frame_seg.py
# M5: YOLOv8m-seg 기하학 프레임 인스턴스 세그멘테이션
# 클래스: wall_edge, ceiling_edge, door_frame, window_frame
# G1 기하학 모듈이 세그멘테이션 결과로 수직수평/직각도 분석
# 출력: models_weights/m5_yolo_seg_frames.onnx
#
# 사용법:
#   cd backend/training
#   python train_m5_frame_seg.py
# =============================================

from __future__ import annotations

import shutil
from pathlib import Path

from ultralytics import YOLO

EPOCHS = 150
BATCH = 4           # RTX 5070 Laptop 8GB VRAM (seg는 메모리 큼 → OOM 시 2로)
IMGSZ = 768         # 엣지 정밀 vs 속도/메모리 절충 (960→768)
PATIENCE = 30
LR0 = 1e-4
DATA_YAML = "configs/frame_seg.yaml"
PROJECT = "runs/m5_frame_seg"
NAME = "seg_v2"
WEIGHTS_DIR = Path("../models_weights")
OUTPUT_NAME = "m5_yolo_seg_frames"


def train():
    print("=" * 60)
    print("[M5-Seg] YOLOv8m-seg 기하학 프레임 세그멘테이션 학습")
    print("=" * 60)

    # frames 라벨 = 유효 polygon 100% (20657/20657) 검증 완료 → seg 전환
    # G1 기하학 모듈이 seg mask로 수직수평/직각도 분석 (본래 설계)
    model = YOLO("yolov8m-seg.pt")      # segmentation pretrained
    model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        batch=BATCH,
        imgsz=IMGSZ,
        patience=PATIENCE,
        optimizer="AdamW",
        lr0=LR0,
        lrf=0.01,
        warmup_epochs=5,
        close_mosaic=20,
        # 기하학 모델: 약한 증강 (엣지 정보 보존 중요)
        degrees=3.0,
        translate=0.1,
        scale=0.3,
        shear=1.0,
        perspective=0.0005,
        flipud=0.0,
        fliplr=0.5,
        mosaic=0.8,
        mixup=0.05,
        erasing=0.0,
        copy_paste=0.2,     # 소형 프레임 복사-붙여넣기
        multi_scale=0.3,    # 기하학은 약한 multi_scale (엣지 보존)
        project=PROJECT,
        name=NAME,
        exist_ok=True,
    )

    # ONNX 변환
    best = YOLO(f"{PROJECT}/{NAME}/weights/best.pt")
    best.export(format="onnx", opset=17, dynamic=True, simplify=True)

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    dst = WEIGHTS_DIR / f"{OUTPUT_NAME}.onnx"
    if dst.exists():
        shutil.copy2(dst, WEIGHTS_DIR / f"{OUTPUT_NAME}_prev.onnx")
    shutil.copy2(f"{PROJECT}/{NAME}/weights/best.onnx", dst)
    print(f"[M5-Seg] ONNX 저장 완료: {WEIGHTS_DIR / OUTPUT_NAME}.onnx")


if __name__ == "__main__":
    train()
