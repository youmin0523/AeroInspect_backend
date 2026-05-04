# =============================================
# train_m3_yolo_floor_window.py
# M3-Stage1: YOLOv8m 바닥·창호 하자 검출 학습
# 클래스: floor_defect, glass_defect, frame_defect
# 출력: models_weights/m3_yolo_floor_window.onnx
# =============================================

from __future__ import annotations

import shutil
from pathlib import Path

from ultralytics import YOLO

EPOCHS_PHASE1 = 10
EPOCHS_PHASE2 = 140
BATCH = 4   # RTX 5070 8GB — imgsz=960에서 batch=4 가능
IMGSZ = 960 # 소형 객체 59% 대응: 640→960 (2.25배 해상도)
PATIENCE = 30
LR0 = 1e-4
DATA_YAML = "configs/floor_window.yaml"
PROJECT = "runs/m3_floor_window"
WEIGHTS_DIR = Path("../models_weights")
OUTPUT_NAME = "m3_yolo_floor_window"


def find_weights(project: str, name: str, prefer: str = "last.pt") -> str:
    """ultralytics가 저장한 weights 경로를 절대경로로 찾기."""
    import glob

    # 프로젝트 루트부터 전체 검색
    root = Path(__file__).resolve().parent.parent.parent  # TEAM_PROJECT_2_Drone_project/
    pattern = f"**/{name}/weights/{prefer}"

    # 여러 위치에서 검색
    for search_root in [Path("."), Path(".."), root]:
        for g in glob.glob(str(search_root / pattern), recursive=True):
            return str(Path(g).resolve())

    raise FileNotFoundError(f"weights not found: {pattern} (searched from {root})")


def train():
    # Phase 1: Backbone freeze (10 epochs)
    model = YOLO("yolov8m.pt")
    model.train(
        data=DATA_YAML, epochs=EPOCHS_PHASE1, batch=BATCH, imgsz=IMGSZ,
        freeze=10, optimizer="AdamW", lr0=LR0,
        project=PROJECT, name="phase1_freeze", exist_ok=True,
    )

    # Phase 2: Full unfreeze — Phase 1의 last.pt에서 이어서 학습
    phase1_weights = find_weights(PROJECT, "phase1_freeze", "last.pt")
    print(f"[M3-YOLO] Phase 2 시작: {phase1_weights}")
    model = YOLO(phase1_weights)
    model.train(
        data=DATA_YAML, epochs=EPOCHS_PHASE2, batch=BATCH, imgsz=IMGSZ,
        optimizer="AdamW", lr0=LR0 * 0.1, lrf=0.01,
        patience=PATIENCE, warmup_epochs=5, close_mosaic=20,
        hsv_h=0.015, hsv_s=0.5, hsv_v=0.4,
        degrees=5.0, translate=0.1, scale=0.5,
        shear=2.0, perspective=0.001,
        flipud=0.0, fliplr=0.5,
        mosaic=1.0, mixup=0.1, erasing=0.0,
        copy_paste=0.3,     # 소형 bbox 복사-붙여넣기 → 소형 객체 노출 빈도 ↑
        multi_scale=0.5,    # 매 batch 해상도 ±50% 변동 → 다양한 크기 학습
        project=PROJECT, name="phase2_full", exist_ok=True,
    )

    # ONNX export
    phase2_best = find_weights(PROJECT, "phase2_full", "best.pt")
    best = YOLO(phase2_best)
    best.export(format="onnx", opset=17, dynamic=True, simplify=True)
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(phase2_best.replace(".pt", ".onnx"),
                 WEIGHTS_DIR / f"{OUTPUT_NAME}.onnx")
    print(f"[M3-YOLO] ONNX 저장 완료: {WEIGHTS_DIR / OUTPUT_NAME}.onnx")


if __name__ == "__main__":
    train()
