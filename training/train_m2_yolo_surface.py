# =============================================
# train_m2_yolo_surface.py
# M2-Stage1: YOLOv8m 마감·표면 하자 검출 학습
# 클래스: surface_defect_wall(벽/천장 표면), baseboard_defect(걸레받이)
# 출력: models_weights/m2_yolo_surface.onnx
# =============================================

from __future__ import annotations

import shutil
from pathlib import Path

from ultralytics import YOLO

EPOCHS_PHASE1 = 10
EPOCHS_PHASE2 = 140
BATCH = 4   # RTX 5070 8GB — imgsz=960에서 batch=4
IMGSZ = 960 # 소형 객체 65% 대응: 640→960
PATIENCE = 30
LR0 = 1e-4
DATA_YAML = "configs/surface.yaml"
PROJECT = "runs/m2_surface"
WEIGHTS_DIR = Path("../models_weights")
OUTPUT_NAME = "m2_yolo_surface"


def find_weights(project: str, name: str, prefer: str = "last.pt") -> str:
    """ultralytics가 저장한 weights 경로를 절대경로로 찾기."""
    import glob
    root = Path(__file__).resolve().parent.parent.parent
    pattern = f"**/{name}/weights/{prefer}"
    for search_root in [Path("."), Path(".."), root]:
        for g in glob.glob(str(search_root / pattern), recursive=True):
            return str(Path(g).resolve())
    raise FileNotFoundError(f"weights not found: {pattern}")


def train():
    # Phase 1: freeze
    model = YOLO("yolov8m.pt")
    model.train(
        data=DATA_YAML, epochs=EPOCHS_PHASE1, batch=BATCH, imgsz=IMGSZ,
        freeze=10, optimizer="AdamW", lr0=LR0,
        project=PROJECT, name="phase1_freeze", exist_ok=True,
    )

    # Phase 2: unfreeze — 표면 하자 특화 증강
    phase1_weights = find_weights(PROJECT, "phase1_freeze", "last.pt")
    print(f"[M2-YOLO] Phase 2 시작: {phase1_weights}")
    model = YOLO(phase1_weights)
    model.train(
        data=DATA_YAML, epochs=EPOCHS_PHASE2, batch=BATCH, imgsz=IMGSZ,
        optimizer="AdamW", lr0=LR0 * 0.1, lrf=0.01,
        patience=PATIENCE, warmup_epochs=5, close_mosaic=20,
        hsv_h=0.02, hsv_s=0.6, hsv_v=0.5,
        degrees=5.0, translate=0.1, scale=0.5,
        shear=2.0, perspective=0.001,
        flipud=0.0, fliplr=0.5,
        mosaic=1.0, mixup=0.15, erasing=0.0,
        copy_paste=0.3,     # 소형 bbox 복사-붙여넣기
        multi_scale=0.5,    # 매 batch 해상도 ±50% 변동
        project=PROJECT, name="phase2_full", exist_ok=True,
    )

    # ONNX 변환
    phase2_best = find_weights(PROJECT, "phase2_full", "best.pt")
    best = YOLO(phase2_best)
    best.export(format="onnx", opset=17, dynamic=True, simplify=True)
    onnx_dst = WEIGHTS_DIR / f"{OUTPUT_NAME}.onnx"
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(phase2_best.replace(".pt", ".onnx"), onnx_dst)
    print(f"[M2-YOLO] ONNX 저장 완료: {onnx_dst}")


if __name__ == "__main__":
    train()
