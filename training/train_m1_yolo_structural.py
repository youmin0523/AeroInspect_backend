# =============================================
# train_m1_yolo_structural.py
# M1-Stage1: YOLOv8m 구조·방수 하자 검출 학습
# 클래스: crack(균열), caulking_defect(코킹불량), waterproof_defect(방수/누수)
# 출력: models_weights/m1_yolo_structural.onnx
#
# 사용법:
#   cd backend/training
#   python train_m1_yolo_structural.py
# =============================================

from __future__ import annotations

import shutil
from pathlib import Path

from ultralytics import YOLO


# ── 하이퍼파라미터 ────���───────────────────────
EPOCHS_PHASE1 = 10          # backbone freeze
EPOCHS_PHASE2 = 190         # full unfreeze
BATCH = 16
IMGSZ = 640
PATIENCE = 30
LR0 = 1e-4
LRF = 0.01                 # cosine 최종 LR 비율
WARMUP_EPOCHS = 5
CLOSE_MOSAIC = 20           # 마지막 20 epoch에서 mosaic 끔
CONF = 0.15                 # 검출 임계값 (높은 재현율)

DATA_YAML = "configs/structural.yaml"
PROJECT = "runs/m1_structural"
WEIGHTS_DIR = Path("../models_weights")
OUTPUT_NAME = "m1_yolo_structural"


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
    """2-Phase 학습: backbone freeze → full unfreeze."""
    print("=" * 60)
    print("[M1-YOLO] Phase 1: Backbone Freeze (10 epochs)")
    print("=" * 60)

    model = YOLO("yolov8m.pt")
    model.train(
        data=DATA_YAML,
        epochs=EPOCHS_PHASE1,
        batch=BATCH,
        imgsz=IMGSZ,
        freeze=10,
        optimizer="AdamW",
        lr0=LR0,
        project=PROJECT,
        name="phase1_freeze",
        exist_ok=True,
    )

    print("=" * 60)
    print("[M1-YOLO] Phase 2: Full Unfreeze (190 epochs)")
    print("=" * 60)

    phase1_weights = find_weights(PROJECT, "phase1_freeze", "last.pt")
    print(f"[M1-YOLO] Phase 2 시작: {phase1_weights}")
    model = YOLO(phase1_weights)
    model.train(
        data=DATA_YAML,
        epochs=EPOCHS_PHASE2,
        batch=BATCH,
        imgsz=IMGSZ,
        optimizer="AdamW",
        lr0=LR0 * 0.1,
        lrf=LRF,
        patience=PATIENCE,
        warmup_epochs=WARMUP_EPOCHS,
        close_mosaic=CLOSE_MOSAIC,
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.4,
        degrees=5.0,
        translate=0.1,
        scale=0.5,
        shear=2.0,
        perspective=0.001,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        erasing=0.0,  # seg 라벨 혼합 데이터 호환
        project=PROJECT,
        name="phase2_full",
        exist_ok=True,
    )

    print("=" * 60)
    print("[M1-YOLO] ONNX 변환")
    print("=" * 60)

    phase2_best = find_weights(PROJECT, "phase2_full", "best.pt")
    export_to_onnx(phase2_best)


def export_to_onnx(pt_path: str):
    """학습된 .pt → ONNX 변환 후 models_weights/로 복사."""
    model = YOLO(pt_path)
    model.export(
        format="onnx",
        opset=17,
        dynamic=True,
        simplify=True,
        half=False,
    )

    onnx_src = pt_path.replace(".pt", ".onnx")
    onnx_dst = WEIGHTS_DIR / f"{OUTPUT_NAME}.onnx"
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(onnx_src, onnx_dst)
    print(f"[M1-YOLO] ONNX 저장 완료: {onnx_dst}")


if __name__ == "__main__":
    train()
