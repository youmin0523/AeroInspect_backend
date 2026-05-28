# =============================================
# train_m4_context_seg.py
# M4 Context (재설계): bbox → segmentation 전환
# - 데이터셋 라벨이 이미 polygon 형식이라 즉시 seg 학습 가능
# - M5의 seg 전환 성공 패턴 (mAP 0.355→0.466, +0.111) 적용
# - region-based(wall/ceiling/floor/window/door)는 polygon이 자연
# - 모델: yolov8m-seg.pt → ONNX
# - 출력: backend/models_weights/m4_yolo_context_elements.onnx (교체)
#
# 사용법: cd backend/training && python -u train_m4_context_seg.py
# =============================================

from __future__ import annotations

import shutil
import sys
from pathlib import Path

from ultralytics import YOLO

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DATA_YAML = "datasets/m4_context/data.yaml"  # 기존 polygon 라벨 그대로 사용
PROJECT = "runs/m4_context_seg"
WEIGHTS_DIR = Path("../models_weights")
OUTPUT_NAME = "m4_yolo_context_elements"  # 기존 ONNX 교체

EPOCHS = 60
BATCH = 4       # 8GB VRAM (M5 seg 검증값)
IMGSZ = 768     # seg는 폴리곤 정밀도 vs 메모리 균형 (M5 검증값)
PATIENCE = 15


def train():
    print("=" * 60)
    print("[M4-Seg] yolov8m-seg 학습 시작 (5클래스: wall/ceiling/floor/window/door)")
    print("=" * 60)

    model = YOLO("yolov8m-seg.pt")
    model.train(
        data=DATA_YAML,
        task="segment",
        epochs=EPOCHS,
        batch=BATCH,
        imgsz=IMGSZ,
        cache="disk",
        workers=4,
        optimizer="AdamW",
        lr0=1e-3,
        lrf=0.01,
        patience=PATIENCE,
        warmup_epochs=3,
        close_mosaic=10,
        amp=True,
        hsv_h=0.015, hsv_s=0.5, hsv_v=0.4,
        degrees=5.0, translate=0.1, scale=0.5,
        fliplr=0.5,
        mosaic=1.0, mixup=0.05,
        copy_paste=0.2,
        save_period=5,
        plots=True,
        project=PROJECT,
        name="train",
        exist_ok=True,
        device=0,
    )

    best_path = Path(f"{PROJECT}/train/weights/best.pt")
    if not best_path.exists():
        print(f"ERROR: best.pt 없음 — {best_path}")
        return

    print(f"\n[M4-Seg] best.pt: {best_path} ({best_path.stat().st_size/1024/1024:.1f}MB)")
    best_model = YOLO(str(best_path))
    best_model.export(format="onnx", opset=17, dynamic=True, simplify=True)

    onnx_path = best_path.with_suffix(".onnx")
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    dst = WEIGHTS_DIR / f"{OUTPUT_NAME}.onnx"
    if dst.exists():
        shutil.copy2(dst, WEIGHTS_DIR / f"{OUTPUT_NAME}_prev.onnx")
    shutil.copy2(onnx_path, dst)
    print(f"[M4-Seg] ONNX 저장: {dst} ({dst.stat().st_size/1024/1024:.1f}MB)")

    print("\n[M4-Seg] val 평가...")
    metrics = best_model.val(data=DATA_YAML, imgsz=IMGSZ, batch=BATCH)
    # seg는 metrics.seg.map / metrics.box.map 둘 다 확인
    print(f"  box mAP50:     {metrics.box.map50:.4f}")
    print(f"  box mAP50-95:  {metrics.box.map:.4f}")
    print(f"  seg mAP50:     {metrics.seg.map50:.4f}")
    print(f"  seg mAP50-95:  {metrics.seg.map:.4f}")
    print(f"  baseline 0.355 돌파? "
          f"{'YES ✅' if metrics.box.map >= 0.355 else 'NO'}")


if __name__ == "__main__":
    train()
