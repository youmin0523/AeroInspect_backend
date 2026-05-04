# =============================================
# train_m2_fast.py
# 역할: M2 빠른 다단계 재학습 (5/4 04시 전 완료 목표)
#       - Stage 1: yolov11l fresh + 30ep + batch=16 + imgsz=640
#       - Stage 2: fine-tune + 10ep + freeze=10 + lr=1e-5
#       - 자동으로 Stage 1/2 중 더 좋은 best 채택 → ONNX export → 배포
# =============================================

from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path

from ultralytics import YOLO

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


DATA = "datasets/surface/data.yaml"
PROJECT = "runs"
NAME_S1 = "m2_fast/stage1"
NAME_S2 = "m2_fast/stage2"
WEIGHTS_DIR = Path("../models_weights")
ONNX_NAME = "m2_yolo_surface.onnx"


def read_best_map(csv_path: Path) -> float:
    if not csv_path.exists():
        return -1.0
    best = -1.0
    for i, line in enumerate(csv_path.read_text(encoding="utf-8").splitlines()):
        if i == 0:
            continue
        parts = line.split(",")
        if len(parts) < 8:
            continue
        try:
            v = float(parts[7])
            if v > best:
                best = v
        except ValueError:
            continue
    return best


def main():
    start = time.time()
    print("=" * 60)
    print("M2 fast 다단계 재학습")
    print("Stage 1: yolov11l fresh + 30ep + batch=16 + imgsz=640")
    print("=" * 60)

    # Stage 1
    model_s1 = YOLO("yolo11l.pt")
    model_s1.train(
        data=DATA,
        epochs=15,
        batch=24,
        imgsz=480,
        cache="disk",
        workers=4,
        optimizer="AdamW",
        lr0=1e-3,
        lrf=0.01,
        cos_lr=True,
        patience=10,
        warmup_epochs=2,
        close_mosaic=8,
        freeze=0,
        hsv_h=0.015, hsv_s=0.5, hsv_v=0.4,
        degrees=5.0, translate=0.1, scale=0.5,
        shear=2.0, perspective=0.001,
        flipud=0.0, fliplr=0.5,
        mosaic=1.0, mixup=0.1, copy_paste=0.3,
        save_period=5,
        plots=True,
        project=PROJECT,
        name=NAME_S1,
        exist_ok=True,
    )

    # Stage 1 best 위치
    s1_best = None
    for p in Path(".").rglob("m2_fast/stage1/weights/best.pt"):
        s1_best = p
        break
    if s1_best is None or not s1_best.exists():
        print("[ERROR] Stage 1 best.pt 못 찾음")
        return 1
    print(f"\nStage 1 best: {s1_best.resolve()}")

    # Stage 2: fine-tune
    print("\n" + "=" * 60)
    print("Stage 2: fine-tune lr=1e-5 + freeze=10 + 10ep")
    print("=" * 60)
    model_s2 = YOLO(str(s1_best))
    model_s2.train(
        data=DATA,
        epochs=5,
        batch=24,
        imgsz=480,
        cache="disk",
        workers=4,
        optimizer="AdamW",
        lr0=1e-5,
        lrf=0.01,
        cos_lr=True,
        patience=8,
        warmup_epochs=1,
        close_mosaic=8,
        freeze=10,
        mosaic=0.5, mixup=0.0, copy_paste=0.2,
        save_period=5,
        plots=True,
        project=PROJECT,
        name=NAME_S2,
        exist_ok=True,
    )

    s2_best = None
    for p in Path(".").rglob("m2_fast/stage2/weights/best.pt"):
        s2_best = p
        break

    # Stage 1 vs 2 mAP 비교
    s1_csv = s1_best.parent.parent / "results.csv"
    s2_csv = s2_best.parent.parent / "results.csv" if s2_best else None
    s1_map = read_best_map(s1_csv)
    s2_map = read_best_map(s2_csv) if s2_csv else -1.0
    print(f"\nStage 1 best mAP50: {s1_map:.4f}")
    print(f"Stage 2 best mAP50: {s2_map:.4f}")

    chosen_pt = s2_best if (s2_best and s2_map >= s1_map) else s1_best
    chosen_label = "Stage 2" if chosen_pt == s2_best else "Stage 1"
    print(f"채택: {chosen_label}")

    # 평가
    print("\n검증...")
    final_model = YOLO(str(chosen_pt))
    metrics = final_model.val(
        data=DATA, imgsz=480, batch=24, device=0,
        workers=0, plots=False, save_json=False, verbose=False,
        split="test",
    )
    print(f"\n=== M2 최종 ===")
    print(f"  mAP50:    {metrics.box.map50:.4f}")
    print(f"  mAP50-95: {metrics.box.map:.4f}")
    print(f"  precision: {metrics.box.mp:.4f}")
    print(f"  recall:    {metrics.box.mr:.4f}")
    print(f"  0.85+? {'YES' if metrics.box.map50 >= 0.85 else 'NO'}")

    # ONNX export
    print("\nONNX export...")
    final_model.export(format="onnx", opset=17, dynamic=True, simplify=True)
    onnx_src = chosen_pt.with_suffix(".onnx")
    if onnx_src.exists():
        WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
        dst = WEIGHTS_DIR / ONNX_NAME
        shutil.copy2(onnx_src, dst)
        size_mb = dst.stat().st_size / (1024 * 1024)
        print(f"\n배포: {dst} ({size_mb:.1f} MB)")

    elapsed = time.time() - start
    print(f"\n총 소요: {elapsed/3600:.2f}시간")
    return 0


if __name__ == "__main__":
    sys.exit(main())
