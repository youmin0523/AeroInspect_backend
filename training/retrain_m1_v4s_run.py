"""M1 structural v4s — yolov8s, batch32, 50ep, box=10.0, patience=15"""
import os, sys, shutil, time, multiprocessing
from pathlib import Path
from datetime import datetime

TRAIN_DIR = Path(__file__).parent.resolve()
WEIGHTS_DIR = TRAIN_DIR.parent / "models_weights"
LOG = TRAIN_DIR / "retrain_m1_v4s_log.txt"

def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")

def main():
    os.chdir(str(TRAIN_DIR))
    WEIGHTS_DIR.mkdir(exist_ok=True)

    import torch
    log(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    log("=" * 50)
    log("M1 structural v4s 시작 (yolov8s, 20393장, box=10.0)")
    log("=" * 50)

    try:
        from ultralytics import YOLO
        model = YOLO("yolov8s.pt")
        model.train(
            data="datasets/structural/data.yaml",
            epochs=50,
            imgsz=640,
            batch=32,
            patience=15,
            project="runs/m1_structural_v4s",
            name="v4s",
            device=0,
            optimizer="AdamW",
            lr0=0.001,
            cos_lr=True,
            box=10.0,
            save=True,
            exist_ok=True,
        )
        metrics = model.val()
        map50 = metrics.box.map50
        recall = metrics.box.mr
        log(f"M1 v4s 완료 — mAP50={map50:.4f} Recall={1-recall:.4f}")

        model.export(format="onnx", imgsz=640, dynamic=True, simplify=True)
        best_onnx = TRAIN_DIR / "runs" / "m1_structural_v4s" / "v4s" / "weights" / "best.onnx"
        if best_onnx.exists():
            dst = WEIGHTS_DIR / "m1_structural_v4s.onnx"
            shutil.copy(best_onnx, dst)
            log(f"ONNX 저장: {dst} ({dst.stat().st_size/1024/1024:.1f}MB)")

    except Exception as e:
        log(f"M1 v4s 에러: {e}")

    log("=" * 50)
    log("M1 structural v4s 종료")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
