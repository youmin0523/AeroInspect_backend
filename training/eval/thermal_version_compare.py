"""
thermal 모델 버전 비교 — v11(현 배포) vs prev vs v3. 진짜 배포 결정용 mAP 실측.
thermal_yolo test split 에서 각 ONNX val → mAP50 / mAP50-95.
실행: backend/venv/Scripts/python.exe backend/training/eval/thermal_version_compare.py
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
DATA = ROOT / "backend/training/datasets/thermal_yolo/data.yaml"
CANDS = {
    "v11_current": ROOT / "backend/models_weights/thermal_yolo.onnx",
    "prev":        ROOT / "backend/models_weights/thermal_yolo_prev.onnx",
    "v3":          ROOT / "backend/models_weights/thermal_v3.onnx",
}

from ultralytics import YOLO

out = []
for name, p in CANDS.items():
    if not p.exists():
        out.append(f"{name}: MISSING")
        continue
    try:
        m = YOLO(str(p), task="detect")
        r = m.val(data=str(DATA), split="test", imgsz=640, conf=0.001,
                  iou=0.6, verbose=False, plots=False)
        out.append(f"{name}: mAP50={r.box.map50:.4f} mAP50-95={r.box.map:.4f} "
                   f"P={r.box.mp:.4f} R={r.box.mr:.4f}")
    except Exception as e:
        out.append(f"{name}: FAIL {type(e).__name__} {str(e)[:120]}")

txt = "\n".join(out)
(Path(__file__).parent / "thermal_version_compare.txt").write_text(txt, encoding="utf-8")
print(txt)
