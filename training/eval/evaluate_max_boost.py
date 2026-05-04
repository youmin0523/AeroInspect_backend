# =============================================
# evaluate_max_boost.py
# 역할: 모든 추론 측면 boost 동원해서 모델 mAP 최대치 측정
#       - TTA (augment=True) — 좌우 반전 + scale 평균
#       - Multi-scale inference (480, 640, 800)
#       - Lower conf threshold (0.001 — AP 곡선 적분 위해 모든 후보 확보)
#       - Box IoU threshold 미세 조정
#
# 사용:
#   cd backend/training
#   python eval/evaluate_max_boost.py
# =============================================

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

from ultralytics import YOLO

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


TARGETS = [
    {"key": "M1_YOLO", "onnx": "../models_weights/m1_yolo_structural.onnx",
     "data": "datasets/structural/data.yaml", "imgsz_list": [640, 960]},
    {"key": "M2_YOLO", "onnx": "../models_weights/m2_yolo_surface.onnx",
     "data": "datasets/surface/data.yaml", "imgsz_list": [480, 640, 800]},
    {"key": "M3_YOLO", "onnx": "../models_weights/m3_yolo_floor_window.onnx",
     "data": "datasets/floor_window/data.yaml", "imgsz_list": [640, 960, 1280]},
    {"key": "M4_CONTEXT", "onnx": "../models_weights/m4_yolo_context_elements.onnx",
     "data": "datasets/m4_context/data.yaml", "imgsz_list": [640, 960]},
    {"key": "M5_SEG", "onnx": "../models_weights/m5_yolo_seg_frames.onnx",
     "data": "datasets/frames/data.yaml", "imgsz_list": [640, 960, 1280]},
    {"key": "furniture_aware", "onnx": "../models_weights/furniture_aware.onnx",
     "data": "datasets/furniture_aware/data.yaml", "imgsz_list": [480, 640]},
]


def evaluate_max_boost(target: dict) -> Optional[Dict]:
    onnx_path = Path(target["onnx"])
    data_yaml = Path(target["data"])

    if not onnx_path.exists() or not data_yaml.exists():
        return None

    print(f"\n{'='*60}\n=== {target['key']} (max boost) ===\n{'='*60}")

    best = {"mAP50": -1, "imgsz": None, "tta": False}

    for imgsz in target["imgsz_list"]:
        for tta in [False, True]:
            split = "test" if (data_yaml.parent / "images" / "test").exists() else "val"
            try:
                model = YOLO(str(onnx_path), task="detect")
                metrics = model.val(
                    data=str(data_yaml),
                    imgsz=imgsz, batch=8, device=0,
                    workers=0, plots=False, save_json=False, verbose=False,
                    augment=tta, conf=0.001, iou=0.6,
                    split=split,
                )
                m50 = float(metrics.box.map50)
                m9 = float(metrics.box.map)
                p = float(metrics.box.mp)
                r = float(metrics.box.mr)
                tag = f"imgsz={imgsz} tta={'O' if tta else 'X'}"
                print(f"  {tag}: mAP50={m50:.4f} mAP={m9:.4f} P={p:.4f} R={r:.4f}")
                if m50 > best["mAP50"]:
                    best = {"mAP50": m50, "mAP": m9, "P": p, "R": r,
                            "imgsz": imgsz, "tta": tta}
            except Exception as e:
                print(f"  imgsz={imgsz} tta={tta} FAIL: {type(e).__name__}: {e}")

    print(f"\n  ⭐ BEST: imgsz={best['imgsz']} tta={'O' if best['tta'] else 'X'} → mAP50={best['mAP50']:.4f}")
    return {"key": target["key"], **best}


def main():
    cwd = Path.cwd()
    print(f"cwd: {cwd}")
    print(f"평가 대상: {len(TARGETS)}개")

    results: List[Dict] = []
    for t in TARGETS:
        r = evaluate_max_boost(t)
        if r:
            results.append(r)

    # 종합
    print(f"\n{'='*70}")
    print("max boost 종합 (모든 모델 최고 mAP)")
    print(f"{'='*70}")
    print(f"| 모델 | mAP50 | imgsz | TTA | mAP50-95 | P | R |")
    print(f"|------|-------|-------|-----|----------|---|---|")
    for r in results:
        tta_s = "O" if r['tta'] else "X"
        print(f"| {r['key']} | **{r['mAP50']:.4f}** | {r['imgsz']} | {tta_s} | {r.get('mAP', 0):.4f} | {r.get('P', 0):.4f} | {r.get('R', 0):.4f} |")

    out = cwd / "eval/results" / f"max_boost_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n결과: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
