# =============================================
# evaluate_ultralytics_val.py
# 역할: ultralytics 표준 model.val()로 진짜 mAP@0.5 측정
#       - 우리 custom IoU evaluator는 너무 strict (단일 threshold F1 비슷)
#       - ultralytics val() = COCO-style AP (recall-precision curve 적분)
#       - conf=0.001 (default) 사용 → recall 극대화 후 PR curve로 AP 계산
#
# 사용:
#   cd backend/training
#   python eval/evaluate_ultralytics_val.py
# =============================================

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

from ultralytics import YOLO

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# 평가 대상 — 각 ONNX + 데이터셋 매핑
TARGETS = [
    {
        "key": "M1_YOLO",
        "onnx": "../models_weights/m1_yolo_structural.onnx",
        "data": "datasets/structural/data.yaml",
        "imgsz": 1280,
    },
    {
        "key": "M2_YOLO",
        "onnx": "../models_weights/m2_yolo_surface.onnx",
        "data": "datasets/surface/data.yaml",
        "imgsz": 960,
    },
    {
        "key": "M3_YOLO",
        "onnx": "../models_weights/m3_yolo_floor_window.onnx",
        "data": "datasets/floor_window/data.yaml",
        "imgsz": 1280,
    },
    {
        "key": "M4_CONTEXT",
        "onnx": "../models_weights/m4_yolo_context_elements.onnx",
        "data": "datasets/m4_context/data.yaml",
        "imgsz": 960,
    },
    {
        "key": "M5_SEG",
        "onnx": "../models_weights/m5_yolo_seg_frames.onnx",
        "data": "datasets/frames/data.yaml",
        "imgsz": 1280,
    },
]


def evaluate_one(target: dict) -> Optional[Dict]:
    onnx_path = Path(target["onnx"])
    data_yaml = Path(target["data"])

    if not onnx_path.exists():
        print(f"[SKIP] {target['key']} — ONNX 없음: {onnx_path}")
        return None
    if not data_yaml.exists():
        print(f"[SKIP] {target['key']} — data.yaml 없음: {data_yaml}")
        return None

    print(f"\n{'='*60}")
    print(f"=== {target['key']} ===")
    print(f"{'='*60}")
    print(f"ONNX: {onnx_path.resolve()}")
    print(f"data: {data_yaml}")
    print(f"imgsz: {target['imgsz']}")

    start = time.time()
    try:
        # ultralytics는 ONNX 모델도 task='detect'로 val 가능
        model = YOLO(str(onnx_path), task="detect")
        metrics = model.val(
            data=str(data_yaml),
            imgsz=target["imgsz"],
            batch=8,
            device=0,           # GPU
            workers=0,
            plots=False,
            save_json=False,
            verbose=False,
            augment=True,       # TTA 적용 — mAP +1~3% 가능
            split="test" if (data_yaml.parent / "images" / "test").exists() else "val",
        )
        elapsed = time.time() - start

        result = {
            "key": target["key"],
            "mAP50": float(metrics.box.map50),
            "mAP50_95": float(metrics.box.map),
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
            "elapsed_min": round(elapsed / 60, 1),
        }

        # per-class
        per_class = {}
        try:
            class_names = model.names
            maps = metrics.box.maps  # per-class mAP50-95
            ap50_per_class = metrics.box.ap50  # per-class AP@0.5
            for i, name in class_names.items() if isinstance(class_names, dict) else enumerate(class_names):
                if i < len(ap50_per_class):
                    per_class[name] = {
                        "mAP50": float(ap50_per_class[i]),
                        "mAP50_95": float(maps[i]) if i < len(maps) else 0.0,
                    }
        except Exception:
            pass
        result["per_class"] = per_class

        print(f"\n  mAP50:    {result['mAP50']:.4f}")
        print(f"  mAP50-95: {result['mAP50_95']:.4f}")
        print(f"  precision: {result['precision']:.4f}")
        print(f"  recall:    {result['recall']:.4f}")
        print(f"  소요: {elapsed/60:.1f}min")
        if per_class:
            print(f"  per-class mAP50:")
            for n, m in per_class.items():
                print(f"    {n}: {m['mAP50']:.4f}")
        return result
    except Exception as e:
        print(f"[ERROR] {target['key']}: {type(e).__name__}: {e}")
        return None


def main():
    cwd = Path.cwd()
    print(f"cwd: {cwd}")
    print(f"평가 대상: {len(TARGETS)}개 모델")

    results: List[Dict] = []
    for t in TARGETS:
        r = evaluate_one(t)
        if r:
            results.append(r)

    # 종합 표
    print(f"\n{'='*70}")
    print("종합 (ultralytics 표준 mAP)")
    print(f"{'='*70}")
    print(f"| 모델 | mAP50 | mAP50-95 | Precision | Recall | 시간 |")
    print(f"|------|-------|----------|-----------|--------|------|")
    for r in results:
        print(f"| {r['key']} | {r['mAP50']:.4f} | {r['mAP50_95']:.4f} | {r['precision']:.4f} | {r['recall']:.4f} | {r['elapsed_min']}min |")

    # 저장
    out_dir = cwd / "eval" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"ultralytics_val_{ts}.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n결과 저장: {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
