# =============================================
# eval_new_onnx.py
# 새로 다운로드한 ONNX 3개 (M2v2, M3v2, m5v2_v2) 한 번에 평가
# CPU 평가 (GPU는 M4v2 학습 중)
# =============================================

import sys
import time
from pathlib import Path

from ultralytics import YOLO

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

EVALUATIONS = [
    {
        "name": "M2v2 (surface)",
        "onnx": "../models_weights/m2_yolo_surface.onnx",
        "yaml": "configs/surface_eval.yaml",
        "imgsz": 960,
        "previous_baseline": 0.794,  # M2 (구)
    },
    {
        "name": "M3v2 (floor_window)",
        "onnx": "../models_weights/m3_yolo_floor_window.onnx",
        "yaml": "configs/floor_window_eval.yaml",
        "imgsz": 1280,
        "previous_baseline": 0.804,  # M3 (구)
    },
    {
        "name": "m5v2_v2 (frames)",
        "onnx": "../models_weights/m5_yolo_seg_frames.onnx",
        "yaml": "configs/frame_seg_eval.yaml",
        "imgsz": 1280,
        "previous_baseline": 0.626,  # M5 (구)
    },
]


def evaluate(cfg):
    name = cfg["name"]
    onnx_path = Path(cfg["onnx"])
    print(f"\n{'='*60}")
    print(f"Evaluating: {name}")
    print(f"  ONNX: {onnx_path.resolve()}")
    print(f"  yaml: {cfg['yaml']}")
    print(f"  imgsz: {cfg['imgsz']}")
    print(f"  previous baseline: {cfg['previous_baseline']:.4f}")
    print(f"{'='*60}")

    if not onnx_path.exists():
        print(f"  [ERROR] ONNX 없음")
        return None

    start = time.time()
    try:
        model = YOLO(str(onnx_path), task="detect")
        metrics = model.val(
            data=cfg["yaml"],
            imgsz=cfg["imgsz"],
            batch=4,
            device="cpu",
            workers=0,
            plots=False,
            save_json=False,
            verbose=False,
        )
        elapsed = time.time() - start
        result = {
            "name": name,
            "mAP50": metrics.box.map50,
            "mAP50-95": metrics.box.map,
            "precision": metrics.box.mp,
            "recall": metrics.box.mr,
            "previous": cfg["previous_baseline"],
            "improvement": metrics.box.map50 - cfg["previous_baseline"],
            "time": elapsed,
        }
        print(f"\n[{name}] 결과:")
        print(f"  mAP50:    {result['mAP50']:.4f}")
        print(f"  mAP50-95: {result['mAP50-95']:.4f}")
        print(f"  precision: {result['precision']:.4f}")
        print(f"  recall:    {result['recall']:.4f}")
        print(f"  vs baseline {cfg['previous_baseline']:.4f}: {result['improvement']:+.4f}")
        print(f"  소요: {elapsed/60:.1f}분")
        return result
    except Exception as e:
        print(f"  [ERROR] {e}")
        return None


def main():
    print("=" * 60)
    print("새 ONNX 3개 평가 (CPU)")
    print("=" * 60)

    results = []
    total_start = time.time()

    for cfg in EVALUATIONS:
        r = evaluate(cfg)
        if r:
            results.append(r)

    print(f"\n{'='*60}")
    print(f"전체 평가 완료 (총 {(time.time()-total_start)/60:.1f}분)")
    print(f"{'='*60}\n")

    # 종합 표
    print(f"{'모델':<30} {'mAP50':>10} {'mAP50-95':>10} {'baseline':>10} {'개선':>10} {'0.9?':>8}")
    print("-" * 90)
    for r in results:
        ach09 = "✅ YES" if r["mAP50"] >= 0.9 else "❌ NO"
        print(f"{r['name']:<30} {r['mAP50']:>10.4f} {r['mAP50-95']:>10.4f} {r['previous']:>10.4f} {r['improvement']:>+10.4f} {ach09:>8}")

    # 최종 권장
    print("\n=== 권장 (이전 baseline vs 새 ONNX) ===")
    for r in results:
        if r["improvement"] > 0:
            print(f"  {r['name']}: 새 ONNX 사용 (개선 +{r['improvement']:.4f})")
        else:
            print(f"  {r['name']}: 이전 ONNX 그대로 사용 ({r['improvement']:+.4f})")


if __name__ == "__main__":
    main()
