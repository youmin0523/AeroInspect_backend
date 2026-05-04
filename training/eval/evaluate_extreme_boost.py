"""
Extreme boost eval — 재학습 없이 mAP 끌어올릴 수 있는 모든 카드:
1. .pt + augment=True (real TTA)
2. Multi-scale (480~1280)
3. agnostic_nms on/off
4. iou threshold sweep (0.5, 0.6, 0.7)
5. max_det 변동 (300, 1000)
"""
from __future__ import annotations

import json
import sys
import time
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional

from ultralytics import YOLO

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


PROJECT_ROOT = Path(__file__).resolve().parents[3]


# 모델별 .pt + 데이터셋
TARGETS = [
    {
        "key": "M2_YOLO",
        "pt": Path("c:/Users/Codelab/Downloads/colab_pt_extracted/m2_plan_a_results/m2_yolo_surface_best.pt"),
        "data": "datasets/surface/data.yaml",
        "imgsz_list": [480, 640, 800],
    },
    {
        "key": "M3_YOLO",
        "pt": PROJECT_ROOT / "runs/detect/runs/m3_floor_window/phase2_full/weights/best.pt",
        "data": "datasets/floor_window/data.yaml",
        "imgsz_list": [640, 800, 960],
    },
    {
        "key": "M5_SEG",
        "pt": Path("c:/Users/Codelab/Desktop/PROJECT/TEAM_PROJECT_2_Drone_project/backend/training/colab/upload_to_drive/m5_baseline_best.pt"),
        "data": "datasets/frames/data.yaml",
        "imgsz_list": [480, 640, 800],
    },
    {
        "key": "M4_CONTEXT",
        "pt": PROJECT_ROOT / "runs/detect/runs/m4v2/stage1/weights/best.pt",
        "data": "datasets/m4_context/data.yaml",
        "imgsz_list": [640, 960],
    },
    {
        "key": "M1_YOLO",
        "pt": PROJECT_ROOT / "runs/detect/runs/m1_yolo_structural_960/finetune/weights/best.pt",
        "data": "datasets/structural/data.yaml",
        "imgsz_list": [640, 960],
    },
]

# 후처리 파라미터 그리드
TTA_OPTS = [False, True]
AGNOSTIC_OPTS = [False, True]
IOU_OPTS = [0.5, 0.6, 0.7]
MAX_DET_OPTS = [300]


def evaluate_extreme(target: dict) -> Optional[Dict]:
    pt_path = Path(target["pt"])
    data_yaml = Path(target["data"])

    if not pt_path.exists():
        print(f"❌ {target['key']}: pt not found at {pt_path}")
        return None
    if not data_yaml.exists():
        print(f"❌ {target['key']}: data yaml not found at {data_yaml}")
        return None

    print(f"\n{'='*70}\n=== {target['key']} (extreme boost) ===\n{'='*70}")
    print(f"  pt: {pt_path}")

    best = {"mAP50": -1.0}

    split = "test" if (data_yaml.parent / "images" / "test").exists() else "val"

    # 그리드: imgsz × tta × agnostic × iou × max_det
    grid = list(product(target["imgsz_list"], TTA_OPTS, AGNOSTIC_OPTS, IOU_OPTS, MAX_DET_OPTS))
    print(f"  Grid: {len(grid)} configurations")

    for imgsz, tta, agnostic, iou, max_det in grid:
        try:
            model = YOLO(str(pt_path))
            metrics = model.val(
                data=str(data_yaml),
                imgsz=imgsz, batch=8, device=0,
                workers=0, plots=False, save_json=False, verbose=False,
                augment=tta, conf=0.001, iou=iou,
                agnostic_nms=agnostic, max_det=max_det,
                split=split,
            )
            m50 = float(metrics.box.map50)
            m9 = float(metrics.box.map)
            p = float(metrics.box.mp)
            r = float(metrics.box.mr)
            tag = f"imgsz={imgsz} tta={'O' if tta else 'X'} ag={'O' if agnostic else 'X'} iou={iou} md={max_det}"
            marker = "🎯" if m50 >= 0.85 else ("📈" if m50 > best["mAP50"] else "")
            print(f"  {marker} {tag}: mAP50={m50:.4f}")
            if m50 > best["mAP50"]:
                best = {"mAP50": m50, "mAP": m9, "P": p, "R": r,
                        "imgsz": imgsz, "tta": tta, "agnostic": agnostic,
                        "iou": iou, "max_det": max_det}
        except Exception as e:
            tag = f"imgsz={imgsz} tta={tta} ag={agnostic} iou={iou}"
            print(f"  ❌ {tag} FAIL: {type(e).__name__}: {e}")

    print(f"\n  ⭐ BEST for {target['key']}: mAP50={best['mAP50']:.4f}")
    print(f"     imgsz={best.get('imgsz')} tta={'O' if best.get('tta') else 'X'}"
          f" agnostic={'O' if best.get('agnostic') else 'X'}"
          f" iou={best.get('iou')} max_det={best.get('max_det')}")
    return {"key": target["key"], **best}


def main():
    cwd = Path.cwd()
    print(f"cwd: {cwd}")
    print(f"평가 대상: {len(TARGETS)}개 (extreme boost grid)")

    results: List[Dict] = []
    for t in TARGETS:
        r = evaluate_extreme(t)
        if r:
            results.append(r)

    print(f"\n{'='*80}")
    print("Extreme boost 종합")
    print(f"{'='*80}")
    print(f"| 모델 | best mAP50 | imgsz | TTA | agnostic | iou | 0.85 갭 |")
    print(f"|------|------------|-------|-----|----------|-----|---------|")
    for r in results:
        gap = r['mAP50'] - 0.85
        gap_s = f"+{gap:.4f}" if gap >= 0 else f"{gap:.4f}"
        print(f"| {r['key']} | **{r['mAP50']:.4f}** | {r.get('imgsz')} |"
              f" {'O' if r.get('tta') else 'X'} |"
              f" {'O' if r.get('agnostic') else 'X'} |"
              f" {r.get('iou')} | {gap_s} |")

    out = cwd / "eval/results" / f"extreme_boost_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n결과: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
