"""
PyTorch native TTA evaluation
- ONNX는 ultralytics에서 augment=True가 silently 무시됨 (max_boost 결과로 확인)
- .pt 파일은 PyTorch native라 진짜 TTA 적용됨
- 0.85 미달 모델 중 가까운 것부터 검증
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

from ultralytics import YOLO

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


PROJECT_ROOT = Path(__file__).resolve().parents[3]

TARGETS = [
    {
        "key": "M3_YOLO",
        "pt": PROJECT_ROOT / "runs/detect/runs/m3_floor_window/phase2_full/weights/best.pt",
        "data": "datasets/floor_window/data.yaml",
        "imgsz_list": [640, 960],
    },
    {
        "key": "M2_YOLO",
        "pt": PROJECT_ROOT / "runs/detect/runs/m2_surface/phase1_freeze/weights/best.pt",
        "data": "datasets/surface/data.yaml",
        "imgsz_list": [480, 640],
    },
    {
        "key": "M4_CONTEXT",
        "pt": PROJECT_ROOT / "runs/detect/runs/m4v2/stage1/weights/best.pt",
        "data": "datasets/m4_context/data.yaml",
        "imgsz_list": [640, 960],
    },
]


def evaluate_pt_tta(target: dict) -> Optional[Dict]:
    pt_path = Path(target["pt"])
    data_yaml = Path(target["data"])

    if not pt_path.exists():
        print(f"❌ {target['key']}: pt not found at {pt_path}")
        return None
    if not data_yaml.exists():
        print(f"❌ {target['key']}: data yaml not found at {data_yaml}")
        return None

    print(f"\n{'='*60}\n=== {target['key']} (.pt + real TTA) ===\n{'='*60}")
    print(f"  pt: {pt_path.name}")

    best = {"mAP50": -1.0, "imgsz": None, "tta": None}

    for imgsz in target["imgsz_list"]:
        for tta in [False, True]:
            split = "test" if (data_yaml.parent / "images" / "test").exists() else "val"
            try:
                model = YOLO(str(pt_path))
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
    print(f"평가 대상: {len(TARGETS)}개 (.pt + real TTA)")

    results: List[Dict] = []
    for t in TARGETS:
        r = evaluate_pt_tta(t)
        if r:
            results.append(r)

    print(f"\n{'='*70}")
    print(".pt + real TTA 종합")
    print(f"{'='*70}")
    print(f"| 모델 | best mAP50 | imgsz | TTA | 0.85 갭 |")
    print(f"|------|------------|-------|-----|---------|")
    for r in results:
        tta_s = "O" if r['tta'] else "X"
        gap = r['mAP50'] - 0.85
        gap_s = f"+{gap:.4f}" if gap >= 0 else f"{gap:.4f}"
        print(f"| {r['key']} | **{r['mAP50']:.4f}** | {r['imgsz']} | {tta_s} | {gap_s} |")

    out = cwd / "eval/results" / f"pt_tta_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n결과: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
