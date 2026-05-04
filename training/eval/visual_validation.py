# =============================================
# visual_validation.py
# 역할: bbox + 객체감지 라벨 시각 검증
#       - GT (초록) + Pred (빨강) bbox 겹쳐 그림
#       - 클래스 라벨 + conf score 표시
#       - IoU 0.5 매칭/불일치 색상 구분
#       - 결과 디렉토리에 시각화 PNG 저장
#
# 사용:
#   cd backend/training
#   python eval/visual_validation.py --n-per-dataset 5 --out eval/results/visual
# =============================================

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
BACKEND = ROOT / "backend"
sys.path.insert(0, str(BACKEND))

EVAL_DIR = Path(__file__).parent
sys.path.insert(0, str(EVAL_DIR))
from evaluate_integrated import load_yolo_labels, iou  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# 색상 (BGR)
COLOR_GT_MATCHED = (0, 255, 0)        # 초록 — GT 매칭됨
COLOR_GT_MISSED = (0, 200, 200)       # 노랑 — GT 미탐지 (FN)
COLOR_PRED_MATCHED = (0, 100, 0)      # 진초록 — Pred 매칭됨 (TP)
COLOR_PRED_FP = (0, 0, 255)           # 빨강 — Pred FP


DATASETS = [
    ("structural", "datasets/structural/images/test", "datasets/structural/labels/test", "datasets/structural/data.yaml"),
    ("surface", "datasets/surface/images/test", "datasets/surface/labels/test", "datasets/surface/data.yaml"),
    ("floor_window", "datasets/floor_window/images/test", "datasets/floor_window/labels/test", "datasets/floor_window/data.yaml"),
    ("frames", "datasets/frames/images/test", "datasets/frames/labels/test", "datasets/frames/data.yaml"),
    ("m4_context", "datasets/m4_context/images/test", "datasets/m4_context/labels/test", "datasets/m4_context/data.yaml"),
]


def _load_class_names(yaml_path: Path) -> List[str]:
    """data.yaml에서 names: 추출."""
    if not yaml_path.exists():
        return []
    text = yaml_path.read_text(encoding="utf-8")
    names: List[str] = []
    in_names = False
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("names:"):
            in_names = True
            if "[" in s and "]" in s:
                inner = s.split("[", 1)[1].rsplit("]", 1)[0]
                names = [x.strip().strip("'\"") for x in inner.split(",") if x.strip()]
                break
            continue
        if in_names:
            if not s or s.startswith("#"):
                break
            if ":" in s:
                _, name = s.split(":", 1)
                names.append(name.strip().strip("'\""))
            elif s.startswith("-"):
                names.append(s.lstrip("- ").strip().strip("'\""))
            else:
                break
    return names


def _draw_bbox_with_label(
    img: np.ndarray,
    bbox: List[float],
    label: str,
    color: Tuple[int, int, int],
    thickness: int = 2,
) -> None:
    """이미지에 bbox + 라벨 그림 (in-place)."""
    x1, y1, x2, y2 = [int(v) for v in bbox]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

    # 라벨 배경
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(img, (x1, max(0, y1 - th - 4)), (x1 + tw + 4, y1), color, -1)
    cv2.putText(
        img, label, (x1 + 2, y1 - 4),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
    )


def visualize_image(
    img_bgr: np.ndarray,
    gt_boxes: List[Tuple[List[float], str]],   # [(xyxy, class_name)]
    pred_boxes: List[Tuple[List[float], str, float]],  # [(xyxy, class_name, conf)]
    iou_threshold: float = 0.5,
) -> np.ndarray:
    """GT + Pred 시각화. 매칭된 것은 초록, FP/FN은 빨강/노랑."""
    out = img_bgr.copy()

    # IoU 매칭 (class-agnostic)
    n_pred = len(pred_boxes)
    n_gt = len(gt_boxes)
    pred_matched = [False] * n_pred
    gt_matched = [False] * n_gt

    pred_order = sorted(range(n_pred), key=lambda i: pred_boxes[i][2], reverse=True)
    for pi in pred_order:
        best_iou = 0.0
        best_gi = -1
        for gi in range(n_gt):
            if gt_matched[gi]:
                continue
            io = iou(pred_boxes[pi][0], gt_boxes[gi][0])
            if io > best_iou:
                best_iou = io
                best_gi = gi
        if best_iou >= iou_threshold and best_gi >= 0:
            pred_matched[pi] = True
            gt_matched[best_gi] = True

    # GT 그리기
    for gi, (gbox, gcls) in enumerate(gt_boxes):
        if gt_matched[gi]:
            color = COLOR_GT_MATCHED
            label = f"GT: {gcls}"
        else:
            color = COLOR_GT_MISSED
            label = f"GT-MISSED: {gcls}"
        _draw_bbox_with_label(out, gbox, label, color, thickness=2)

    # Pred 그리기
    for pi, (pbox, pcls, pconf) in enumerate(pred_boxes):
        if pred_matched[pi]:
            color = COLOR_PRED_MATCHED
            label = f"TP: {pcls} {pconf:.2f}"
        else:
            color = COLOR_PRED_FP
            label = f"FP: {pcls} {pconf:.2f}"
        _draw_bbox_with_label(out, pbox, label, color, thickness=1)

    # 통계 텍스트 좌상단
    tp = sum(pred_matched)
    fp = n_pred - tp
    fn = sum(1 for m in gt_matched if not m)
    stats_text = f"GT={n_gt} | Pred={n_pred} | TP={tp} FP={fp} FN={fn}"
    cv2.rectangle(out, (5, 5), (5 + 380, 30), (0, 0, 0), -1)
    cv2.putText(
        out, stats_text, (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA,
    )
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-per-dataset", type=int, default=5, help="데이터셋당 시각화할 이미지 수")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--out", type=str, default="eval/results/visual")
    args = parser.parse_args()

    cwd = Path.cwd()
    backend_dir = ROOT / "backend"
    out_dir = cwd / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    # 데이터셋별 샘플 수집
    samples: List[Tuple[str, Path, Path, List[str]]] = []
    for name, img_rel, lbl_rel, yaml_rel in DATASETS:
        img_dir = cwd / img_rel
        lbl_dir = cwd / lbl_rel
        yaml_path = cwd / yaml_rel
        if not img_dir.exists():
            continue
        class_names = _load_class_names(yaml_path)
        all_imgs = sorted([
            f for f in img_dir.iterdir()
            if f.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ])
        if not all_imgs:
            continue
        chosen = rng.sample(all_imgs, min(args.n_per_dataset, len(all_imgs)))
        for img in chosen:
            lbl = lbl_dir / (img.stem + ".txt")
            samples.append((name, img, lbl, class_names))

    print(f"시각화 대상: {len(samples)}장")

    # Pipeline 로딩 (cwd=backend/)
    orig_cwd = os.getcwd()
    try:
        os.chdir(backend_dir)
        from app.services.inference_pipeline_20 import InferencePipeline20
        pipe = InferencePipeline20()
        pipe.load_models()
        if not pipe.is_loaded:
            print("[ERROR] Pipeline not loaded")
            return 1

        n_total_tp = 0
        n_total_fp = 0
        n_total_fn = 0
        n_total_gt = 0

        for ds_name, img_path, lbl_path, class_names in samples:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            H, W = img.shape[:2]

            # GT
            gts_raw = load_yolo_labels(lbl_path)
            gt_boxes_with_class: List[Tuple[List[float], str]] = []
            for g in gts_raw:
                cname = class_names[g.class_id] if 0 <= g.class_id < len(class_names) else f"cls_{g.class_id}"
                gt_boxes_with_class.append((g.to_xyxy(W, H), cname))

            # Pred — Tier 2 (M1+M2+M3+M4 Context+M5+gates)
            try:
                result = pipe.detect(img, tier=2)
            except Exception as e:
                print(f"[ERROR] {img_path.name}: {e}")
                continue

            pred_boxes_with_class: List[Tuple[List[float], str, float]] = []
            for d in result.detections:
                pred_boxes_with_class.append((list(d.bbox_xyxy), d.class_, d.conf))

            # 시각화
            vis = visualize_image(
                img, gt_boxes_with_class, pred_boxes_with_class,
                iou_threshold=args.iou,
            )

            # 통계 누적
            n_pred = len(pred_boxes_with_class)
            tp = sum(
                1 for _ in range(n_pred)
                if any(iou(pred_boxes_with_class[pi][0], g[0]) >= args.iou
                       for pi, g in [(pi, g) for pi in range(n_pred) for g in gt_boxes_with_class])
            )
            # 정확한 TP 계산은 visualize_image와 동일하게 다시 한번
            # (위 추정은 부정확 — visualize 내부 매칭 로직 사용)
            # 결과 텍스트는 이미지에 박혀 있으므로 별도 stats는 지금은 생략
            # 다만 GT 수만 누적
            n_total_gt += len(gt_boxes_with_class)

            # 저장
            out_subdir = out_dir / ds_name
            out_subdir.mkdir(parents=True, exist_ok=True)
            out_path = out_subdir / f"{img_path.stem}_visual.jpg"
            cv2.imwrite(str(out_path), vis)

        print(f"\n시각화 완료: {out_dir.resolve()}")
        print(f"  데이터셋별 하위 폴더에 GT(초록/노랑) + Pred(진초록/빨강) bbox 저장됨")
        print(f"  - 초록 = GT (매칭 성공)")
        print(f"  - 노랑 = GT 미탐지 (FN)")
        print(f"  - 진초록 = Pred 정확 (TP)")
        print(f"  - 빨강 = Pred 오탐 (FP)")
    finally:
        os.chdir(orig_cwd)
    return 0


if __name__ == "__main__":
    sys.exit(main())
