# =============================================
# evaluate_postprocess_ablation.py
# 역할: 각 모델에 대해 후처리 단계별 mAP 변화 측정 (ablation)
#
# 측정 단계:
#   1. raw            : YOLO predict 그대로
#   2. raw + SAHI     : SAHI 타일링 추론
#   3. raw + TTA      : Test-Time Augmentation (hflip + scale)
#   4. SAHI + TTA     : 타일링 + TTA 둘 다
#
# 출력: 각 모델에 대해 4단계 mAP 비교 표 + JSON
#
# 사용:
#   cd backend/training
#   python eval/evaluate_postprocess_ablation.py --model M2_YOLO --max-images 100
#   python eval/evaluate_postprocess_ablation.py  # 전체
# =============================================

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
BACKEND = ROOT / "backend"
sys.path.insert(0, str(BACKEND))

# evaluate_integrated.py와 같은 유틸 재사용
EVAL_DIR = Path(__file__).parent
sys.path.insert(0, str(EVAL_DIR))
from evaluate_integrated import (  # noqa: E402
    ClassMetrics, ModelEvalResult, PredictionBox,
    iou, match_predictions_to_gt, load_yolo_labels, load_class_names_from_yaml,
    MODEL_EVAL_TARGETS,
)

from app.services.tta import TTAEnsemble  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# ─────────────────────────────────────────────
# SAHI 타일링 (단순 구현, backend의 tiled_inference.py와 동일 컨셉)
# ─────────────────────────────────────────────

def _sahi_tile_predict(
    model_predict_fn: Callable[[np.ndarray], List[dict]],
    image_bgr: np.ndarray,
    tile_size: int = 640,
    overlap: float = 0.2,
    nms_iou: float = 0.5,
) -> List[dict]:
    """
    SAHI 스타일 타일 추론 + cross-tile NMS.
    """
    H, W = image_bgr.shape[:2]
    if H <= tile_size and W <= tile_size:
        return model_predict_fn(image_bgr)

    stride = int(tile_size * (1 - overlap))
    all_dets: List[dict] = []

    for y in range(0, max(1, H - tile_size + 1), stride):
        for x in range(0, max(1, W - tile_size + 1), stride):
            x2 = min(x + tile_size, W)
            y2 = min(y + tile_size, H)
            tile = image_bgr[y:y2, x:x2]
            dets = model_predict_fn(tile)
            for d in dets:
                bbox = d.get("bbox_xyxy")
                if bbox is None:
                    continue
                # 타일 → 원본 좌표
                global_bbox = [
                    bbox[0] + x, bbox[1] + y,
                    bbox[2] + x, bbox[3] + y,
                ]
                all_dets.append({**d, "bbox_xyxy": global_bbox, "_tile": (x, y)})

    # 우측/하단 가장자리 타일 (stride 커버 못함)
    if H > tile_size:
        y = H - tile_size
        for x in range(0, max(1, W - tile_size + 1), stride):
            x2 = min(x + tile_size, W)
            y2 = H
            tile = image_bgr[y:y2, x:x2]
            dets = model_predict_fn(tile)
            for d in dets:
                bbox = d.get("bbox_xyxy")
                if bbox is None:
                    continue
                global_bbox = [bbox[0] + x, bbox[1] + y, bbox[2] + x, bbox[3] + y]
                all_dets.append({**d, "bbox_xyxy": global_bbox, "_tile": (x, y)})

    if W > tile_size:
        x = W - tile_size
        for y in range(0, max(1, H - tile_size + 1), stride):
            tile = image_bgr[y:y + tile_size, x:W]
            dets = model_predict_fn(tile)
            for d in dets:
                bbox = d.get("bbox_xyxy")
                if bbox is None:
                    continue
                global_bbox = [bbox[0] + x, bbox[1] + y, bbox[2] + x, bbox[3] + y]
                all_dets.append({**d, "bbox_xyxy": global_bbox, "_tile": (x, y)})

    # Cross-tile NMS (class별)
    by_class: Dict[str, List[dict]] = {}
    for d in all_dets:
        by_class.setdefault(d["class"], []).append(d)

    merged: List[dict] = []
    for cls, dets in by_class.items():
        sorted_dets = sorted(dets, key=lambda x: x["conf"], reverse=True)
        kept: List[dict] = []
        for d in sorted_dets:
            is_dup = False
            for k in kept:
                if iou(d["bbox_xyxy"], k["bbox_xyxy"]) >= nms_iou:
                    is_dup = True
                    break
            if not is_dup:
                kept.append(d)
        merged.extend(kept)
    return merged


# ─────────────────────────────────────────────
# 모델 추론 래퍼 (ultralytics 결과 → dict)
# ─────────────────────────────────────────────

def _make_predict_fn(model, class_names: List[str], imgsz: int, conf: float):
    """ultralytics model + class_names → predict 함수."""
    def predict(image: np.ndarray) -> List[dict]:
        results = model.predict(
            source=image, conf=conf, iou=0.5,
            imgsz=imgsz, verbose=False, save=False, device="cpu",
        )
        if not results:
            return []
        res = results[0]
        out: List[dict] = []
        if res.boxes is None or len(res.boxes) == 0:
            return out
        xyxy = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        cls = res.boxes.cls.cpu().numpy().astype(int)
        for j in range(len(xyxy)):
            cid = int(cls[j])
            if not (0 <= cid < len(class_names)):
                continue
            out.append({
                "class": class_names[cid],
                "class_id": cid,
                "conf": float(confs[j]),
                "bbox_xyxy": [float(v) for v in xyxy[j]],
            })
        return out
    return predict


# ─────────────────────────────────────────────
# 단일 모델 ablation 평가
# ─────────────────────────────────────────────

@dataclass
class AblationResult:
    """후처리 단계별 결과 묶음."""
    model_name: str
    raw: Optional[ModelEvalResult] = None
    sahi: Optional[ModelEvalResult] = None
    tta: Optional[ModelEvalResult] = None
    sahi_tta: Optional[ModelEvalResult] = None
    error: Optional[str] = None


def _compute_metrics_for_predictions(
    images: List[Path],
    pred_per_image: Dict[str, List[dict]],
    lbl_dir: Path,
    class_names: List[str],
    iou_thr: float,
    label: str,
) -> ModelEvalResult:
    """이미지별 예측 dict를 받아서 mAP 계산."""
    result = ModelEvalResult(
        model_name=label, onnx_path="", test_size=len(images),
        imgsz=0, iou_threshold=iou_thr, conf_threshold=0.0,
    )
    for cid, cname in enumerate(class_names):
        result.per_class[cname] = ClassMetrics(class_id=cid, class_name=cname)

    for img_path in images:
        gts = load_yolo_labels(lbl_dir / (img_path.stem + ".txt"))
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        H, W = img.shape[:2]

        for g in gts:
            if 0 <= g.class_id < len(class_names):
                result.per_class[class_names[g.class_id]].n_gt += 1

        # 예측
        preds_raw = pred_per_image.get(str(img_path), [])
        preds: List[PredictionBox] = []
        for d in preds_raw:
            cid = d.get("class_id")
            if cid is None or not (0 <= cid < len(class_names)):
                continue
            preds.append(PredictionBox(
                class_id=cid, class_name=class_names[cid],
                conf=d["conf"], bbox_xyxy=d["bbox_xyxy"],
            ))

        pm, gm = match_predictions_to_gt(preds, gts, W, H, iou_threshold=iou_thr)
        for pi, p in enumerate(preds):
            cm = result.per_class[p.class_name]
            cm.confidences.append(p.conf)
            cm.is_correct.append(pm[pi])
            if pm[pi]:
                cm.tp += 1
            else:
                cm.fp += 1
        for gi, g in enumerate(gts):
            if 0 <= g.class_id < len(class_names) and not gm[gi]:
                result.per_class[class_names[g.class_id]].fn += 1

    return result


def evaluate_ablation(
    target: dict,
    weights_dir: Path,
    cwd: Path,
    iou_thr: float = 0.5,
    max_images: Optional[int] = None,
    skip_sahi: bool = False,
    skip_tta: bool = False,
) -> AblationResult:
    """단일 모델에 대해 4단계 ablation 평가."""
    from ultralytics import YOLO

    onnx_path = weights_dir / target["onnx"]
    dataset_dir = cwd / target["dataset"]
    yaml_path = dataset_dir / "data.yaml"
    img_dir = dataset_dir / "images" / "test"
    lbl_dir = dataset_dir / "labels" / "test"

    ablation = AblationResult(model_name=target["key"])

    if not onnx_path.exists():
        ablation.error = f"ONNX missing: {onnx_path}"
        return ablation
    if not yaml_path.exists() or not img_dir.exists():
        ablation.error = f"Dataset missing: {dataset_dir}"
        return ablation

    class_names = load_class_names_from_yaml(yaml_path)
    if not class_names:
        ablation.error = f"Class names not loaded: {yaml_path}"
        return ablation

    images = sorted([
        f for f in img_dir.iterdir()
        if f.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ])
    if max_images:
        images = images[:max_images]

    print(f"\n[{target['key']}] {len(images)} images, classes={class_names}")

    model = YOLO(str(onnx_path), task="detect")
    predict_fn = _make_predict_fn(model, class_names, target["imgsz"], target["conf"])

    # === Stage 1: raw ===
    print("  raw...")
    t0 = time.time()
    raw_preds: Dict[str, List[dict]] = {}
    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        raw_preds[str(img_path)] = predict_fn(img)
    raw_result = _compute_metrics_for_predictions(images, raw_preds, lbl_dir, class_names, iou_thr, f"{target['key']}_raw")
    raw_result.elapsed_sec = time.time() - t0
    ablation.raw = raw_result
    print(f"    mAP={raw_result.mAP50:.4f} R={raw_result.macro_recall:.4f} P={raw_result.macro_precision:.4f} ({raw_result.elapsed_sec/60:.1f}min)")

    # === Stage 2: SAHI ===
    if not skip_sahi:
        print("  sahi tiling...")
        t0 = time.time()
        sahi_preds: Dict[str, List[dict]] = {}
        for img_path in images:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            sahi_preds[str(img_path)] = _sahi_tile_predict(
                predict_fn, img,
                tile_size=min(640, target["imgsz"]), overlap=0.2,
            )
        sahi_result = _compute_metrics_for_predictions(images, sahi_preds, lbl_dir, class_names, iou_thr, f"{target['key']}_sahi")
        sahi_result.elapsed_sec = time.time() - t0
        ablation.sahi = sahi_result
        print(f"    mAP={sahi_result.mAP50:.4f} R={sahi_result.macro_recall:.4f} P={sahi_result.macro_precision:.4f} ({sahi_result.elapsed_sec/60:.1f}min)")

    # === Stage 3: TTA ===
    if not skip_tta:
        print("  tta...")
        t0 = time.time()
        tta = TTAEnsemble(
            augmentations=["horizontal_flip", "scale_0_8", "scale_1_2"],
            merge_method="wbf", iou_merge_threshold=0.5, include_original=True,
        )
        tta_preds: Dict[str, List[dict]] = {}
        for img_path in images:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            tta_preds[str(img_path)] = tta.predict(predict_fn, img)
        tta_result = _compute_metrics_for_predictions(images, tta_preds, lbl_dir, class_names, iou_thr, f"{target['key']}_tta")
        tta_result.elapsed_sec = time.time() - t0
        ablation.tta = tta_result
        print(f"    mAP={tta_result.mAP50:.4f} R={tta_result.macro_recall:.4f} P={tta_result.macro_precision:.4f} ({tta_result.elapsed_sec/60:.1f}min)")

    # === Stage 4: SAHI + TTA ===
    if not skip_sahi and not skip_tta:
        print("  sahi + tta...")
        t0 = time.time()
        sahi_tta_predict = lambda img: _sahi_tile_predict(predict_fn, img, tile_size=min(640, target["imgsz"]), overlap=0.2)
        tta_full = TTAEnsemble(
            augmentations=["horizontal_flip"],  # 타일 + TTA는 hflip만 (시간 폭증 방지)
            merge_method="wbf", iou_merge_threshold=0.5, include_original=True,
        )
        st_preds: Dict[str, List[dict]] = {}
        for img_path in images:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            st_preds[str(img_path)] = tta_full.predict(sahi_tta_predict, img)
        st_result = _compute_metrics_for_predictions(images, st_preds, lbl_dir, class_names, iou_thr, f"{target['key']}_sahi_tta")
        st_result.elapsed_sec = time.time() - t0
        ablation.sahi_tta = st_result
        print(f"    mAP={st_result.mAP50:.4f} R={st_result.macro_recall:.4f} P={st_result.macro_precision:.4f} ({st_result.elapsed_sec/60:.1f}min)")

    return ablation


# ─────────────────────────────────────────────
# 보고서
# ─────────────────────────────────────────────

def render_ablation_md(ablations: List[AblationResult]) -> str:
    lines = []
    lines.append("# 후처리 Ablation 평가\n")
    lines.append(f"생성: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append("## 단계별 mAP 비교\n")
    lines.append("| 모델 | raw mAP | +SAHI | +TTA | +SAHI+TTA | 최대 개선분 |")
    lines.append("|------|---------|-------|------|-----------|------------|")

    for a in ablations:
        if a.error:
            lines.append(f"| {a.model_name} | ❌ {a.error} | | | | |")
            continue
        raw = a.raw.mAP50 if a.raw else 0.0
        sahi = a.sahi.mAP50 if a.sahi else None
        tta = a.tta.mAP50 if a.tta else None
        st = a.sahi_tta.mAP50 if a.sahi_tta else None
        max_val = max([v for v in [raw, sahi, tta, st] if v is not None])
        delta = max_val - raw

        sahi_s = f"{sahi:.4f}" if sahi is not None else "—"
        tta_s = f"{tta:.4f}" if tta is not None else "—"
        st_s = f"{st:.4f}" if st is not None else "—"

        lines.append(f"| {a.model_name} | {raw:.4f} | {sahi_s} | {tta_s} | {st_s} | +{delta:.4f} |")

    lines.append("\n## 시간 비용\n")
    lines.append("| 모델 | raw | +SAHI | +TTA | +SAHI+TTA |")
    lines.append("|------|------|-------|------|-----------|")
    for a in ablations:
        if a.error:
            continue
        raw_t = f"{a.raw.elapsed_sec/60:.1f}min" if a.raw else "—"
        sahi_t = f"{a.sahi.elapsed_sec/60:.1f}min" if a.sahi else "—"
        tta_t = f"{a.tta.elapsed_sec/60:.1f}min" if a.tta else "—"
        st_t = f"{a.sahi_tta.elapsed_sec/60:.1f}min" if a.sahi_tta else "—"
        lines.append(f"| {a.model_name} | {raw_t} | {sahi_t} | {tta_t} | {st_t} |")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None, help="단일 모델만 (M1_YOLO, M2_YOLO, ...)")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--quick", action="store_true", help="--max-images 50")
    parser.add_argument("--skip-sahi", action="store_true")
    parser.add_argument("--skip-tta", action="store_true")
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--out-dir", type=str, default="eval/results")
    args = parser.parse_args()

    if args.quick:
        args.max_images = args.max_images or 50

    cwd = Path.cwd()
    weights_dir = cwd.parent / "models_weights"
    out_dir = cwd / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"후처리 Ablation 평가 (IoU={args.iou})")
    print(f"  cwd: {cwd}")
    print(f"  weights_dir: {weights_dir}")
    print(f"  max_images: {args.max_images or 'all'}")
    print(f"  skip_sahi: {args.skip_sahi}, skip_tta: {args.skip_tta}")
    print("=" * 70)

    targets = MODEL_EVAL_TARGETS
    if args.model:
        targets = [t for t in targets if t["key"] == args.model]
        if not targets:
            print(f"[ERROR] Model not found: {args.model}")
            return 1

    ablations: List[AblationResult] = []
    for t in targets:
        ab = evaluate_ablation(
            t, weights_dir=weights_dir, cwd=cwd,
            iou_thr=args.iou, max_images=args.max_images,
            skip_sahi=args.skip_sahi, skip_tta=args.skip_tta,
        )
        ablations.append(ab)

    ts = time.strftime("%Y%m%d_%H%M%S")
    json_path = out_dir / f"postprocess_ablation_{ts}.json"
    md_path = out_dir / f"postprocess_ablation_{ts}.md"

    json_data = {
        "timestamp": ts,
        "iou_threshold": args.iou,
        "max_images": args.max_images,
        "ablations": [
            {
                "model_name": a.model_name,
                "error": a.error,
                "raw": {"mAP50": a.raw.mAP50, "recall": a.raw.macro_recall, "precision": a.raw.macro_precision, "elapsed_min": a.raw.elapsed_sec/60} if a.raw else None,
                "sahi": {"mAP50": a.sahi.mAP50, "recall": a.sahi.macro_recall, "precision": a.sahi.macro_precision, "elapsed_min": a.sahi.elapsed_sec/60} if a.sahi else None,
                "tta": {"mAP50": a.tta.mAP50, "recall": a.tta.macro_recall, "precision": a.tta.macro_precision, "elapsed_min": a.tta.elapsed_sec/60} if a.tta else None,
                "sahi_tta": {"mAP50": a.sahi_tta.mAP50, "recall": a.sahi_tta.macro_recall, "precision": a.sahi_tta.macro_precision, "elapsed_min": a.sahi_tta.elapsed_sec/60} if a.sahi_tta else None,
            }
            for a in ablations
        ],
    }
    json_path.write_text(json.dumps(json_data, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(render_ablation_md(ablations), encoding="utf-8")

    print("\n결과 저장:")
    print(f"  JSON: {json_path}")
    print(f"  MD:   {md_path}")
    print()
    print(render_ablation_md(ablations))
    return 0


if __name__ == "__main__":
    sys.exit(main())
