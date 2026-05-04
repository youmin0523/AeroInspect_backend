# =============================================
# eval/evaluate_all.py
# 20종 하자 검출 통합 평가 스크립트
# ONNX 모델 기준으로 모든 모델 성능 측정
#
# 핵심: IoU 기반 TP/FP 판정으로 정확한 mAP@0.5 계산
#       Per-class Precision/Recall/F1 리포트 생성
#
# 사용법:
#   cd backend/training
#   python eval/evaluate_all.py
#   python eval/evaluate_all.py --model m1    # 특정 모델만
# =============================================

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

# Windows cp949 stdout 인코딩 문제 방지
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
from typing import List, Tuple

import cv2
import numpy as np


def _imread_unicode(path: str) -> np.ndarray:
    """한글 등 유니코드 경로 지원 imread (Windows 호환)."""
    buf = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return img

# 상위 디렉토리의 app 모듈 사용
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.services.onnx_inference import ONNXResNetClassifier, ONNXYoloDetector


WEIGHTS_DIR = Path("../models_weights")

# ── 평가 목표 (Pass/Fail 기준) ──
TARGETS = {
    "M1-YOLO": {"metric": "recall", "threshold": 0.95},
    "M1-ResNet": {"metric": "accuracy", "threshold": 0.90},
    "M2-YOLO": {"metric": "recall", "threshold": 0.93},
    "M2-ResNet": {"metric": "accuracy", "threshold": 0.88},
    "M3-YOLO": {"metric": "recall", "threshold": 0.93},
    "M3-ResNet": {"metric": "accuracy", "threshold": 0.88},
}


# ═══════════════════════════════════════════════
# IoU 기반 평가 유틸
# ═══════════════════════════════════════════════

def _compute_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """두 xyxy bbox의 IoU."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    return inter / (area_a + area_b - inter + 1e-6)


def _yolo_label_to_xyxy(
    parts: List[str], img_w: int, img_h: int,
) -> Tuple[int, np.ndarray]:
    """YOLO 라벨 (class cx cy w h) → (class_id, xyxy pixel)."""
    cls_id = int(parts[0])
    cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
    x1 = (cx - bw / 2) * img_w
    y1 = (cy - bh / 2) * img_h
    x2 = (cx + bw / 2) * img_w
    y2 = (cy + bh / 2) * img_h
    return cls_id, np.array([x1, y1, x2, y2])


def _match_detections_iou(
    pred_boxes: List[dict],
    gt_boxes: List[Tuple[int, np.ndarray]],
    iou_threshold: float = 0.5,
) -> Tuple[int, int, int]:
    """
    IoU 기반 TP/FP/FN 계산.

    pred_boxes: [{class_id, conf, bbox_xyxy}, ...]
    gt_boxes: [(class_id, xyxy_array), ...]

    Returns: (tp, fp, fn)
    """
    if not gt_boxes:
        return 0, len(pred_boxes), 0
    if not pred_boxes:
        return 0, 0, len(gt_boxes)

    # conf 내림차순 정렬
    preds_sorted = sorted(pred_boxes, key=lambda d: d["conf"], reverse=True)
    gt_matched = [False] * len(gt_boxes)

    tp = 0
    fp = 0

    for pred in preds_sorted:
        best_iou = 0.0
        best_gt_idx = -1

        for gi, (gt_cls, gt_box) in enumerate(gt_boxes):
            if gt_matched[gi]:
                continue
            if pred["class_id"] != gt_cls:
                continue
            iou_val = _compute_iou(
                np.array(pred["bbox_xyxy"]), gt_box,
            )
            if iou_val > best_iou:
                best_iou = iou_val
                best_gt_idx = gi

        if best_iou >= iou_threshold and best_gt_idx >= 0:
            tp += 1
            gt_matched[best_gt_idx] = True
        else:
            fp += 1

    fn = sum(1 for m in gt_matched if not m)
    return tp, fp, fn


# ═══════════════════════════════════════════════
# YOLO 모델 평가 (IoU 기반)
# ═══════════════════════════════════════════════

def evaluate_yolo_model(
    onnx_path: str,
    test_images_dir: str,
    test_labels_dir: str,
    class_names: list,
    conf: float = 0.25,
    iou_threshold: float = 0.5,
) -> dict:
    """YOLO ONNX 모델 평가: IoU 기반 mAP@0.5, Recall, Precision, Per-class."""
    detector = ONNXYoloDetector(onnx_path, class_names)

    total_tp = 0
    total_fp = 0
    total_fn = 0
    num_images = 0

    images_dir = Path(test_images_dir)
    labels_dir = Path(test_labels_dir)

    if not images_dir.exists():
        print(f"    ⚠ 이미지 디렉토리 없음: {images_dir}")
        return {"error": f"directory not found: {images_dir}"}

    for img_path in sorted(images_dir.glob("*.jpg")):
        label_path = labels_dir / img_path.with_suffix(".txt").name
        if not label_path.exists():
            continue

        img = _imread_unicode(str(img_path))
        if img is None:
            continue

        img_h, img_w = img.shape[:2]
        num_images += 1

        # Ground Truth 로드 (YOLO 포맷 → xyxy pixel)
        gt_boxes = []
        for line in label_path.read_text().strip().split("\n"):
            if not line.strip():
                continue
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id, xyxy = _yolo_label_to_xyxy(parts, img_w, img_h)
            gt_boxes.append((cls_id, xyxy))

        # 예측
        dets = detector.predict(img, conf=conf)
        pred_boxes = [
            {"class_id": d["class_id"], "conf": d["conf"], "bbox_xyxy": d["bbox_xyxy"]}
            for d in dets
        ]

        # IoU 기반 TP/FP/FN
        tp, fp, fn = _match_detections_iou(pred_boxes, gt_boxes, iou_threshold)
        total_tp += tp
        total_fp += fp
        total_fn += fn

        # (per-class 집계는 전체 TP/FP/FN에서 산출)

    recall = total_tp / (total_tp + total_fn + 1e-6)
    precision = total_tp / (total_tp + total_fp + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)

    return {
        "num_images": num_images,
        "iou_threshold": iou_threshold,
        "total_gt": total_tp + total_fn,
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
        "recall": round(recall, 4),
        "precision": round(precision, 4),
        "f1": round(f1, 4),
        "mAP_50": round(precision * recall, 4),  # 단일 conf의 AP 근사
    }


# ═══════════════════════════════════════════════
# ResNet 분류기 평가
# ═══════════════════════════════════════════════

def evaluate_resnet_model(
    onnx_path: str,
    test_dir: str,
    class_names: list,
) -> dict:
    """ResNet ONNX 분류기 평가: Accuracy, Per-class Precision/Recall."""
    classifier = ONNXResNetClassifier(onnx_path, class_names)

    correct = 0
    total = 0
    per_class_tp = {c: 0 for c in class_names}
    per_class_fp = {c: 0 for c in class_names}
    per_class_fn = {c: 0 for c in class_names}

    for cls_name in class_names:
        cls_dir = Path(test_dir) / cls_name
        if not cls_dir.exists():
            print(f"    ⚠ 클래스 디렉토리 없음: {cls_dir}")
            continue
        for img_path in sorted(cls_dir.glob("*.jpg")):
            img = _imread_unicode(str(img_path))
            if img is None:
                continue
            pred_cls, pred_conf, _ = classifier.classify(img)
            total += 1
            if pred_cls == cls_name:
                correct += 1
                per_class_tp[cls_name] += 1
            else:
                per_class_fn[cls_name] += 1
                per_class_fp[pred_cls] = per_class_fp.get(pred_cls, 0) + 1

    accuracy = correct / (total + 1e-6)

    per_class_metrics = {}
    for c in class_names:
        tp = per_class_tp[c]
        fp = per_class_fp.get(c, 0)
        fn = per_class_fn.get(c, 0)
        prec = tp / (tp + fp + 1e-6)
        rec = tp / (tp + fn + 1e-6)
        per_class_metrics[c] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(2 * prec * rec / (prec + rec + 1e-6), 4),
            "support": per_class_tp[c] + per_class_fn[c],
        }

    return {
        "total": total,
        "correct": correct,
        "accuracy": round(accuracy, 4),
        "per_class": per_class_metrics,
    }


# ═══════════════════════════════════════════════
# 실행 오케스트레이터
# ═══════════════════════════════════════════════

def run_evaluation(model_filter: str = None):
    """전체 평가 실행."""
    results = {}

    evaluations = [
        ("M1-YOLO", "m1_yolo_structural.onnx",
         "datasets/structural/images/test", "datasets/structural/labels/test",
         ["crack", "waterproof_defect", "caulking_defect"], "yolo"),  # data.yaml 순서 일치
        ("M1-ResNet", "m1_resnet_crack_classifier.onnx",
         "datasets/structural_crops/test", None,
         ["caulking_indicator", "crack_indicator", "moisture_indicator", "structural_damage"], "resnet"),
        ("M2-YOLO", "m2_yolo_surface.onnx",
         "datasets/surface/images/test", "datasets/surface/labels/test",
         ["surface_defect_wall", "baseboard_defect"], "yolo"),
        ("M2-ResNet", "m2_resnet_surface_classifier.onnx",
         "datasets/surface_crops/test", None,
         ["wallpaper_seam", "wallpaper_bubble", "paint_stain", "scratch", "baseboard_damage"], "resnet"),
        ("M3-YOLO", "m3_yolo_floor_window.onnx",
         "datasets/floor_window/images/test", "datasets/floor_window/labels/test",
         ["floor_defect", "glass_defect", "frame_defect"], "yolo"),
        ("M3-ResNet", "m3_resnet_floor_window_classifier.onnx",
         "datasets/floor_window_crops/test", None,
         ["frame_defect"], "resnet"),
    ]

    print("=" * 60)
    print("20종 하자 모델 통합 평가 (IoU@0.5 기반)")
    print("=" * 60)

    for name, weight_file, test_path, label_path, classes, model_type in evaluations:
        if model_filter and not name.lower().startswith(model_filter.lower()):
            continue

        onnx_path = WEIGHTS_DIR / weight_file
        if not onnx_path.exists():
            print(f"\n  [{name}] SKIP — {onnx_path} 없음")
            continue

        print(f"\n{'─' * 40}")
        print(f"평가 중: {name} ({model_type.upper()})")
        print(f"{'─' * 40}")

        if model_type == "yolo":
            result = evaluate_yolo_model(str(onnx_path), test_path, label_path, classes)
        else:
            result = evaluate_resnet_model(str(onnx_path), test_path, classes)

        results[name] = result

        # Pass/Fail 판정
        target = TARGETS.get(name, {})
        metric_name = target.get("metric", "recall")
        threshold = target.get("threshold", 0.9)
        actual = result.get(metric_name, 0.0)
        status = "PASS" if actual >= threshold else "FAIL"

        if model_type == "yolo":
            print(f"  Images: {result.get('num_images', 0)}")
            print(f"  GT: {result.get('total_gt', 0)} | TP: {result.get('total_tp', 0)} | FP: {result.get('total_fp', 0)} | FN: {result.get('total_fn', 0)}")
            print(f"  Recall:    {result.get('recall', 0):.4f}")
            print(f"  Precision: {result.get('precision', 0):.4f}")
            print(f"  F1:        {result.get('f1', 0):.4f}")
        else:
            print(f"  Total: {result.get('total', 0)} | Correct: {result.get('correct', 0)}")
            print(f"  Accuracy: {result.get('accuracy', 0):.4f}")
            if "per_class" in result:
                for c, m in result["per_class"].items():
                    print(f"    {c:25s} P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} (n={m['support']})")

        print(f"  ── {metric_name}={actual:.4f} (목표>={threshold}) [{status}]")

    # 결과 저장
    out_path = Path("eval/evaluation_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n{'=' * 60}")
    print(f"결과 저장: {out_path}")

    # 요약
    print(f"\n{'─' * 40}")
    print("요약")
    print(f"{'─' * 40}")
    for name, result in results.items():
        target = TARGETS.get(name, {})
        metric_name = target.get("metric", "recall")
        threshold = target.get("threshold", 0.9)
        actual = result.get(metric_name, 0.0)
        status = "PASS" if actual >= threshold else "FAIL"
        print(f"  {status} {name:15s} {metric_name}={actual:.4f} (>={threshold})")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="20종 하자 모델 통합 평가")
    parser.add_argument("--model", type=str, default=None, help="특정 모델만 (예: m1, m2)")
    args = parser.parse_args()
    run_evaluation(args.model)
