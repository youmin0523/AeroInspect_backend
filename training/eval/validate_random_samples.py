# =============================================
# validate_random_samples.py
# 역할: 무작위 이미지 샘플로 통합 파이프라인 end-to-end 검증
#       - 무작위 샘플 선택 (각 test set에서 N장)
#       - 통합 파이프라인 (M1+M2+M3+M4 Context+M5+gates) 실행
#       - 검출 비율 (% of images with detections matching GT)
#       - 정확도 (IoU>=0.5 매칭 비율)
#       - 화질별 성능 (원본/중간/낮음 — 다운샘플링 시뮬레이션)
#
# 사용:
#   cd backend/training
#   python eval/validate_random_samples.py
#   python eval/validate_random_samples.py --n-per-dataset 30 --seed 42
# =============================================

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
BACKEND = ROOT / "backend"
sys.path.insert(0, str(BACKEND))

EVAL_DIR = Path(__file__).parent
sys.path.insert(0, str(EVAL_DIR))
from evaluate_integrated import (  # noqa: E402
    GroundTruthBox, load_yolo_labels, iou,
)

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# ─────────────────────────────────────────────
# 데이터 구조
# ─────────────────────────────────────────────

@dataclass
class ImageValidation:
    """이미지 1장 검증 결과."""
    img_path: str
    quality: str        # "original" | "medium" | "low"
    gt_count: int
    pred_count: int
    tp: int = 0         # IoU>=0.5 매칭
    fp: int = 0         # 매칭 안 된 검출
    fn: int = 0         # 매칭 안 된 GT
    inference_ms: float = 0.0


@dataclass
class QualityReport:
    """화질별 종합 결과."""
    quality: str
    n_images: int = 0
    n_with_gt: int = 0          # GT가 있는 이미지 수
    n_detected: int = 0         # 검출 1개 이상 발생한 이미지 수
    n_correct: int = 0          # GT와 IoU>=0.5 매칭된 검출 1개 이상
    total_tp: int = 0
    total_fp: int = 0
    total_fn: int = 0
    total_gt: int = 0
    total_pred: int = 0
    avg_inference_ms: float = 0.0
    per_image: List[ImageValidation] = field(default_factory=list)

    @property
    def detection_rate(self) -> float:
        """GT가 있는 이미지 중 검출에 성공한 비율."""
        return self.n_detected / self.n_with_gt if self.n_with_gt > 0 else 0.0

    @property
    def accuracy_rate(self) -> float:
        """GT가 있는 이미지 중 IoU 매칭에 성공한 비율."""
        return self.n_correct / self.n_with_gt if self.n_with_gt > 0 else 0.0

    @property
    def precision(self) -> float:
        denom = self.total_tp + self.total_fp
        return self.total_tp / denom if denom > 0 else 0.0

    @property
    def recall(self) -> float:
        denom = self.total_tp + self.total_fn
        return self.total_tp / denom if denom > 0 else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


# ─────────────────────────────────────────────
# 화질 변형
# ─────────────────────────────────────────────

def degrade_quality(img: np.ndarray, level: str) -> np.ndarray:
    """이미지 화질 저하 시뮬레이션."""
    if level == "original":
        return img
    elif level == "medium":
        # 50% 다운샘플 후 원래 크기로 복원 (해상도 손실)
        h, w = img.shape[:2]
        small = cv2.resize(img, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    elif level == "low":
        # 25% 다운샘플 + JPEG 압축 (품질 30)
        h, w = img.shape[:2]
        small = cv2.resize(img, (w // 4, h // 4), interpolation=cv2.INTER_AREA)
        upsampled = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
        # JPEG 압축
        _, enc = cv2.imencode(".jpg", upsampled, [cv2.IMWRITE_JPEG_QUALITY, 30])
        decoded = cv2.imdecode(enc, cv2.IMREAD_COLOR)
        return decoded
    else:
        raise ValueError(f"Unknown quality level: {level}")


# ─────────────────────────────────────────────
# 데이터셋 → 이미지 + GT 라벨
# ─────────────────────────────────────────────

DATASETS = [
    ("structural", "datasets/structural/images/test", "datasets/structural/labels/test"),
    ("surface", "datasets/surface/images/test", "datasets/surface/labels/test"),
    ("floor_window", "datasets/floor_window/images/test", "datasets/floor_window/labels/test"),
    ("frames", "datasets/frames/images/test", "datasets/frames/labels/test"),
    ("m4_context", "datasets/m4_context/images/test", "datasets/m4_context/labels/test"),
]


def collect_random_samples(
    cwd: Path, n_per_dataset: int, seed: int = 42
) -> List[Tuple[str, Path, Path]]:
    """각 데이터셋에서 N장 무작위 샘플."""
    rng = random.Random(seed)
    samples: List[Tuple[str, Path, Path]] = []
    for name, img_rel, lbl_rel in DATASETS:
        img_dir = cwd / img_rel
        lbl_dir = cwd / lbl_rel
        if not img_dir.exists():
            continue
        all_imgs = sorted([
            f for f in img_dir.iterdir()
            if f.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ])
        if not all_imgs:
            continue
        chosen = rng.sample(all_imgs, min(n_per_dataset, len(all_imgs)))
        for img in chosen:
            lbl = lbl_dir / (img.stem + ".txt")
            samples.append((name, img, lbl))
    return samples


# ─────────────────────────────────────────────
# 통합 파이프라인 검증
# ─────────────────────────────────────────────

def _class_agnostic_nms(
    boxes: List[List[float]], confs: List[float], iou_thr: float = 0.5,
) -> List[int]:
    """
    Class-agnostic NMS — 클래스 무관 spatial 중복 제거.
    같은 영역에 여러 모델이 다른 클래스로 검출 시 최고 conf 1개만 유지.

    Returns: kept indices (in confidence-descending order).
    """
    if not boxes:
        return []
    order = sorted(range(len(boxes)), key=lambda i: confs[i], reverse=True)
    kept: List[int] = []
    for i in order:
        is_dup = False
        for k in kept:
            if iou(boxes[i], boxes[k]) >= iou_thr:
                is_dup = True
                break
        if not is_dup:
            kept.append(i)
    return kept


def validate_with_pipeline(
    samples: List[Tuple[str, Path, Path]],
    quality: str,
    iou_threshold: float = 0.5,
    apply_cross_class_nms: bool = True,
    cross_class_nms_iou: float = 0.4,    # 더 적극적 중복 제거 (precision↑)
) -> QualityReport:
    """
    통합 파이프라인으로 검증.
    cwd가 backend/이어야 settings.AEROINSPECT_WEIGHTS_DIR 정상 동작.

    apply_cross_class_nms=True면 파이프라인 출력에 class-agnostic NMS 적용
    (멀티 모델 같은 영역 중복 제거 — 사용자 체감 정확도 측정용).
    """
    from app.services.inference_pipeline_20 import InferencePipeline20

    pipe = InferencePipeline20()
    pipe.load_models()

    if not pipe.is_loaded:
        print("[ERROR] Pipeline not loaded — abort")
        return QualityReport(quality=quality)

    report = QualityReport(quality=quality)
    inference_times = []

    for ds_name, img_path, lbl_path in samples:
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        # 화질 저하
        img_q = degrade_quality(img, quality)
        H, W = img_q.shape[:2]

        # GT 로딩
        gts = load_yolo_labels(lbl_path)
        gt_count = len(gts)

        # Tier 2 추론 (M1+M2+M3+M4 Context+M5+gates 적용)
        t0 = time.time()
        try:
            result = pipe.detect(img_q, tier=2)
        except Exception as e:
            print(f"[ERROR] {img_path.name}: {e}")
            continue
        elapsed_ms = (time.time() - t0) * 1000
        inference_times.append(elapsed_ms)

        # 예측 박스 + conf
        all_boxes: List[List[float]] = [list(d.bbox_xyxy) for d in result.detections]
        all_confs: List[float] = [d.conf for d in result.detections]

        # Class-agnostic NMS — 다중 모델 같은 영역 중복 제거
        if apply_cross_class_nms and all_boxes:
            kept_idx = _class_agnostic_nms(all_boxes, all_confs, cross_class_nms_iou)
            pred_boxes = [all_boxes[i] for i in kept_idx]
            pred_confs = [all_confs[i] for i in kept_idx]
        else:
            pred_boxes = all_boxes
            pred_confs = all_confs

        # IoU 매칭 (class_agnostic — 위치만 봄)
        gt_xyxy = [g.to_xyxy(W, H) for g in gts]
        gt_matched = [False] * len(gts)
        pred_matched = [False] * len(pred_boxes)

        # confidence sort
        pred_order = sorted(range(len(pred_boxes)), key=lambda i: pred_confs[i], reverse=True)

        for pi in pred_order:
            best_iou = 0.0
            best_gi = -1
            for gi in range(len(gt_xyxy)):
                if gt_matched[gi]:
                    continue
                io = iou(pred_boxes[pi], gt_xyxy[gi])
                if io > best_iou:
                    best_iou = io
                    best_gi = gi
            if best_iou >= iou_threshold and best_gi >= 0:
                pred_matched[pi] = True
                gt_matched[best_gi] = True

        tp = sum(pred_matched)
        fp = len(pred_matched) - tp
        fn = sum(1 for m in gt_matched if not m)

        iv = ImageValidation(
            img_path=str(img_path), quality=quality,
            gt_count=gt_count, pred_count=len(pred_boxes),
            tp=tp, fp=fp, fn=fn, inference_ms=elapsed_ms,
        )
        report.per_image.append(iv)
        report.total_tp += tp
        report.total_fp += fp
        report.total_fn += fn
        report.total_gt += gt_count
        report.total_pred += len(pred_boxes)
        report.n_images += 1
        if gt_count > 0:
            report.n_with_gt += 1
            if len(pred_boxes) > 0:
                report.n_detected += 1
            if tp > 0:
                report.n_correct += 1

    if inference_times:
        report.avg_inference_ms = sum(inference_times) / len(inference_times)
    return report


def render_report_md(reports: List[QualityReport]) -> str:
    lines = []
    lines.append("# 무작위 샘플 + 화질별 검증 리포트\n")
    lines.append(f"생성: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    lines.append("## 화질별 종합\n")
    lines.append("| 화질 | 샘플 | GT 있는 이미지 | 검출 비율 | 정확 검출 비율 | precision | recall | F1 | 추론 평균 |")
    lines.append("|------|------|---------------|----------|---------------|-----------|--------|-----|----------|")
    for r in reports:
        lines.append(
            f"| {r.quality} | {r.n_images} | {r.n_with_gt} | "
            f"{r.detection_rate:.3f} | {r.accuracy_rate:.3f} | "
            f"{r.precision:.3f} | {r.recall:.3f} | {r.f1:.3f} | "
            f"{r.avg_inference_ms:.0f}ms |"
        )

    lines.append("\n## 화질별 상세\n")
    for r in reports:
        lines.append(f"### {r.quality}")
        lines.append(f"- 총 샘플: {r.n_images}")
        lines.append(f"- GT 있는 이미지: {r.n_with_gt}")
        lines.append(f"- 1개 이상 검출 발생: {r.n_detected} ({r.detection_rate*100:.1f}%)")
        lines.append(f"- 정확 매칭(IoU≥0.5) 발생: {r.n_correct} ({r.accuracy_rate*100:.1f}%)")
        lines.append(f"- 전체 TP/FP/FN: {r.total_tp}/{r.total_fp}/{r.total_fn}")
        lines.append(f"- 평균 추론 시간: {r.avg_inference_ms:.0f}ms")
        lines.append("")
    return "\n".join(lines)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-per-dataset", type=int, default=30, help="데이터셋당 샘플 수")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--qualities", nargs="+", default=["original", "medium", "low"])
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--out-dir", type=str, default="eval/results")
    args = parser.parse_args()

    cwd = Path.cwd()
    backend_dir = ROOT / "backend"
    out_dir = cwd / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # 무작위 샘플 (cwd=backend/training 기준)
    samples = collect_random_samples(cwd, args.n_per_dataset, args.seed)
    print(f"무작위 샘플: {len(samples)}장 (각 데이터셋 최대 {args.n_per_dataset}장)")
    if not samples:
        print("[ERROR] 샘플 없음")
        return 1

    # Pipeline은 cwd=backend/에서 동작해야 함
    orig_cwd = os.getcwd()
    reports: List[QualityReport] = []
    try:
        os.chdir(backend_dir)
        for q in args.qualities:
            print(f"\n=== 화질: {q} ===")
            t0 = time.time()
            report = validate_with_pipeline(samples, q, args.iou)
            elapsed = time.time() - t0
            print(f"  완료 ({elapsed/60:.1f}min)")
            print(f"  검출 비율: {report.detection_rate:.3f}")
            print(f"  정확 비율: {report.accuracy_rate:.3f}")
            print(f"  precision/recall/F1: {report.precision:.3f}/{report.recall:.3f}/{report.f1:.3f}")
            reports.append(report)
    finally:
        os.chdir(orig_cwd)

    # 결과 저장
    ts = time.strftime("%Y%m%d_%H%M%S")
    json_path = out_dir / f"validate_random_{ts}.json"
    md_path = out_dir / f"validate_random_{ts}.md"

    json_data = {
        "timestamp": ts,
        "n_per_dataset": args.n_per_dataset,
        "seed": args.seed,
        "iou_threshold": args.iou,
        "n_samples": len(samples),
        "reports": [
            {
                "quality": r.quality,
                "n_images": r.n_images,
                "n_with_gt": r.n_with_gt,
                "n_detected": r.n_detected,
                "n_correct": r.n_correct,
                "detection_rate": r.detection_rate,
                "accuracy_rate": r.accuracy_rate,
                "precision": r.precision,
                "recall": r.recall,
                "f1": r.f1,
                "total_tp": r.total_tp,
                "total_fp": r.total_fp,
                "total_fn": r.total_fn,
                "total_gt": r.total_gt,
                "total_pred": r.total_pred,
                "avg_inference_ms": r.avg_inference_ms,
            }
            for r in reports
        ],
    }
    json_path.write_text(json.dumps(json_data, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(render_report_md(reports), encoding="utf-8")

    print(f"\n결과 저장:")
    print(f"  JSON: {json_path}")
    print(f"  MD:   {md_path}")
    print()
    print(render_report_md(reports))
    return 0


if __name__ == "__main__":
    sys.exit(main())
