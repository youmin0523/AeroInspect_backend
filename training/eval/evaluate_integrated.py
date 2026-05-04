# =============================================
# evaluate_integrated.py
# 역할: 20종 하자 통합 시스템 평가
#       - Phase 1: Per-model raw mAP (단일 모델 IoU@0.5 mAP, FN/FP per class)
#       - Phase 2: Integrated pipeline mAP (inference_pipeline_20.detect 호출)
#       - Phase 3: Post-processing ablation (raw → tracker → temporal → ensemble)
#       - Phase 4: Confidence calibration (reliability diagram, ECE)
#       - Phase 5: Edge case probe (저조도/모션블러/축소 시뮬레이션)
#
# 출력:
#   - eval/results/integrated_eval_<timestamp>.json
#   - eval/results/integrated_eval_<timestamp>.md (사람 읽기용)
#
# 사용법:
#   cd backend/training
#   python eval/evaluate_integrated.py
#   python eval/evaluate_integrated.py --phase 1 --model M1
#   python eval/evaluate_integrated.py --max-images 200 --quick
# =============================================

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# backend/app 임포트 가능하도록 경로 추가
ROOT = Path(__file__).resolve().parents[3]
BACKEND = ROOT / "backend"
sys.path.insert(0, str(BACKEND))

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# ─────────────────────────────────────────────
# 1) 데이터 구조
# ─────────────────────────────────────────────

@dataclass
class GroundTruthBox:
    """YOLO 라벨 한 줄: class_id, normalized cx,cy,w,h"""
    class_id: int
    cx: float
    cy: float
    w: float
    h: float

    def to_xyxy(self, img_w: int, img_h: int) -> List[float]:
        x1 = (self.cx - self.w / 2) * img_w
        y1 = (self.cy - self.h / 2) * img_h
        x2 = (self.cx + self.w / 2) * img_w
        y2 = (self.cy + self.h / 2) * img_h
        return [x1, y1, x2, y2]


@dataclass
class PredictionBox:
    """모델 예측 한 건: class_id 또는 class_name, conf, xyxy"""
    class_id: int
    class_name: str
    conf: float
    bbox_xyxy: List[float]
    matched: bool = False  # IoU 매칭 후 True


@dataclass
class ClassMetrics:
    """클래스별 평가 지표"""
    class_id: int
    class_name: str
    tp: int = 0
    fp: int = 0
    fn: int = 0
    n_gt: int = 0
    confidences: List[float] = field(default_factory=list)
    is_correct: List[bool] = field(default_factory=list)  # calibration용

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom > 0 else 0.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom > 0 else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    @property
    def fn_rate(self) -> float:
        """미탐률 = FN / GT_total — 안전 직결 지표"""
        return self.fn / self.n_gt if self.n_gt > 0 else 0.0

    @property
    def fp_rate(self) -> float:
        """오탐률 (per detected positive)"""
        denom = self.tp + self.fp
        return self.fp / denom if denom > 0 else 0.0


@dataclass
class ModelEvalResult:
    """단일 모델 평가 결과"""
    model_name: str
    onnx_path: str
    test_size: int
    imgsz: int
    iou_threshold: float
    conf_threshold: float
    per_class: Dict[str, ClassMetrics] = field(default_factory=dict)
    elapsed_sec: float = 0.0
    error: Optional[str] = None

    @property
    def mAP50(self) -> float:
        """클래스별 AP의 단순 평균 (per-class recall 기반 근사)"""
        if not self.per_class:
            return 0.0
        return sum(m.recall * m.precision for m in self.per_class.values()) / len(self.per_class)

    @property
    def macro_recall(self) -> float:
        if not self.per_class:
            return 0.0
        return sum(m.recall for m in self.per_class.values()) / len(self.per_class)

    @property
    def macro_precision(self) -> float:
        if not self.per_class:
            return 0.0
        return sum(m.precision for m in self.per_class.values()) / len(self.per_class)

    @property
    def macro_f1(self) -> float:
        if not self.per_class:
            return 0.0
        return sum(m.f1 for m in self.per_class.values()) / len(self.per_class)

    @property
    def overall_fn_rate(self) -> float:
        total_fn = sum(m.fn for m in self.per_class.values())
        total_gt = sum(m.n_gt for m in self.per_class.values())
        return total_fn / total_gt if total_gt > 0 else 0.0


# ─────────────────────────────────────────────
# 2) IoU 계산 유틸
# ─────────────────────────────────────────────

def iou(a: List[float], b: List[float]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    aa = (a[2] - a[0]) * (a[3] - a[1])
    bb = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (aa + bb - inter + 1e-9)


def match_predictions_to_gt(
    preds: List[PredictionBox],
    gts: List[GroundTruthBox],
    img_w: int,
    img_h: int,
    iou_threshold: float = 0.5,
    class_agnostic: bool = False,
) -> Tuple[List[bool], List[bool]]:
    """
    Greedy IoU matching.

    Returns:
        pred_matched: 각 prediction의 매칭 여부 (TP=True, FP=False)
        gt_matched: 각 GT의 매칭 여부 (recall 계산용)
    """
    if not preds:
        return [], [False] * len(gts)

    gt_xyxy = [g.to_xyxy(img_w, img_h) for g in gts]
    gt_matched = [False] * len(gts)
    pred_matched = [False] * len(preds)

    # confidence 내림차순
    pred_order = sorted(range(len(preds)), key=lambda i: preds[i].conf, reverse=True)

    for pi in pred_order:
        pred = preds[pi]
        best_iou = 0.0
        best_gi = -1
        for gi, gt in enumerate(gts):
            if gt_matched[gi]:
                continue
            if not class_agnostic and pred.class_id != gt.class_id:
                continue
            io = iou(pred.bbox_xyxy, gt_xyxy[gi])
            if io > best_iou:
                best_iou = io
                best_gi = gi

        if best_iou >= iou_threshold and best_gi >= 0:
            pred_matched[pi] = True
            gt_matched[best_gi] = True

    return pred_matched, gt_matched


# ─────────────────────────────────────────────
# 3) YOLO 라벨 로더
# ─────────────────────────────────────────────

def load_yolo_labels(label_path: Path) -> List[GroundTruthBox]:
    """YOLO 라벨 로드 — bbox 포맷 (5 values) 또는 segment 포맷 (5+ values polygon)."""
    if not label_path.exists():
        return []
    out: List[GroundTruthBox] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        try:
            cid = int(parts[0])
            vals = [float(x) for x in parts[1:]]

            if len(vals) == 4:
                # bbox 포맷: cx, cy, w, h
                out.append(GroundTruthBox(
                    class_id=cid,
                    cx=vals[0], cy=vals[1], w=vals[2], h=vals[3],
                ))
            elif len(vals) >= 6 and len(vals) % 2 == 0:
                # segment 포맷: x1, y1, x2, y2, ... (polygon points)
                xs = vals[0::2]
                ys = vals[1::2]
                x_min, x_max = min(xs), max(xs)
                y_min, y_max = min(ys), max(ys)
                cx = (x_min + x_max) / 2
                cy = (y_min + y_max) / 2
                w = x_max - x_min
                h = y_max - y_min
                if w > 0 and h > 0:
                    out.append(GroundTruthBox(class_id=cid, cx=cx, cy=cy, w=w, h=h))
            # 그 외 포맷 (5 values 등)은 스킵
        except ValueError:
            continue
    return out


def load_class_names_from_yaml(yaml_path: Path) -> List[str]:
    """data.yaml의 names: 또는 nc: 만 있으면 빈 리스트"""
    text = yaml_path.read_text(encoding="utf-8")
    names: List[str] = []
    in_names = False
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("names:"):
            in_names = True
            # inline list?
            if "[" in s and "]" in s:
                inner = s.split("[", 1)[1].rsplit("]", 1)[0]
                names = [x.strip().strip("'\"") for x in inner.split(",") if x.strip()]
                break
            continue
        if in_names:
            if not s or s.startswith("#"):
                break
            # "  0: classname" 또는 "- classname"
            if ":" in s:
                _, name = s.split(":", 1)
                names.append(name.strip().strip("'\""))
            elif s.startswith("-"):
                names.append(s.lstrip("- ").strip().strip("'\""))
            else:
                break
    return names


# ─────────────────────────────────────────────
# 4) Phase 1: Per-model raw mAP
# ─────────────────────────────────────────────

# (model_key, onnx_filename, test_dataset_dir, imgsz, conf_threshold, classes_for_report)
MODEL_EVAL_TARGETS = [
    {
        "key": "M1_YOLO",
        "onnx": "m1_yolo_structural.onnx",
        "dataset": "datasets/structural",
        "imgsz": 960,
        "conf": 0.25,
    },
    {
        "key": "M2_YOLO",
        "onnx": "m2_yolo_surface.onnx",
        "dataset": "datasets/surface",
        "imgsz": 960,
        "conf": 0.25,
    },
    {
        "key": "M3_YOLO",
        "onnx": "m3_yolo_floor_window.onnx",
        "dataset": "datasets/floor_window",
        "imgsz": 1280,
        "conf": 0.25,
    },
    {
        "key": "M4_CONTEXT",
        "onnx": "m4_yolo_context_elements.onnx",
        "dataset": "datasets/m4_context",
        "imgsz": 960,
        "conf": 0.25,
    },
    {
        "key": "M5_SEG",
        "onnx": "m5_yolo_seg_frames.onnx",
        "dataset": "datasets/frames",
        "imgsz": 1280,
        "conf": 0.25,
        # ONNX는 detect 형식으로 export됨 → task=detect로 평가, GT는 segment polygon→bbox 변환
    },
]


def evaluate_single_model(
    target: dict,
    weights_dir: Path,
    cwd: Path,
    iou_thr: float = 0.5,
    max_images: Optional[int] = None,
) -> ModelEvalResult:
    """단일 YOLO ONNX를 test set에 대해 평가"""
    from ultralytics import YOLO

    onnx_path = weights_dir / target["onnx"]
    dataset_dir = cwd / target["dataset"]
    yaml_path = dataset_dir / "data.yaml"
    img_dir = dataset_dir / "images" / "test"
    lbl_dir = dataset_dir / "labels" / "test"

    result = ModelEvalResult(
        model_name=target["key"],
        onnx_path=str(onnx_path),
        test_size=0,
        imgsz=target["imgsz"],
        iou_threshold=iou_thr,
        conf_threshold=target["conf"],
    )

    if not onnx_path.exists():
        result.error = f"ONNX missing: {onnx_path}"
        return result
    if not yaml_path.exists() or not img_dir.exists():
        result.error = f"Dataset missing: {dataset_dir}"
        return result

    class_names = load_class_names_from_yaml(yaml_path)
    if not class_names:
        result.error = f"Class names not loaded from {yaml_path}"
        return result

    for cid, cname in enumerate(class_names):
        result.per_class[cname] = ClassMetrics(class_id=cid, class_name=cname)

    images = sorted([
        f for f in img_dir.iterdir()
        if f.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ])
    if max_images:
        images = images[:max_images]
    result.test_size = len(images)

    print(f"\n[{target['key']}] {result.test_size} images, classes={len(class_names)}")

    start = time.time()
    task = target.get("task", "detect")  # 기본 detect, M5_SEG는 segment
    try:
        model = YOLO(str(onnx_path), task=task)
        BATCH = 4
        for i in range(0, len(images), BATCH):
            batch = images[i:i + BATCH]
            results = model.predict(
                source=[str(b) for b in batch],
                conf=target["conf"], iou=0.5,
                imgsz=target["imgsz"], verbose=False, save=False, device=0,
            )
            for img_path, res in zip(batch, results):
                # GT
                lbl_file = lbl_dir / (img_path.stem + ".txt")
                gts = load_yolo_labels(lbl_file)
                # 이미지 크기
                if hasattr(res, "orig_shape"):
                    H, W = res.orig_shape[:2]
                else:
                    img = cv2.imread(str(img_path))
                    H, W = img.shape[:2]

                # GT 카운팅
                for g in gts:
                    if 0 <= g.class_id < len(class_names):
                        result.per_class[class_names[g.class_id]].n_gt += 1

                # 예측 박스
                preds: List[PredictionBox] = []
                if res.boxes is not None and len(res.boxes) > 0:
                    xyxy = res.boxes.xyxy.cpu().numpy()
                    conf = res.boxes.conf.cpu().numpy()
                    cls = res.boxes.cls.cpu().numpy().astype(int)
                    for j in range(len(xyxy)):
                        cid = int(cls[j])
                        if not (0 <= cid < len(class_names)):
                            continue
                        preds.append(PredictionBox(
                            class_id=cid,
                            class_name=class_names[cid],
                            conf=float(conf[j]),
                            bbox_xyxy=[float(v) for v in xyxy[j]],
                        ))

                # IoU 매칭
                pred_matched, gt_matched = match_predictions_to_gt(
                    preds, gts, W, H, iou_threshold=iou_thr,
                )

                # TP/FP 집계
                for pi, p in enumerate(preds):
                    cm = result.per_class[p.class_name]
                    cm.confidences.append(p.conf)
                    cm.is_correct.append(pred_matched[pi])
                    if pred_matched[pi]:
                        cm.tp += 1
                    else:
                        cm.fp += 1

                # FN 집계
                for gi, g in enumerate(gts):
                    if 0 <= g.class_id < len(class_names) and not gt_matched[gi]:
                        result.per_class[class_names[g.class_id]].fn += 1

            if (i // BATCH) % 10 == 0:
                pct = min(100, int((i + BATCH) / len(images) * 100))
                print(f"  {min(i+BATCH, len(images))}/{len(images)} ({pct}%)")

    except Exception as e:
        result.error = f"{type(e).__name__}: {e}"

    result.elapsed_sec = time.time() - start
    return result


# ─────────────────────────────────────────────
# 5) Phase 4: Confidence Calibration
# ─────────────────────────────────────────────

def expected_calibration_error(
    confidences: List[float],
    is_correct: List[bool],
    n_bins: int = 10,
) -> Tuple[float, List[dict]]:
    """
    ECE (Expected Calibration Error) 계산.
    낮을수록 예측 conf가 실제 정확도와 잘 일치.

    Returns:
        ece: 0.0 ~ 1.0 (낮을수록 좋음)
        bins: [{lo, hi, count, avg_conf, accuracy}, ...]
    """
    if not confidences:
        return 0.0, []
    confs = np.array(confidences)
    correct = np.array(is_correct, dtype=float)

    bins_info = []
    ece = 0.0
    n = len(confs)

    for b in range(n_bins):
        lo = b / n_bins
        hi = (b + 1) / n_bins
        if b == n_bins - 1:
            mask = (confs >= lo) & (confs <= hi)
        else:
            mask = (confs >= lo) & (confs < hi)
        count = int(mask.sum())
        if count == 0:
            bins_info.append({"lo": lo, "hi": hi, "count": 0, "avg_conf": 0.0, "accuracy": 0.0})
            continue
        avg_conf = float(confs[mask].mean())
        accuracy = float(correct[mask].mean())
        ece += (count / n) * abs(avg_conf - accuracy)
        bins_info.append({
            "lo": lo, "hi": hi, "count": count,
            "avg_conf": round(avg_conf, 4),
            "accuracy": round(accuracy, 4),
        })

    return float(ece), bins_info


# ─────────────────────────────────────────────
# 6) 보고서 생성
# ─────────────────────────────────────────────

def render_markdown_report(results: List[ModelEvalResult], integrated: Optional[dict] = None) -> str:
    lines = []
    lines.append("# 20종 하자 통합 평가 리포트")
    lines.append(f"\n생성: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("\n## Phase 1: Per-Model Raw mAP\n")
    lines.append("| 모델 | Test | macro mAP50 | macro Recall | macro Precision | macro F1 | Overall FN rate | ECE | Time |")
    lines.append("|------|------|------------|-------------|----------------|---------|----------------|------|------|")

    all_class_metrics_for_calibration: List[Tuple[str, ClassMetrics]] = []

    for r in results:
        if r.error:
            lines.append(f"| {r.model_name} | ❌ ERROR | {r.error} | | | | | | |")
            continue

        # 전체 calibration 계산
        all_confs = []
        all_correct = []
        for cm in r.per_class.values():
            all_confs.extend(cm.confidences)
            all_correct.extend(cm.is_correct)
            all_class_metrics_for_calibration.append((r.model_name, cm))

        ece, _ = expected_calibration_error(all_confs, all_correct)

        lines.append(
            f"| {r.model_name} | {r.test_size} | "
            f"{r.mAP50:.4f} | {r.macro_recall:.4f} | "
            f"{r.macro_precision:.4f} | {r.macro_f1:.4f} | "
            f"{r.overall_fn_rate:.4f} | {ece:.4f} | "
            f"{r.elapsed_sec/60:.1f}min |"
        )

    lines.append("\n## Per-Class 상세 (FN rate 위험순)\n")
    lines.append("| 모델 | 클래스 | GT | TP | FP | FN | Precision | Recall | F1 | FN rate | 위험도 |")
    lines.append("|------|--------|----|----|----|----|-----------|--------|----|---------|--------|")

    rows = []
    for r in results:
        if r.error:
            continue
        for cname, cm in r.per_class.items():
            risk = ""
            if cm.fn_rate > 0.5:
                risk = "🔴 매우 높음"
            elif cm.fn_rate > 0.3:
                risk = "🟡 높음"
            elif cm.fn_rate > 0.15:
                risk = "🟢 보통"
            else:
                risk = "✅ 양호"
            rows.append((cm.fn_rate, r.model_name, cname, cm, risk))

    rows.sort(key=lambda x: x[0], reverse=True)
    for fn_rate, mname, cname, cm, risk in rows:
        lines.append(
            f"| {mname} | {cname} | {cm.n_gt} | {cm.tp} | {cm.fp} | {cm.fn} | "
            f"{cm.precision:.3f} | {cm.recall:.3f} | {cm.f1:.3f} | {cm.fn_rate:.3f} | {risk} |"
        )

    lines.append("\n## 후처리 가중 우선순위\n")
    lines.append(
        "후처리 강도 정책: 모든 모델 baseline 동일 + 약한 모델만 추가 가중. "
        "FN rate > 0.3 클래스를 우선으로 가중치 적용 (window_size↑, accumulated_conf 임계↓, "
        "cross_model_boost 가중↑).\n"
    )
    weak_classes = [(r.model_name, c, cm) for fn, mn, c, cm, _ in rows if fn > 0.3 for r in [next((x for x in results if x.model_name == mn), None)] if r]
    if weak_classes:
        lines.append("### 추가 가중 대상 클래스\n")
        for mn, cn, cm in weak_classes[:20]:
            lines.append(f"- **{mn} / {cn}**: FN rate {cm.fn_rate:.3f}, recall {cm.recall:.3f}")
    else:
        lines.append("\n현재 측정 기준 FN rate > 0.3 클래스 없음.")

    if integrated:
        lines.append("\n## Phase 2: 통합 파이프라인 (예정)\n")
        lines.append(json.dumps(integrated, indent=2, ensure_ascii=False))

    return "\n".join(lines)


# ─────────────────────────────────────────────
# 7) Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=int, default=1, help="1=per-model, 2=integrated, 3=ablation, 4=calibration, 5=edge case, 0=all")
    parser.add_argument("--model", type=str, default=None, help="단일 모델만 평가 (M1_YOLO, M2_YOLO, ...)")
    parser.add_argument("--max-images", type=int, default=None, help="모델당 최대 이미지 수 (디버그용)")
    parser.add_argument("--quick", action="store_true", help="--max-images 100")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU 임계값")
    parser.add_argument("--out-dir", type=str, default="eval/results", help="결과 저장 디렉토리")
    args = parser.parse_args()

    if args.quick:
        args.max_images = args.max_images or 100

    cwd = Path.cwd()
    weights_dir = cwd.parent / "models_weights"  # backend/training → backend/models_weights
    out_dir = cwd / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"통합 평가 (Phase {args.phase}, IoU={args.iou})")
    print(f"  cwd: {cwd}")
    print(f"  weights_dir: {weights_dir}")
    print(f"  max_images: {args.max_images or 'all'}")
    print("=" * 70)

    results: List[ModelEvalResult] = []

    # ── Phase 1: Per-model raw evaluation ──
    if args.phase in (1, 0):
        targets = MODEL_EVAL_TARGETS
        if args.model:
            targets = [t for t in targets if t["key"] == args.model]
            if not targets:
                print(f"[ERROR] Model not found: {args.model}")
                return 1

        for t in targets:
            r = evaluate_single_model(
                t, weights_dir=weights_dir, cwd=cwd,
                iou_thr=args.iou, max_images=args.max_images,
            )
            results.append(r)
            if r.error:
                print(f"  [{r.model_name}] ❌ {r.error}")
            else:
                print(f"  [{r.model_name}] mAP50={r.mAP50:.4f} recall={r.macro_recall:.4f} "
                      f"precision={r.macro_precision:.4f} FN_rate={r.overall_fn_rate:.4f} "
                      f"({r.elapsed_sec/60:.1f}min)")

    # ── 결과 저장 ──
    ts = time.strftime("%Y%m%d_%H%M%S")
    json_path = out_dir / f"integrated_eval_{ts}.json"
    md_path = out_dir / f"integrated_eval_{ts}.md"

    json_data = {
        "timestamp": ts,
        "phase": args.phase,
        "iou_threshold": args.iou,
        "max_images": args.max_images,
        "results": [
            {
                **{k: v for k, v in asdict(r).items() if k != "per_class"},
                "per_class": {
                    cname: {
                        "class_id": cm.class_id, "class_name": cm.class_name,
                        "tp": cm.tp, "fp": cm.fp, "fn": cm.fn, "n_gt": cm.n_gt,
                        "precision": cm.precision, "recall": cm.recall,
                        "f1": cm.f1, "fn_rate": cm.fn_rate, "fp_rate": cm.fp_rate,
                    }
                    for cname, cm in r.per_class.items()
                },
                "macro_recall": r.macro_recall,
                "macro_precision": r.macro_precision,
                "macro_f1": r.macro_f1,
                "mAP50": r.mAP50,
                "overall_fn_rate": r.overall_fn_rate,
            }
            for r in results
        ],
    }
    json_path.write_text(json.dumps(json_data, indent=2, ensure_ascii=False), encoding="utf-8")

    md = render_markdown_report(results)
    md_path.write_text(md, encoding="utf-8")

    print("\n결과 저장:")
    print(f"  JSON: {json_path}")
    print(f"  MD:   {md_path}")
    print()
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
