"""
tests/test_floorplan_accuracy.py
역할: 평면도 가구·벽 검출 정확도 정량 측정 (precision / recall / IoU).
      합성 평면도 N개를 ground truth 와 함께 생성하고, 검출 결과를 매칭.

실행: pytest tests/test_floorplan_accuracy.py -v -s
또는: python tests/test_floorplan_accuracy.py
"""
from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass

import cv2
import numpy as np
import pytest

from app.services.floorplan_processor import extract_walls_from_bytes


# ────────────────────────────────────────────
# 합성 평면도 + ground truth 생성기
# ────────────────────────────────────────────

@dataclass
class GTFurniture:
    """Ground truth 가구 — 정규화 0-1 bbox."""
    cx: float; cy: float; w: float; h: float
    label: str

    def bbox_xyxy(self):
        return (
            self.cx - self.w / 2, self.cy - self.h / 2,
            self.cx + self.w / 2, self.cy + self.h / 2,
        )


def _draw_rect_gt(img, x1, y1, x2, y2, color):
    cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
    return GTFurniture(
        cx=(x1 + x2) / 2 / img.shape[1],
        cy=(y1 + y2) / 2 / img.shape[0],
        w=abs(x2 - x1) / img.shape[1],
        h=abs(y2 - y1) / img.shape[0],
        label='rectangular',
    )


def _draw_circle_gt(img, cx, cy, r, color):
    cv2.circle(img, (cx, cy), r, color, -1)
    return GTFurniture(
        cx=cx / img.shape[1], cy=cy / img.shape[0],
        w=(2 * r) / img.shape[1], h=(2 * r) / img.shape[0],
        label='circular',
    )


def make_case_simple():
    """단순 — 가구 4종 (침대·소파·식탁·책상)."""
    W, H = 1600, 1200
    img = np.full((H, W, 3), 252, dtype=np.uint8)
    BLACK = (15, 15, 15); GRAY = (90, 90, 90); THK = 14
    cv2.rectangle(img, (100, 100), (1500, 1100), BLACK, THK)
    cv2.line(img, (800, 100), (800, 1100), BLACK, THK)
    gt = []
    gt.append(_draw_rect_gt(img, 200, 200, 600, 500, GRAY))     # 침대
    gt.append(_draw_rect_gt(img, 900, 200, 1400, 350, GRAY))    # 소파
    gt.append(_draw_circle_gt(img, 1100, 700, 120, GRAY))       # 식탁
    gt.append(_draw_rect_gt(img, 200, 850, 600, 1000, GRAY))    # 책상
    return img, gt, 'simple'


def make_case_dense():
    """빽빽 — 가구 8종, 인접해서 분리 난이도 ↑."""
    W, H = 1600, 1200
    img = np.full((H, W, 3), 252, dtype=np.uint8)
    BLACK = (15, 15, 15); GRAY = (90, 90, 90); THK = 14
    cv2.rectangle(img, (100, 100), (1500, 1100), BLACK, THK)
    gt = []
    # 6개의 의자 4 (식탁 주변)
    for cx, cy in [(700, 400), (900, 400), (700, 600), (900, 600), (500, 500), (1100, 500)]:
        gt.append(_draw_circle_gt(img, cx, cy, 35, GRAY))
    gt.append(_draw_rect_gt(img, 200, 800, 600, 1050, GRAY))     # 침대
    gt.append(_draw_rect_gt(img, 1000, 800, 1400, 1000, GRAY))   # 소파
    return img, gt, 'dense'


def make_case_noisy():
    """노이즈 + 텍스트 라벨 — 실제 스캔본 모방."""
    W, H = 1400, 1000
    img = np.full((H, W, 3), 250, dtype=np.uint8)
    BLACK = (15, 15, 15); GRAY = (95, 95, 95); THK = 12
    cv2.rectangle(img, (80, 80), (W - 80, H - 80), BLACK, THK)
    cv2.line(img, (700, 80), (700, H - 80), BLACK, THK)
    gt = []
    gt.append(_draw_rect_gt(img, 150, 150, 500, 400, GRAY))
    gt.append(_draw_circle_gt(img, 950, 300, 100, GRAY))
    gt.append(_draw_rect_gt(img, 800, 600, 1200, 750, GRAY))
    # 노이즈 + 텍스트
    gauss = np.random.normal(0, 5, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + gauss, 0, 255).astype(np.uint8)
    cv2.putText(img, 'BEDROOM', (200, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 60, 60), 2)
    cv2.putText(img, '15.4 sqm', (200, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 80), 1)
    return img, gt, 'noisy'


def make_case_dark_furniture():
    """진한 회색 가구 — 다중 threshold 효과 검증."""
    W, H = 1500, 1000
    img = np.full((H, W, 3), 250, dtype=np.uint8)
    BLACK = (15, 15, 15); DARK = (60, 60, 60); THK = 14
    cv2.rectangle(img, (100, 100), (W - 100, H - 100), BLACK, THK)
    gt = []
    gt.append(_draw_rect_gt(img, 200, 200, 600, 450, DARK))     # 진한 침대
    gt.append(_draw_circle_gt(img, 1100, 500, 120, DARK))       # 진한 식탁
    gt.append(_draw_rect_gt(img, 800, 700, 1300, 850, DARK))    # 진한 책상
    return img, gt, 'dark'


# ────────────────────────────────────────────
# 매칭 + 메트릭
# ────────────────────────────────────────────

def _bbox_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def evaluate(detections, gts, iou_threshold=0.3):
    """
    Hungarian 없이 greedy 매칭 (간이) — IoU 가장 높은 것부터 매칭.

    Returns:
        dict { tp, fp, fn, precision, recall, f1, mean_iou }
    """
    det_boxes = [(d['cx'] - d['w']/2, d['cy'] - d['h']/2,
                  d['cx'] + d['w']/2, d['cy'] + d['h']/2) for d in detections]
    gt_boxes = [g.bbox_xyxy() for g in gts]

    matched_det = set()
    matched_gt = set()
    matched_iou = []

    # 모든 (det, gt) 페어의 IoU 계산
    pairs = []
    for di, db in enumerate(det_boxes):
        for gi, gb in enumerate(gt_boxes):
            iou = _bbox_iou(db, gb)
            if iou >= iou_threshold:
                pairs.append((iou, di, gi))
    pairs.sort(reverse=True)

    for iou, di, gi in pairs:
        if di in matched_det or gi in matched_gt:
            continue
        matched_det.add(di)
        matched_gt.add(gi)
        matched_iou.append(iou)

    tp = len(matched_iou)
    fp = len(det_boxes) - tp
    fn = len(gt_boxes) - tp
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    mean_iou = sum(matched_iou) / len(matched_iou) if matched_iou else 0.0

    return {
        'tp': tp, 'fp': fp, 'fn': fn,
        'precision': round(precision, 3),
        'recall': round(recall, 3),
        'f1': round(f1, 3),
        'mean_iou': round(mean_iou, 3),
    }


# ────────────────────────────────────────────
# Pytest 테스트
# ────────────────────────────────────────────

CASES = [make_case_simple, make_case_dense, make_case_noisy, make_case_dark_furniture]


@pytest.mark.parametrize("case_fn", CASES)
def test_furniture_detection_recall(case_fn):
    img, gts, name = case_fn()
    path = os.path.join(tempfile.gettempdir(), f'acc_{name}.png')
    cv2.imwrite(path, img)
    with open(path, 'rb') as f:
        b = f.read()

    result = extract_walls_from_bytes(b)
    metrics = evaluate(result['furniture'], gts, iou_threshold=0.3)

    print(f"\n[{name}] GT={len(gts)}, detected={len(result['furniture'])}, "
          f"precision={metrics['precision']}, recall={metrics['recall']}, "
          f"F1={metrics['f1']}, mean_IoU={metrics['mean_iou']}")

    # 회피 안전 입장 — recall 우선 (놓치는 가구가 적어야 함)
    # 안전 임계값: dense/noisy 는 recall ≥ 0.5, simple/dark 는 ≥ 0.7
    threshold = 0.5 if name in ('dense', 'noisy') else 0.7
    assert metrics['recall'] >= threshold, \
        f"{name} recall {metrics['recall']} < {threshold} (충돌 회피 안전 임계 미달)"


def test_aggregate_summary(capsys):
    """모든 케이스 종합 요약 — 콘솔에만 출력."""
    print('\n' + '=' * 72)
    print('  가구 검출 정확도 종합 (IoU threshold = 0.3, recall 우선)')
    print('=' * 72)
    print(f"  {'CASE':<12} {'GT':>4} {'DET':>4} {'TP':>4} {'FP':>4} {'FN':>4}  "
          f"{'PRECISION':>9}  {'RECALL':>7}  {'F1':>5}  {'mIoU':>5}")
    print('  ' + '-' * 70)

    aggregate = {'tp': 0, 'fp': 0, 'fn': 0, 'iou_sum': 0, 'iou_n': 0}
    for case_fn in CASES:
        img, gts, name = case_fn()
        path = os.path.join(tempfile.gettempdir(), f'agg_{name}.png')
        cv2.imwrite(path, img)
        with open(path, 'rb') as f:
            b = f.read()
        result = extract_walls_from_bytes(b)
        m = evaluate(result['furniture'], gts, iou_threshold=0.3)
        print(f"  {name:<12} {len(gts):>4} {len(result['furniture']):>4} "
              f"{m['tp']:>4} {m['fp']:>4} {m['fn']:>4}  "
              f"{m['precision']:>9}  {m['recall']:>7}  {m['f1']:>5}  {m['mean_iou']:>5}")
        aggregate['tp'] += m['tp']
        aggregate['fp'] += m['fp']
        aggregate['fn'] += m['fn']
        if m['mean_iou'] > 0:
            aggregate['iou_sum'] += m['mean_iou']
            aggregate['iou_n'] += 1

    tp = aggregate['tp']; fp = aggregate['fp']; fn = aggregate['fn']
    macro_p = tp / (tp + fp) if (tp + fp) > 0 else 0
    macro_r = tp / (tp + fn) if (tp + fn) > 0 else 0
    macro_f = 2 * macro_p * macro_r / (macro_p + macro_r) if (macro_p + macro_r) > 0 else 0
    macro_iou = aggregate['iou_sum'] / aggregate['iou_n'] if aggregate['iou_n'] else 0
    print('  ' + '-' * 70)
    print(f"  {'TOTAL':<12} {'':>4} {'':>4} {tp:>4} {fp:>4} {fn:>4}  "
          f"{macro_p:>9.3f}  {macro_r:>7.3f}  {macro_f:>5.3f}  {macro_iou:>5.3f}")
    print()


if __name__ == '__main__':
    # pytest 외 직접 실행
    print('=== 정량 정확도 측정 ===\n')
    for case_fn in CASES:
        img, gts, name = case_fn()
        path = os.path.join(tempfile.gettempdir(), f'direct_{name}.png')
        cv2.imwrite(path, img)
        with open(path, 'rb') as f:
            b = f.read()
        result = extract_walls_from_bytes(b)
        m = evaluate(result['furniture'], gts, iou_threshold=0.3)
        print(f"[{name}]  GT={len(gts)}  DET={len(result['furniture'])}  "
              f"P={m['precision']}  R={m['recall']}  F1={m['f1']}  IoU={m['mean_iou']}  "
              f"(TP={m['tp']} FP={m['fp']} FN={m['fn']})")
