# =============================================
# app/services/ensemble.py
# 역할: 크로스 모델 NMS + PatchCore 앙상블 + 신뢰도 교정
#       - cross_model_nms: 서로 다른 모델 간 중복 검출 제거
#       - ensemble_with_patchcore: PatchCore로 YOLO/ResNet 저신뢰 검출 승격
#       - compute_combined_confidence: 독립 사건 결합 신뢰도
# =============================================

from __future__ import annotations

from typing import List, Optional

import numpy as np


def _iou(box_a: List[float], box_b: List[float]) -> float:
    """두 xyxy bbox의 IoU 계산."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    return inter / (area_a + area_b - inter + 1e-6)


def cross_model_nms(
    detections: List[dict],
    iou_threshold: float = 0.5,
) -> List[dict]:
    """
    서로 다른 모델의 검출 결과 간 중복 제거.

    규칙:
    1. 동일 class 중복 → 높은 confidence 유지
    2. 다른 class 겹침 → 둘 다 보고 (복합 하자 가능)
    """
    if len(detections) <= 1:
        return detections

    # 동일 class끼리 그룹핑
    by_class: dict = {}
    for det in detections:
        by_class.setdefault(det["class"], []).append(det)

    result: List[dict] = []
    for cls, dets in by_class.items():
        if len(dets) <= 1:
            result.extend(dets)
            continue

        # 같은 class 내에서 NMS
        sorted_dets = sorted(dets, key=lambda d: d["conf"], reverse=True)
        keep: List[dict] = []
        for det in sorted_dets:
            is_dup = False
            for kept in keep:
                if _iou(det["bbox_xyxy"], kept["bbox_xyxy"]) >= iou_threshold:
                    is_dup = True
                    break
            if not is_dup:
                keep.append(det)
        result.extend(keep)

    return result


def ensemble_with_patchcore(
    detections: List[dict],
    anomaly_mask: Optional[np.ndarray],
    anomaly_score: float,
    low_conf_threshold: float = 0.35,
) -> List[dict]:
    """
    PatchCore 이상 탐지와 교차 검증.

    저신뢰 검출(conf < low_conf_threshold)이지만 PatchCore가
    동일 영역을 이상으로 판정 → confidence 승격.
    """
    if anomaly_mask is None:
        return detections

    h, w = anomaly_mask.shape[:2]

    for det in detections:
        if det["conf"] >= low_conf_threshold:
            continue

        x1, y1, x2, y2 = [int(v) for v in det["bbox_xyxy"]]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 <= x1 or y2 <= y1:
            continue

        roi_score = anomaly_mask[y1:y2, x1:x2].mean() / 255.0
        if roi_score > 0.5:
            # 독립 사건 결합
            combined = 1.0 - (1.0 - det["conf"]) * (1.0 - roi_score)
            det["conf"] = min(1.0, combined)
            det["ensemble_boosted"] = True

    return detections


def compute_combined_confidence(conf1: float, conf2: float) -> float:
    """두 독립 검출의 결합 신뢰도."""
    return 1.0 - (1.0 - conf1) * (1.0 - conf2)


def cross_model_spatial_boost(
    detections: List[dict],
    iou_threshold: float = 0.4,     # 0.25 → 0.4 (실제 같은 위치만 boost)
    boost_factor: float = 0.15,     # 0.50 → 0.15 (소폭 boost — 측정 결과 0.50은 mAP 저하)
    min_conf: float = 0.3,          # 신규: baseline conf 미달 시 boost X (noise FP 보호)
) -> List[dict]:
    """
    서로 다른 모델(defect_source)이 동일 위치를 탐지했을 때 신뢰도 부스팅.

    측정 기반 조정 (2026-05-04 multi_model_voting eval):
    - 이전 boost_factor=0.50, iou=0.25 조합은 M3 -0.045, M2 -0.058 mAP 저하
    - 원인: 낮은 IoU + 강한 boost → false positive conf 증폭
    - 해결: iou_threshold 0.4 (실제 같은 위치만), boost 0.15 (소폭),
            baseline conf < min_conf면 boost X (noise 보호)

    규칙:
    - 서로 다른 defect_source의 검출이 IoU >= iou_threshold로 겹침
    - 두 검출 모두 conf >= min_conf 일 때만 boost (noise 증폭 방지)
    - 둘 다 conf를 boost_factor만큼 승격 (최대 1.0)
    - 동일 source 간은 이미 cross_model_nms로 처리하므로 스킵
    """
    if len(detections) <= 1:
        return detections

    boosted = set()

    for i in range(len(detections)):
        for j in range(i + 1, len(detections)):
            src_i = detections[i].get("defect_source", "")
            src_j = detections[j].get("defect_source", "")

            if src_i == src_j:
                continue

            bbox_i = detections[i].get("bbox_xyxy")
            bbox_j = detections[j].get("bbox_xyxy")
            if bbox_i is None or bbox_j is None:
                continue

            # noise FP 보호: 둘 중 하나라도 baseline conf 미달 시 boost X
            conf_i = detections[i].get("conf", 0.0)
            conf_j = detections[j].get("conf", 0.0)
            if conf_i < min_conf or conf_j < min_conf:
                continue

            if _iou(bbox_i, bbox_j) >= iou_threshold:
                if i not in boosted:
                    detections[i]["conf"] = min(
                        1.0, detections[i]["conf"] + boost_factor,
                    )
                    detections[i]["cross_model_boosted"] = True
                    boosted.add(i)
                if j not in boosted:
                    detections[j]["conf"] = min(
                        1.0, detections[j]["conf"] + boost_factor,
                    )
                    detections[j]["cross_model_boosted"] = True
                    boosted.add(j)

    return detections
