# =============================================
# app/services/tta.py
# 역할: Test-Time Augmentation — 약한 모델 추론 강화
#       - 원본 + augmented 이미지로 N회 추론
#       - 좌표 역변환 (augmented → original)
#       - 결과 병합 (max_conf | wbf)
#
# 사용:
#   tta = TTAEnsemble(
#     augmentations=['horizontal_flip', 'scale_0_8', 'scale_1_2'],
#     merge_method='max_conf',
#   )
#   merged_dets = tta.predict(model, image_bgr, conf=0.25)
#
# 정책:
#   - 약한 모델(M4 Context 0.527, M5 Seg 0.626 등)에만 적용
#   - 강한 모델은 baseline 그대로 (추론 시간 증가)
#   - postprocess_config.yaml의 tta 섹션에서 설정 로드
# =============================================

from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np


# ─────────────────────────────────────────────
# Augmentation primitives + inverse coord transforms
# ─────────────────────────────────────────────

def _flip_horizontal(img: np.ndarray) -> Tuple[np.ndarray, dict]:
    """수평 반전. 좌표 역변환에 image width 필요."""
    flipped = cv2.flip(img, 1)
    return flipped, {"type": "hflip", "img_w": img.shape[1]}


def _flip_horizontal_inverse_bbox(bbox: List[float], meta: dict) -> List[float]:
    """xyxy 좌표를 hflip 역변환."""
    w = meta["img_w"]
    x1, y1, x2, y2 = bbox
    return [w - x2, y1, w - x1, y2]


def _scale(img: np.ndarray, factor: float) -> Tuple[np.ndarray, dict]:
    """선형 스케일링. 좌표 역변환에 factor 필요."""
    h, w = img.shape[:2]
    new_h, new_w = int(h * factor), int(w * factor)
    scaled = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return scaled, {"type": "scale", "factor": factor}


def _scale_inverse_bbox(bbox: List[float], meta: dict) -> List[float]:
    """스케일링 역변환. 좌표를 1/factor만큼 곱해서 원본 좌표로."""
    f = meta["factor"]
    x1, y1, x2, y2 = bbox
    return [x1 / f, y1 / f, x2 / f, y2 / f]


# augmentation 이름 → (augment 함수, inverse 함수)
AUGMENTATIONS = {
    "horizontal_flip": (_flip_horizontal, _flip_horizontal_inverse_bbox),
    "scale_0_8": (lambda img: _scale(img, 0.8), _scale_inverse_bbox),
    "scale_1_2": (lambda img: _scale(img, 1.2), _scale_inverse_bbox),
}


# ─────────────────────────────────────────────
# IoU
# ─────────────────────────────────────────────

def _iou(a: List[float], b: List[float]) -> float:
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


# ─────────────────────────────────────────────
# Merge methods
# ─────────────────────────────────────────────

def _merge_max_conf(
    all_dets: List[dict],
    iou_threshold: float = 0.5,
) -> List[dict]:
    """
    같은 위치에서 여러 augmentation의 검출이 겹치면 max conf만 유지.
    NMS와 비슷하지만 augmentation별 결과 합치는 용도.

    bbox 없는 검출(분류기 결과)은 IoU 매칭 불가 → class 동일 시 최고 conf만 유지.
    """
    if not all_dets:
        return []

    # class별 그룹화
    by_class: dict = {}
    for d in all_dets:
        by_class.setdefault(d.get("class"), []).append(d)

    merged: List[dict] = []
    for cls, dets in by_class.items():
        # bbox 있는 것 / 없는 것 분리
        with_bbox = [d for d in dets if d.get("bbox_xyxy") is not None]
        no_bbox = [d for d in dets if d.get("bbox_xyxy") is None]

        # bbox 없는 검출: class 단위 최고 conf 1개만
        if no_bbox:
            best_no_bbox = max(no_bbox, key=lambda x: x["conf"])
            merged.append(best_no_bbox)

        # bbox 있는 검출: 기존 IoU NMS
        sorted_dets = sorted(with_bbox, key=lambda x: x["conf"], reverse=True)
        kept: List[dict] = []
        for d in sorted_dets:
            is_dup = False
            for k in kept:
                if _iou(d["bbox_xyxy"], k["bbox_xyxy"]) >= iou_threshold:
                    is_dup = True
                    break
            if not is_dup:
                kept.append(d)
        merged.extend(kept)
    return merged


def _merge_wbf(
    all_dets: List[dict],
    iou_threshold: float = 0.5,
) -> List[dict]:
    """
    Weighted Box Fusion: 겹치는 검출들의 좌표를 conf 가중 평균.
    max_conf보다 robust하지만 약간 더 비쌈.

    bbox 없는 검출(분류기)은 class 단위 최고 conf만 유지.

    Reference: https://arxiv.org/abs/1910.13302
    """
    if not all_dets:
        return []

    by_class: dict = {}
    for d in all_dets:
        by_class.setdefault(d.get("class"), []).append(d)

    merged: List[dict] = []
    for cls, dets in by_class.items():
        with_bbox = [d for d in dets if d.get("bbox_xyxy") is not None]
        no_bbox = [d for d in dets if d.get("bbox_xyxy") is None]

        if no_bbox:
            merged.append(max(no_bbox, key=lambda x: x["conf"]))

        sorted_dets = sorted(with_bbox, key=lambda x: x["conf"], reverse=True)
        clusters: List[List[dict]] = []  # 각 클러스터는 겹치는 검출 그룹

        for d in sorted_dets:
            placed = False
            for cluster in clusters:
                # 클러스터의 anchor(첫 검출)와 IoU 비교
                if _iou(d["bbox_xyxy"], cluster[0]["bbox_xyxy"]) >= iou_threshold:
                    cluster.append(d)
                    placed = True
                    break
            if not placed:
                clusters.append([d])

        # 각 클러스터 내 가중 평균
        for cluster in clusters:
            total_conf = sum(c["conf"] for c in cluster)
            if total_conf <= 0:
                continue
            wx1 = sum(c["bbox_xyxy"][0] * c["conf"] for c in cluster) / total_conf
            wy1 = sum(c["bbox_xyxy"][1] * c["conf"] for c in cluster) / total_conf
            wx2 = sum(c["bbox_xyxy"][2] * c["conf"] for c in cluster) / total_conf
            wy2 = sum(c["bbox_xyxy"][3] * c["conf"] for c in cluster) / total_conf
            avg_conf = total_conf / len(cluster)
            # 겹친 횟수만큼 conf 부스트 (cluster 크기 / 전체 augmentation 수)
            merged.append({
                **cluster[0],
                "bbox_xyxy": [wx1, wy1, wx2, wy2],
                "conf": min(1.0, avg_conf * (1.0 + 0.1 * (len(cluster) - 1))),
                "tta_cluster_size": len(cluster),
            })
    return merged


MERGE_METHODS = {
    "max_conf": _merge_max_conf,
    "wbf": _merge_wbf,
}


# ─────────────────────────────────────────────
# TTAEnsemble
# ─────────────────────────────────────────────

class TTAEnsemble:
    """
    Test-Time Augmentation 앙상블.

    Args:
        augmentations: 적용할 augmentation 이름 리스트
                       (AUGMENTATIONS dict의 키)
        merge_method: 'max_conf' | 'wbf'
        iou_merge_threshold: 결과 병합 IoU 임계
        include_original: True면 원본 이미지 결과도 포함 (권장)
    """

    def __init__(
        self,
        augmentations: Optional[List[str]] = None,
        merge_method: str = "max_conf",
        iou_merge_threshold: float = 0.5,
        include_original: bool = True,
    ):
        self.augmentations = augmentations or []
        # 알 수 없는 augmentation 검증
        for aug in self.augmentations:
            if aug not in AUGMENTATIONS:
                raise ValueError(f"Unknown augmentation: {aug}. Available: {list(AUGMENTATIONS.keys())}")

        self.merge_method = merge_method
        if merge_method not in MERGE_METHODS:
            raise ValueError(f"Unknown merge method: {merge_method}. Available: {list(MERGE_METHODS.keys())}")

        self.iou_merge_threshold = iou_merge_threshold
        self.include_original = include_original

    def predict(
        self,
        model_predict_fn: Callable[[np.ndarray], List[dict]],
        image_bgr: np.ndarray,
    ) -> List[dict]:
        """
        TTA 추론.

        Args:
            model_predict_fn: 모델 추론 함수 (image → List[detection dict])
                              detection dict는 {class, conf, bbox_xyxy, ...} 포함
            image_bgr: 원본 BGR 이미지

        Returns:
            병합된 검출 리스트. 각 검출에 'tta_aug_count' 필드 추가 (선택).
        """
        all_dets: List[dict] = []

        # 원본 추론
        if self.include_original:
            orig_dets = model_predict_fn(image_bgr)
            for d in orig_dets:
                all_dets.append({**d, "tta_source": "original"})

        # 각 augmentation 추론
        for aug_name in self.augmentations:
            aug_fn, inv_fn = AUGMENTATIONS[aug_name]
            aug_img, meta = aug_fn(image_bgr)
            aug_dets = model_predict_fn(aug_img)

            for d in aug_dets:
                bbox = d.get("bbox_xyxy")
                if bbox is None:
                    # bbox 없는 검출은 그대로 통과
                    all_dets.append({**d, "tta_source": aug_name})
                    continue
                # 좌표 역변환
                inv_bbox = inv_fn(bbox, meta)
                all_dets.append({
                    **d,
                    "bbox_xyxy": inv_bbox,
                    "tta_source": aug_name,
                })

        # 결과 병합
        merge_fn = MERGE_METHODS[self.merge_method]
        merged = merge_fn(all_dets, iou_threshold=self.iou_merge_threshold)
        return merged


def load_tta_from_config(config: dict, model_key: str) -> Optional[TTAEnsemble]:
    """
    postprocess_config.yaml의 tta 섹션에서 인스턴스 생성.
    해당 model_key가 enabled_for에 없으면 None 반환 (TTA 미적용).
    """
    enabled_for = config.get("enabled_for", [])
    if model_key not in enabled_for:
        return None

    return TTAEnsemble(
        augmentations=config.get("augmentations", []),
        merge_method=config.get("merge_method", "max_conf"),
        iou_merge_threshold=config.get("iou_merge_threshold", 0.5),
        include_original=True,
    )


__all__ = ["TTAEnsemble", "load_tta_from_config", "AUGMENTATIONS", "MERGE_METHODS"]
