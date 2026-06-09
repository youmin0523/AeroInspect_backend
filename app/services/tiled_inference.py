# =============================================
# app/services/tiled_inference.py
# 역할: 고해상도 이미지를 타일로 분할하여 YOLO 추론 (SAHI 방식)
#       - 4K 드론 영상에서 소형 하자(크랙, 핀홀) Recall 극대화
#       - 겹침(overlap) 영역의 중복 검출은 cross-tile NMS로 제거
#       - 실배치 추론(타일을 [N,3,S,S]로 스택, ONNX 1회 호출)으로
#         타일 순차 처리 대비 속도 향상 (dynamic-batch ONNX 전제)
#       - Tier 3 프레임에서만 선택적 적용 (실시간 예산 보호)
#
# 사용: inference_pipeline_20.py의 _run_m1/m2/m3에서 호출
# =============================================

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import cv2
import numpy as np

from app.services.onnx_inference import ONNXYoloDetector


def generate_tiles(
    img_h: int,
    img_w: int,
    tile_size: int = 640,
    overlap_ratio: float = 0.2,
) -> List[Tuple[int, int, int, int]]:
    """
    이미지를 타일로 분할할 좌표 리스트 생성.

    Returns:
        [(x1, y1, x2, y2), ...] — 각 타일의 원본 이미지 좌표
    """
    stride = int(tile_size * (1 - overlap_ratio))
    tiles = []

    # 이미지가 타일 1개 크기 이하면 전체를 1타일로
    if img_h <= tile_size and img_w <= tile_size:
        return [(0, 0, img_w, img_h)]

    n_rows = max(1, math.ceil((img_h - tile_size) / stride) + 1)
    n_cols = max(1, math.ceil((img_w - tile_size) / stride) + 1)

    # 중복 타일 버그 방지: (img_h - tile_size) 가 stride 의 배수가 아니면
    # 마지막 행/열에서 y1/x1 이 img_h-tile_size 로 clamp 되어, 직전 행/열과
    # *동일한* 좌표가 되는 경우가 생긴다(예: 1000px, tile=640, stride=512 →
    # row0 y1=0, row1 y1=360, row2 y1=360 중복). 이러면 같은 crop 을 ONNX 로
    # 두 번 추론하는 낭비 + cross-tile NMS 부하만 늘어난다.
    # set 으로 좌표를 dedupe 하되, 결정적 순서를 위해 삽입 순서를 유지한다.
    seen: set = set()
    for row in range(n_rows):
        for col in range(n_cols):
            y1 = min(row * stride, img_h - tile_size)
            x1 = min(col * stride, img_w - tile_size)
            y1 = max(0, y1)
            x1 = max(0, x1)
            y2 = min(y1 + tile_size, img_h)
            x2 = min(x1 + tile_size, img_w)
            coord = (x1, y1, x2, y2)
            if coord in seen:
                continue  # 클램프로 생긴 중복 타일 제거
            seen.add(coord)
            tiles.append(coord)

    return tiles


def _cross_tile_nms(
    detections: List[dict],
    iou_threshold: float = 0.5,
) -> List[dict]:
    """타일 간 중복 검출 제거 (동일 클래스 기준)."""
    if len(detections) <= 1:
        return detections

    by_class: dict = {}
    for det in detections:
        by_class.setdefault(det["class"], []).append(det)

    result: List[dict] = []
    for cls, dets in by_class.items():
        if len(dets) <= 1:
            result.extend(dets)
            continue

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


def _iou(box_a: List[float], box_b: List[float]) -> float:
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    return inter / (area_a + area_b - inter + 1e-6)


def tiled_predict(
    frame_bgr: np.ndarray,
    detector: ONNXYoloDetector,
    conf: float = 0.25,
    iou: float = 0.45,
    tile_size: int = 640,
    overlap_ratio: float = 0.2,
    nms_iou: float = 0.5,
    min_resolution: int = 1280,
) -> List[dict]:
    """
    타일 분할 추론 → 좌표 리맵 → cross-tile NMS.

    고해상도(min_resolution 이상)에서만 타일링 적용.
    저해상도면 일반 full-frame 추론으로 fallback.

    Args:
        frame_bgr: 원본 BGR 이미지
        detector: ONNXYoloDetector 인스턴스
        conf: 신뢰도 임계값
        iou: 타일 내 NMS IoU
        tile_size: 타일 크기 (px)
        overlap_ratio: 타일 간 겹침 비율 (0.0~0.5)
        nms_iou: cross-tile NMS IoU
        min_resolution: 이 해상도 미만이면 타일링 스킵

    Returns:
        병합된 검출 리스트 [{class, conf, bbox_xyxy, ...}]
    """
    h, w = frame_bgr.shape[:2]

    # 저해상도 → full-frame fallback
    if max(h, w) < min_resolution:
        return detector.predict(frame_bgr, conf=conf, iou=iou)

    tiles = generate_tiles(h, w, tile_size, overlap_ratio)

    # 타일 1개면 full-frame과 동일
    if len(tiles) == 1:
        return detector.predict(frame_bgr, conf=conf, iou=iou)

    all_dets: List[dict] = []

    # ── 실배치 추론 (타일을 한꺼번에 스택 → 단일 ONNX 호출) ──
    # 조사 결과: 운영 YOLO ONNX 는 dynamic=True 로 export 되어 입력 batch 축이
    # 심볼릭('batch')이다(m1/m2/m3/m4 모두 [batch,3,H,W] 확인). 따라서 모든
    # 타일을 [N,3,S,S] 로 묶어 ONNXYoloDetector.predict_batch 로 단 한 번
    # 추론하고 결과를 타일별로 분리한다. (고정 batch=1 export 모델이면
    # predict_batch 내부에서 자동으로 순차 폴백하므로 호출부는 동일.)
    #
    # 좌표 리맵 + cross-tile NMS 로직은 기존과 동일하게 유지.
    tile_crops: List[np.ndarray] = []
    tile_origins: List[Tuple[int, int]] = []  # 각 crop 의 원본 좌상단 (tx1, ty1)
    for tx1, ty1, tx2, ty2 in tiles:
        tile_crop = frame_bgr[ty1:ty2, tx1:tx2]

        # 타일이 너무 작으면 스킵
        th, tw = tile_crop.shape[:2]
        if th < 32 or tw < 32:
            continue

        tile_crops.append(tile_crop)
        tile_origins.append((tx1, ty1))

    # Full-frame 추론도 같은 배치에 포함 (대형 하자는 full-frame이 유리).
    # 마지막 슬롯이 full-frame, origin (0,0) 이지만 tiled=False 로 구분 표시.
    full_frame_idx = len(tile_crops)
    batch_frames: List[np.ndarray] = [*tile_crops, frame_bgr]

    # 단일(또는 폴백 시 순차) ONNX 호출로 전 타일 + full-frame 일괄 추론
    batched_results = detector.predict_batch(batch_frames, conf=conf, iou=iou)

    for idx, dets in enumerate(batched_results):
        is_full = idx == full_frame_idx
        ox, oy = (0, 0) if is_full else tile_origins[idx]
        for det in dets:
            bx1, by1, bx2, by2 = det["bbox_xyxy"]
            det["bbox_xyxy"] = [bx1 + ox, by1 + oy, bx2 + ox, by2 + oy]
            det["tiled"] = not is_full
            all_dets.append(det)

    # cross-tile NMS로 중복 제거
    return _cross_tile_nms(all_dets, nms_iou)


__all__ = ["tiled_predict", "generate_tiles"]
