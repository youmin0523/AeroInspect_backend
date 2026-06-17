# =============================================
# app/utils/image_utils.py
# 역할: 이미지 처리 유틸리티 함수 모음
#       - crop_roi: 바운딩 박스 기반 ROI 크롭
#       - encode_frame_to_base64: 프레임 → Base64 JPEG 인코딩
#       - decode_base64_to_frame: Base64 → numpy 배열 디코딩
#       - draw_detection_overlay: 탐지 결과 박스/라벨 오버레이
# =============================================

import base64
from typing import Optional, Tuple

import cv2
import numpy as np

# 심각도별 박스 색상 (BGR)
SEVERITY_COLORS = {
    "HIGH": (0, 0, 255),    # 빨강
    "MED":  (0, 165, 255),  # 주황
    "LOW":  (0, 255, 255),  # 노랑
}


def crop_roi(
    frame: np.ndarray,
    bbox: Tuple[float, float, float, float],
    padding: float = 0.05,
) -> Optional[np.ndarray]:
    """
    정규화 바운딩 박스 기준으로 ROI 크롭.

    Args:
        frame: BGR 입력 이미지
        bbox: (cx, cy, w, h) 정규화 좌표 (0.0~1.0)
        padding: 크롭 영역 여백 비율

    Returns:
        크롭된 이미지 또는 None
    """
    h, w = frame.shape[:2]
    cx, cy, bw, bh = bbox

    # 여백은 박스 크기에 비례(전체 이미지 비율이 아님). 과거엔 padding 을 정규화 좌표에
    # 그대로 더해 박스 크기와 무관하게 이미지의 padding*100% 를 더했고, 작은 박스는 과도하게
    # 크롭됐다. alignment_detector._crop_roi 와 동일하게 박스 변에 비례시킨다.
    pad_x = bw * padding
    pad_y = bh * padding

    # 픽셀 좌표 변환 (여백 포함)
    x1 = int((cx - bw / 2 - pad_x) * w)
    y1 = int((cy - bh / 2 - pad_y) * h)
    x2 = int((cx + bw / 2 + pad_x) * w)
    y2 = int((cy + bh / 2 + pad_y) * h)

    # 이미지 경계 클리핑
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    if x2 <= x1 or y2 <= y1:
        return None

    return frame[y1:y2, x1:x2].copy()


def encode_frame_to_base64(
    frame: np.ndarray,
    quality: int = 80,
) -> Optional[str]:
    """
    numpy 배열 이미지를 Base64 인코딩된 JPEG 문자열로 변환.

    Args:
        frame: BGR numpy 배열
        quality: JPEG 압축 품질 (0~100)

    Returns:
        "data:image/jpeg;base64,..." 형식 문자열
    """
    if frame is None or frame.size == 0:
        return None

    success, buf = cv2.imencode(
        ".jpg", frame,
        [cv2.IMWRITE_JPEG_QUALITY, quality]
    )
    if not success:
        return None

    b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def decode_base64_to_frame(b64_string: str) -> Optional[np.ndarray]:
    """
    Base64 이미지 문자열을 numpy 배열로 디코딩.

    Args:
        b64_string: "data:image/jpeg;base64,..." 또는 순수 base64

    Returns:
        BGR numpy 배열 또는 None
    """
    try:
        if b64_string.startswith("data:"):
            b64_string = b64_string.split(",", 1)[1]
        img_bytes = base64.b64decode(b64_string)
        arr = np.frombuffer(img_bytes, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        return None


def draw_detection_overlay(
    frame: np.ndarray,
    detections: list,
) -> np.ndarray:
    """
    탐지 결과를 프레임에 바운딩 박스와 라벨로 시각화.

    Args:
        frame: BGR 원본 프레임
        detections: DetectionResult 목록

    Returns:
        오버레이가 적용된 프레임 복사본
    """
    overlay = frame.copy()
    h, w = overlay.shape[:2]

    for det in detections:
        color = SEVERITY_COLORS.get(det.severity, (128, 128, 128))

        # 픽셀 좌표 변환
        x1 = int((det.bbox_x - det.bbox_w / 2) * w)
        y1 = int((det.bbox_y - det.bbox_h / 2) * h)
        x2 = int((det.bbox_x + det.bbox_w / 2) * w)
        y2 = int((det.bbox_y + det.bbox_h / 2) * h)

        # 박스 그리기
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)

        # 라벨 (카테고리 코드 + 신뢰도)
        label = f"{det.category_code} {det.confidence:.0%}"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(overlay, (x1, y1 - lh - 4), (x1 + lw, y1), color, -1)
        cv2.putText(
            overlay, label,
            (x1, y1 - 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
        )

    return overlay


def resize_frame(
    frame: np.ndarray,
    width: int = 640,
    height: int = 480,
) -> np.ndarray:
    """프레임을 지정 크기로 리사이즈"""
    return cv2.resize(frame, (width, height))
