# =============================================
# app/services/thermal.py
# 역할: IRC-256CA 열화상 카메라 프레임 처리 서비스
#       - 16bit ADC 원시값 → 섭씨 온도 변환
#       - COLORMAP_INFERNO 의사색상 오버레이 생성
#       - 온도 통계 추출 (max/min/avg)
#       - blend_frames(): RGB + 열화상 알파 합성 (스트리밍 API에서 사용)
#
# IRC-256CA 스펙:
#   해상도: 256x192 (기본), 384x288, 640x512
#   열감도: <50mK NETD
#   USB Capture Card를 통해 cv2.VideoCapture로 수신
# =============================================

import asyncio
from typing import Optional, Tuple

import cv2
import numpy as np

from app.services.camera import thermal_camera_service


class ThermalProcessor:
    """
    열화상 프레임 처리 서비스.
    원시 프레임을 온도 맵으로 변환하고 의사색상 이미지를 생성한다.
    """

    # IRC-256CA 온도 변환 계수 (캘리브레이션 후 조정 필요)
    TEMP_SCALE = 0.04       # ADC 단위당 온도 변화 (°C)
    TEMP_OFFSET = -273.15   # 절대온도 → 섭씨 변환 오프셋

    # 경보 임계값
    INSULATION_ALERT_DELTA = 3.0  # 주변 대비 온도차 임계값 (°C) — 단열 결함 판정
    FLOOR_HEATING_ALERT_DELTA = 2.0  # 바닥 난방 불균일 판정

    def process_frame(
        self,
        raw_frame: np.ndarray,
        colormap: int = cv2.COLORMAP_INFERNO,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        원시 열화상 프레임을 처리하여 의사색상 이미지와 온도 맵 반환.

        Args:
            raw_frame: USB Capture Card에서 수신한 원시 프레임 (BGR 또는 GRAY)
            colormap: OpenCV 컬러맵 (기본: INFERNO — 열화상 표준)

        Returns:
            (colored_frame, temp_map):
                colored_frame: BGR 의사색상 이미지
                temp_map: float32 온도 맵 (°C)
        """
        # 그레이스케일로 변환
        if len(raw_frame.shape) == 3:
            gray = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = raw_frame

        # 16bit → 8bit 정규화 (디스플레이용)
        normalized = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

        # 의사색상 적용
        colored = cv2.applyColorMap(normalized, colormap)

        # 온도 맵 생성 (16bit ADC → 섭씨)
        temp_map = gray.astype(np.float32) * self.TEMP_SCALE + self.TEMP_OFFSET

        return colored, temp_map

    def get_roi_temperature(
        self,
        temp_map: np.ndarray,
        bbox: Optional[Tuple[float, float, float, float]] = None,
    ) -> Tuple[float, float, float]:
        """
        관심 영역(ROI)의 온도 통계 추출.

        Args:
            temp_map: 온도 맵 (°C)
            bbox: 정규화 바운딩 박스 (x, y, w, h) — None이면 전체 영역

        Returns:
            (max_temp, min_temp, avg_temp) in °C
        """
        if bbox is None:
            roi = temp_map
        else:
            h, w = temp_map.shape[:2]
            x, y, bw, bh = bbox
            x1 = int((x - bw / 2) * w)
            y1 = int((y - bh / 2) * h)
            x2 = int((x + bw / 2) * w)
            y2 = int((y + bh / 2) * h)
            roi = temp_map[
                max(0, y1):min(h, y2),
                max(0, x1):min(w, x2),
            ]

        if roi.size == 0:
            return 0.0, 0.0, 0.0

        return float(roi.max()), float(roi.min()), float(roi.mean())

    def detect_insulation_defect(
        self,
        temp_map: np.ndarray,
        threshold_delta: float = None,
    ) -> np.ndarray:
        """
        단열 결함 영역 마스크 생성.
        주변 온도 대비 임계값 이상 차이나는 영역을 결함으로 판정.

        Returns:
            결함 영역 바이너리 마스크 (uint8, 0 또는 255)
        """
        delta = threshold_delta or self.INSULATION_ALERT_DELTA
        mean_temp = float(temp_map.mean())
        diff = np.abs(temp_map - mean_temp)
        mask = (diff > delta).astype(np.uint8) * 255
        return mask

    async def get_processed_frame(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        열화상 카메라에서 프레임을 비동기로 획득하고 처리.

        Returns:
            (colored_frame, temp_map) 또는 (None, None)
        """
        raw = await thermal_camera_service.get_single_frame()
        if raw is None:
            return None, None
        return await asyncio.to_thread(self.process_frame, raw)


def blend_frames(
    rgb_frame: np.ndarray,
    thermal_frame: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """
    RGB 프레임과 열화상 의사색상 프레임 알파 블렌딩.
    streaming.py와 동일한 로직 (서비스 레이어에서 직접 사용할 때)

    Args:
        rgb_frame: BGR RGB 프레임
        thermal_frame: BGR 의사색상 열화상 프레임
        alpha: 열화상 가중치 (0.0=RGB만, 1.0=열화상만)

    Returns:
        합성된 BGR 프레임
    """
    h, w = rgb_frame.shape[:2]
    if thermal_frame.shape[:2] != (h, w):
        thermal_frame = cv2.resize(thermal_frame, (w, h))
    if len(thermal_frame.shape) == 2:
        thermal_frame = cv2.cvtColor(thermal_frame, cv2.COLOR_GRAY2BGR)
    return cv2.addWeighted(rgb_frame, 1.0 - alpha, thermal_frame, alpha, 0)


# 모듈 레벨 싱글톤
thermal_processor = ThermalProcessor()
