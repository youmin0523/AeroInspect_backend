# =============================================
# app/services/insulation_detector.py
# 역할: M4 열화상-RGB 퓨전 기반 단열/기밀/난방 하자 검출
#       - U-Net ONNX로 열화상 세그멘테이션 (4 class)
#       - RGB Context YOLO로 건물 요소 검출 (열원 오탐 필터링)
#       - 건물 요소별 적응적 온도차 임계값 적용
#       - Connected Component → bbox + severity + delta_T
#
# 커버 하자:
#   B-01: 창호 단열 불량 (결로·냉교)
#   B-02: 벽체 단열 공백·탈락
#   B-05: 창호 기밀 불량 (틈새)
#   D-01: 바닥 난방 불량 (온도 편차)
# =============================================

from __future__ import annotations

import asyncio
import json
import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from app.config import settings
from app.services.onnx_inference import ONNXUNetSegmenter, ONNXYoloDetector


# ── 클래스 인덱스 ↔ 하자 코드 ──
THERMAL_CLASS_MAP = {
    1: {"code": "B-01", "class_name": "window_insulation_defect", "display_ko": "창호 단열 불량"},
    2: {"code": "B-02", "class_name": "wall_insulation_gap", "display_ko": "벽체 단열 공백·탈락"},
    3: {"code": "B-05", "class_name": "window_airtight_defect", "display_ko": "창호 기밀 불량"},
    4: {"code": "D-01", "class_name": "floor_heating_defect", "display_ko": "바닥 난방 불량"},
}


class InsulationDetector:
    """
    열화상-RGB 퓨전 기반 단열/기밀/난방 하자 검출기.

    이중 브랜치 구조:
    - Branch A: U-Net 열화상 세그멘테이션 → 하자 마스크
    - Branch B: YOLO RGB 컨텍스트 → 건물 요소 (배관, 라디에이터 등 오탐 필터)
    - Fusion: 열원 영역 제거 + 요소별 적응적 임계값 → 최종 검출
    """

    def __init__(self):
        self._unet: Optional[ONNXUNetSegmenter] = None
        self._context_yolo: Optional[ONNXYoloDetector] = None
        self._homography: Optional[np.ndarray] = None

    @property
    def is_loaded(self) -> bool:
        return self._unet is not None

    def load_models(self) -> None:
        """M4 모델 로드."""
        weights_dir = settings.AEROINSPECT_WEIGHTS_DIR

        unet_path = os.path.join(weights_dir, settings.M4_UNET_ONNX)
        if os.path.exists(unet_path):
            self._unet = ONNXUNetSegmenter(
                unet_path,
                class_names=["background", "window_insulation", "wall_insulation",
                             "window_airtight", "floor_heating"],
            )
            print(f"[M4-UNet] 로드 완료: {unet_path}")

        context_path = os.path.join(weights_dir, settings.M4_CONTEXT_ONNX)
        if os.path.exists(context_path):
            self._context_yolo = ONNXYoloDetector(
                context_path,
                class_names=["window", "wall", "door", "pipe", "radiator", "floor"],
            )
            print(f"[M4-Context] 로드 완료: {context_path}")

        # Homography 행렬 로드
        h_path = os.path.join(weights_dir, settings.THERMAL_RGB_HOMOGRAPHY)
        if os.path.exists(h_path):
            with open(h_path) as f:
                h_data = json.load(f)
            self._homography = np.array(h_data, dtype=np.float64)
            print(f"[M4] Homography 로드 완료")

    def detect(
        self,
        frame_bgr: np.ndarray,
        temp_map: np.ndarray,
    ) -> List[dict]:
        """
        열화상+RGB 퓨전 검출.

        Args:
            frame_bgr: RGB 프레임 (H, W, 3)
            temp_map: 온도맵 float32 (h, w) °C

        Returns:
            [{class, class_name, code, conf, bbox_xyxy,
              delta_temperature, max_temp, min_temp, severity}]
        """
        if self._unet is None:
            return []

        # U-Net 세그멘테이션
        class_mask, prob_map = self._unet.segment(temp_map)

        # RGB 컨텍스트로 열원(배관/라디에이터) 영역 마스크 생성
        heat_source_mask = self._get_heat_source_mask(frame_bgr, temp_map.shape)

        # 열원 영역 제거
        if heat_source_mask is not None:
            class_mask[heat_source_mask > 0] = 0

        # 하자 클래스별 Connected Component → bbox
        results: List[dict] = []
        for cls_idx, info in THERMAL_CLASS_MAP.items():
            binary = (class_mask == cls_idx).astype(np.uint8)
            if binary.sum() == 0:
                continue

            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < 50:  # 너무 작은 영역 무시
                    continue

                x, y, w, h = cv2.boundingRect(cnt)
                roi_temp = temp_map[y : y + h, x : x + w]
                roi_prob = prob_map[cls_idx, y : y + h, x : x + w]

                delta_t = float(np.abs(roi_temp.mean() - temp_map.mean()))
                threshold = self._get_threshold(cls_idx)

                if delta_t < threshold:
                    continue

                conf = float(roi_prob.mean())
                max_t = float(roi_temp.max())
                min_t = float(roi_temp.min())

                # bbox를 원본 RGB 좌표로 변환 (Homography 역변환)
                bbox_xyxy = self._thermal_to_rgb_bbox(
                    [x, y, x + w, y + h], frame_bgr.shape, temp_map.shape,
                )

                severity = "HIGH" if delta_t > threshold * 1.5 else "MED"

                results.append({
                    "class": info["class_name"],
                    "code": info["code"],
                    "display_ko": info["display_ko"],
                    "conf": conf,
                    "bbox_xyxy": bbox_xyxy,
                    "delta_temperature": delta_t,
                    "max_temperature": max_t,
                    "min_temperature": min_t,
                    "severity": severity,
                    "defect_source": "thermal_unet",
                })

        return results

    async def detect_async(
        self, frame_bgr: np.ndarray, temp_map: np.ndarray,
    ) -> List[dict]:
        """비동기 래퍼."""
        return await asyncio.to_thread(self.detect, frame_bgr, temp_map)

    # ── 내부 ─────────────────────────────────
    def _get_threshold(self, cls_idx: int) -> float:
        """하자 유형별 온도차 임계값."""
        thresholds = {
            1: settings.M4_INSULATION_WINDOW_DELTA,
            2: settings.M4_INSULATION_WALL_DELTA,
            3: settings.M4_AIRTIGHT_DELTA,
            4: settings.M4_FLOOR_HEATING_DELTA,
        }
        return thresholds.get(cls_idx, 3.0)

    def _get_heat_source_mask(
        self,
        frame_bgr: np.ndarray,
        thermal_shape: Tuple[int, int],
    ) -> Optional[np.ndarray]:
        """RGB 컨텍스트 YOLO로 열원(배관, 라디에이터) 영역 마스크 생성."""
        if self._context_yolo is None:
            return None

        dets = self._context_yolo.predict(frame_bgr, conf=0.3)
        mask = np.zeros(thermal_shape[:2], dtype=np.uint8)

        for det in dets:
            if det["class"] not in ("pipe", "radiator"):
                continue
            # RGB bbox → 열화상 좌표 변환
            bbox = self._rgb_to_thermal_bbox(
                det["bbox_xyxy"], frame_bgr.shape, thermal_shape,
            )
            x1, y1, x2, y2 = [int(v) for v in bbox]
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(thermal_shape[1], x2)
            y2 = min(thermal_shape[0], y2)
            mask[y1:y2, x1:x2] = 255

        return mask

    def _thermal_to_rgb_bbox(
        self,
        bbox: List[float],
        rgb_shape: Tuple[int, ...],
        thermal_shape: Tuple[int, ...],
    ) -> List[float]:
        """열화상 좌표 bbox → RGB 좌표 bbox (단순 비율 변환)."""
        th, tw = thermal_shape[:2]
        rh, rw = rgb_shape[:2]
        sx, sy = rw / tw, rh / th
        return [bbox[0] * sx, bbox[1] * sy, bbox[2] * sx, bbox[3] * sy]

    def _rgb_to_thermal_bbox(
        self,
        bbox: List[float],
        rgb_shape: Tuple[int, ...],
        thermal_shape: Tuple[int, ...],
    ) -> List[float]:
        """RGB 좌표 bbox → 열화상 좌표 bbox."""
        rh, rw = rgb_shape[:2]
        th, tw = thermal_shape[:2]
        sx, sy = tw / rw, th / rh
        return [bbox[0] * sx, bbox[1] * sy, bbox[2] * sx, bbox[3] * sy]


# 모듈 레벨 싱글톤
insulation_detector = InsulationDetector()
