# =============================================
# app/services/anomaly_detection.py
# 역할: Anomalib PatchCore 기반 이상 탐지 서비스
#       - 정상 표면 패턴을 학습하여 이상 영역 탐지
#       - 도배지 기포/들뜸(C-02), 찍힘/스크래치(C-04), 바닥 오염(D-03) 탐지
#       - 이상 점수 히트맵 생성 후 임계값 기반 마스크 추출
#       - asyncio.to_thread로 블로킹 추론을 비동기 래핑
#
# 사용 모델: anomalib PatchCore (EfficientNet 백본)
# 학습: 정상 표면 이미지로만 학습 (비지도 학습)
# =============================================

import asyncio
import os
from typing import Optional, Tuple

import cv2
import numpy as np


class AnomalyDetectionService:
    """
    PatchCore 기반 이상 탐지 서비스.
    정상 표면과 비교하여 이상 영역을 히트맵으로 표현.
    """

    ANOMALY_THRESHOLD = 0.5  # 이상 점수 임계값 (0.0~1.0)

    def __init__(self):
        self._model = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load_model(self, model_path: str) -> None:
        """
        PatchCore 모델 로드.
        anomalib 라이브러리 설치 필요.
        """
        if not os.path.exists(model_path):
            print(f"[Anomaly] 경고: 모델 파일 없음 ({model_path}). 더미 모드 실행.")
            return
        try:
            # anomalib 임포트 (무거운 의존성, 지연 로드)
            from anomalib.deploy import OpenVINOInferencer
            self._model = OpenVINOInferencer(path=model_path)
            print(f"[Anomaly] PatchCore 모델 로드 완료: {model_path}")
        except Exception as e:
            print(f"[Anomaly] 모델 로드 실패: {e}")

    async def detect(
        self,
        frame: np.ndarray,
    ) -> Tuple[Optional[np.ndarray], float]:
        """
        프레임에서 이상 영역 탐지.

        Args:
            frame: BGR 입력 이미지

        Returns:
            (anomaly_mask, anomaly_score):
                anomaly_mask: 이상 영역 이진 마스크 (uint8)
                anomaly_score: 전체 이상 점수 (0.0~1.0)
        """
        if self._model is None:
            return None, 0.0

        result = await asyncio.to_thread(self._infer, frame)
        return result

    def _infer(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], float]:
        """동기 추론 (to_thread에서 실행)"""
        try:
            output = self._model(frame)
            score = float(output.pred_score)
            mask = output.pred_mask.numpy().astype(np.uint8) * 255
            return mask, score
        except Exception as e:
            print(f"[Anomaly] 추론 오류: {e}")
            return None, 0.0

    def generate_heatmap_overlay(
        self,
        frame: np.ndarray,
        mask: np.ndarray,
        alpha: float = 0.4,
    ) -> np.ndarray:
        """
        원본 이미지 위에 이상 영역 히트맵 오버레이.

        Args:
            frame: 원본 BGR 이미지
            mask: 이상 영역 마스크
            alpha: 오버레이 투명도

        Returns:
            히트맵이 합성된 BGR 이미지
        """
        if mask is None:
            return frame

        # 마스크를 컬러맵으로 변환 (빨강 = 이상)
        heatmap = cv2.applyColorMap(mask, cv2.COLORMAP_JET)

        # 원본과 합성
        h, w = frame.shape[:2]
        if heatmap.shape[:2] != (h, w):
            heatmap = cv2.resize(heatmap, (w, h))

        return cv2.addWeighted(frame, 1.0 - alpha, heatmap, alpha, 0)


# 모듈 레벨 싱글톤
anomaly_service = AnomalyDetectionService()
