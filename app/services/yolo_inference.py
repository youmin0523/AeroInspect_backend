# =============================================
# app/services/yolo_inference.py
# 역할: 레거시 호환 shim — 실제 추론은 inference_pipeline가 담당
#       - YOLOInferenceService: 기존 API 유지 (load_model, infer, is_loaded)
#       - DetectionResult: 기존 dataclass 필드 유지 (LegacyDetection alias)
#       - 모델 로드는 inference_pipeline.pipeline 싱글톤에 위임 (중복 로드 금지)
#       - 추론 결과는 신규 3-모델 파이프라인 → A-E taxonomy 레거시 포맷 변환
#
# 기존 호출자: [defect_processor.py](defect_processor.py), [dependencies.py](../dependencies.py)
# 신규 코드: inference_pipeline.detect_defects() 또는 pipeline.detect() 사용
# =============================================

from __future__ import annotations

import asyncio
from typing import List

import numpy as np

from app.services.inference_pipeline import (
    LegacyDetection as DetectionResult,  # 기존 심볼명 유지
    detect_defects_legacy,
    pipeline,
)

__all__ = ["DetectionResult", "YOLOInferenceService", "yolo_service"]


class YOLOInferenceService:
    """
    레거시 shim. 내부적으로 inference_pipeline 싱글톤을 참조한다.
    기존 호출자의 메서드 시그니처를 그대로 유지.
    """

    @property
    def is_loaded(self) -> bool:
        """하위 호환: 3개 모델 중 하나라도 로드됐으면 True."""
        return pipeline.is_loaded

    def load_model(self) -> None:
        """
        하위 호환 진입점. main.py의 lifespan이 이 메서드를 호출해왔으므로
        여기서 신규 3-모델 일괄 로드를 트리거한다.
        가중치 누락 등 에러는 상위로 전파한다.
        """
        pipeline.load_models()

    async def infer(self, frame: np.ndarray) -> List[DetectionResult]:
        """
        기존 호출자 시그니처 유지 — BGR ndarray 입력 → LegacyDetection 리스트.
        신규 코드는 `pipeline.detect_async(image)` 사용 권장.
        """
        if not pipeline.is_loaded:
            return []
        return await asyncio.to_thread(detect_defects_legacy, frame)


# ── 모듈 레벨 싱글톤 ─────────────────────────
# 기존 코드: `from app.services.yolo_inference import yolo_service`
yolo_service = YOLOInferenceService()
