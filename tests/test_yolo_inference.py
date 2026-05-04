# =============================================
# tests/test_yolo_inference.py
# 역할: YOLOv8 추론 서비스 단위 테스트
#       - 모델 로드 (파일 없을 시 더미 모드)
#       - 추론 결과 구조 검증
#       - severity_mapper 매핑 검증
# 실행: pytest tests/test_yolo_inference.py -v
# =============================================

import numpy as np
import pytest

from app.services.yolo_inference import YOLOInferenceService
from app.utils.severity_mapper import (
    get_severity_by_code,
    DEFECT_CATALOG,
    DEFECT_CLASS_NAMES,
)


@pytest.fixture
def service():
    """테스트용 YOLO 서비스 (더미 모드)"""
    svc = YOLOInferenceService()
    svc.load_model()  # 가중치 없으면 더미 모드
    return svc


@pytest.fixture
def dummy_frame():
    """테스트용 더미 프레임 (640x480 랜덤 이미지)"""
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


@pytest.mark.asyncio
async def test_infer_dummy_mode(service, dummy_frame):
    """더미 모드에서 추론 → 빈 결과 반환"""
    results = await service.infer(dummy_frame)
    assert isinstance(results, list)
    # 더미 모드에서는 빈 결과
    assert len(results) == 0


def test_severity_mapper_all_codes():
    """20종 하자 코드 모두 매핑 가능한지 검증"""
    for code in DEFECT_CATALOG.keys():
        result = get_severity_by_code(code)
        assert result["code"] == code
        assert result["severity"] in ("HIGH", "MED", "LOW")
        assert result["area"] in ("A", "B", "C", "D", "E")


def test_severity_mapper_class_name():
    """class_name으로도 매핑 가능한지 검증"""
    result = get_severity_by_code("crack_structural")
    assert result["code"] == "A-02"
    assert result["severity"] == "HIGH"


def test_severity_mapper_unknown():
    """알 수 없는 클래스 → 기본값 반환"""
    result = get_severity_by_code("unknown_defect_xyz")
    assert result["code"] == "X-00"
    assert result["severity"] == "MED"


def test_defect_class_names_count():
    """YOLOv8 클래스 ID 매핑 수가 20인지 검증"""
    assert len(DEFECT_CLASS_NAMES) == 20


def test_service_is_loaded_false_without_weights(service):
    """가중치 파일 없으면 is_loaded = False"""
    assert service.is_loaded is False
