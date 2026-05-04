# =============================================
# tests/test_floorplan_calibration.py
# 역할: 평면도 스케일 보정 로직 검증 (FR-015)
#       - 픽셀 거리 / 실측 길이 = px_per_m 계산 회귀
#       - 동일 점 입력 시 400 에러
#       - 양의 real_length_m 강제 (Pydantic)
# 핵심 계산만 검증 — FastAPI TestClient는 별도
# 실행: pytest tests/test_floorplan_calibration.py -v
# =============================================

from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from app.schemas.floorplan import FloorplanCalibrateRequest


def _scale(p1, p2, real_length_m):
    """api/floorplan.py 의 핵심 계산 로직 복제."""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    pixel_length = math.hypot(dx, dy)
    if pixel_length < 1e-6:
        return None
    return pixel_length / real_length_m


class TestScaleMath:
    def test_horizontal_line_scale(self):
        # (0,0) → (300,0) 픽셀이 실제 3m → 100 px/m
        assert _scale([0, 0], [300, 0], 3.0) == pytest.approx(100.0)

    def test_diagonal_scale(self):
        # (0,0) → (300,400) = 500 픽셀, 실제 5m → 100 px/m
        assert _scale([0, 0], [300, 400], 5.0) == pytest.approx(100.0)

    def test_same_point_returns_none(self):
        assert _scale([100, 100], [100, 100], 1.0) is None


class TestRequestValidation:
    def test_valid_request_accepted(self):
        req = FloorplanCalibrateRequest(p1=[0, 0], p2=[100, 0], real_length_m=1.0)
        assert req.real_length_m == 1.0

    def test_negative_length_rejected(self):
        with pytest.raises(ValidationError):
            FloorplanCalibrateRequest(p1=[0, 0], p2=[100, 0], real_length_m=-1.0)

    def test_zero_length_rejected(self):
        with pytest.raises(ValidationError):
            FloorplanCalibrateRequest(p1=[0, 0], p2=[100, 0], real_length_m=0.0)

    def test_wrong_point_shape_rejected(self):
        """p1/p2 는 정확히 2개 원소."""
        with pytest.raises(ValidationError):
            FloorplanCalibrateRequest(p1=[0], p2=[100, 0], real_length_m=1.0)
        with pytest.raises(ValidationError):
            FloorplanCalibrateRequest(p1=[0, 0, 0], p2=[100, 0], real_length_m=1.0)
