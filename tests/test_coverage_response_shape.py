# =============================================
# tests/test_coverage_response_shape.py
# 역할: CoverageResponse Pydantic 스키마 검증
#       - site_id UUID 필수
#       - coverage_ratio 0~1 범위 허용값 통과
#       - hull 직렬화가 [[x, y], ...] 형태로 유지
#       - 샘플 부족 fallback 구조 유효성
# 실행: pytest tests/test_coverage_response_shape.py -v
# =============================================

from __future__ import annotations

from uuid import uuid4

import pytest

from app.schemas.monitoring import (
    CoverageResponse,
    LidarStats,
    StreamStatsResponse,
    TelemetryCacheStats,
    WorkerStats,
)


class TestCoverageResponse:
    def test_full_response_validates(self):
        r = CoverageResponse(
            site_id=uuid4(),
            covered_area_m2=76.3,
            supplied_area_m2=84.0,
            coverage_ratio=0.9083,
            uncovered_area_m2=7.7,
            sample_count=1200,
            hull=[[0.0, 0.0], [10.0, 0.0], [10.0, 5.0], [0.0, 5.0]],
        )
        dumped = r.model_dump(mode="json")
        assert dumped["covered_area_m2"] == 76.3
        assert dumped["hull"][0] == [0.0, 0.0]
        assert dumped["note"] is None

    def test_fallback_when_insufficient_samples(self):
        """샘플 < 3 시 반환하는 구조가 스키마에 맞는지."""
        r = CoverageResponse(
            site_id=uuid4(),
            covered_area_m2=0.0,
            supplied_area_m2=84.0,
            coverage_ratio=None,
            uncovered_area_m2=84.0,
            sample_count=2,
            hull=[],
            note="텔레메트리 부족",
        )
        assert r.sample_count == 2
        assert r.hull == []
        assert r.note.startswith("텔레메트리")

    def test_supplied_area_optional(self):
        """사용자가 total_area 미입력한 site도 커버리지 계산 가능해야 함."""
        r = CoverageResponse(
            site_id=uuid4(),
            covered_area_m2=50.0,
            supplied_area_m2=None,
            coverage_ratio=None,
            uncovered_area_m2=None,
            sample_count=100,
            hull=[[0, 0], [10, 0], [5, 5]],
        )
        assert r.supplied_area_m2 is None
        assert r.coverage_ratio is None


class TestStreamStatsResponse:
    def test_happy_path(self):
        r = StreamStatsResponse(
            worker=WorkerStats(
                running=True, submitted=100, processed=30, dropped=2,
                queue_size=0, frame_skip=3,
            ),
            telemetry_cache=TelemetryCacheStats(ready=True, age_sec=0.12),
            lidar=LidarStats(connected=True, distance_m=2.43),
        )
        dumped = r.model_dump()
        assert dumped["worker"]["submitted"] == 100
        assert dumped["lidar"]["distance_m"] == 2.43

    def test_cold_start_all_nulls(self):
        """서버 기동 직후 텔레메트리·LiDAR 둘 다 없을 때."""
        r = StreamStatsResponse(
            worker=WorkerStats(
                running=True, submitted=0, processed=0, dropped=0,
                queue_size=0, frame_skip=3,
            ),
            telemetry_cache=TelemetryCacheStats(ready=False, age_sec=None),
            lidar=LidarStats(connected=False, distance_m=None),
        )
        assert r.telemetry_cache.age_sec is None
        assert r.lidar.distance_m is None
