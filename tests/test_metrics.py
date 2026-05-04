# =============================================
# tests/test_metrics.py
# 역할: Prometheus 메트릭 파이프라인 회귀
#       - Counter inc / Gauge set
#       - render_metrics가 OpenMetrics 텍스트 반환
#       - refresh_sensor_gauges 가 싱글톤 상태를 반영
# 실행: pytest tests/test_metrics.py -v
# =============================================

from __future__ import annotations

import pytest

from app.core.metrics import (
    defect_detected_total,
    lidar_distance_meters,
    refresh_sensor_gauges,
    render_metrics,
    stream_frames_dropped,
    stream_frames_submitted,
    telemetry_cache_age_seconds,
)
from app.services.lidar import lidar_service
from app.services.telemetry_cache import telemetry_cache


def test_counter_labels_increment():
    """severity 라벨 Counter 증감."""
    before = defect_detected_total.labels(severity="HIGH")._value.get()
    defect_detected_total.labels(severity="HIGH").inc()
    after = defect_detected_total.labels(severity="HIGH")._value.get()
    assert after == before + 1


def test_unlabeled_counter_inc():
    before = stream_frames_submitted._value.get()
    stream_frames_submitted.inc(5)
    after = stream_frames_submitted._value.get()
    assert after == before + 5


def test_render_metrics_returns_openmetrics_text():
    """/metrics 응답이 Prometheus 파서가 먹을 수 있는 텍스트인지."""
    defect_detected_total.labels(severity="MED").inc()
    stream_frames_dropped.inc()

    response = render_metrics()
    assert response.status_code == 200
    assert response.media_type.startswith("text/plain")

    body = response.body.decode("utf-8")
    # 대표 시리즈 몇 개만 확인
    assert "aeroinspect_defect_detected_total" in body
    assert "aeroinspect_stream_frames_dropped_total" in body
    assert "aeroinspect_lidar_distance_meters" in body


def test_refresh_sensor_gauges_sets_minus_one_when_empty():
    """센서 미연결 시 sentinel -1 세팅 (Prom 쿼리에서 구분 가능)."""
    # 캐시 비우고 LiDAR 상태 리셋
    telemetry_cache.clear()
    lidar_service._latest_distance = None

    refresh_sensor_gauges()

    assert lidar_distance_meters._value.get() == -1
    assert telemetry_cache_age_seconds._value.get() == -1


@pytest.mark.asyncio
async def test_refresh_sensor_gauges_reflects_updates():
    """캐시 업데이트 후 gauge 에 반영."""
    await telemetry_cache.update(pos_x=1.0, pos_y=2.0, pos_z=3.0)
    lidar_service._latest_distance = 2.5

    refresh_sensor_gauges()

    assert lidar_distance_meters._value.get() == pytest.approx(2.5)
    # age 는 정확한 값보다 "0 이상 & 몇 초 내" 로 느슨히 검증
    age = telemetry_cache_age_seconds._value.get()
    assert 0 <= age < 5

    # 테스트 후 정리
    telemetry_cache.clear()
    lidar_service._latest_distance = None
