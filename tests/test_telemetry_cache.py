# =============================================
# tests/test_telemetry_cache.py
# 역할: TelemetryCache 싱글톤 동작 검증
#       - 초기 상태, update, snapshot, stale 판정, clear
# 실행: pytest tests/test_telemetry_cache.py -v
# =============================================

from __future__ import annotations

import time

import pytest

from app.services.telemetry_cache import DronePose, TelemetryCache


@pytest.fixture
def cache():
    """각 테스트마다 독립된 캐시 인스턴스."""
    return TelemetryCache()


class TestTelemetryCache:
    def test_initial_state_is_empty(self, cache):
        assert cache.is_ready is False
        assert cache.snapshot() is None
        assert cache.snapshot_fresh() is None
        assert cache.age_sec is None

    @pytest.mark.asyncio
    async def test_update_sets_pose(self, cache):
        await cache.update(pos_x=1.0, pos_y=2.0, pos_z=3.0)
        assert cache.is_ready is True
        pose = cache.snapshot()
        assert pose is not None
        assert pose.pos_x == 1.0
        assert pose.pos_y == 2.0
        assert pose.pos_z == 3.0
        assert pose.has_attitude is False  # roll/pitch/yaw 없음

    @pytest.mark.asyncio
    async def test_update_with_attitude(self, cache):
        await cache.update(
            pos_x=0.0, pos_y=0.0, pos_z=1.5,
            roll=0.1, pitch=0.2, yaw=0.3,
            lidar_distance=2.0,
        )
        pose = cache.snapshot()
        assert pose.has_attitude is True
        assert pose.lidar_distance == 2.0

    @pytest.mark.asyncio
    async def test_update_overwrites_previous(self, cache):
        await cache.update(pos_x=1.0, pos_y=1.0, pos_z=1.0)
        await cache.update(pos_x=9.0, pos_y=9.0, pos_z=9.0)
        pose = cache.snapshot()
        assert pose.pos_x == 9.0

    @pytest.mark.asyncio
    async def test_snapshot_fresh_returns_none_when_stale(self, cache):
        await cache.update(pos_x=0.0, pos_y=0.0, pos_z=0.0)
        # 강제로 타임스탬프 조작 (threshold=5초 초과)
        old_pose = cache._pose
        cache._pose = DronePose(
            pos_x=old_pose.pos_x,
            pos_y=old_pose.pos_y,
            pos_z=old_pose.pos_z,
            received_at=time.time() - 100.0,
        )
        assert cache.snapshot_fresh() is None
        # snapshot()은 stale 여부 무시하고 반환
        assert cache.snapshot() is not None

    @pytest.mark.asyncio
    async def test_clear_resets(self, cache):
        await cache.update(pos_x=1.0, pos_y=2.0, pos_z=3.0)
        cache.clear()
        assert cache.is_ready is False
        assert cache.snapshot() is None

    @pytest.mark.asyncio
    async def test_age_sec_monotonic(self, cache):
        await cache.update(pos_x=0.0, pos_y=0.0, pos_z=0.0)
        age1 = cache.age_sec
        assert age1 is not None and age1 >= 0
        time.sleep(0.01)
        age2 = cache.age_sec
        assert age2 > age1
