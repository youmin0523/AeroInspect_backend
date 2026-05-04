# =============================================
# app/services/telemetry_cache.py
# 역할: 최신 드론 pose(위치/자세) 메모리 캐시 싱글톤
#       - POST /telemetry 수신 시마다 갱신
#       - StreamInferenceWorker가 프레임 캡처 시점에 snapshot() 호출
#       - DB 조회 없이 O(1)로 최신값 제공 (실시간 추론 경로 블로킹 방지)
#
# 스레드 안전성: asyncio.Lock 사용. FastAPI는 단일 이벤트 루프이므로
# 사실상 critical section은 dict 쓰기 한 번 — 경합 거의 없음.
# =============================================

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, replace
from typing import Optional


@dataclass(frozen=True)
class DronePose:
    """단일 시점의 드론 상태 스냅샷."""
    pos_x: float
    pos_y: float
    pos_z: float
    roll: Optional[float] = None
    pitch: Optional[float] = None
    yaw: Optional[float] = None
    lidar_distance: Optional[float] = None
    received_at: float = 0.0  # epoch seconds

    @property
    def has_attitude(self) -> bool:
        return self.roll is not None and self.pitch is not None and self.yaw is not None


class TelemetryCache:
    """
    최신 드론 pose 캐시. 모듈 레벨 싱글톤.
    """

    STALE_THRESHOLD_SEC = 5.0  # 5초 이상 미갱신 시 stale 간주

    def __init__(self):
        self._pose: Optional[DronePose] = None
        self._lock = asyncio.Lock()

    async def update(
        self,
        pos_x: float,
        pos_y: float,
        pos_z: float,
        roll: Optional[float] = None,
        pitch: Optional[float] = None,
        yaw: Optional[float] = None,
        lidar_distance: Optional[float] = None,
    ) -> None:
        """텔레메트리 수신 시마다 호출. O(1)."""
        pose = DronePose(
            pos_x=pos_x,
            pos_y=pos_y,
            pos_z=pos_z,
            roll=roll,
            pitch=pitch,
            yaw=yaw,
            lidar_distance=lidar_distance,
            received_at=time.time(),
        )
        async with self._lock:
            self._pose = pose

    def snapshot(self) -> Optional[DronePose]:
        """동기 호출. 최신 pose 복사본 반환 (없으면 None)."""
        return self._pose

    def snapshot_fresh(self) -> Optional[DronePose]:
        """stale(오래된) pose는 None 반환. 좌표 정확도가 중요할 때 사용."""
        p = self._pose
        if p is None:
            return None
        if (time.time() - p.received_at) > self.STALE_THRESHOLD_SEC:
            return None
        return p

    def clear(self) -> None:
        """테스트/종료용."""
        self._pose = None

    @property
    def is_ready(self) -> bool:
        return self._pose is not None

    @property
    def age_sec(self) -> Optional[float]:
        if self._pose is None:
            return None
        return time.time() - self._pose.received_at


# ── 모듈 레벨 싱글톤 ─────────────────────────
telemetry_cache = TelemetryCache()


__all__ = ["TelemetryCache", "DronePose", "telemetry_cache"]
