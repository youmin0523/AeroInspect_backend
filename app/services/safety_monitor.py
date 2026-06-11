# =============================================
# app/services/safety_monitor.py
# 역할: 자율비행 안전 단일 진입점
#       - 배터리/통신/SLAM/지오펜스/E-stop 임계 검사
#       - 임계 위반 시 SafetyDecision 발행 → mission_orchestrator 가 즉시 LAND/RTL 강제
#       - 모든 안전 결정은 본 모듈을 통해서만 발생 (분산 의사결정 금지)
# =============================================
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Awaitable, Callable, List, Optional

from app.core.logging import get_logger

logger = get_logger(__name__)


# ── 임계값 (플랜 §7.5 / §9 / §11) ─────────────────
class Thresholds:
    BATTERY_PCT_RTL = 35.0
    VOLTAGE_PER_CELL_RTL = 3.55  # 4S 기준 셀당
    COMM_LOSS_SEC = 2.0
    SLAM_POS_VAR_M = 0.5
    GEOFENCE_MARGIN_M = 0.5
    OBSTACLE_THRESHOLD_M = 0.35   # 76mm 프롭 외경 22cm + 마진 15cm 환산
    TILT_LIMIT_DEG = 60.0


class SafetyAction(str, Enum):
    CONTINUE = "continue"
    HOVER = "hover"
    RTL = "rtl"
    LAND = "land"
    ESTOP = "estop"


@dataclass
class TelemetrySnapshot:
    battery_pct: Optional[float]
    voltage_per_cell: Optional[float]
    tilt_deg: Optional[float]
    pos_var_m: Optional[float]
    last_telemetry_ts: float          # epoch sec
    last_pi_heartbeat_ts: float
    inside_geofence: bool
    obstacle_min_dist_m: Optional[float]
    slam_confidence: Optional[float]


@dataclass
class SafetyDecision:
    action: SafetyAction
    reason: str
    detail: dict = field(default_factory=dict)


SafetyCallback = Callable[[SafetyDecision], Awaitable[None]]


class SafetyMonitor:
    """단일 인스턴스(싱글톤). MissionOrchestrator 가 생성 후 콜백 등록."""

    def __init__(self) -> None:
        self._callbacks: List[SafetyCallback] = []
        self._estop_flag = False
        self._lock = asyncio.Lock()

    def register(self, cb: SafetyCallback) -> None:
        self._callbacks.append(cb)

    def trigger_estop(self, reason: str = "user-estop") -> None:
        """UI/외부 트리거. 다음 check() 호출에서 즉시 ESTOP 결정."""
        self._estop_flag = True
        logger.warning("safety.estop_triggered", reason=reason)

    async def check(self, snap: TelemetrySnapshot) -> SafetyDecision:
        """
        매 100~500ms 호출. 가장 위험도 높은 결정 1개 반환.
        E-STOP > LAND > RTL > HOVER > CONTINUE 우선순위.
        """
        if self._estop_flag:
            return await self._emit(SafetyDecision(SafetyAction.ESTOP, "estop_flag"))

        now = time.time()

        # comm-loss
        if now - snap.last_pi_heartbeat_ts > Thresholds.COMM_LOSS_SEC:
            return await self._emit(SafetyDecision(
                SafetyAction.LAND, "comm_loss",
                {"gap_sec": now - snap.last_pi_heartbeat_ts},
            ))

        # geofence
        if not snap.inside_geofence:
            return await self._emit(SafetyDecision(SafetyAction.RTL, "geofence_breach"))

        # battery
        if snap.battery_pct is not None and snap.battery_pct < Thresholds.BATTERY_PCT_RTL:
            return await self._emit(SafetyDecision(
                SafetyAction.RTL, "battery_low",
                {"battery_pct": snap.battery_pct},
            ))
        if snap.voltage_per_cell is not None and snap.voltage_per_cell < Thresholds.VOLTAGE_PER_CELL_RTL:
            return await self._emit(SafetyDecision(
                SafetyAction.RTL, "voltage_low",
                {"v_per_cell": snap.voltage_per_cell},
            ))

        # tilt anomaly
        if snap.tilt_deg is not None and abs(snap.tilt_deg) > Thresholds.TILT_LIMIT_DEG:
            return await self._emit(SafetyDecision(
                SafetyAction.LAND, "tilt_anomaly", {"tilt_deg": snap.tilt_deg},
            ))

        # SLAM 신뢰도 / 위치 분산
        if snap.pos_var_m is not None and snap.pos_var_m > Thresholds.SLAM_POS_VAR_M:
            return await self._emit(SafetyDecision(
                SafetyAction.HOVER, "slam_pos_var_high",
                {"pos_var_m": snap.pos_var_m},
            ))

        # 장애물 안전회랑 침입
        if (
            snap.obstacle_min_dist_m is not None
            and snap.obstacle_min_dist_m < Thresholds.OBSTACLE_THRESHOLD_M
        ):
            return await self._emit(SafetyDecision(
                SafetyAction.HOVER, "obstacle_corridor_breach",
                {"min_dist_m": snap.obstacle_min_dist_m},
            ))

        return SafetyDecision(SafetyAction.CONTINUE, "ok")

    async def _emit(self, decision: SafetyDecision) -> SafetyDecision:
        async with self._lock:
            for cb in list(self._callbacks):
                try:
                    await cb(decision)
                except Exception as e:
                    logger.error("safety.callback_failed", error=str(e), action=decision.action.value)
        return decision


# ── 싱글톤 인스턴스 ──────────────────────────────
safety_monitor = SafetyMonitor()
