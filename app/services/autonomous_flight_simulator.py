# =============================================
# app/services/autonomous_flight_simulator.py
# 역할: L3 자율비행 미션 시뮬레이터 (Gazebo 없는 환경용 backend 폴백).
#       도면 벽체 → boustrophedon(왕복 격자) 커버리지 경로 생성 →
#       WebSocket(mission:{id})으로 진행 좌표 스트리밍.
#   - run_autonomous_scan(): 미션 시작 (백그라운드 asyncio task)
#   - get_mission / list_active_missions / cancel_mission: 상태 관리
#   - 메모리 기반 (프로세스 재시작 시 초기화)
# =============================================

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

from app.core.ws_manager import ws_manager

# boustrophedon 레인 간격(m) — 카메라 FOV 커버 가정
DEFAULT_LANE_SPACING_M = 2.0
TICK_INTERVAL_S = 0.5          # 좌표 emit 주기
POINTS_PER_TICK = 1           # tick당 진행 waypoint 수


@dataclass
class MissionState:
    mission_id: str
    status: str = "running"        # running | completed | cancelled | error
    progress: float = 0.0          # 0.0 ~ 1.0
    points_emitted: int = 0
    floorplan_id: Optional[str] = None
    started_at: float = field(default_factory=time.time)
    ended_at: Optional[float] = None
    waypoints: List[Dict[str, float]] = field(default_factory=list)
    _cancel: bool = False


# 미션 레지스트리 (메모리)
_missions: Dict[str, MissionState] = {}
_tasks: Dict[str, asyncio.Task] = {}


def _plan_boustrophedon(
    world_w: float, world_d: float, altitude: float,
    lane_spacing: float = DEFAULT_LANE_SPACING_M,
) -> List[Dict[str, float]]:
    """왕복 격자(보행렬) 커버리지 경로 waypoint 생성."""
    if world_w <= 0 or world_d <= 0:
        return []
    n_lanes = max(int(world_d / lane_spacing), 1)
    pts: List[Dict[str, float]] = []
    for i in range(n_lanes + 1):
        y = min(i * lane_spacing, world_d)
        # 짝수 레인 좌→우, 홀수 레인 우→좌 (왕복)
        xs = (0.0, world_w) if i % 2 == 0 else (world_w, 0.0)
        pts.append({"x": round(xs[0], 3), "y": round(y, 3), "z": round(altitude, 3)})
        pts.append({"x": round(xs[1], 3), "y": round(y, 3), "z": round(altitude, 3)})
    return pts


async def _run_mission(state: MissionState, waypoints: List[Dict[str, float]]) -> None:
    """미션 실행 루프 — waypoint를 순차 emit (WebSocket mission:{id})."""
    channel = f"mission:{state.mission_id}"
    total = len(waypoints) or 1
    try:
        await ws_manager.broadcast(channel, {
            "type": "mission_start",
            "mission_id": state.mission_id,
            "total_waypoints": total,
            "floorplan_id": state.floorplan_id,
        })
        for idx, wp in enumerate(waypoints):
            if state._cancel:
                state.status = "cancelled"
                break
            state.points_emitted = idx + 1
            state.progress = (idx + 1) / total
            await ws_manager.broadcast(channel, {
                "type": "waypoint",
                "mission_id": state.mission_id,
                "index": idx,
                "point": wp,
                "progress": round(state.progress, 4),
            })
            await asyncio.sleep(TICK_INTERVAL_S)
        if not state._cancel:
            state.status = "completed"
            state.progress = 1.0
    except Exception as e:  # noqa: BLE001
        state.status = "error"
        try:
            await ws_manager.broadcast(channel, {
                "type": "mission_error", "mission_id": state.mission_id, "error": str(e)[:200],
            })
        except Exception:
            pass
    finally:
        state.ended_at = time.time()
        try:
            await ws_manager.broadcast(channel, {
                "type": "mission_end",
                "mission_id": state.mission_id,
                "status": state.status,
                "points_emitted": state.points_emitted,
            })
        except Exception:
            pass


async def run_autonomous_scan(
    walls: List[Dict[str, float]],
    outline: Optional[List[Dict[str, float]]],
    world_w: float,
    world_d: float,
    floorplan_id: Optional[str] = None,
    altitude: float = 1.5,
    speed: float = 1.0,
) -> str:
    """자율 스캔 미션 시작 → mission_id 반환. 백그라운드 task로 경로 emit."""
    mission_id = uuid.uuid4().hex[:12]
    waypoints = _plan_boustrophedon(world_w, world_d, altitude)
    state = MissionState(
        mission_id=mission_id, floorplan_id=floorplan_id, waypoints=waypoints,
    )
    _missions[mission_id] = state
    # 백그라운드 실행 (요청 응답을 막지 않음)
    _tasks[mission_id] = asyncio.create_task(_run_mission(state, waypoints))
    return mission_id


def get_mission(mission_id: str) -> Optional[MissionState]:
    return _missions.get(mission_id)


def list_active_missions() -> List[Dict[str, Any]]:
    return [
        {
            "mission_id": m.mission_id,
            "status": m.status,
            "progress": round(m.progress, 4),
            "points_emitted": m.points_emitted,
            "floorplan_id": m.floorplan_id,
            "started_at": m.started_at,
            "ended_at": m.ended_at,
        }
        for m in _missions.values()
    ]


def cancel_mission(mission_id: str) -> bool:
    """미션 취소 요청 → 다음 tick에서 cancelled 전이. 종료된 미션은 False."""
    m = _missions.get(mission_id)
    if not m or m.status != "running":
        return False
    m._cancel = True
    return True
