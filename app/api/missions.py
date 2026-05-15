# =============================================
# app/api/missions.py
# 역할: L3 자율비행 + LiDAR 스캔 미션 제어 API
#       - POST /missions/autonomous-scan/start  → 미션 시작 (백그라운드 시뮬레이션)
#       - POST /missions/{id}/cancel            → 미션 취소
#       - GET  /missions/{id}                   → 미션 상태 조회
#       - GET  /missions                        → 활성 미션 목록
#
# 미션 데이터 흐름:
#   클라이언트 → POST /missions/autonomous-scan/start (walls + outline + world_size)
#   백엔드 시뮬레이터 → ws_manager.broadcast('defects', {type:'lidar.points', ...})
#   클라이언트 useWebSocket → droneStore.appendLidarPoints() → BuildingMesh L3 렌더
#
# Gazebo 환경에서는 이 시뮬레이터 대신 ros2 lidar 토픽 → ws_manager.broadcast 로 교체.
# =============================================

from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import get_current_org_member, get_current_user, get_db
from app.models.floorplan import Floorplan
from app.services.autonomous_flight_simulator import (
    cancel_mission,
    get_mission,
    list_active_missions,
    run_autonomous_scan,
)
from app.services.gazebo_world_generator import derive_world_size

router = APIRouter()


class AutonomousScanRequest(BaseModel):
    """
    L3 자율비행 미션 시작 요청.
    walls/outline 은 정규화 0-1 좌표. world_w/world_d 가 주어지면 우선 사용,
    아니면 image_width/height + scale_px_per_meter 로 산출, 둘 다 없으면 폴백 12×9.
    """
    floorplan_id: Optional[UUID] = Field(None, description="DB 의 평면도 id (있으면 우선 사용)")
    walls: Optional[list[dict]] = Field(None, description="벽체 좌표 (정규화 0-1) — floorplan_id 미지정 시 필수")
    outline: Optional[list[dict]] = Field(default_factory=list, description="외곽 다각형 (정규화 0-1)")
    # //* [Modified Code 2026-05-13] 가구 — 자율비행 충돌 회피용
    furniture: Optional[list[dict]] = Field(default_factory=list, description="가구 회전 사각형 (정규화 0-1) [{cx,cy,w,h,angle,label}, ...]")
    image_width: Optional[int] = None
    image_height: Optional[int] = None
    scale_px_per_meter: Optional[float] = None
    world_w: Optional[float] = None
    world_d: Optional[float] = None
    altitude: float = Field(1.5, gt=0, le=10, description="단일층 모드 고도 (altitude_layers 미지정 시 사용)")
    speed: float = Field(0.8, gt=0, le=3, description="비행 속도 (m/s)")
    # //* [Modified Code 2026-05-13 v3] 다층 sweep + lane spacing 옵션
    altitude_layers: Optional[list[float]] = Field(
        default=None,
        description="다층 비행 고도 리스트 (m). 예: [0.4, 1.5, 2.5] = 걸레받이/일반/천장. 미지정 시 (0.4, 1.5, 2.5)",
    )
    lane_spacing: float = Field(0.5, gt=0.05, le=3, description="격자 라인 간격 (m). 작을수록 빈틈 적음")
    ceiling_height: float = Field(2.7, gt=1.5, le=10, description="천장 높이 (m)")


class AutonomousScanResponse(BaseModel):
    mission_id: str
    walls_count: int
    furniture_count: int = 0
    world_w: float
    world_d: float
    size_source: str
    estimated_duration_s: float = Field(..., description="대략 예상 비행 시간(초)")


@router.post("/autonomous-scan/start", response_model=AutonomousScanResponse)
async def start_autonomous_scan(
    payload: AutonomousScanRequest,
    db: AsyncSession = Depends(get_db),
    org_tuple=Depends(get_current_org_member),
):
    """
    자율비행 시뮬레이션 시작.
    - floorplan_id 가 있으면 DB 에서 walls 를 가져온다 (현재 조직 소유분만).
    - 없으면 payload.walls 사용.
    - 백그라운드 태스크로 실행되고 즉시 mission_id 반환.
    - 진행은 WebSocket 'defects' 채널로 telemetry.update / lidar.points 이벤트 발행.
    """
    _user, _member, org = org_tuple
    walls = payload.walls
    outline = payload.outline or []
    furniture = payload.furniture or []
    image_width = payload.image_width
    image_height = payload.image_height
    scale_px_per_meter = payload.scale_px_per_meter
    floorplan_id_str: Optional[str] = None

    if payload.floorplan_id:
        floorplan_id_str = str(payload.floorplan_id)
        result = await db.execute(
            select(Floorplan)
            .where(Floorplan.id == payload.floorplan_id)
            .where(Floorplan.organization_id == org.id)
        )
        fp = result.scalar_one_or_none()
        if not fp:
            raise HTTPException(status_code=404, detail="평면도를 찾을 수 없습니다.")
        if not fp.walls_data:
            raise HTTPException(
                status_code=409,
                detail="평면도에 벽체 데이터가 없습니다. /floorplan/{id}/process 를 먼저 호출하세요.",
            )
        walls = fp.walls_data
        # //* [Modified Code 2026-05-13] DB 가구도 함께 로드 (충돌 회피용)
        if not furniture and fp.furniture_data:
            furniture = fp.furniture_data
        scale_px_per_meter = scale_px_per_meter or fp.scale_px_per_meter

    if not walls:
        raise HTTPException(
            status_code=400,
            detail="walls 가 필요합니다. floorplan_id 또는 payload.walls 중 하나를 제공하세요.",
        )

    # world 크기 결정
    if payload.world_w and payload.world_d:
        world_w, world_d, size_source = payload.world_w, payload.world_d, "explicit"
    else:
        world_w, world_d, size_source = derive_world_size(
            image_width, image_height, scale_px_per_meter,
        )

    # 다층 sweep 옵션 — 미지정 시 모듈 기본 (0.4 / 1.5 / 2.5)
    from app.services.autonomous_flight_simulator import DEFAULT_ALTITUDE_LAYERS
    layers = tuple(payload.altitude_layers) if payload.altitude_layers else DEFAULT_ALTITUDE_LAYERS

    mission_id = await run_autonomous_scan(
        walls=walls,
        outline=outline,
        world_w=world_w,
        world_d=world_d,
        floorplan_id=floorplan_id_str,
        altitude=payload.altitude,
        speed=payload.speed,
        furniture=furniture,
        altitude_layers=layers,
        lane_spacing=payload.lane_spacing,
        ceiling_height=payload.ceiling_height,
    )

    # 대략적인 비행 시간 예측: 격자 라인 길이 합 / 속도
    # boustrophedon: 라인 수 ≈ world_d / lane_spacing, 라인당 길이 ≈ world_w
    from app.services.autonomous_flight_simulator import DEFAULT_LANE_SPACING_M
    lanes = max(world_d / DEFAULT_LANE_SPACING_M, 1)
    total_len = lanes * world_w
    est = total_len / payload.speed

    return AutonomousScanResponse(
        mission_id=mission_id,
        walls_count=len(walls),
        furniture_count=len(furniture),
        world_w=round(world_w, 4),
        world_d=round(world_d, 4),
        size_source=size_source,
        estimated_duration_s=round(est, 1),
    )


class MissionStatusResponse(BaseModel):
    mission_id: str
    status: str
    progress: float = Field(..., description="0.0 ~ 1.0")
    points_emitted: int
    floorplan_id: Optional[str] = None
    started_at: float
    ended_at: Optional[float] = None


@router.get("/{mission_id}", response_model=MissionStatusResponse)
async def get_mission_status(
    mission_id: str,
    _user=Depends(get_current_user),
):
    """미션 상태 조회 (폴링 폴백 — 주된 진행은 WebSocket 으로)."""
    m = get_mission(mission_id)
    if not m:
        raise HTTPException(status_code=404, detail="미션을 찾을 수 없습니다.")
    return MissionStatusResponse(
        mission_id=m.mission_id,
        status=m.status,
        progress=round(m.progress, 4),
        points_emitted=m.points_emitted,
        floorplan_id=m.floorplan_id,
        started_at=m.started_at,
        ended_at=m.ended_at,
    )


@router.post("/{mission_id}/cancel", response_model=MissionStatusResponse)
async def cancel_mission_endpoint(
    mission_id: str,
    _user=Depends(get_current_user),
):
    """미션 취소 — 다음 tick 에서 cancelled 상태로 전이."""
    ok = cancel_mission(mission_id)
    if not ok:
        raise HTTPException(status_code=409, detail="미션을 취소할 수 없습니다 (이미 종료 또는 미존재).")
    m = get_mission(mission_id)
    return MissionStatusResponse(
        mission_id=m.mission_id,
        status=m.status,
        progress=round(m.progress, 4),
        points_emitted=m.points_emitted,
        floorplan_id=m.floorplan_id,
        started_at=m.started_at,
        ended_at=m.ended_at,
    )


@router.get("")
async def list_missions(_user=Depends(get_current_user)):
    """현재 알려진 미션 목록 (메모리 — 프로세스 재시작 시 초기화)."""
    return {"items": list_active_missions()}
