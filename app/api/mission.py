# =============================================
# app/api/mission.py
# 역할: 자율비행 미션 제어 REST API + Pi 측 fc_bridge WebSocket
#       - POST /mission/start   : 미션 생성 + FSM 시작
#       - POST /mission/abort   : 미션 중단(LAND)
#       - POST /mission/estop   : 비상정지(즉시 LAND, 어느 phase 에서나)
#       - POST /mission/rtl     : Return-To-Launch (LAND 로 종료)
#       - POST /mission/pause   : PositionHold 강제
#       - GET  /mission/state   : 현재 인메모리 FSM 상태 + DB phase 동기
#       - GET  /mission/{id}/state : 특정 미션 DB 상태 조회
#       - WS   /mission/fc-bridge  : Pi Zero 가 reverse-WS 클라이언트로 접속
# 본 todo(4번)에서는 라우터 골격 + ORM 생성 + orchestrator 위임. FSM 본체 구현은 todo 3.
# =============================================
from __future__ import annotations

import uuid
from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, WebSocket, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.dependencies import get_current_org_member, get_db
from app.models.mission_plan import MissionPlan
from app.models.site import Site
from app.services.fc_bridge import fc_bridge
from app.services.mission_orchestrator import MissionPhase, mission_orchestrator
from app.services.path_planner import PlanParams

router = APIRouter()
logger = get_logger(__name__)


# ── 스키마 (inline — 추후 schemas/mission.py 분리 가능) ─
class StartRequest(BaseModel):
    site_id: UUID
    schedule_id: Optional[UUID] = None
    fov_h_deg: float = Field(default=80.0, ge=30.0, le=170.0)
    d_inspect_m: float = Field(default=1.5, ge=0.5, le=5.0)
    overlap: float = Field(default=0.30, ge=0.0, lt=0.95)
    # 사전모델(CAD/평면도) 룸 폴리곤(미터 단위). None이면 SLAM-only.
    # 있어도 VERIFICATION 단계에서 SLAM과 정합 비교 후 차이영역을 PATH_PLAN에 반영.
    # 추후 mission API 가 site_id 로 floorplan 모듈에서 자동 조회하도록 확장 예정(todo#5).
    prior_polygons: Optional[List[List[List[float]]]] = None


class StateResponse(BaseModel):
    mission_id: Optional[str]
    phase: str
    current_room_idx: Optional[int]
    failure_reason: Optional[str]
    fc_attached: bool
    verification: Optional[dict] = None
    captured_cells: int = 0


class AckResponse(BaseModel):
    ok: bool
    detail: str = ""


# ── REST 엔드포인트 ─────────────────────────
@router.post("/start", response_model=StateResponse, status_code=status.HTTP_201_CREATED)
async def start_mission(
    body: StartRequest,
    db: AsyncSession = Depends(get_db),
    current=Depends(get_current_org_member),
):
    # 현장 존재 검증 (멀티테넌트 격리는 dependencies 쪽에서 처리 가정)
    result = await db.execute(select(Site).where(Site.id == body.site_id))
    site = result.scalar_one_or_none()
    if site is None:
        raise HTTPException(status_code=404, detail="site_not_found")

    # 자동 주입: 사전모델/창호/공급면적
    # body.prior_polygons 가 명시되면 그대로 사용. 없으면 floorplan 모듈에서 자동 조회.
    auto_prior_polygons = body.prior_polygons
    auto_window_polygons = None
    if auto_prior_polygons is None:
        auto_prior_polygons, auto_window_polygons = await _autoload_floorplan(
            db=db, site_id=body.site_id,
        )
    supplied_area_m2 = float(site.total_area) if getattr(site, "total_area", None) else None

    # 기존 진행 중 미션 차단 — 단일 드론 가정
    if mission_orchestrator.state.phase is not MissionPhase.IDLE:
        raise HTTPException(
            status_code=409,
            detail=f"mission_already_running:{mission_orchestrator.state.phase.value}",
        )

    # Pi 미접속 시 즉시 거부 (soft-fail) — 플랜 §11
    if not fc_bridge.is_attached:
        raise HTTPException(status_code=412, detail="fc_bridge_not_attached")

    mission_id = uuid.uuid4()
    plan_row = MissionPlan(
        id=mission_id,
        site_id=body.site_id,
        schedule_id=body.schedule_id,
        operator_user_id=getattr(current, "user_id", None),
        status="running",
        current_phase="idle",
        params_json={
            "fov_h_deg": body.fov_h_deg,
            "d_inspect_m": body.d_inspect_m,
            "overlap": body.overlap,
        },
    )
    db.add(plan_row)
    await db.commit()

    try:
        await mission_orchestrator.start(
            mission_id=str(mission_id),
            params=PlanParams(
                fov_h_deg=body.fov_h_deg,
                d_inspect_m=body.d_inspect_m,
                overlap=body.overlap,
            ),
            prior_polygons=auto_prior_polygons,
            window_polygons_per_room=auto_window_polygons,
            supplied_area_m2=supplied_area_m2,
        )
    except RuntimeError as e:
        # orchestrator 가 거부 → DB 상태 정리
        plan_row.status = "aborted"
        plan_row.failure_reason = str(e)
        await db.commit()
        raise HTTPException(status_code=412, detail=f"orchestrator_refused:{e}")

    return _state_response()


@router.post("/abort", response_model=AckResponse)
async def abort_mission(_=Depends(get_current_org_member)):
    await mission_orchestrator.abort("user_abort")
    return AckResponse(ok=True, detail="abort_requested")


@router.post("/estop", response_model=AckResponse)
async def estop_mission(_=Depends(get_current_org_member)):
    """비상정지. 어느 phase 에서도 즉시 LAND. 인증된 직원이면 누구나 누를 수 있어야 함."""
    await mission_orchestrator.estop()
    return AckResponse(ok=True, detail="estop_requested")


@router.post("/rtl", response_model=AckResponse)
async def rtl_mission(_=Depends(get_current_org_member)):
    await mission_orchestrator.rtl()
    return AckResponse(ok=True, detail="rtl_requested")


@router.post("/pause", response_model=AckResponse)
async def pause_mission(_=Depends(get_current_org_member)):
    await mission_orchestrator.pause()
    return AckResponse(ok=True, detail="paused")


@router.get("/state", response_model=StateResponse)
async def current_state(_=Depends(get_current_org_member)):
    return _state_response()


@router.get("/{mission_id}/state", response_model=StateResponse)
async def mission_state(
    mission_id: UUID,
    db: AsyncSession = Depends(get_db),
    _=Depends(get_current_org_member),
):
    result = await db.execute(select(MissionPlan).where(MissionPlan.id == mission_id))
    row = result.scalar_one_or_none()
    if row is None:
        raise HTTPException(status_code=404, detail="mission_not_found")
    return StateResponse(
        mission_id=str(row.id),
        phase=str(row.current_phase),
        current_room_idx=row.current_room_idx,
        failure_reason=row.failure_reason,
        fc_attached=fc_bridge.is_attached,
    )


# ── Pi 측 fc_bridge WebSocket ───────────────
# 인증: 사내망 가정 + 간단 토큰 헤더 검증(추후 강화). 단일 Pi 만 attach.
@router.websocket("/fc-bridge")
async def pi_fc_bridge_ws(websocket: WebSocket):
    # TODO(todo#7 통합): 토큰 검증 (env: AEROINSPECT_PI_TOKEN)
    await fc_bridge.attach(websocket)


# ── 내부 유틸 ───────────────────────────────
async def _autoload_floorplan(
    db: AsyncSession,
    site_id: UUID,
) -> tuple[Optional[list], Optional[dict]]:
    """
    site_id 기반 사전모델 폴리곤·창호 폴리곤 자동 조회.

    Returns:
        (room_polygons, window_polygons_per_room)
        - room_polygons: List[List[List[float]]] (룸별 [x,y] 점들)
        - window_polygons_per_room: Dict[room_idx, List[List[List[float]]]]
        둘 다 floorplan 메타에서 추출 실패하면 None.

    TODO: floorplan 스키마(Floorplan.metadata_ JSONB) 의 정식 키 확정 후 본 구현.
          현재는 안전한 폴백(None) — mission_orchestrator 가 사전모델 없음으로 진행.
    """
    try:
        from app.models.floorplan import Floorplan
        result = await db.execute(
            select(Floorplan).where(Floorplan.site_id == site_id).limit(1)
        )
        fp = result.scalar_one_or_none()
        if fp is None:
            return None, None
        meta = getattr(fp, "metadata_", None) or getattr(fp, "metadata", None)
        if not isinstance(meta, dict):
            return None, None
        room_polygons = meta.get("room_polygons")
        window_polygons_per_room = meta.get("window_polygons_per_room")
        # 형 변환 검증
        if room_polygons and isinstance(room_polygons, list):
            return room_polygons, window_polygons_per_room
        return None, None
    except Exception:
        return None, None


def _state_response() -> StateResponse:
    s = mission_orchestrator.get_state()
    return StateResponse(
        mission_id=s["mission_id"],
        phase=s["phase"],
        current_room_idx=s["current_room_idx"],
        failure_reason=s["failure_reason"],
        fc_attached=s["fc_attached"],
        verification=s.get("verification"),
        captured_cells=s.get("captured_cells", 0),
    )
