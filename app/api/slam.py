# =============================================
# app/api/slam.py
# 역할: SLAM 맵 데이터 API (조직 단위 격리)
#       - POST   /slam          → 새 맵 세션 생성 (소유 조직 자동 기록)
#       - GET    /slam          → 맵 목록 조회 (메타데이터만, 현재 조직 분만)
#       - GET    /slam/{id}     → 맵 상세 조회 (이미지 포함)
#       - PATCH  /slam/{id}     → 맵 업데이트 (실시간 매핑 중 갱신)
#       - DELETE /slam/{id}     → 맵 삭제
# =============================================

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select, func, desc
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import get_current_org_member, get_db, get_ws_manager
from app.models.slam_map import SlamMap
from app.schemas.slam_map import (
    SlamMapCreate,
    SlamMapUpdate,
    SlamMapResponse,
    SlamMapMetaResponse,
    SlamMapListResponse,
)
from app.core.ws_manager import ConnectionManager

router = APIRouter()


async def _get_org_slam_map(db: AsyncSession, org_id, map_id: UUID) -> SlamMap:
    """현재 조직 소유의 SLAM 맵만 조회. 없거나 타 조직 것이면 404."""
    result = await db.execute(
        select(SlamMap)
        .where(SlamMap.id == map_id)
        .where(SlamMap.organization_id == org_id)
    )
    slam_map = result.scalar_one_or_none()
    if not slam_map:
        raise HTTPException(status_code=404, detail="SLAM 맵을 찾을 수 없습니다.")
    return slam_map


@router.get("", response_model=SlamMapListResponse)
async def list_slam_maps(
    db: AsyncSession = Depends(get_db),
    org_tuple=Depends(get_current_org_member),
):
    """SLAM 맵 목록 조회 (이미지 제외, 현재 조직 소유분만)."""
    _user, _member, org = org_tuple
    base = select(SlamMap).where(SlamMap.organization_id == org.id)
    total = await db.scalar(
        select(func.count()).select_from(base.subquery())
    )
    result = await db.execute(base.order_by(desc(SlamMap.created_at)))
    items = result.scalars().all()

    return SlamMapListResponse(
        items=[SlamMapMetaResponse.model_validate(item) for item in items],
        total=total or 0,
    )


@router.get("/{map_id}", response_model=SlamMapResponse)
async def get_slam_map(
    map_id: UUID,
    db: AsyncSession = Depends(get_db),
    org_tuple=Depends(get_current_org_member),
):
    """SLAM 맵 상세 조회 (이미지 포함, 현재 조직 소유분만)."""
    _user, _member, org = org_tuple
    slam_map = await _get_org_slam_map(db, org.id, map_id)
    return SlamMapResponse.model_validate(slam_map)


@router.post("", response_model=SlamMapResponse, status_code=201)
async def create_slam_map(
    payload: SlamMapCreate,
    db: AsyncSession = Depends(get_db),
    manager: ConnectionManager = Depends(get_ws_manager),
    org_tuple=Depends(get_current_org_member),
):
    """새 SLAM 맵 세션 생성 (소유 조직 자동 기록)."""
    _user, _member, org = org_tuple
    slam_map = SlamMap(
        organization_id=org.id,
        name=payload.name,
        resolution=payload.resolution,
        width=payload.width,
        height=payload.height,
        origin_x=payload.origin_x,
        origin_y=payload.origin_y,
        origin_yaw=payload.origin_yaw,
        map_image=payload.map_image,
        metadata_=payload.metadata_,
        status=payload.status,
    )

    db.add(slam_map)
    await db.flush()

    response = SlamMapResponse.model_validate(slam_map)

    # WS로 새 맵 생성 알림
    await manager.broadcast("telemetry", {
        "type": "slam.created",
        "data": {"id": str(slam_map.id), "name": slam_map.name},
    })

    return response


@router.patch("/{map_id}", response_model=SlamMapResponse)
async def update_slam_map(
    map_id: UUID,
    payload: SlamMapUpdate,
    db: AsyncSession = Depends(get_db),
    manager: ConnectionManager = Depends(get_ws_manager),
    org_tuple=Depends(get_current_org_member),
):
    """
    SLAM 맵 업데이트 (실시간 매핑 중 지도 이미지 갱신, 현재 조직 소유분만).
    SLAM 노드에서 주기적으로 호출하여 웹 미니맵에 반영.
    """
    _user, _member, org = org_tuple
    slam_map = await _get_org_slam_map(db, org.id, map_id)

    update_data = payload.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(slam_map, field, value)

    await db.flush()

    response = SlamMapResponse.model_validate(slam_map)

    # WS로 맵 업데이트 Push (프론트 미니맵 갱신용)
    await manager.broadcast("telemetry", {
        "type": "slam.updated",
        "data": {
            "id": str(slam_map.id),
            "status": slam_map.status,
            "width": slam_map.width,
            "height": slam_map.height,
        },
    })

    return response


@router.delete("/{map_id}", status_code=204)
async def delete_slam_map(
    map_id: UUID,
    db: AsyncSession = Depends(get_db),
    org_tuple=Depends(get_current_org_member),
):
    """SLAM 맵 삭제 (현재 조직 소유분만)."""
    _user, _member, org = org_tuple
    slam_map = await _get_org_slam_map(db, org.id, map_id)
    await db.delete(slam_map)
