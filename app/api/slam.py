# =============================================
# app/api/slam.py
# 역할: SLAM 맵 데이터 API
#       - POST   /slam          → 새 맵 세션 생성
#       - GET    /slam          → 맵 목록 조회 (메타데이터만)
#       - GET    /slam/{id}     → 맵 상세 조회 (이미지 포함)
#       - PATCH  /slam/{id}     → 맵 업데이트 (실시간 매핑 중 갱신)
#       - DELETE /slam/{id}     → 맵 삭제
# =============================================

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select, func, desc
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import get_current_user, get_db, get_ws_manager
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


@router.get("", response_model=SlamMapListResponse)
async def list_slam_maps(
    db: AsyncSession = Depends(get_db),
    _user=Depends(get_current_user),  # TODO: 조직별 SLAM 분리 시 get_current_org_member로 교체
):
    """SLAM 맵 목록 조회 (이미지 제외, 메타데이터만)"""
    query = select(SlamMap).order_by(desc(SlamMap.created_at))

    total = await db.scalar(select(func.count()).select_from(SlamMap))
    result = await db.execute(query)
    items = result.scalars().all()

    return SlamMapListResponse(
        items=[SlamMapMetaResponse.model_validate(item) for item in items],
        total=total or 0,
    )


@router.get("/{map_id}", response_model=SlamMapResponse)
async def get_slam_map(
    map_id: UUID,
    db: AsyncSession = Depends(get_db),
    _user=Depends(get_current_user),
):
    """SLAM 맵 상세 조회 (이미지 포함)"""
    result = await db.execute(select(SlamMap).where(SlamMap.id == map_id))
    slam_map = result.scalar_one_or_none()
    if not slam_map:
        raise HTTPException(status_code=404, detail="SLAM 맵을 찾을 수 없습니다.")
    return SlamMapResponse.model_validate(slam_map)


@router.post("", response_model=SlamMapResponse, status_code=201)
async def create_slam_map(
    payload: SlamMapCreate,
    db: AsyncSession = Depends(get_db),
    manager: ConnectionManager = Depends(get_ws_manager),
    _user=Depends(get_current_user),
):
    """새 SLAM 맵 세션 생성"""
    slam_map = SlamMap(
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
    _user=Depends(get_current_user),
):
    """
    SLAM 맵 업데이트 (실시간 매핑 중 지도 이미지 갱신).
    SLAM 노드에서 주기적으로 호출하여 웹 미니맵에 반영.
    """
    result = await db.execute(select(SlamMap).where(SlamMap.id == map_id))
    slam_map = result.scalar_one_or_none()
    if not slam_map:
        raise HTTPException(status_code=404, detail="SLAM 맵을 찾을 수 없습니다.")

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
    _user=Depends(get_current_user),
):
    """SLAM 맵 삭제"""
    result = await db.execute(select(SlamMap).where(SlamMap.id == map_id))
    slam_map = result.scalar_one_or_none()
    if not slam_map:
        raise HTTPException(status_code=404, detail="SLAM 맵을 찾을 수 없습니다.")
    await db.delete(slam_map)
