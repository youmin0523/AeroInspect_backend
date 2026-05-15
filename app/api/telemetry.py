# =============================================
# app/api/telemetry.py
# 역할: 드론 좌표/센서 로그 API
#       - POST /telemetry       → 텔레메트리 저장 + WS Push
#       - GET  /telemetry       → 텔레메트리 목록 조회
#       - GET  /telemetry/latest → 최신 텔레메트리 조회
# =============================================

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select, func, desc
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import get_current_user, get_db, get_ws_manager, verify_ai_webhook
from app.models.telemetry import TelemetryLog
from app.schemas.telemetry import (
    TelemetryCreate,
    TelemetryResponse,
    TelemetryListResponse,
)
from app.core.ws_manager import ConnectionManager
from app.services.lidar import lidar_service
from app.services.telemetry_cache import telemetry_cache

router = APIRouter()


@router.get("/latest", response_model=Optional[TelemetryResponse])
async def get_latest_telemetry(
    db: AsyncSession = Depends(get_db),
    _user=Depends(get_current_user),
):
    """최신 드론 텔레메트리 1건 조회"""
    result = await db.execute(
        select(TelemetryLog).order_by(desc(TelemetryLog.timestamp)).limit(1)
    )
    row = result.scalar_one_or_none()
    if not row:
        raise HTTPException(status_code=404, detail="텔레메트리 데이터가 없습니다.")
    return TelemetryResponse.model_validate(row)


@router.get("", response_model=TelemetryListResponse)
async def list_telemetry(
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
    _user=Depends(get_current_user),
):
    """텔레메트리 로그 목록 조회 (최신순)"""
    query = select(TelemetryLog).order_by(desc(TelemetryLog.timestamp))

    count_query = select(func.count()).select_from(TelemetryLog)
    total = await db.scalar(count_query)

    result = await db.execute(query.offset(offset).limit(limit))
    items = result.scalars().all()

    return TelemetryListResponse(
        items=[TelemetryResponse.model_validate(item) for item in items],
        total=total or 0,
        limit=limit,
        offset=offset,
    )


@router.post("", response_model=TelemetryResponse, status_code=201)
async def create_telemetry(
    payload: TelemetryCreate,
    db: AsyncSession = Depends(get_db),
    manager: ConnectionManager = Depends(get_ws_manager),
    _wh=Depends(verify_ai_webhook),
):
    """
    드론 텔레메트리 저장 + WebSocket 'telemetry' 채널로 실시간 Push.
    ROS2 브릿지 또는 MAVLink 파서에서 주기적으로 호출.

    인증: X-AI-Webhook-Secret 헤더 필수 (내부 서비스 토큰).
    ROS2 브릿지/MAVLink 파서가 settings.AI_WEBHOOK_SECRET 와 동일한 시크릿을 보내야 함.

    추가 동작:
      - telemetry_cache 갱신 (stream_inference가 프레임 캡처 시점에 snapshot)
      - LiDAR 서비스에 드론 자세 전파 (3D 좌표 계산용)
    """
    log = TelemetryLog(
        site_id=payload.site_id,
        pos_x=payload.pos_x,
        pos_y=payload.pos_y,
        pos_z=payload.pos_z,
        roll=payload.roll,
        pitch=payload.pitch,
        yaw=payload.yaw,
        vel_x=payload.vel_x,
        vel_y=payload.vel_y,
        vel_z=payload.vel_z,
        battery_percent=payload.battery_percent,
        flight_mode=payload.flight_mode,
        is_armed=payload.is_armed,
        lidar_distance=payload.lidar_distance,
        sensor_status=payload.sensor_status,
    )

    db.add(log)
    await db.flush()

    # 메모리 캐시 갱신 (실시간 추론 경로에서 DB 조회 없이 O(1) 접근)
    await telemetry_cache.update(
        pos_x=payload.pos_x,
        pos_y=payload.pos_y,
        pos_z=payload.pos_z,
        roll=payload.roll,
        pitch=payload.pitch,
        yaw=payload.yaw,
        lidar_distance=payload.lidar_distance,
    )

    # LiDAR 서비스에 자세 전파 (compute_3d_position에서 사용)
    if payload.roll is not None and payload.pitch is not None and payload.yaw is not None:
        lidar_service.update_attitude(payload.roll, payload.pitch, payload.yaw)

    response = TelemetryResponse.model_validate(log)

    # WS "telemetry" 채널에 실시간 브로드캐스트
    await manager.broadcast("telemetry", {
        "type": "telemetry.update",
        "data": response.model_dump(mode="json"),
    })

    return response
