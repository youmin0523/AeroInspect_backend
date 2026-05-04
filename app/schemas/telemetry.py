# =============================================
# app/schemas/telemetry.py
# 역할: 드론 텔레메트리 Pydantic 입출력 스키마
# =============================================

from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, Field


class TelemetryCreate(BaseModel):
    """드론 텔레메트리 데이터 저장 요청"""
    site_id: Optional[UUID] = Field(None, description="연결 현장 ID (없으면 전역 기록)")
    pos_x: float = Field(..., description="드론 X 좌표 (m)")
    pos_y: float = Field(..., description="드론 Y 좌표 (m)")
    pos_z: float = Field(..., description="드론 Z 좌표 / 고도 (m)")
    roll: Optional[float] = None
    pitch: Optional[float] = None
    yaw: Optional[float] = None
    vel_x: Optional[float] = None
    vel_y: Optional[float] = None
    vel_z: Optional[float] = None
    battery_percent: Optional[float] = Field(None, ge=0, le=100)
    flight_mode: Optional[str] = None
    is_armed: bool = False
    lidar_distance: Optional[float] = None
    sensor_status: Optional[dict] = None


class TelemetryResponse(BaseModel):
    """드론 텔레메트리 응답"""
    id: UUID
    site_id: Optional[UUID] = None
    pos_x: float
    pos_y: float
    pos_z: float
    roll: Optional[float]
    pitch: Optional[float]
    yaw: Optional[float]
    vel_x: Optional[float]
    vel_y: Optional[float]
    vel_z: Optional[float]
    battery_percent: Optional[float]
    flight_mode: Optional[str]
    is_armed: bool
    lidar_distance: Optional[float]
    sensor_status: Optional[dict]
    timestamp: datetime

    class Config:
        from_attributes = True


class TelemetryListResponse(BaseModel):
    """텔레메트리 목록 페이지네이션 응답"""
    items: list[TelemetryResponse]
    total: int
    limit: int
    offset: int
