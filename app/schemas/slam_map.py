# =============================================
# app/schemas/slam_map.py
# 역할: SLAM 맵 데이터 Pydantic 입출력 스키마
# =============================================

from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, Field


class SlamMapCreate(BaseModel):
    """SLAM 맵 스냅샷 저장 요청"""
    name: Optional[str] = Field(None, max_length=200)
    resolution: float = Field(..., description="격자 해상도 (m/pixel)")
    width: int = Field(..., description="맵 너비 (pixels)")
    height: int = Field(..., description="맵 높이 (pixels)")
    origin_x: float = 0.0
    origin_y: float = 0.0
    origin_yaw: float = 0.0
    map_image: str = Field(..., description="점유 격자 이미지 (Base64 PNG)")
    metadata_: Optional[dict] = Field(None, alias="metadata")
    status: str = Field(default="mapping", pattern="^(mapping|completed|failed)$")


class SlamMapUpdate(BaseModel):
    """SLAM 맵 업데이트 요청 (실시간 매핑 중 갱신)"""
    map_image: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    status: Optional[str] = Field(None, pattern="^(mapping|completed|failed)$")
    metadata_: Optional[dict] = Field(None, alias="metadata")


class SlamMapResponse(BaseModel):
    """SLAM 맵 응답"""
    id: UUID
    name: Optional[str]
    resolution: Optional[float]
    width: Optional[int]
    height: Optional[int]
    origin_x: Optional[float]
    origin_y: Optional[float]
    origin_yaw: Optional[float]
    map_image: Optional[str]
    status: Optional[str]
    created_at: datetime
    updated_at: Optional[datetime]

    class Config:
        from_attributes = True


class SlamMapMetaResponse(BaseModel):
    """SLAM 맵 메타데이터 응답 (목록 조회용, 이미지 제외)"""
    id: UUID
    name: Optional[str]
    resolution: Optional[float]
    width: Optional[int]
    height: Optional[int]
    status: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True


class SlamMapListResponse(BaseModel):
    """SLAM 맵 목록 응답 (이미지 제외, 메타데이터만)"""
    items: list[SlamMapMetaResponse]
    total: int
