# =============================================
# app/schemas/floorplan.py
# 역할: 평면도 업로드/처리 Pydantic 스키마
# =============================================

from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, Field


class FloorplanUploadResponse(BaseModel):
    """평면도 업로드 응답"""
    id: UUID
    filename: str
    content_type: str
    status: str  # uploaded / processing / completed / failed
    created_at: datetime

    class Config:
        from_attributes = True


class FloorplanProcessResponse(BaseModel):
    """OpenCV 처리 결과 응답"""
    id: UUID
    filename: str
    status: str
    wall_count: Optional[int] = Field(None, description="추출된 벽체 라인 수")
    walls: Optional[list] = Field(None, description="벽체 좌표 리스트 [{x1,y1,x2,y2}, ...]")
    gazebo_world: Optional[str] = Field(None, description="생성된 .world 파일 경로")


class FloorplanAnalyzeResponse(BaseModel):
    """벽체 추출 분석 결과 (Stateless — DB 불필요)"""
    walls: list[dict] = Field(default_factory=list, description="벽체 좌표 [{x1,y1,x2,y2}, ...] (0-1 정규화)")
    outline: list[dict] = Field(default_factory=list, description="건물 외곽 다각형 [{x,y}, ...] (0-1 정규화, 닫힘)")
    image_width: int
    image_height: int
    wall_count: int


class FloorplanListResponse(BaseModel):
    """평면도 목록 응답"""
    items: list[FloorplanUploadResponse]
    total: int


# ── FR-015 스케일 보정 ──────────────────────────
class FloorplanQualityCheckDetail(BaseModel):
    """개별 품질 체크 결과"""
    pass_check: bool = Field(..., alias="pass")
    value: Optional[object] = None
    message: str

    class Config:
        populate_by_name = True


class FloorplanValidateResponse(BaseModel):
    """도면 이미지 품질 검증 응답"""
    status: str = Field(..., description="ok | warning | rejected")
    score: float = Field(..., description="종합 점수 (0-100)")
    checks: dict = Field(default_factory=dict, description="항목별 검증 결과")
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class FloorplanCalibrateRequest(BaseModel):
    """
    평면도 위 두 점 + 실측 거리 → px/m 환산.
    p1, p2: 평면도 이미지 픽셀 좌표 (원본 해상도 기준)
    real_length_m: p1-p2 구간의 실제 길이(미터)
    """
    p1: list[float] = Field(..., min_length=2, max_length=2, description="[x, y] 픽셀 좌표")
    p2: list[float] = Field(..., min_length=2, max_length=2, description="[x, y] 픽셀 좌표")
    real_length_m: float = Field(..., gt=0, description="p1-p2 구간 실측 길이 (m)")


class FloorplanCalibrateResponse(BaseModel):
    """스케일 보정 결과"""
    id: UUID
    scale_px_per_meter: float = Field(..., description="1m 당 픽셀 수")
    pixel_length: float = Field(..., description="p1-p2 픽셀 거리 (검증용)")
    real_length_m: float
