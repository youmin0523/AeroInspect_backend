# =============================================
# app/schemas/site.py
# 역할: 현장(Site) Pydantic 입출력 스키마 정의
#       - SiteCreate: POST 요청 시 입력 검증
#       - SiteUpdate: PATCH 요청 시 부분 업데이트
#       - SiteResponse: GET 응답 직렬화
#       - SiteListResponse: 페이지네이션 래퍼
# =============================================

from datetime import date, datetime
from typing import Optional, List
from uuid import UUID

from pydantic import BaseModel, Field


# ── 중첩 모델 ───────────────────────────────

class AssignedMember(BaseModel):
    """배정 팀원"""
    id: str
    name: str
    role: str


class Recording(BaseModel):
    """촬영 영상 메타"""
    id: str
    date: str
    type: str
    duration_sec: int
    url: Optional[str] = None


# ── 입력 스키마 ──────────────────────────────

class SiteCreate(BaseModel):
    """POST /api/v1/sites — 새 현장 등록"""
    name: str = Field(..., max_length=200)
    inspection_type: Optional[str] = Field(default="사전점검")
    address: Optional[str] = Field(None, max_length=500)
    building_type: str = Field(default="아파트")
    total_area: Optional[float] = None
    building_count: Optional[int] = None
    unit_count: Optional[int] = None
    client_type: str = Field(default="B2B")
    client_name: Optional[str] = Field(None, max_length=200)
    client_contact: Optional[str] = Field(None, max_length=100)
    contract_start: Optional[date] = None
    contract_end: Optional[date] = None
    status: str = Field(default="pending")
    assigned_members: Optional[List[AssignedMember]] = []
    memo: Optional[str] = None

    model_config = {
        "json_schema_extra": {
            "example": {
                "name": "래미안 송파 101동",
                "inspection_type": "사전점검",
                "address": "서울시 송파구 OO로 123",
                "building_type": "아파트",
                "total_area": 84.5,
                "building_count": 1,
                "unit_count": 24,
                "client_type": "B2B",
                "client_name": "삼성물산 건설부문",
                "client_contact": "010-1234-5678",
                "contract_start": "2026-05-10",
                "contract_end": "2026-06-30",
                "status": "pending",
                "memo": "입주 전 사전점검 — 1차 비행 5/15 예정",
            },
        },
    }


class SiteUpdate(BaseModel):
    """PATCH /api/v1/sites/{id} — 부분 업데이트"""
    name: Optional[str] = Field(None, max_length=200)
    inspection_type: Optional[str] = None
    address: Optional[str] = Field(None, max_length=500)
    building_type: Optional[str] = None
    total_area: Optional[float] = None
    building_count: Optional[int] = None
    unit_count: Optional[int] = None
    client_type: Optional[str] = None
    client_name: Optional[str] = Field(None, max_length=200)
    client_contact: Optional[str] = Field(None, max_length=100)
    contract_start: Optional[date] = None
    contract_end: Optional[date] = None
    status: Optional[str] = None
    assigned_members: Optional[List[AssignedMember]] = None
    memo: Optional[str] = None
    inspection_count: Optional[int] = None
    last_inspection_date: Optional[date] = None
    recordings: Optional[List[Recording]] = None


# ── 출력 스키마 ──────────────────────────────

class SiteResponse(BaseModel):
    """GET 응답 — 현장 전체 정보"""
    id: UUID
    seq: int
    name: str
    inspection_type: Optional[str] = None
    address: Optional[str] = None
    building_type: str
    total_area: Optional[float] = None
    building_count: Optional[int] = None
    unit_count: Optional[int] = None
    client_type: str
    client_name: Optional[str] = None
    client_contact: Optional[str] = None
    contract_start: Optional[date] = None
    contract_end: Optional[date] = None
    status: str
    assigned_members: Optional[List[AssignedMember]] = []
    memo: Optional[str] = None
    inspection_count: int = 0
    last_inspection_date: Optional[date] = None
    recordings: Optional[List[Recording]] = []
    created_by: Optional[UUID] = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class SiteListResponse(BaseModel):
    """GET /api/v1/sites — 목록 + 페이지네이션"""
    items: List[SiteResponse]
    total: int
    limit: int
    offset: int
