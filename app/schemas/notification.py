# =============================================
# app/schemas/notification.py
# 역할: 알림 Pydantic 입출력 스키마 정의
#       - NotificationCreate: 알림 생성 요청 (서비스 내부 호출용)
#       - NotificationResponse: 알림 응답 직렬화
#       - NotificationListResponse: 페이지네이션 목록 응답
#       - NotificationUnreadCount: 미읽음 카운트
# 사용: API 라우터, notification_service 에서 사용
# =============================================

from datetime import datetime
from typing import Optional, List
from uuid import UUID

from pydantic import BaseModel, Field


VALID_CATEGORIES = {
    "schedule", "site", "blueprint", "work", "defect",
    "report", "drone", "team", "system", "compliance",
}


# ── 생성 요청 스키마 ─────────────────────────
class NotificationCreate(BaseModel):
    """알림 생성 요청 (서비스 레이어에서 사용)"""
    user_id: UUID
    category: str = Field(..., description="알림 카테고리 (schedule, site, blueprint, ...)")
    title: str = Field(..., max_length=300, description="알림 제목")
    message: Optional[str] = Field(None, description="알림 상세 메시지")
    metadata: Optional[dict] = Field(None, description="부가 정보 JSON")


# ── 응답 스키마 ──────────────────────────────
class NotificationResponse(BaseModel):
    """알림 단건 응답"""
    id: UUID
    user_id: UUID
    category: str
    title: str
    message: Optional[str]
    metadata: Optional[dict] = Field(None, alias="metadata_")
    is_read: bool
    created_at: datetime

    class Config:
        from_attributes = True
        populate_by_name = True


# ── 목록 응답 ────────────────────────────────
class NotificationListResponse(BaseModel):
    """알림 목록 페이지네이션 응답"""
    items: List[NotificationResponse]
    total: int
    limit: int
    offset: int


# ── 미읽음 카운트 ────────────────────────────
class NotificationUnreadCount(BaseModel):
    """미읽음 알림 수"""
    count: int
