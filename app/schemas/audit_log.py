# =============================================
# app/schemas/audit_log.py
# 역할: 감사 로그 Pydantic 입출력 스키마
#       - AuditLogResponse: GET 응답 시 직렬화
#       - AuditLogListResponse: 페이지네이션 응답
#       - AuditLogFilter: 목록 조회 필터
# 사용: app/api/audit_logs.py 의 응답 모델
# =============================================

from datetime import datetime
from typing import Optional, List, Any
from uuid import UUID

from pydantic import BaseModel, Field


class AuditLogResponse(BaseModel):
    """감사 로그 단건 응답."""
    id: UUID
    user_id: Optional[UUID] = Field(None, description="행위 주체 사용자 (NULL=시스템)")
    organization_id: Optional[UUID] = Field(None, description="조직 컨텍스트")
    action: str = Field(description="행위 식별자 (점 구분, 예: defect.review.approve)")
    resource_type: str = Field(description="대상 자원 종류")
    resource_id: Optional[UUID] = Field(None, description="대상 자원 ID")
    before: Optional[Any] = Field(None, description="변경 전 상태 (요약 JSON)")
    after: Optional[Any] = Field(None, description="변경 후 상태 (요약 JSON)")
    ip: Optional[str] = Field(None, description="클라이언트 IP")
    user_agent: Optional[str] = Field(None, description="User-Agent")
    request_id: Optional[str] = Field(None, description="요청 ID (structlog)")
    note: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True


class AuditLogListResponse(BaseModel):
    """감사 로그 페이지네이션 응답."""
    items: List[AuditLogResponse]
    total: int
    limit: int
    offset: int


class AuditLogFilter(BaseModel):
    """감사 로그 목록 조회 필터."""
    action: Optional[str] = None
    resource_type: Optional[str] = None
    resource_id: Optional[UUID] = None
    user_id: Optional[UUID] = None
    since: Optional[datetime] = None
    until: Optional[datetime] = None
    limit: int = Field(default=50, ge=1, le=500)
    offset: int = Field(default=0, ge=0)
