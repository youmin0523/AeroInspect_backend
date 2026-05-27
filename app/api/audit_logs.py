# =============================================
# app/api/audit_logs.py
# 역할: 감사 로그 조회 API (read-only)
#       - GET /audit-logs        → 페이지네이션 + 필터 목록 (admin/owner/superadmin)
#       - GET /audit-logs/{id}   → 단건 상세
# 권한:
#   - 일반 owner/admin: 자기 조직 로그만
#   - superadmin: 모든 조직 로그 (organization_id 필터로 좁힘 가능)
# 사용처: 내부 감사 화면, 분쟁 대응, 보안 incident 분석
# =============================================

from datetime import datetime
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select, func, desc, and_
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import (
    get_db,
    get_current_user,
    require_admin_or_superadmin,
)
from app.models.audit_log import AuditLog
from app.models.organization import OrganizationMember
from app.schemas.audit_log import AuditLogResponse, AuditLogListResponse


router = APIRouter()


async def _resolve_visible_org_ids(db: AsyncSession, user) -> Optional[list[UUID]]:
    """현재 사용자가 볼 수 있는 organization_id 목록을 반환.
    superadmin → None (모든 조직 조회 허용, 필터 없음).
    그 외 → 활성 멤버십이 있는 조직 ID 리스트.
    """
    if user.is_superadmin:
        return None
    result = await db.execute(
        select(OrganizationMember.organization_id)
        .where(OrganizationMember.user_id == user.id)
        .where(OrganizationMember.status == "active")
    )
    org_ids = [row[0] for row in result.all()]
    return org_ids


@router.get("", response_model=AuditLogListResponse)
async def list_audit_logs(
    action: Optional[str] = Query(None, description="행위 식별자 prefix 일치 (예: defect.review)"),
    resource_type: Optional[str] = Query(None, description="자원 종류 (defect/report/site/user 등)"),
    resource_id: Optional[UUID] = Query(None, description="특정 자원 ID 의 이력만"),
    user_id: Optional[UUID] = Query(None, description="특정 사용자 행위만"),
    organization_id: Optional[UUID] = Query(None, description="(superadmin 전용) 특정 조직 한정"),
    since: Optional[datetime] = Query(None, description="이 시각 이후 (UTC)"),
    until: Optional[datetime] = Query(None, description="이 시각 이전 (UTC)"),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    current_user=Depends(require_admin_or_superadmin),
    db: AsyncSession = Depends(get_db),
):
    """감사 로그 목록 조회. 조직 격리 + 페이지네이션 + 다중 필터."""
    visible_org_ids = await _resolve_visible_org_ids(db, current_user)

    conditions = []
    if visible_org_ids is not None:
        # 일반 admin: 자기 조직(들) 한정. NULL organization_id 인 시스템 로그는 노출하지 않음.
        if not visible_org_ids:
            return AuditLogListResponse(items=[], total=0, limit=limit, offset=offset)
        conditions.append(AuditLog.organization_id.in_(visible_org_ids))
        if organization_id is not None and organization_id not in visible_org_ids:
            raise HTTPException(status_code=403, detail="해당 조직 감사 로그 조회 권한이 없습니다.")
        if organization_id is not None:
            conditions.append(AuditLog.organization_id == organization_id)
    elif organization_id is not None:
        # superadmin 이 특정 조직 선택
        conditions.append(AuditLog.organization_id == organization_id)

    if action:
        conditions.append(AuditLog.action.like(f"{action}%"))
    if resource_type:
        conditions.append(AuditLog.resource_type == resource_type)
    if resource_id is not None:
        conditions.append(AuditLog.resource_id == resource_id)
    if user_id is not None:
        conditions.append(AuditLog.user_id == user_id)
    if since is not None:
        conditions.append(AuditLog.created_at >= since)
    if until is not None:
        conditions.append(AuditLog.created_at <= until)

    where_clause = and_(*conditions) if conditions else None

    base = select(AuditLog)
    count_base = select(func.count(AuditLog.id))
    if where_clause is not None:
        base = base.where(where_clause)
        count_base = count_base.where(where_clause)

    total = await db.scalar(count_base) or 0
    result = await db.execute(
        base.order_by(desc(AuditLog.created_at)).offset(offset).limit(limit)
    )
    items = result.scalars().all()

    return AuditLogListResponse(
        items=[AuditLogResponse.model_validate(i) for i in items],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/{log_id}", response_model=AuditLogResponse)
async def get_audit_log(
    log_id: UUID,
    current_user=Depends(require_admin_or_superadmin),
    db: AsyncSession = Depends(get_db),
):
    """감사 로그 단건 조회. 조직 격리 검증."""
    visible_org_ids = await _resolve_visible_org_ids(db, current_user)
    result = await db.execute(select(AuditLog).where(AuditLog.id == log_id))
    log = result.scalar_one_or_none()
    if log is None:
        raise HTTPException(status_code=404, detail="감사 로그를 찾을 수 없습니다.")
    if visible_org_ids is not None and log.organization_id not in visible_org_ids:
        raise HTTPException(status_code=403, detail="해당 감사 로그 조회 권한이 없습니다.")
    return AuditLogResponse.model_validate(log)
