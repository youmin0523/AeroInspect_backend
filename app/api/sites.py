# =============================================
# app/api/sites.py
# 역할: 현장(Site) 관리 REST CRUD API 엔드포인트
#       - GET    /sites           → 목록 조회 (필터, 검색, 페이지네이션)
#       - GET    /sites/{id}      → 단건 조회
#       - POST   /sites           → 신규 현장 등록
#       - PATCH  /sites/{id}      → 부분 업데이트
#       - DELETE /sites/{id}      → 삭제
# =============================================

from uuid import UUID
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select, func, desc, delete as sa_delete
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import get_db, get_current_org_member
from app.models.site import Site
from app.models.defect import DefectLog
from app.models.report import Report
from app.models.telemetry import TelemetryLog
from app.services.image_storage import image_storage
from app.schemas.site import (
    SiteCreate,
    SiteUpdate,
    SiteResponse,
    SiteListResponse,
)

router = APIRouter()


@router.get("/", response_model=SiteListResponse)
async def list_sites(
    status_filter: Optional[str] = Query(None, alias="status"),
    building_type: Optional[str] = Query(None),
    client_type: Optional[str] = Query(None),
    search: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    org_tuple=Depends(get_current_org_member),
    db: AsyncSession = Depends(get_db),
):
    """현장 목록 조회 (소속 조직 한정, 필터링 + 페이지네이션)"""
    user, member, org = org_tuple
    query = select(Site).where(Site.organization_id == org.id)

    if status_filter:
        query = query.where(Site.status == status_filter)
    if building_type:
        query = query.where(Site.building_type == building_type)
    if client_type:
        query = query.where(Site.client_type == client_type)
    if search:
        query = query.where(Site.name.ilike(f"%{search}%"))

    # 전체 카운트
    count_query = select(func.count()).select_from(query.subquery())
    total = await db.scalar(count_query)

    # 정렬 + 페이지네이션
    query = query.order_by(desc(Site.updated_at)).limit(limit).offset(offset)
    result = await db.execute(query)
    items = result.scalars().all()

    return SiteListResponse(
        items=[SiteResponse.model_validate(s) for s in items],
        total=total or 0,
        limit=limit,
        offset=offset,
    )


@router.get("/{site_id}", response_model=SiteResponse)
async def get_site(
    site_id: UUID,
    org_tuple=Depends(get_current_org_member),
    db: AsyncSession = Depends(get_db),
):
    """단건 조회 (소속 조직 검증)"""
    user, member, org = org_tuple
    result = await db.execute(
        select(Site).where(Site.id == site_id, Site.organization_id == org.id)
    )
    site = result.scalar_one_or_none()
    if not site:
        raise HTTPException(status_code=404, detail="현장을 찾을 수 없습니다.")
    return SiteResponse.model_validate(site)


@router.post("/", response_model=SiteResponse, status_code=status.HTTP_201_CREATED)
async def create_site(
    body: SiteCreate,
    org_tuple=Depends(get_current_org_member),
    db: AsyncSession = Depends(get_db),
):
    """새 현장 등록 (소속 조직에 자동 배정)"""
    user, member, org = org_tuple
    site = Site(
        name=body.name,
        address=body.address,
        building_type=body.building_type,
        total_area=body.total_area,
        building_count=body.building_count,
        unit_count=body.unit_count,
        client_type=body.client_type,
        client_name=body.client_name,
        client_contact=body.client_contact,
        contract_start=body.contract_start,
        contract_end=body.contract_end,
        status=body.status,
        assigned_members=[m.model_dump() for m in body.assigned_members] if body.assigned_members else [],
        memo=body.memo,
        organization_id=org.id,
        created_by=user.id,
    )
    db.add(site)
    await db.commit()
    await db.refresh(site)
    return SiteResponse.model_validate(site)


@router.patch("/{site_id}", response_model=SiteResponse)
async def update_site(
    site_id: UUID,
    body: SiteUpdate,
    org_tuple=Depends(get_current_org_member),
    db: AsyncSession = Depends(get_db),
):
    """부분 업데이트 (소속 조직 검증)"""
    user, member, org = org_tuple
    result = await db.execute(
        select(Site).where(Site.id == site_id, Site.organization_id == org.id)
    )
    site = result.scalar_one_or_none()
    if not site:
        raise HTTPException(status_code=404, detail="현장을 찾을 수 없습니다.")

    update_data = body.model_dump(exclude_unset=True)
    # JSONB 필드 변환
    if "assigned_members" in update_data and update_data["assigned_members"] is not None:
        update_data["assigned_members"] = [m.model_dump() if hasattr(m, 'model_dump') else m for m in update_data["assigned_members"]]
    if "recordings" in update_data and update_data["recordings"] is not None:
        update_data["recordings"] = [r.model_dump() if hasattr(r, 'model_dump') else r for r in update_data["recordings"]]

    for key, value in update_data.items():
        setattr(site, key, value)

    await db.commit()
    await db.refresh(site)
    return SiteResponse.model_validate(site)


@router.delete("/{site_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_site(
    site_id: UUID,
    org_tuple=Depends(get_current_org_member),
    db: AsyncSession = Depends(get_db),
):
    """삭제 (소속 조직 검증)"""
    user, member, org = org_tuple
    result = await db.execute(
        select(Site).where(Site.id == site_id, Site.organization_id == org.id)
    )
    site = result.scalar_one_or_none()
    if not site:
        raise HTTPException(status_code=404, detail="현장을 찾을 수 없습니다.")

    # ── 자식 레코드 정리 ──────────────────────────────────────────────
    # defect_logs / reports / telemetry_logs 의 site_id FK 에는 ondelete 가 없어
    # (Postgres 기본 RESTRICT) 자식이 1건이라도 있으면 site 삭제가 IntegrityError(500)로
    # 터졌다. DB 스키마(create_all/alembic 혼용) 상태와 무관하게 동작하도록 앱 레벨에서
    # 자식을 먼저 제거한다. inspection_schedules 는 FK 가 ondelete=CASCADE 라 자동 정리됨.
    crop_rows = await db.execute(
        select(DefectLog.image_crop_path).where(
            DefectLog.site_id == site.id,
            DefectLog.image_crop_path.isnot(None),
        )
    )
    crop_paths = [p for (p,) in crop_rows.all() if p]

    await db.execute(sa_delete(DefectLog).where(DefectLog.site_id == site.id))
    await db.execute(sa_delete(Report).where(Report.site_id == site.id))
    await db.execute(sa_delete(TelemetryLog).where(TelemetryLog.site_id == site.id))

    await db.delete(site)
    await db.commit()

    # 트랜잭션 커밋 성공 이후에만 크롭 이미지 파일 정리 (롤백 시 파일 보존)
    for p in crop_paths:
        image_storage.delete(p)
