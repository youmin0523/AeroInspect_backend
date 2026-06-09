# =============================================
# app/api/defects.py
# 역할: 하자 탐지 로그 REST CRUD API 엔드포인트
#       - GET  /defects        → 목록 조회 (필터링, 페이지네이션)
#       - GET  /defects/{id}   → 단건 조회
#       - POST /defects        → 신규 하자 저장 + WS 브로드캐스트
#       - GET  /defects/summary → 대시보드용 요약 통계
#       - DELETE /defects/{id} → 하자 삭제
# =============================================

from datetime import datetime, timezone
from uuid import UUID
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy import select, func, desc
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import get_db, get_ws_manager, get_current_org_member
from app.models.audit_log import AuditLog
from app.models.defect import DefectLog
from app.models.site import Site
from app.schemas.audit_log import AuditLogResponse, AuditLogListResponse
from app.schemas.defect import (
    DefectLogCreate,
    DefectLogResponse,
    DefectLogListResponse,
    DefectReviewRequest,
    DefectSummary,
)
from app.core.ws_manager import ConnectionManager
from app.services.audit_logger import write_audit
from app.services.image_storage import image_storage


def _build_response(defect: DefectLog) -> DefectLogResponse:
    """ORM → 응답 스키마. image_crop_path가 있으면 URL까지 채움."""
    resp = DefectLogResponse.model_validate(defect)
    if defect.image_crop_path:
        resp.image_crop_url = image_storage.get_url(defect.image_crop_path)
    return resp


router = APIRouter()


@router.get("/summary", response_model=DefectSummary)
async def get_defect_summary(
    org_tuple=Depends(get_current_org_member),
    db: AsyncSession = Depends(get_db),
):
    """
    대시보드용 하자 요약 통계 반환 (소속 조직 한정).
    전체 건수, 심각도별 건수, 영역별 건수, 최신 탐지 결과.
    """
    user, member, org = org_tuple
    org_filter = DefectLog.site_id.in_(
        select(Site.id).where(Site.organization_id == org.id)
    )

    total = await db.scalar(
        select(func.count(DefectLog.id)).where(org_filter)
    )

    # 심각도별 카운트
    severity_rows = await db.execute(
        select(DefectLog.severity, func.count(DefectLog.id))
        .where(org_filter)
        .group_by(DefectLog.severity)
    )
    by_severity = {row[0]: row[1] for row in severity_rows}

    # 영역별 카운트
    area_rows = await db.execute(
        select(DefectLog.area, func.count(DefectLog.id))
        .where(org_filter)
        .group_by(DefectLog.area)
    )
    by_area = {row[0]: row[1] for row in area_rows}

    # 최신 탐지 결과
    latest_result = await db.execute(
        select(DefectLog).where(org_filter)
        .order_by(desc(DefectLog.timestamp)).limit(1)
    )
    latest = latest_result.scalar_one_or_none()

    return DefectSummary(
        total=total or 0,
        by_severity=by_severity,
        by_area=by_area,
        latest=_build_response(latest) if latest else None,
    )


@router.get("/recent", response_model=DefectLogListResponse)
async def list_recent_defects(
    severity: Optional[str] = Query(None, description="심각도 필터 (HIGH/MED/LOW)"),
    limit: int = Query(50, ge=1, le=500),
    db: AsyncSession = Depends(get_db),
):
    """
    최신순 하자 로그 N건 조회.
    실시간 대시보드용 — severity 단일 필터만 지원하는 경량 엔드포인트.
    """
    query = select(DefectLog).order_by(desc(DefectLog.timestamp))
    if severity:
        query = query.where(DefectLog.severity == severity.upper())
    query = query.limit(limit)
    result = await db.execute(query)
    items = result.scalars().all()
    return DefectLogListResponse(
        items=[_build_response(item) for item in items],
        total=len(items),
        limit=limit,
        offset=0,
    )


@router.get("", response_model=DefectLogListResponse)
async def list_defects(
    area: Optional[str] = Query(None, description="영역 코드 (A-E)"),
    severity: Optional[str] = Query(None, description="심각도 (HIGH/MED/LOW)"),
    category_code: Optional[str] = Query(None, description="카테고리 코드 (예: A-01)"),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    org_tuple=Depends(get_current_org_member),
    db: AsyncSession = Depends(get_db),
):
    """하자 탐지 로그 목록 조회 (소속 조직 한정, 필터링 + 페이지네이션)"""
    user, member, org = org_tuple
    query = select(DefectLog).where(
        DefectLog.site_id.in_(
            select(Site.id).where(Site.organization_id == org.id)
        )
    )

    if area:
        query = query.where(DefectLog.area == area.upper())
    if severity:
        query = query.where(DefectLog.severity == severity.upper())
    if category_code:
        query = query.where(DefectLog.category_code == category_code)

    # 최신 순 정렬
    query = query.order_by(desc(DefectLog.timestamp))

    # 전체 건수
    count_query = select(func.count()).select_from(query.subquery())
    total = await db.scalar(count_query)

    # 페이지네이션
    result = await db.execute(query.offset(offset).limit(limit))
    items = result.scalars().all()

    return DefectLogListResponse(
        items=[_build_response(item) for item in items],
        total=total or 0,
        limit=limit,
        offset=offset,
    )


@router.get("/{defect_id}", response_model=DefectLogResponse)
async def get_defect(
    defect_id: UUID,
    org_tuple=Depends(get_current_org_member),
    db: AsyncSession = Depends(get_db),
):
    """하자 탐지 로그 단건 조회 (소속 조직 검증)"""
    user, member, org = org_tuple
    result = await db.execute(
        select(DefectLog).where(
            DefectLog.id == defect_id,
            DefectLog.site_id.in_(
                select(Site.id).where(Site.organization_id == org.id)
            ),
        )
    )
    defect = result.scalar_one_or_none()
    if not defect:
        raise HTTPException(status_code=404, detail="하자 탐지 기록을 찾을 수 없습니다.")
    return _build_response(defect)


@router.post("", response_model=DefectLogResponse, status_code=201)
async def create_defect(
    payload: DefectLogCreate,
    db: AsyncSession = Depends(get_db),
    manager: ConnectionManager = Depends(get_ws_manager),
):
    """
    새 하자 탐지 결과 저장 후 WebSocket으로 실시간 브로드캐스트.
    AI 파이프라인(defect_processor.py)에서 탐지 시 호출.
    """
    # Base64 이미지 → 파일 저장 (실패 시 None)
    image_crop_path = await image_storage.save_base64_jpeg(payload.image_crop)

    defect = DefectLog(
        area=payload.area.upper() if payload.area else None,
        category_code=payload.category_code,
        defect_type=payload.defect_type,
        severity=payload.severity.upper(),
        confidence=payload.confidence,
        defect_source=payload.defect_source,
        defect_class=payload.defect_class,
        defect_class_display_en=payload.defect_class_display_en,
        defect_class_display_ko=payload.defect_class_display_ko,
        bbox_x=payload.bbox.x if payload.bbox else None,
        bbox_y=payload.bbox.y if payload.bbox else None,
        bbox_w=payload.bbox.w if payload.bbox else None,
        bbox_h=payload.bbox.h if payload.bbox else None,
        lidar_x=payload.lidar_position.x if payload.lidar_position else None,
        lidar_y=payload.lidar_position.y if payload.lidar_position else None,
        lidar_z=payload.lidar_position.z if payload.lidar_position else None,
        image_crop=None,  # Base64는 더이상 DB에 저장 안 함
        image_crop_path=image_crop_path,
        thermal_max=payload.thermal_data.max if payload.thermal_data else None,
        thermal_min=payload.thermal_data.min if payload.thermal_data else None,
        thermal_avg=payload.thermal_data.avg if payload.thermal_data else None,
        frame_id=payload.frame_id,
        raw_payload=payload.raw_payload,
    )

    db.add(defect)
    await db.flush()  # ID 생성을 위해 flush (commit은 get_db에서)

    response = _build_response(defect)

    # WS "defects" 채널에 실시간 브로드캐스트
    await manager.broadcast("defects", {
        "type": "defect.new",
        "data": response.model_dump(mode="json"),
    })

    return response


@router.delete("/{defect_id}", status_code=204)
async def delete_defect(
    defect_id: UUID,
    org_tuple=Depends(get_current_org_member),
    db: AsyncSession = Depends(get_db),
):
    """
    하자 탐지 기록 삭제 (연결된 크롭 파일도 함께 제거).
    소속 조직의 site에 연결된 레코드만 삭제 가능.
    """
    user, member, org = org_tuple

    # 내 조직 site에 연결된 것만 조회 — 다른 조직 데이터 노출·파일 삭제 방지
    result = await db.execute(
        select(DefectLog).where(
            DefectLog.id == defect_id,
            DefectLog.site_id.in_(
                select(Site.id).where(Site.organization_id == org.id)
            ),
        )
    )
    defect = result.scalar_one_or_none()
    if not defect:
        raise HTTPException(status_code=404, detail="하자 탐지 기록을 찾을 수 없습니다.")

    # DB 레코드 먼저 제거 → 파일 정리. 파일 삭제 실패해도 트랜잭션 영향 없음.
    crop_path = defect.image_crop_path
    snapshot = {
        "id": str(defect.id),
        "site_id": str(defect.site_id) if defect.site_id else None,
        "category_code": defect.category_code,
        "severity": defect.severity,
        "confidence": defect.confidence,
        "review_status": defect.review_status,
        "image_crop_path": crop_path,
    }
    await db.delete(defect)

    if crop_path:
        image_storage.delete(crop_path)

    # 감사 로그 — 삭제는 책임 추적 핵심 사건
    await write_audit(
        db,
        action="defect.delete",
        resource_type="defect",
        resource_id=defect_id,
        user_id=user.id,
        organization_id=org.id,
        before=snapshot,
        after=None,
    )


# ──────────────────────────────────────────────
# Track C — 현장/사무실 인라인 검수
# ──────────────────────────────────────────────
def _validate_review_note(payload: DefectReviewRequest) -> None:
    """반려/오탐 플래그는 review_note 필수 (감사 추적용)."""
    if payload.review_status in ("rejected", "flagged_false_positive"):
        if not payload.review_note or not payload.review_note.strip():
            raise HTTPException(
                status_code=400,
                detail=f"{payload.review_status} 검수는 review_note(사유)가 필수입니다.",
            )


@router.patch("/{defect_id}/review", response_model=DefectLogResponse)
async def review_defect(
    defect_id: UUID,
    payload: DefectReviewRequest,
    request: Request,
    org_tuple=Depends(get_current_org_member),
    db: AsyncSession = Depends(get_db),
    manager: ConnectionManager = Depends(get_ws_manager),
):
    """
    하자 검수 (승인/반려/오탐 플래그).
    현장 작업자/사무실 검수자 누구나 가능 — 조직 격리만 검증.
    flagged_false_positive 는 Active Learning hard example 큐로 별도 적재 가능.
    """
    _validate_review_note(payload)
    user, member, org = org_tuple

    result = await db.execute(
        select(DefectLog).where(
            DefectLog.id == defect_id,
            DefectLog.site_id.in_(
                select(Site.id).where(Site.organization_id == org.id)
            ),
        )
    )
    defect = result.scalar_one_or_none()
    if not defect:
        raise HTTPException(status_code=404, detail="하자 탐지 기록을 찾을 수 없습니다.")

    before = {
        "review_status": defect.review_status,
        "reviewed_by_user_id": str(defect.reviewed_by_user_id) if defect.reviewed_by_user_id else None,
        "reviewed_at": defect.reviewed_at.isoformat() if defect.reviewed_at else None,
        "review_note": defect.review_note,
    }

    defect.review_status = payload.review_status
    defect.review_note = payload.review_note
    defect.reviewed_by_user_id = user.id
    defect.reviewed_at = datetime.now(timezone.utc)

    await db.flush()
    response = _build_response(defect)

    # 감사 로그 (action 을 review_status 별로 세분화 → 통계/검색 용이)
    action_map = {
        "approved": "defect.review.approve",
        "rejected": "defect.review.reject",
        "flagged_false_positive": "defect.review.flag_false_positive",
        "pending": "defect.review.reset",
    }
    await write_audit(
        db,
        action=action_map.get(payload.review_status, "defect.review"),
        resource_type="defect",
        resource_id=defect.id,
        user_id=user.id,
        organization_id=org.id,
        before=before,
        after={
            "review_status": defect.review_status,
            "reviewed_by_user_id": str(user.id),
            "reviewed_at": defect.reviewed_at.isoformat(),
            "review_note": defect.review_note,
        },
        note=payload.review_note,
        request=request,
    )

    # WS broadcast — 다른 사용자도 검수 결과 실시간 반영
    await manager.broadcast("defects", {
        "type": "defect.reviewed",
        "data": response.model_dump(mode="json"),
    })

    return response


@router.get("/{defect_id}/audit-trail", response_model=AuditLogListResponse)
async def get_defect_audit_trail(
    defect_id: UUID,
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    org_tuple=Depends(get_current_org_member),
    db: AsyncSession = Depends(get_db),
):
    """단일 하자의 감사 이력 — 검수 기록·삭제·수정 시간순(최신 우선)."""
    user, member, org = org_tuple

    # 자기 조직 site 의 defect 인지 먼저 검증 (없는 ID 에 대한 정보 누설 차단)
    exists_result = await db.execute(
        select(DefectLog.id).where(
            DefectLog.id == defect_id,
            DefectLog.site_id.in_(
                select(Site.id).where(Site.organization_id == org.id)
            ),
        )
    )
    if exists_result.scalar_one_or_none() is None:
        raise HTTPException(status_code=404, detail="하자 탐지 기록을 찾을 수 없습니다.")

    base = select(AuditLog).where(
        AuditLog.resource_type == "defect",
        AuditLog.resource_id == defect_id,
    )
    total = await db.scalar(
        select(func.count(AuditLog.id)).where(
            AuditLog.resource_type == "defect",
            AuditLog.resource_id == defect_id,
        )
    ) or 0

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
