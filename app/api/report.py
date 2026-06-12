# =============================================
# app/api/report.py
# 역할: LLM 기반 하자 점검 보고서 생성/저장/조회/다운로드 API
#       - POST /report/generate   → 스트리밍 방식 보고서 생성 (Claude/Gemini)
#       - POST /report/preview    → 비스트리밍 방식 보고서 미리보기
#       - POST /report/save       → 생성된 보고서 DB 저장
#       - GET  /report            → 저장된 보고서 목록 조회
#       - GET  /report/{id}       → 보고서 단건 조회
#       - GET  /report/{id}/download → 마크다운 파일 다운로드
#       - DELETE /report/{id}     → 보고서 삭제
# =============================================

from datetime import datetime
from typing import List, Optional
from urllib.parse import quote
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse, Response
from pydantic import BaseModel
from sqlalchemy import select, func, desc
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import get_db, get_current_org_member
from app.models.organization import OrganizationMember
from app.models.report import Report
from app.models.site import Site
from app.schemas.report import (
    ReportRequest,
    ReportResponse,
    ReportSaveRequest,
    ReportSavedResponse,
    ReportListResponse,
)
from app.services.llm_report import LLMReportService
from app.services.notification_service import notification_service

router = APIRouter()
report_service = LLMReportService()


def _extract_assigned_user_ids(assigned_members) -> set[UUID]:
    """Site.assigned_members JSONB ([{id, name, role}, ...]) 에서 user UUID 추출."""
    ids: set[UUID] = set()
    if not assigned_members:
        return ids
    for m in assigned_members:
        if not isinstance(m, dict):
            continue
        raw = m.get("id") or m.get("user_id")
        if not raw:
            continue
        try:
            ids.add(raw if isinstance(raw, UUID) else UUID(str(raw)))
        except (ValueError, TypeError, AttributeError):
            continue
    return ids


@router.post("/generate")
async def generate_report_stream(
    request: ReportRequest,
    org_tuple=Depends(get_current_org_member),
    db: AsyncSession = Depends(get_db),
):
    """
    LLM 기반 하자 점검 보고서 스트리밍 생성 (소속 조직의 하자 로그 한정).
    텍스트 청크를 순차적으로 전송하여 프론트엔드에서 실시간 표시.

    프론트엔드 수신 방법:
        const response = await fetch('/api/v1/report/generate', {...})
        const reader = response.body.getReader()
        while (true) {
            const { done, value } = await reader.read()
            if (done) break
            // value를 텍스트로 디코딩하여 화면에 추가
        }
    """
    _user, _member, org = org_tuple
    try:
        # StreamingResponse 제너레이터는 라우트 반환 후 실행되므로
        # org.id(스칼라)만 캡처해 서비스가 자체 세션으로 조회하도록 위임한다.
        generator = report_service.generate_stream(request, org_id=org.id)
        return StreamingResponse(
            generator,
            media_type="text/plain; charset=utf-8",
            headers={"X-Content-Type-Options": "nosniff"},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"보고서 생성 실패: {str(e)}")


@router.post("/preview", response_model=ReportResponse)
async def preview_report(
    request: ReportRequest,
    org_tuple=Depends(get_current_org_member),
    db: AsyncSession = Depends(get_db),
):
    """
    LLM 기반 하자 점검 보고서 비스트리밍 생성 (소속 조직의 하자 로그 한정).
    전체 내용을 한 번에 반환 (소규모 탐지 결과 또는 테스트용).
    """
    _user, _member, org = org_tuple
    try:
        report = await report_service.generate_full(request, db, org_id=org.id)
        return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"보고서 생성 실패: {str(e)}")


# ── 보고서 저장/조회/다운로드 ──────────────────


@router.post("/save", response_model=ReportSavedResponse, status_code=201)
async def save_report(
    payload: ReportSaveRequest,
    org_tuple=Depends(get_current_org_member),
    db: AsyncSession = Depends(get_db),
):
    """생성된 보고서를 DB에 저장 (site_id 조직 검증) + 관계자에게 알림."""
    user, member, org = org_tuple
    # site_id가 있으면 소속 조직 검증 + 전체 객체 로드 (현장 관리자 추출용)
    site = None
    if payload.site_id:
        site = await db.scalar(
            select(Site).where(Site.id == payload.site_id, Site.organization_id == org.id)
        )
        if not site:
            raise HTTPException(status_code=404, detail="해당 현장을 찾을 수 없습니다.")

    report = Report(
        site_id=payload.site_id,
        title=payload.title,
        building_name=payload.building_name,
        inspector_name=payload.inspector_name,
        provider=payload.provider,
        content=payload.content,
        defect_count=payload.defect_count,
        high_count=payload.high_count,
        med_count=payload.med_count,
        low_count=payload.low_count,
    )
    db.add(report)
    await db.flush()

    # ── 알림 수신자 합집합 (A: 요청자 / B: 현장 관리자 / C: 조직 관리자) ─
    recipient_ids: set[UUID] = set()

    # A: 요청자
    recipient_ids.add(user.id)

    # B: 현장 관리자 (등록자 + 배정 팀원)
    if site is not None:
        if site.created_by:
            recipient_ids.add(site.created_by)
        recipient_ids |= _extract_assigned_user_ids(site.assigned_members)

    # C: 조직의 owner / admin 전부
    admin_rows = await db.execute(
        select(OrganizationMember.user_id)
        .where(OrganizationMember.organization_id == org.id)
        .where(OrganizationMember.role.in_(("owner", "admin")))
        .where(OrganizationMember.status == "active")
    )
    for (uid,) in admin_rows.all():
        recipient_ids.add(uid)

    if recipient_ids:
        await notification_service.create_for_many(
            db=db,
            user_ids=list(recipient_ids),
            category="report",
            title=f"보고서 '{report.title or '제목 없음'}'가 저장되었습니다",
            message=(
                f"하자 {report.defect_count}건"
                f" (HIGH {report.high_count} / MED {report.med_count} / LOW {report.low_count})"
            ),
            metadata={
                "report_id": str(report.id),
                "site_id": str(report.site_id) if report.site_id else None,
            },
        )

    return ReportSavedResponse.model_validate(report)


@router.get("", response_model=ReportListResponse)
async def list_reports(
    org_tuple=Depends(get_current_org_member),
    db: AsyncSession = Depends(get_db),
):
    """저장된 보고서 목록 조회 (소속 조직 한정, 최신순)"""
    user, member, org = org_tuple
    org_filter = Report.site_id.in_(
        select(Site.id).where(Site.organization_id == org.id)
    )
    total = await db.scalar(
        select(func.count()).select_from(select(Report).where(org_filter).subquery())
    )
    result = await db.execute(
        select(Report).where(org_filter).order_by(desc(Report.created_at))
    )
    items = result.scalars().all()
    return ReportListResponse(
        items=[ReportSavedResponse.model_validate(item) for item in items],
        total=total or 0,
    )


@router.get("/{report_id}", response_model=ReportSavedResponse)
async def get_report(
    report_id: UUID,
    org_tuple=Depends(get_current_org_member),
    db: AsyncSession = Depends(get_db),
):
    """보고서 단건 조회 (소속 조직 검증)"""
    user, member, org = org_tuple
    result = await db.execute(
        select(Report).where(
            Report.id == report_id,
            Report.site_id.in_(
                select(Site.id).where(Site.organization_id == org.id)
            ),
        )
    )
    report = result.scalar_one_or_none()
    if not report:
        raise HTTPException(status_code=404, detail="보고서를 찾을 수 없습니다.")
    return ReportSavedResponse.model_validate(report)


@router.get("/{report_id}/download")
async def download_report(
    report_id: UUID,
    org_tuple=Depends(get_current_org_member),
    db: AsyncSession = Depends(get_db),
):
    """보고서 마크다운 파일 다운로드 (소속 조직 검증)"""
    user, member, org = org_tuple
    result = await db.execute(
        select(Report).where(
            Report.id == report_id,
            Report.site_id.in_(
                select(Site.id).where(Site.organization_id == org.id)
            ),
        )
    )
    report = result.scalar_one_or_none()
    if not report:
        raise HTTPException(status_code=404, detail="보고서를 찾을 수 없습니다.")

    filename = f"{report.title or 'report'}_{report.created_at.strftime('%Y%m%d')}.md"
    return Response(
        content=report.content,
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.delete("/{report_id}", status_code=204)
async def delete_report(
    report_id: UUID,
    org_tuple=Depends(get_current_org_member),
    db: AsyncSession = Depends(get_db),
):
    """보고서 삭제 (소속 조직 검증)"""
    user, member, org = org_tuple
    result = await db.execute(
        select(Report).where(
            Report.id == report_id,
            Report.site_id.in_(
                select(Site.id).where(Site.organization_id == org.id)
            ),
        )
    )
    report = result.scalar_one_or_none()
    if not report:
        raise HTTPException(status_code=404, detail="보고서를 찾을 수 없습니다.")
    await db.delete(report)


# ── 엑셀 양식 보고서(이미지 포함) ─────────────────────────────
class ExcelDefect(BaseModel):
    category_code: Optional[str] = None
    defect_type: Optional[str] = None
    severity: Optional[str] = None
    confidence: Optional[float] = None
    image_crop: Optional[str] = None          # base64(또는 data URL) — 양식 시트2 사진


class ExcelReportRequest(BaseModel):
    defects: List[ExcelDefect] = []
    site_name: Optional[str] = ""
    unit: Optional[str] = ""
    inspector: Optional[str] = ""
    inspect_area: Optional[str] = ""


@router.post("/excel")
async def generate_excel_report(
    request: ExcelReportRequest,
    org_tuple=Depends(get_current_org_member),
):
    """검출 하자 리스트(프론트의 testDetections/defects, 이미지 포함) → 제출용 엑셀 양식.

    DB 가 아니라 요청 payload 로 받는다 — 프론트가 이미 들고 있는 검출+크롭을 그대로
    양식에 채워 돌려준다(GPU/Fly 노드·image_crop 영속화와 무관하게 동작). 시트1=상세표,
    시트2=하자 사진. 심각도 내림차순 정렬."""
    from app.services.excel_report import build_excel_report

    user, member, org = org_tuple
    defects = [d.model_dump() for d in request.defects]
    order = {"HIGH": 0, "MED": 1, "LOW": 2}
    defects.sort(key=lambda x: (order.get((x.get("severity") or "LOW").upper(), 3),
                                -(x.get("confidence") or 0)))
    xlsx = await run_in_threadpool(
        build_excel_report,
        defects,
        site_name=request.site_name or org.name,
        unit=request.unit or "",
        inspector=request.inspector or getattr(user, "name", "") or "",
        inspect_area=request.inspect_area or "",
    )
    fn = f"하자점검_결과보고서_{datetime.now().strftime('%Y%m%d')}.xlsx"
    return Response(
        content=xlsx,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f"attachment; filename*=UTF-8''{quote(fn)}"},
    )
