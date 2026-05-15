# =============================================
# app/api/employee.py
# 역할: EmployeeLanding(/employee) 페이지가 사용하는 통합 엔드포인트
#       - GET /api/v1/employee/schedule/today  → 오늘 일정
#       - GET /api/v1/employee/kpi/monthly     → 이번 달 KPI 집계
#       - GET /api/v1/employee/activities      → 최근 활동 (notifications 변환)
#
# 모두 현재 사용자가 소속된 조직 단위로 격리.
# =============================================

from datetime import datetime, time, timezone
from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field
from sqlalchemy import and_, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import get_current_org_member, get_db
from app.models.defect import DefectLog
from app.models.inspection_schedule import InspectionSchedule
from app.models.notification import Notification
from app.models.report import Report
from app.models.site import Site
from app.models.telemetry import TelemetryLog
from app.models.user import User

router = APIRouter()


# ── 응답 스키마 ────────────────────────────────

class TodayScheduleItem(BaseModel):
    id: str
    time: str = Field(..., description="HH:MM (현지 시각, KST)")
    site: str
    site_id: Optional[str] = None
    status: str
    operator: str
    operator_id: Optional[str] = None


class MonthlyKpi(BaseModel):
    inspections_completed: int = Field(..., description="이번 달 completed 사이트 수")
    reports_published: int = Field(..., description="이번 달 발행된 보고서 수")
    average_flight_minutes: int = Field(..., description="이번 달 평균 비행 분 (실측 없으면 0)")
    defects_found: int = Field(..., description="이번 달 검출 하자 총 건수")
    high_severity_count: int = Field(..., description="이번 달 HIGH 심각도 건수")


class ActivityItem(BaseModel):
    id: str
    date: str
    kind: str
    label: str
    actor: str


# ── 헬퍼 ───────────────────────────────────────

def _month_start_utc(now: Optional[datetime] = None) -> datetime:
    n = now or datetime.now(timezone.utc)
    return datetime(n.year, n.month, 1, tzinfo=timezone.utc)


def _today_kst_bounds() -> tuple[datetime, datetime]:
    """오늘 00:00 ~ 24:00 KST 를 UTC 로 변환한 (start, end) 튜플."""
    from datetime import timedelta
    kst = timezone(timedelta(hours=9))
    today_kst = datetime.now(kst).date()
    start_kst = datetime.combine(today_kst, time.min, tzinfo=kst)
    end_kst = datetime.combine(today_kst, time.max, tzinfo=kst)
    return start_kst.astimezone(timezone.utc), end_kst.astimezone(timezone.utc)


# ── 1) 오늘 일정 ───────────────────────────────

@router.get("/schedule/today", response_model=List[TodayScheduleItem])
async def get_today_schedule(
    org_tuple=Depends(get_current_org_member),
    db: AsyncSession = Depends(get_db),
):
    """
    오늘(KST 기준) 점검 일정 목록.
    현재 조직의 schedule 만 시각순으로 반환.
    """
    user, member, org = org_tuple
    start_utc, end_utc = _today_kst_bounds()

    stmt = (
        select(InspectionSchedule, Site, User)
        .join(Site, Site.id == InspectionSchedule.site_id)
        .outerjoin(User, User.id == InspectionSchedule.operator_user_id)
        .where(InspectionSchedule.organization_id == org.id)
        .where(and_(
            InspectionSchedule.scheduled_at >= start_utc,
            InspectionSchedule.scheduled_at <= end_utc,
        ))
        .order_by(InspectionSchedule.scheduled_at.asc())
    )
    rows = (await db.execute(stmt)).all()

    from datetime import timedelta
    kst = timezone(timedelta(hours=9))

    return [
        TodayScheduleItem(
            id=str(sched.id),
            time=sched.scheduled_at.astimezone(kst).strftime("%H:%M"),
            site=site.name,
            site_id=str(site.id),
            status=sched.status,
            operator=op_user.name if op_user else "미배정",
            operator_id=str(op_user.id) if op_user else None,
        )
        for sched, site, op_user in rows
    ]


# ── 2) 이번 달 KPI ─────────────────────────────

@router.get("/kpi/monthly", response_model=MonthlyKpi)
async def get_monthly_kpi(
    org_tuple=Depends(get_current_org_member),
    db: AsyncSession = Depends(get_db),
):
    """
    이번 달 누적 KPI.
    - inspections_completed: status='completed' 인 sites 카운트
    - reports_published: 이번 달 생성된 보고서
    - defects_found / high_severity_count: 이번 달 timestamp 결함 집계
    - average_flight_minutes: 비행 시간 측정 데이터가 아직 없으므로 0 (추후 telemetry로 계산)
    """
    user, member, org = org_tuple
    month_start = _month_start_utc()

    # site 카운트
    inspections_completed = await db.scalar(
        select(func.count(Site.id))
        .where(Site.organization_id == org.id)
        .where(Site.status == "completed")
        .where(Site.updated_at >= month_start)
    ) or 0

    # 보고서 카운트 (Report 모델에 created_at 있다고 가정; site→org 조인)
    site_ids_subq = select(Site.id).where(Site.organization_id == org.id).subquery()
    reports_published = await db.scalar(
        select(func.count(Report.id))
        .where(Report.site_id.in_(select(site_ids_subq)))
        .where(Report.created_at >= month_start)
    ) or 0

    # 결함 집계
    defects_found = await db.scalar(
        select(func.count(DefectLog.id))
        .where(DefectLog.site_id.in_(select(site_ids_subq)))
        .where(DefectLog.timestamp >= month_start)
    ) or 0

    high_severity_count = await db.scalar(
        select(func.count(DefectLog.id))
        .where(DefectLog.site_id.in_(select(site_ids_subq)))
        .where(DefectLog.timestamp >= month_start)
        .where(DefectLog.severity == "HIGH")
    ) or 0

    # ── 평균 비행 분 (이번 달, site 단위 first-last timestamp diff 의 평균) ──
    # 노이즈/유령 site 제외를 위해 텔레메트리 샘플 5건 이상인 site 만 집계.
    # 비행 사이 휴식 시간이 포함되어 다소 과대평가될 수 있으나, 데모/대시보드 용도로 충분.
    per_site_subq = (
        select(
            TelemetryLog.site_id,
            (func.max(TelemetryLog.timestamp) - func.min(TelemetryLog.timestamp))
                .label("duration"),
        )
        .where(TelemetryLog.site_id.in_(select(site_ids_subq)))
        .where(TelemetryLog.timestamp >= month_start)
        .group_by(TelemetryLog.site_id)
        .having(func.count(TelemetryLog.id) >= 5)
        .subquery()
    )
    avg_seconds = await db.scalar(
        select(func.avg(func.extract("epoch", per_site_subq.c.duration)))
    )
    average_flight_minutes = int(round((avg_seconds or 0) / 60))

    return MonthlyKpi(
        inspections_completed=inspections_completed,
        reports_published=reports_published,
        average_flight_minutes=average_flight_minutes,
        defects_found=defects_found,
        high_severity_count=high_severity_count,
    )


# ── 3) 최근 활동 ───────────────────────────────

# 알림 카테고리 → activity kind 매핑 (프론트 마커 색/아이콘 일관)
_CAT_TO_KIND = {
    "report": "report",
    "defect": "inspection",
    "blueprint": "upload",
    "schedule": "schedule",
    "site": "schedule",
    "drone": "drone",
    "team": "team",
    "work": "work",
    "system": "system",
    "compliance": "system",
}


@router.get("/activities", response_model=List[ActivityItem])
async def get_recent_activities(
    limit: int = Query(default=5, ge=1, le=50),
    org_tuple=Depends(get_current_org_member),
    db: AsyncSession = Depends(get_db),
):
    """
    최근 활동 피드. 별도 ActivityLog 모델을 두지 않고
    notifications 테이블을 변환해서 반환 (schedule/site/blueprint/report/defect 등).
    """
    user, member, org = org_tuple

    # 같은 조직 사용자 ID 들 (organization_members 통해)
    from app.models.organization import OrganizationMember
    member_ids_subq = select(OrganizationMember.user_id).where(
        OrganizationMember.organization_id == org.id
    ).subquery()

    stmt = (
        select(Notification, User)
        .outerjoin(User, User.id == Notification.user_id)
        .where(Notification.user_id.in_(select(member_ids_subq)))
        .order_by(Notification.created_at.desc())
        .limit(limit)
    )
    rows = (await db.execute(stmt)).all()

    return [
        ActivityItem(
            id=str(n.id),
            date=n.created_at.date().isoformat(),
            kind=_CAT_TO_KIND.get(n.category, "system"),
            label=n.title,
            actor=u.name if u else "시스템",
        )
        for n, u in rows
    ]
