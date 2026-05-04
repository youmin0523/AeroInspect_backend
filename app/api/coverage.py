# =============================================
# app/api/coverage.py
# 역할: 점검 커버리지 산출 API
#       - 현장(site) 내 텔레메트리 경로의 2D convex hull → 점검 면적 추정
#       - sites.total_area(공급 면적) 대비 커버리지율 반환
#       - 미점검 구역은 보고서·3D 미니맵 음영 처리 재료로 쓰임
#
# 텔레메트리 필터 정책 (마이그레이션 e4c9a8b27f10 이후):
#   - 우선: telemetry_logs.site_id == :site_id 레코드만 사용
#   - fallback: 해당 site의 telemetry가 0건이면 전역 최근 N건으로 계산
#     (기존 비행 데이터 호환 — site_id FK 추가 전 로그 보존)
# =============================================

from __future__ import annotations

from typing import List, Tuple
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import get_current_org_member, get_db
from app.models.site import Site
from app.models.telemetry import TelemetryLog
from app.schemas.monitoring import CoverageResponse

router = APIRouter()


# ── 순수 기하 유틸 (의존성 최소화) ────────────────────
def _convex_hull(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Andrew's monotone chain. O(n log n). 중복·공선 처리."""
    pts = sorted(set(points))
    if len(pts) <= 1:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: List[Tuple[float, float]] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper: List[Tuple[float, float]] = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    # 마지막 점은 시작점과 겹침 — 제거
    return lower[:-1] + upper[:-1]


def _polygon_area(poly: List[Tuple[float, float]]) -> float:
    """Shoelace. 부호 없는 면적(㎡)."""
    n = len(poly)
    if n < 3:
        return 0.0
    area2 = 0.0
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        area2 += x1 * y2 - x2 * y1
    return abs(area2) / 2.0


# ── 엔드포인트 ──────────────────────────────────────
@router.get("/{site_id}", response_model=CoverageResponse)
async def get_site_coverage(
    site_id: UUID,
    sample_limit: int = Query(
        2000,
        ge=50,
        le=20000,
        description="convex hull 계산에 쓸 최근 텔레메트리 샘플 수",
    ),
    org_tuple=Depends(get_current_org_member),
    db: AsyncSession = Depends(get_db),
) -> CoverageResponse:
    """
    현장별 점검 커버리지 산출.

    반환:
      - covered_area_m2: 드론 비행 경로 convex hull 면적
      - supplied_area_m2: sites.total_area (사용자가 사전 입력)
      - coverage_ratio: covered / supplied (0.0 ~ 1.0)
      - uncovered_area_m2: supplied - covered (음수는 0으로 clamp)
      - sample_count: hull 계산에 쓰인 샘플 수
      - hull: 외곽 폴리곤 꼭짓점 [[x, y], ...] (프론트 3D 미니맵 음영용)
    """
    user, member, org = org_tuple

    # 1) 현장이 내 조직 소속인지 검증
    site = await db.scalar(
        select(Site).where(
            Site.id == site_id,
            Site.organization_id == org.id,
        )
    )
    if site is None:
        raise HTTPException(status_code=404, detail="현장을 찾을 수 없습니다.")

    # 2) 텔레메트리 로드. 우선순위:
    #    a) telemetry_logs.site_id == site.id 인 레코드 (신규 경로)
    #    b) 해당 site 레코드 0건이면 전역 최근 N건 (마이그레이션 이전 비행 호환)
    #    pos_x/pos_y는 월드 좌표계(m) — LiDAR 3D 좌표 배선 때와 동일 기준
    used_fallback = False
    rows = await db.execute(
        select(TelemetryLog.pos_x, TelemetryLog.pos_y)
        .where(TelemetryLog.site_id == site.id)
        .order_by(desc(TelemetryLog.timestamp))
        .limit(sample_limit)
    )
    points = [
        (float(x), float(y))
        for x, y in rows.all()
        if x is not None and y is not None
    ]

    if not points:
        # fallback: 기존 비행 데이터(site_id=NULL)에서 최근 N건
        used_fallback = True
        rows = await db.execute(
            select(TelemetryLog.pos_x, TelemetryLog.pos_y)
            .order_by(desc(TelemetryLog.timestamp))
            .limit(sample_limit)
        )
        points = [
            (float(x), float(y))
            for x, y in rows.all()
            if x is not None and y is not None
        ]

    if len(points) < 3:
        return CoverageResponse(
            site_id=site.id,
            covered_area_m2=0.0,
            supplied_area_m2=site.total_area,
            coverage_ratio=None,
            uncovered_area_m2=site.total_area,
            sample_count=len(points),
            hull=[],
            note="convex hull 계산에 필요한 텔레메트리(≥3점)가 부족합니다.",
        )

    hull = _convex_hull(points)
    covered = _polygon_area(hull)

    supplied = site.total_area
    ratio = None
    uncovered = None
    if supplied and supplied > 0:
        ratio = max(0.0, min(1.0, covered / supplied))
        uncovered = max(0.0, supplied - covered)

    fallback_note = (
        "이 현장에 연결된 텔레메트리가 없어 전역 최근 샘플로 계산된 근사치입니다."
        if used_fallback
        else None
    )

    return CoverageResponse(
        site_id=site.id,
        covered_area_m2=round(covered, 3),
        supplied_area_m2=supplied,
        coverage_ratio=round(ratio, 4) if ratio is not None else None,
        uncovered_area_m2=round(uncovered, 3) if uncovered is not None else None,
        sample_count=len(points),
        hull=[[round(x, 3), round(y, 3)] for x, y in hull],
        note=fallback_note,
    )
