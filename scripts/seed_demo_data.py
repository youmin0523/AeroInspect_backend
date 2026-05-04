# =============================================
# scripts/seed_demo_data.py
# 역할: 배포 데모/시연용 시드 데이터 생성 (idempotent)
#
# 채워지는 도메인:
#   - Organization (1개) + Departments (3개)
#   - Users (백승희 / 오희진) + admin 시드 멤버십 연결
#   - Sites (8개 — 다양한 시행사·건물유형·상태 분산)
#   - DefectLogs (현장당 25~60건, 최근 6개월에 분산 배포)
#   - Notifications (사용자별 8건)
#
# 안전장치:
#   - APP_ENV=production 일 때 abort (실수로 운영 DB 시드 방지).
#     강제 실행하려면 --force-prod 플래그.
#   - 모든 INSERT는 사전 SELECT로 중복 검사 → 재실행 안전.
#
# 사용:
#   cd backend
#   venv\Scripts\python.exe -m scripts.seed_demo_data           (개발/스테이징)
#   venv\Scripts\python.exe -m scripts.seed_demo_data --force-prod   (운영, 비상시)
#   venv\Scripts\python.exe -m scripts.seed_demo_data --reset   (시드 데이터만 삭제 후 재생성)
# =============================================

from __future__ import annotations

import argparse
import asyncio
import os
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# scripts/ 에서 실행해도 app.* import 가능하게 backend/ 를 path 에 추가
_BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from sqlalchemy import select, delete

from app.config import APP_ENV_VAR, PROD_ENV_VALUES
from app.core.security import hash_password
from app.db.session import async_session_factory
from app.models.defect import DefectLog
from app.models.department import Department
from app.models.inspection_schedule import InspectionSchedule
from app.models.notification import Notification
from app.models.organization import Organization, OrganizationMember
from app.models.report import Report
from app.models.site import Site
from app.models.user import User


# ── 결정적 난수 (재실행 시에도 같은 분포) ──────
_RNG = random.Random(20260503)


# ── 시드 사용자 정의 ───────────────────────────
DEMO_ORG_NAME = "DRONE INSPECT 데모 조직"
DEMO_ORG_BIZ = "0000000001"  # placeholder 사업자번호 (UNIQUE 충족)

DEMO_DEPARTMENTS = ["안전진단 1팀", "드론운용팀", "리포트팀"]

DEMO_USERS = [
    # username, name, phone, password, role, department, position
    ("baeksh", "백승희",  "010-2000-1001", "demo!Pass1", "admin",  "안전진단 1팀", "대리"),
    ("ohhj",   "오희진",  "010-2000-1002", "demo!Pass1", "member", "안전진단 1팀", "대리"),
]
# 슈퍼어드민 admin 계정은 init_db._ensure_superadmin_seed 에서 만들어진 것과 매칭하여 owner 지정.
ADMIN_USERNAME = "admin"


# ── 시드 현장 정의 (8개, 시행사/유형/상태 분산) ──
# 동일 이름으로 이미 존재하면 skip.
DEMO_SITES = [
    {
        "name": "송파 헬리오시티 101동~109동", "inspection_type": "사전점검",
        "address": "서울 송파구 송파대로 345", "building_type": "아파트",
        "total_area": 84.5, "building_count": 9, "unit_count": 1080,
        "client_type": "B2B", "client_name": "현대건설",
        "client_contact": "02-3441-7000",
        "contract_start": "2026-04-01", "contract_end": "2026-05-15",
        "status": "active",
        "memo": "9개동 순차 비행, 단열·결로 집중 점검",
        "members": ["유민수", "백승희"],
    },
    {
        "name": "위례 자이 201동~205동", "inspection_type": "하자점검",
        "address": "경기 성남시 수정구 위례중앙로 12", "building_type": "아파트",
        "total_area": 102.3, "building_count": 5, "unit_count": 410,
        "client_type": "B2B", "client_name": "GS건설",
        "client_contact": "031-720-3000",
        "contract_start": "2026-02-10", "contract_end": "2026-03-05",
        "status": "completed",
        "memo": "외벽 균열·계단실 결로 다수 검출",
        "members": ["유민수"],
    },
    {
        "name": "판교 알파돔시티 A·B·C동", "inspection_type": "정기점검",
        "address": "경기 성남시 분당구 분당내곡로 117", "building_type": "오피스",
        "total_area": 156.8, "building_count": 3, "unit_count": 0,
        "client_type": "B2B", "client_name": "삼성물산",
        "client_contact": "031-696-1234",
        "contract_start": "2026-05-10", "contract_end": "2026-06-30",
        "status": "pending",
        "memo": "도면 수령 대기 — 5/12 회신 예정",
        "members": ["오희진"],
    },
    {
        "name": "잠실 리센츠 303동 503호", "inspection_type": "입주점검",
        "address": "서울 송파구 올림픽로 99", "building_type": "아파트",
        "total_area": 84.0, "building_count": 1, "unit_count": 1,
        "client_type": "B2C", "client_name": "이정아",
        "client_contact": "010-3300-5050",
        "contract_start": "2026-04-22", "contract_end": "2026-04-23",
        "status": "active",
        "memo": "B2C — 입주 전 단독 점검 의뢰",
        "members": ["백승희"],
    },
    {
        "name": "잠실 엘스 208동 2102호", "inspection_type": "입주점검",
        "address": "서울 송파구 올림픽로 240", "building_type": "아파트",
        "total_area": 116.0, "building_count": 1, "unit_count": 1,
        "client_type": "B2C", "client_name": "박민호",
        "client_contact": "010-9988-2102",
        "contract_start": "2026-04-25", "contract_end": "2026-04-25",
        "status": "active",
        "memo": "B2C — 도배 후 점검",
        "members": ["오희진"],
    },
    {
        "name": "성북구 성북로 23-5", "inspection_type": "하자점검",
        "address": "서울 성북구 성북로 23-5", "building_type": "단독주택",
        "total_area": 220.0, "building_count": 1, "unit_count": 1,
        "client_type": "B2C", "client_name": "최영수",
        "client_contact": "010-7711-2345",
        "contract_start": "2026-04-12", "contract_end": "2026-04-15",
        "status": "completed",
        "memo": "지붕 누수 의심 → 열화상 위주",
        "members": ["백승희"],
    },
    {
        "name": "강남 래미안 1단지 103동 1201호", "inspection_type": "사전점검",
        "address": "서울 강남구 도곡로 410", "building_type": "아파트",
        "total_area": 98.0, "building_count": 1, "unit_count": 1,
        "client_type": "B2C", "client_name": "정해진",
        "client_contact": "010-5588-1201",
        "contract_start": "2026-04-25", "contract_end": "2026-04-25",
        "status": "active",
        "memo": "B2C 긴급 의뢰 — 백승희 주도, 유민수 백업",
        "members": ["백승희", "유민수"],
    },
    {
        "name": "광교 자연앤힐스테이트 105동", "inspection_type": "정기점검",
        "address": "경기 수원시 영통구 광교중앙로 142", "building_type": "주상복합",
        "total_area": 132.5, "building_count": 1, "unit_count": 96,
        "client_type": "B2B", "client_name": "현대건설",
        "client_contact": "031-220-5000",
        "contract_start": "2026-03-15", "contract_end": "2026-03-25",
        "status": "completed",
        "memo": "준공 1년차 정기점검",
        "members": ["오희진", "유민수"],
    },
]


# ── 하자 카테고리 풀 (mockTrendData 와 일치) ──
DEFECT_POOL = [
    # (area, code, defect_type, defect_source, defect_class, ko, severity 가중치 분포)
    ("B", "B-02", "결로·곰팡이",       "yolo_thermal", "Moisture",     "결로", {"HIGH": 0.45, "MED": 0.40, "LOW": 0.15}),
    ("B", "B-03", "단열재 시공불량",   "yolo_thermal", "Insulation",   "단열불량", {"HIGH": 0.50, "MED": 0.35, "LOW": 0.15}),
    ("A", "A-01", "균열 (0.3mm 이상)", "yolo_thermal", "Crack",        "균열", {"HIGH": 0.30, "MED": 0.50, "LOW": 0.20}),
    ("C", "C-01", "도장 불량",         "wallpaper",    "Paint",        "도장불량", {"HIGH": 0.05, "MED": 0.35, "LOW": 0.60}),
    ("D", "D-01", "바닥 레벨 불량",    "yolo_delam",   "FloorLevel",   "바닥불량", {"HIGH": 0.20, "MED": 0.55, "LOW": 0.25}),
    ("B", "B-01", "누수·침투",         "yolo_thermal", "Leak",         "누수", {"HIGH": 0.55, "MED": 0.35, "LOW": 0.10}),
    ("C", "C-03", "타일 파손·들뜸",    "yolo_delam",   "TileBurst",    "타일파손", {"HIGH": 0.10, "MED": 0.50, "LOW": 0.40}),
    ("E", "E-01", "창호 기밀 불량",    "yolo_thermal", "WindowSeal",   "창호기밀", {"HIGH": 0.20, "MED": 0.55, "LOW": 0.25}),
    ("A", "A-02", "철근 노출",         "yolo_thermal", "Rebar",        "철근노출", {"HIGH": 0.70, "MED": 0.25, "LOW": 0.05}),
    ("D", "D-02", "마루 들뜸",         "yolo_delam",   "Floor",        "마루들뜸", {"HIGH": 0.05, "MED": 0.40, "LOW": 0.55}),
    ("C", "C-02", "벽지 터짐",         "wallpaper",    "good",         "터짐(Burst)", {"HIGH": 0.05, "MED": 0.30, "LOW": 0.65}),
]

# 사이트별 하자 건수 분포 — KPI 가 채워지도록 충분히 생성
DEFECTS_PER_SITE_RANGE = (25, 60)


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _is_production() -> bool:
    return os.environ.get(APP_ENV_VAR, "").strip().lower() in PROD_ENV_VALUES


# ──────────────────────────────────────────────
# Seed helpers (idempotent)
# ──────────────────────────────────────────────

async def _ensure_organization(session) -> Organization:
    org = await session.scalar(
        select(Organization).where(Organization.biz_number == DEMO_ORG_BIZ)
    )
    if org:
        return org
    org = Organization(name=DEMO_ORG_NAME, biz_number=DEMO_ORG_BIZ)
    session.add(org)
    await session.flush()
    return org


async def _ensure_departments(session, org_id) -> dict[str, Department]:
    existing = (await session.execute(
        select(Department).where(Department.organization_id == org_id)
    )).scalars().all()
    by_name = {d.name: d for d in existing}
    for name in DEMO_DEPARTMENTS:
        if name not in by_name:
            d = Department(organization_id=org_id, name=name)
            session.add(d)
            await session.flush()
            by_name[name] = d
    return by_name


async def _ensure_user(session, username, name, phone, password):
    user = await session.scalar(select(User).where(User.username == username))
    if user:
        return user, False
    user = User(
        account_type="business",
        username=username,
        email=f"{username}@droneinspect.demo",
        password_hash=hash_password(password),
        name=name,
        phone=phone,
        is_superadmin=False,
    )
    session.add(user)
    await session.flush()
    return user, True


async def _ensure_membership(session, org_id, user_id, role, department, position):
    existing = await session.scalar(
        select(OrganizationMember).where(
            OrganizationMember.organization_id == org_id,
            OrganizationMember.user_id == user_id,
        )
    )
    if existing:
        # role/부서/직위 동기화만
        existing.role = role
        existing.department = department
        existing.position = position
        existing.status = "active"
        return existing
    m = OrganizationMember(
        organization_id=org_id,
        user_id=user_id,
        role=role,
        department=department,
        position=position,
        status="active",
    )
    session.add(m)
    await session.flush()
    return m


async def _ensure_sites(session, org_id, owner_user_id) -> list[Site]:
    """name 기준 idempotent. 이미 있으면 그대로 반환, 없으면 생성."""
    sites: list[Site] = []
    # 다음 seq 계산 (기존 max + 1)
    max_seq = await session.scalar(select(Site.seq).order_by(Site.seq.desc()).limit(1)) or 0
    next_seq = max_seq + 1

    for spec in DEMO_SITES:
        existing = await session.scalar(select(Site).where(Site.name == spec["name"]))
        if existing:
            sites.append(existing)
            continue
        from datetime import date as _date
        site = Site(
            seq=next_seq,
            name=spec["name"],
            inspection_type=spec["inspection_type"],
            address=spec["address"],
            building_type=spec["building_type"],
            total_area=spec["total_area"],
            building_count=spec["building_count"],
            unit_count=spec["unit_count"],
            client_type=spec["client_type"],
            client_name=spec["client_name"],
            client_contact=spec["client_contact"],
            contract_start=_date.fromisoformat(spec["contract_start"]),
            contract_end=_date.fromisoformat(spec["contract_end"]),
            status=spec["status"],
            assigned_members=[{"id": f"t{i+1}", "name": n, "role": "operator"} for i, n in enumerate(spec["members"])],
            memo=spec["memo"],
            inspection_count=_RNG.randint(1, 6),
            last_inspection_date=_date.fromisoformat(spec["contract_start"]),
            organization_id=org_id,
            created_by=owner_user_id,
        )
        session.add(site)
        await session.flush()
        sites.append(site)
        next_seq += 1
    return sites


def _weighted_severity(weights: dict) -> str:
    items, probs = zip(*weights.items())
    return _RNG.choices(items, weights=probs, k=1)[0]


async def _ensure_defects(session, sites: list[Site], target_per_site_range=DEFECTS_PER_SITE_RANGE):
    """현장별 결함 시드. site_id 기준 카운트가 적으면 부족분만 채움."""
    now = _now_utc()
    six_months_ago = now - timedelta(days=180)

    total_inserted = 0
    for site in sites:
        existing_count = await session.scalar(
            select(DefectLog.id).where(DefectLog.site_id == site.id).limit(1)
        )
        if existing_count is not None:
            # 이미 시드된 사이트는 건너뜀 (idempotent)
            continue

        n = _RNG.randint(*target_per_site_range)
        for _ in range(n):
            spec = _RNG.choice(DEFECT_POOL)
            area, code, name, source, klass, ko, weights = spec
            severity = _weighted_severity(weights)
            confidence = round(_RNG.uniform(0.62, 0.97), 3)

            # 최근 6개월 내 임의 timestamp (현실감 있게 평일 위주)
            offset_sec = _RNG.randint(0, int((now - six_months_ago).total_seconds()))
            ts = six_months_ago + timedelta(seconds=offset_sec)

            # bbox 정규화 좌표
            bx = round(_RNG.uniform(0.1, 0.85), 3)
            by = round(_RNG.uniform(0.1, 0.85), 3)
            bw = round(_RNG.uniform(0.05, 0.20), 3)
            bh = round(_RNG.uniform(0.05, 0.20), 3)

            # LiDAR 월드좌표 (실내 기준 0~10m)
            lx = round(_RNG.uniform(0, 10), 2)
            ly = round(_RNG.uniform(0, 10), 2)
            lz = round(_RNG.uniform(0, 3), 2)

            # 열화상 (단열·결로 계열만 채움)
            tmax = tmin = tavg = None
            delta = None
            if source == "yolo_thermal" and klass in {"Moisture", "Insulation", "Leak", "WindowSeal"}:
                tavg = round(_RNG.uniform(18, 27), 1)
                tmax = round(tavg + _RNG.uniform(2, 6), 1)
                tmin = round(tavg - _RNG.uniform(1, 3), 1)
                delta = round(tmax - tmin, 1)

            # 균열·기하학 계열만 deviation 채움
            deviation = None
            if klass in {"Crack", "Rebar"}:
                deviation = round(_RNG.uniform(0.1, 0.6), 2)

            session.add(DefectLog(
                site_id=site.id,
                area=area, category_code=code, defect_type=name,
                defect_source=source, defect_class=klass,
                defect_class_display_en=klass, defect_class_display_ko=ko,
                severity=severity, confidence=confidence,
                bbox_x=bx, bbox_y=by, bbox_w=bw, bbox_h=bh,
                lidar_x=lx, lidar_y=ly, lidar_z=lz,
                thermal_max=tmax, thermal_min=tmin, thermal_avg=tavg,
                delta_temperature=delta,
                deviation_degrees=deviation,
                timestamp=ts,
                frame_id=_RNG.randint(1000, 9999),
                accumulated_conf=round(min(1.0, confidence + _RNG.uniform(0, 0.05)), 3),
                tier_executed=_RNG.choice([1, 1, 2, 2, 3]),
            ))
            total_inserted += 1

        await session.flush()

    return total_inserted


# ── 오늘 일정 시드 ─────────────────────────────
# (시각, 사이트 이름 substring, 담당자 username) — 오늘 KST 기준 시각.
DEMO_TODAY_SCHEDULE = [
    ("09:00", "송파 헬리오시티", "admin"),
    ("14:00", "잠실 리센츠", "baeksh"),    # 사용자 요청: 잠실 리센츠 14:00 백승희
    ("16:30", "잠실 엘스",   "ohhj"),
]


async def _ensure_today_schedule(session, org_id, sites: list[Site], user_lookup: dict[str, User]):
    """오늘 KST 일정 시드. (org_id, scheduled_at, site_id) 동일하면 skip."""
    from datetime import time as _time
    kst = timezone(timedelta(hours=9))
    today_kst = datetime.now(kst).date()

    # 사이트 이름 -> Site 매핑 (substring contains)
    def find_site(needle: str) -> Site | None:
        for s in sites:
            if needle in s.name:
                return s
        return None

    inserted = 0
    for hhmm, site_needle, op_username in DEMO_TODAY_SCHEDULE:
        site = find_site(site_needle)
        if site is None:
            continue
        op_user = user_lookup.get(op_username)
        hh, mm = hhmm.split(":")
        sched_at_kst = datetime.combine(today_kst, _time(int(hh), int(mm)), tzinfo=kst)
        sched_at_utc = sched_at_kst.astimezone(timezone.utc)

        existing = await session.scalar(
            select(InspectionSchedule.id).where(
                InspectionSchedule.organization_id == org_id,
                InspectionSchedule.site_id == site.id,
                InspectionSchedule.scheduled_at == sched_at_utc,
            )
        )
        if existing:
            continue

        session.add(InspectionSchedule(
            site_id=site.id,
            operator_user_id=op_user.id if op_user else None,
            organization_id=org_id,
            scheduled_at=sched_at_utc,
            status="upcoming",
            note=f"{site.inspection_type} — {site.name.split()[0]} 정기 비행",
        ))
        inserted += 1

    await session.flush()
    return inserted


# ── 보고서(Report) 시드 ────────────────────────
# 이번 달 발행된 보고서 수가 KPI 에 들어가므로 충분량 보장.
DEMO_REPORTS_PER_COMPLETED_SITE = (3, 5)


async def _ensure_reports(session, sites: list[Site]):
    """status='completed' 인 사이트에 대해 보고서 3~5건씩 시드. (site_id, title) 중복이면 skip."""
    now = _now_utc()
    month_start = datetime(now.year, now.month, 1, tzinfo=timezone.utc)

    inspector_pool = ["유민수", "백승희", "오희진"]
    inserted = 0

    for site in sites:
        if site.status != "completed":
            continue

        n = _RNG.randint(*DEMO_REPORTS_PER_COMPLETED_SITE)
        for i in range(n):
            title = f"{site.name} 점검 보고서 #{i+1}"
            existing = await session.scalar(
                select(Report.id).where(
                    Report.site_id == site.id,
                    Report.title == title,
                )
            )
            if existing:
                continue

            # 절반은 이번 달, 절반은 지난 6개월에 분산 (월간 KPI 도 채우고, 누적도 채움)
            if i < n // 2:
                offset_h = _RNG.randint(0, max(1, int((now - month_start).total_seconds() // 3600)))
                created_at = now - timedelta(hours=offset_h)
            else:
                offset_d = _RNG.randint(31, 180)
                created_at = now - timedelta(days=offset_d)

            high = _RNG.randint(2, 6)
            med  = _RNG.randint(5, 12)
            low  = _RNG.randint(8, 20)

            session.add(Report(
                site_id=site.id,
                title=title,
                building_name=site.name,
                inspector_name=_RNG.choice(inspector_pool),
                provider=_RNG.choice(["claude", "gemini"]),
                content=(
                    f"# {title}\n\n"
                    f"## 1. 점검 개요\n- 현장: {site.name}\n- 주소: {site.address or '미상'}\n"
                    f"- 점검 유형: {site.inspection_type}\n- 점검자: {site.assigned_members or '안전진단팀'}\n\n"
                    f"## 2. 검출 요약\n- 중대(HIGH): {high}건\n- 보통(MED): {med}건\n- 경미(LOW): {low}건\n\n"
                    f"## 3. 권고\n단열·방수 위주 우선 시정조치 권고.\n"
                ),
                defect_count=high + med + low,
                high_count=high,
                med_count=med,
                low_count=low,
                created_at=created_at,
            ))
            inserted += 1
        await session.flush()

    return inserted


async def _ensure_notifications(session, users: list[User], sites: list[Site]):
    """사용자별 8건씩 demo 알림. 동일 (user_id, title) 이미 있으면 skip."""
    now = _now_utc()
    templates = [
        ("schedule",    "내일 09:00 헬리오시티 102동 점검 예정",          "P1 우선순위. 도면·드론 사전 점검 부탁드립니다."),
        ("site",        "강남 래미안 1단지 신규 등록",                   "B2C 의뢰 — 점검일 4/25"),
        ("blueprint",   "광교 자연앤힐스테이트 평면도 업로드 완료",      "L1 CAD 변환 정상"),
        ("work",        "전주 점검 보고서 발행 대기 (3건)",              "리포트팀 검토 요청"),
        ("defect",      "위례 자이 결로 HIGH 2건 추가 검출",             "단열 보강 시정조치 권고"),
        ("report",      "송파 헬리오시티 103동 보고서 발행됨",            "고객사 자동 발송 완료"),
        ("drone",       "DRONE 02 배터리 70% — 충전 권장",                "다음 비행 전 충전 부탁드립니다."),
        ("team",        "오희진 대리 — 판교 알파돔시티 담당 배정",       "도면 수령 후 비행 일정 조율 필요"),
    ]
    inserted = 0
    for user in users:
        for cat, title, msg in templates:
            existing = await session.scalar(
                select(Notification.id).where(
                    Notification.user_id == user.id,
                    Notification.title == title,
                )
            )
            if existing:
                continue
            offset_h = _RNG.randint(1, 96)  # 최근 4일 내
            session.add(Notification(
                user_id=user.id,
                category=cat,
                title=title,
                message=msg,
                metadata_={"site_id": str(_RNG.choice(sites).id)} if sites else None,
                is_read=_RNG.random() < 0.4,
                created_at=now - timedelta(hours=offset_h),
            ))
            inserted += 1
        await session.flush()
    return inserted


# ──────────────────────────────────────────────
# Reset (선택적)
# ──────────────────────────────────────────────

async def _reset_demo_data(session, org_id):
    """시드 데이터만 제거. 슈퍼어드민/기타 사용자는 보존."""
    # 1) defects (site_id 가 demo 사이트인 것)
    site_ids = (await session.execute(
        select(Site.id).where(Site.organization_id == org_id)
    )).scalars().all()
    if site_ids:
        await session.execute(delete(DefectLog).where(DefectLog.site_id.in_(site_ids)))

    # 2) demo 사이트
    await session.execute(delete(Site).where(Site.organization_id == org_id))

    # 3) demo 알림 (demo 사용자에 대한 것만)
    demo_usernames = [u[0] for u in DEMO_USERS]
    demo_user_ids = (await session.execute(
        select(User.id).where(User.username.in_(demo_usernames))
    )).scalars().all()
    if demo_user_ids:
        await session.execute(delete(Notification).where(Notification.user_id.in_(demo_user_ids)))

    # 4) demo 멤버십
    await session.execute(delete(OrganizationMember).where(
        OrganizationMember.organization_id == org_id
    ))

    # 5) demo 부서
    await session.execute(delete(Department).where(Department.organization_id == org_id))

    # 6) demo 사용자
    if demo_user_ids:
        await session.execute(delete(User).where(User.id.in_(demo_user_ids)))

    # 7) demo 조직
    await session.execute(delete(Organization).where(Organization.id == org_id))

    await session.commit()


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

async def run(reset: bool, force_prod: bool):
    if _is_production() and not force_prod:
        print("[seed] APP_ENV=production 감지 — 안전을 위해 abort. 강제 실행: --force-prod")
        sys.exit(1)
    if _is_production() and force_prod:
        print("[seed] ⚠️  운영 환경에서 시드 진행 — 진짜 시연용 시드만 실행됩니다.")

    async with async_session_factory() as session:
        # 1) 조직 / 부서
        org = await _ensure_organization(session)

        # reset 모드면 먼저 비우고 다시 만듦
        if reset:
            print(f"[seed] reset 모드 — 조직 '{org.name}' 의 시드 데이터 제거")
            await _reset_demo_data(session, org.id)
            org = await _ensure_organization(session)

        depts = await _ensure_departments(session, org.id)
        print(f"[seed] org='{org.name}' (biz={org.biz_number}) / depts={len(depts)}")

        # 2) 슈퍼어드민(admin) → owner 멤버십 등록
        admin = await session.scalar(select(User).where(User.username == ADMIN_USERNAME))
        if admin:
            await _ensure_membership(session, org.id, admin.id, "owner", "안전진단 1팀", "과장")
            print(f"[seed] admin (유민수) → org owner 등록")
        owner_user_id = admin.id if admin else None

        # 3) 데모 사용자 (백승희 / 오희진)
        demo_user_objs: list[User] = []
        for username, name, phone, pw, role, dept, position in DEMO_USERS:
            u, created = await _ensure_user(session, username, name, phone, pw)
            await _ensure_membership(session, org.id, u.id, role, dept, position)
            demo_user_objs.append(u)
            print(f"[seed] user {username}/{name} {'생성' if created else '존재'} → {role} ({dept})")

        # admin 도 알림 시드 대상에 포함
        notif_target_users = list(demo_user_objs)
        if admin:
            notif_target_users.append(admin)

        # 4) 현장
        sites = await _ensure_sites(session, org.id, owner_user_id)
        print(f"[seed] sites total={len(sites)}")

        # 5) 하자 (idempotent — 이미 site별로 존재하면 skip)
        defects_inserted = await _ensure_defects(session, sites)
        print(f"[seed] defects newly inserted={defects_inserted}")

        # 6) 알림
        notif_inserted = await _ensure_notifications(session, notif_target_users, sites)
        print(f"[seed] notifications newly inserted={notif_inserted}")

        # 7) 보고서 (월간 KPI 채우는 핵심 — completed 사이트당 3~5건)
        reports_inserted = await _ensure_reports(session, sites)
        print(f"[seed] reports newly inserted={reports_inserted}")

        # 8) 오늘 일정 (EmployeeLanding 의 schedule/today 위젯)
        user_lookup = {ADMIN_USERNAME: admin} if admin else {}
        for u in demo_user_objs:
            user_lookup[u.username] = u
        schedule_inserted = await _ensure_today_schedule(session, org.id, sites, user_lookup)
        print(f"[seed] today schedules newly inserted={schedule_inserted}")

        await session.commit()
        print("[seed] DONE.")


def parse_args():
    p = argparse.ArgumentParser(description="Drone Inspect 데모 시드 데이터 생성")
    p.add_argument("--reset", action="store_true", help="기존 데모 시드 삭제 후 재생성")
    p.add_argument("--force-prod", action="store_true", help="APP_ENV=production 에서도 강행")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(run(reset=args.reset, force_prod=args.force_prod))
