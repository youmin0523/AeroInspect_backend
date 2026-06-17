"""
tests/test_report_notifications.py
역할: save_report 가 (A 요청자 + B 현장 관리자 + C 조직 admin) 합집합에 알림을 보내는지 검증.
실행: pytest tests/test_report_notifications.py -v
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest


def _make_db_with_flush_populating(scalar_side_effect, execute_return):
    """db.add 로 들어온 ORM 객체의 id/created_at 을 db.flush 시점에 채워줌."""
    added = []
    db = AsyncMock()
    db.scalar = AsyncMock(side_effect=scalar_side_effect)
    db.execute = AsyncMock(return_value=execute_return)
    db.add = lambda obj: added.append(obj)

    async def _flush():
        now = datetime.now(timezone.utc)
        for obj in added:
            if hasattr(obj, "id") and getattr(obj, "id", None) is None:
                # ReportSavedResponse.id 가 str 이므로 str(uuid) 로 채움
                obj.id = str(uuid4())
            if hasattr(obj, "created_at") and getattr(obj, "created_at", None) is None:
                obj.created_at = now

    db.flush = AsyncMock(side_effect=_flush)
    return db


@pytest.mark.asyncio
async def test_save_report_notifies_a_b_c_union():
    """A(요청자) + B(현장 관리자) + C(조직 admin) 모두에게 알림, 중복 제거."""
    from app.api import report as report_module
    from app.models.organization import Organization
    from app.models.site import Site
    from app.models.user import User

    requester_id = uuid4()
    site_creator_id = uuid4()
    assigned_user_id = uuid4()
    org_admin1_id = uuid4()
    org_admin2_id = uuid4()
    site_id = uuid4()
    org_id = uuid4()

    fake_org = Organization(id=org_id, name="Acme")
    fake_user = User(
        id=requester_id, name="요청자",
        email="r@x.com", phone="010-0000-0000",
        password_hash="x", account_type="personal",
    )
    fake_member = type("M", (), {"role": "owner"})()

    fake_site = Site(
        id=site_id, name="강남현장",
        organization_id=org_id,
        created_by=site_creator_id,
        assigned_members=[
            {"id": str(assigned_user_id), "name": "배정담당", "role": "manager"},
            {"id": "not-a-uuid", "name": "잘못된", "role": "x"},  # 깨진 항목 무시
        ],
    )

    # 조직 admin/owner 목록 — 한 명은 요청자와 동일(중복 테스트)
    admin_result = type("R", (), {
        "all": lambda self: [(org_admin1_id,), (org_admin2_id,), (requester_id,)],
    })()

    db = _make_db_with_flush_populating(
        scalar_side_effect=[fake_site],
        execute_return=admin_result,
    )

    payload = type("P", (), {
        "site_id": site_id,
        "title": "4월 점검 보고서",
        "building_name": "강남타워",
        "inspector_name": "김검사",
        "provider": "claude",
        "content": "본문",
        "defect_count": 12,
        "high_count": 3,
        "med_count": 5,
        "low_count": 4,
    })()

    with patch.object(
        report_module.notification_service, "create_for_many", new_callable=AsyncMock,
    ) as mock_many:
        await report_module.save_report(
            payload=payload,
            org_tuple=(fake_user, fake_member, fake_org),
            db=db,
        )

        mock_many.assert_awaited_once()
        kwargs = mock_many.call_args.kwargs
        ids = set(kwargs["user_ids"])
        # 합집합: 요청자 + 등록자 + 배정자 + admin1 + admin2 (요청자 중복 제거)
        assert ids == {requester_id, site_creator_id, assigned_user_id, org_admin1_id, org_admin2_id}
        assert kwargs["category"] == "report"
        assert "4월 점검 보고서" in kwargs["title"]


@pytest.mark.asyncio
async def test_save_report_without_site_skips_b():
    """site_id 가 없고 조직에 기존 현장도 없으면, 기본 현장이 생성된다(created_by=요청자).

    그 기본 현장의 관리자(B)는 요청자(A) 한 명뿐이므로 결과적으로 A + C 만 수신.
    (save_report 가 site_id 미지정 시 '최근 현장 자동 연결, 없으면 기본 현장 생성'으로
     바뀐 뒤의 동작을 반영 — scalar 가 None 을 반환하면 기본 현장 생성 경로를 탄다.)
    """
    from app.api import report as report_module
    from app.models.organization import Organization
    from app.models.user import User

    requester_id = uuid4()
    org_admin_id = uuid4()
    org_id = uuid4()

    fake_org = Organization(id=org_id, name="Solo")
    fake_user = User(
        id=requester_id, name="요청자",
        email="r@x.com", phone="010-0000-0000",
        password_hash="x", account_type="personal",
    )
    fake_member = type("M", (), {"role": "admin"})()

    admin_result = type("R", (), {"all": lambda self: [(org_admin_id,)]})()

    # site_id 없음 → 최근 현장 조회 scalar 가 None → 기본 현장 생성(created_by=요청자)
    db = _make_db_with_flush_populating(
        scalar_side_effect=[None],
        execute_return=admin_result,
    )

    payload = type("P", (), {
        "site_id": None,
        "title": "현장없는보고서",
        "building_name": None,
        "inspector_name": None,
        "provider": "claude",
        "content": "본문",
        "defect_count": 0, "high_count": 0, "med_count": 0, "low_count": 0,
    })()

    with patch.object(
        report_module.notification_service, "create_for_many", new_callable=AsyncMock,
    ) as mock_many:
        await report_module.save_report(
            payload=payload,
            org_tuple=(fake_user, fake_member, fake_org),
            db=db,
        )

        kwargs = mock_many.call_args.kwargs
        assert set(kwargs["user_ids"]) == {requester_id, org_admin_id}
