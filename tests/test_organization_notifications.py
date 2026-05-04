"""
tests/test_organization_notifications.py
역할: 조직 멤버 배정 / 초대코드 가입 시 notification_service 가 호출되는지 검증.

- assign_member  → 배정된 본인에게 1건
- join_by_invite_code → 그 조직의 owner/admin 들에게 일괄
실행: pytest tests/test_organization_notifications.py -v
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest


@pytest.mark.asyncio
async def test_assign_member_creates_notification_for_target():
    """assign_member 가 배정 대상자에게 notification_service.create() 를 호출."""
    from app.api import organization as org_module
    from app.models.organization import Organization
    from app.models.user import User

    target_user_id = uuid4()
    org_id = uuid4()

    fake_org = Organization(id=org_id, name="Acme Inc")
    fake_target = User(
        id=target_user_id, name="박철수",
        email="p@x.com", phone="010-0000-0000",
        password_hash="x", account_type="personal",
    )
    fake_caller = User(
        id=uuid4(), name="관리자",
        email="a@x.com", phone="010-1111-1111",
        password_hash="x", account_type="personal",
        is_superadmin=True,
    )

    db = AsyncMock()
    # scalar 호출 순서: target_org / target_user / 기존 멤버십(None)
    db.scalar = AsyncMock(side_effect=[fake_org, fake_target, None])
    db.flush = AsyncMock()
    db.add = lambda obj: None

    payload = type("P", (), {
        "user_id": target_user_id,
        "organization_id": org_id,
        "role": "member",
        "department": "개발",
        "position": None,
    })()

    with patch.object(org_module.notification_service, "create", new_callable=AsyncMock) as mock_create:
        await org_module.assign_member(payload=payload, current_user=fake_caller, db=db)

        mock_create.assert_awaited_once()
        kwargs = mock_create.call_args.kwargs
        assert kwargs["user_id"] == target_user_id
        assert kwargs["category"] == "team"
        assert "Acme Inc" in kwargs["title"]
        assert kwargs["metadata"]["organization_id"] == str(org_id)


@pytest.mark.asyncio
async def test_join_by_invite_code_notifies_admins():
    """join_by_invite_code 가 그 조직의 owner/admin 들에게 알림을 일괄 발송."""
    from app.api import organization as org_module
    from app.models.organization import Organization
    from app.models.user import User

    org_id = uuid4()
    new_user_id = uuid4()
    admin1_id = uuid4()
    admin2_id = uuid4()

    fake_org = Organization(
        id=org_id, name="Beta Corp",
        biz_number="123-45-67890", invite_code="ABC12345",
    )
    fake_org.created_at = datetime(2026, 1, 1)
    fake_user = User(
        id=new_user_id, name="이영희",
        email="y@x.com", phone="010-2222-2222",
        password_hash="x", account_type="personal",
    )

    # admin user_id 목록 + 본인 포함
    admin_result = type("R", (), {
        "all": lambda self: [(admin1_id,), (admin2_id,), (new_user_id,)],
    })()

    db = AsyncMock()
    # scalar 호출 순서: org 조회 / 기존 멤버십(None) / count(3)
    db.scalar = AsyncMock(side_effect=[fake_org, None, 3])
    db.execute = AsyncMock(return_value=admin_result)
    db.add = lambda obj: None
    db.flush = AsyncMock()

    payload = type("P", (), {"invite_code": "abc12345"})()

    with patch.object(org_module.notification_service, "create_for_many", new_callable=AsyncMock) as mock_many:
        await org_module.join_by_invite_code(payload=payload, current_user=fake_user, db=db)

        mock_many.assert_awaited_once()
        kwargs = mock_many.call_args.kwargs
        # 본인은 제외되어야 함
        assert new_user_id not in kwargs["user_ids"]
        assert set(kwargs["user_ids"]) == {admin1_id, admin2_id}
        assert kwargs["category"] == "team"
        assert "이영희" in kwargs["title"]


@pytest.mark.asyncio
async def test_join_by_invite_code_no_admins_no_notification():
    """admin/owner 가 한 명도 없으면 알림 호출이 일어나지 않아야 함."""
    from app.api import organization as org_module
    from app.models.organization import Organization
    from app.models.user import User

    fake_org = Organization(
        id=uuid4(), name="Solo Inc",
        biz_number=None, invite_code="XYZ99999",
    )
    fake_org.created_at = datetime(2026, 1, 1)
    fake_user = User(
        id=uuid4(), name="혼자",
        email="s@x.com", phone="010-3333-3333",
        password_hash="x", account_type="personal",
    )

    empty_result = type("R", (), {"all": lambda self: []})()

    db = AsyncMock()
    db.scalar = AsyncMock(side_effect=[fake_org, None, 1])
    db.execute = AsyncMock(return_value=empty_result)
    db.add = lambda obj: None
    db.flush = AsyncMock()

    payload = type("P", (), {"invite_code": "xyz99999"})()

    with patch.object(org_module.notification_service, "create_for_many", new_callable=AsyncMock) as mock_many:
        await org_module.join_by_invite_code(payload=payload, current_user=fake_user, db=db)
        mock_many.assert_not_awaited()
