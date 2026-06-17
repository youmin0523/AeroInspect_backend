# =============================================
# tests/test_org_member_owner_guard.py
# 역할: PATCH /organizations/members/{id} 의 소유자 보호 가드 검증.
#   회귀 방지: update_member 에 owner 보호가 없어 (1) admin 이 owner 를 강등/비활성화하거나
#   (2) 마지막 owner 가 강등되어 조직이 무소유주(관리 복구 불가)가 될 수 있었다.
# 실행: pytest tests/test_org_member_owner_guard.py -v
# =============================================

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from httpx import AsyncClient, ASGITransport

from app.dependencies import get_current_user, get_db
from app.main import app


def _first_returning(tpl):
    r = MagicMock()
    r.first.return_value = tpl
    return r


def _make_user():
    return SimpleNamespace(id=uuid4())


def _make_member(role):
    return SimpleNamespace(
        role=role, department=None, position=None,
        status="active", started_at=None, ended_at=None,
    )


def _make_target_user():
    return SimpleNamespace(id=uuid4(), name="대상", email="t@example.com", phone="010")


def _setup(my_role, target_role, other_owner_count):
    """update_member 가 호출하는 db.execute/scalar 순서를 모사."""
    org = SimpleNamespace(id=uuid4(), name="Org")
    my_member = _make_member(my_role)
    target_member = _make_member(target_role)
    target_user = _make_target_user()

    db = AsyncMock()
    # 1) _get_user_org → (my_member, org)  2) main query → (target_member, target_user)
    db.execute = AsyncMock(side_effect=[
        _first_returning((my_member, org)),
        _first_returning((target_member, target_user)),
    ])
    db.scalar = AsyncMock(return_value=other_owner_count)
    db.flush = AsyncMock()

    cur = _make_user()

    async def _override_user():
        return cur

    async def _override_db():
        yield db

    app.dependency_overrides[get_current_user] = _override_user
    app.dependency_overrides[get_db] = _override_db
    return db, target_member, target_user


@pytest.fixture(autouse=True)
def _clear_overrides():
    yield
    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_admin_cannot_demote_owner():
    """admin 이 owner 를 강등하려 하면 403."""
    _setup(my_role="admin", target_role="owner", other_owner_count=5)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        res = await ac.patch(f"/api/v1/organizations/members/{uuid4()}", json={"role": "member"})
    assert res.status_code == 403


@pytest.mark.asyncio
async def test_cannot_demote_last_owner():
    """owner 가 마지막 owner 를 강등하면 400 (무소유주 방지)."""
    _setup(my_role="owner", target_role="owner", other_owner_count=0)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        res = await ac.patch(f"/api/v1/organizations/members/{uuid4()}", json={"role": "member"})
    assert res.status_code == 400


@pytest.mark.asyncio
async def test_can_demote_owner_when_another_owner_exists():
    """다른 owner 가 있으면 owner 강등 허용 → 200."""
    db, target_member, _ = _setup(my_role="owner", target_role="owner", other_owner_count=2)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        res = await ac.patch(f"/api/v1/organizations/members/{uuid4()}", json={"role": "member"})
    assert res.status_code == 200
    assert target_member.role == "member"


@pytest.mark.asyncio
async def test_can_update_regular_member():
    """일반 멤버 권한 변경은 소유자 가드와 무관하게 200."""
    db, target_member, _ = _setup(my_role="owner", target_role="member", other_owner_count=1)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        res = await ac.patch(f"/api/v1/organizations/members/{uuid4()}", json={"role": "admin"})
    assert res.status_code == 200
    assert target_member.role == "admin"
    db.scalar.assert_not_awaited()  # 일반 멤버 변경은 owner-count 조회 불필요
