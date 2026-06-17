# =============================================
# tests/test_telemetry_scoping.py
# 역할: GET /telemetry, /telemetry/latest 의 조직 스코핑 회귀 검증.
#   회귀 방지: 한때 두 GET 이 get_current_user 만 의존하고 site/org 필터가 없어,
#   A 조직 사용자가 B 조직 드론의 위치·비행경로 전체를 조회할 수 있었다.
#   이제 get_current_org_member 필수 + (site_id IS NULL OR 내 org site) 로 제한.
# 실행: pytest tests/test_telemetry_scoping.py -v
# =============================================

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from httpx import AsyncClient, ASGITransport

from app.dependencies import get_current_org_member, get_db
from app.main import app


def _make_org_tuple():
    user = SimpleNamespace(id=uuid4(), email="t@example.com", is_superadmin=False)
    org = SimpleNamespace(id=uuid4(), name="Org A")
    member = SimpleNamespace(role="owner", user_id=user.id, organization_id=org.id)
    return user, member, org


@pytest.fixture
def capturing_client():
    """org 인증 + db.execute 로 들어온 statement 를 캡처하는 클라이언트."""
    captured: dict = {}
    org_tuple = _make_org_tuple()

    db = AsyncMock()
    db.scalar = AsyncMock(return_value=0)

    result = MagicMock()
    scalars_mock = MagicMock()
    scalars_mock.all.return_value = []
    result.scalars.return_value = scalars_mock
    result.scalar_one_or_none.return_value = None

    async def _execute(stmt, *a, **k):
        captured["stmt"] = stmt
        return result

    db.execute = _execute

    async def _override_org():
        return org_tuple

    async def _override_db():
        yield db

    app.dependency_overrides[get_current_org_member] = _override_org
    app.dependency_overrides[get_db] = _override_db

    yield AsyncClient(transport=ASGITransport(app=app), base_url="http://test"), captured, org_tuple

    app.dependency_overrides.clear()


@pytest.fixture
def unauth_client():
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


@pytest.mark.asyncio
async def test_list_telemetry_without_auth_returns_401(unauth_client):
    async with unauth_client as ac:
        res = await ac.get("/api/v1/telemetry")
    assert res.status_code == 401


@pytest.mark.asyncio
async def test_latest_telemetry_without_auth_returns_401(unauth_client):
    async with unauth_client as ac:
        res = await ac.get("/api/v1/telemetry/latest")
    assert res.status_code == 401


@pytest.mark.asyncio
async def test_list_telemetry_query_is_org_scoped(capturing_client):
    client, captured, org_tuple = capturing_client
    async with client as ac:
        res = await ac.get("/api/v1/telemetry")
    assert res.status_code == 200
    sql = str(captured["stmt"]).upper()
    # 다른 org 의 site 텔레메트리가 새지 않도록 organization_id 로 스코핑
    assert "ORGANIZATION_ID" in sql
    # 전역(현장 미지정) 비행은 계속 노출 (site_id IS NULL)
    assert "IS NULL" in sql


@pytest.mark.asyncio
async def test_latest_telemetry_query_is_org_scoped(capturing_client):
    client, captured, org_tuple = capturing_client
    async with client as ac:
        res = await ac.get("/api/v1/telemetry/latest")
    # 빈 결과면 404 (스코핑 자체는 통과)
    assert res.status_code in (200, 404)
    sql = str(captured["stmt"]).upper()
    assert "ORGANIZATION_ID" in sql
    assert "IS NULL" in sql
