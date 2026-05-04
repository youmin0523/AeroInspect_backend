# =============================================
# tests/test_defects_api.py
# 역할: 하자 탐지 로그 REST API 라우터 레벨 테스트
#       - 조직 스코핑(get_current_org_member) 적용 이후 버전
#       - 실제 DB·JWT 대신 dependency_overrides + AsyncMock으로 경로 검증
#
# 검증 범위:
#   - 인증/조직 통과 시 200 반환 + 응답 구조
#   - 필터 쿼리 파라미터 수용 여부
#   - 404 케이스 (scalar_one_or_none → None)
#   - 무인증 시 401 (overrides 없는 클라이언트)
# 실행: pytest tests/test_defects_api.py -v
# =============================================

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from httpx import AsyncClient, ASGITransport

from app.dependencies import get_current_org_member, get_db
from app.main import app


# ── Fake 조직/DB 빌더 ─────────────────────────────────
def _make_org_tuple():
    """get_current_org_member 대체용 (user, member, org) 튜플."""
    user = SimpleNamespace(id=uuid4(), email="test@example.com", is_superadmin=False)
    org = SimpleNamespace(id=uuid4(), name="Test Org")
    member = SimpleNamespace(role="owner", user_id=user.id, organization_id=org.id)
    return user, member, org


def _make_empty_db():
    """defects.py 핸들러가 쓰는 scalar / execute 메서드만 빈 값으로 응답."""
    db = AsyncMock()

    # 카운트 쿼리용
    db.scalar = AsyncMock(return_value=0)

    # 목록/단건 조회용 Result 객체
    result = MagicMock()
    scalars_mock = MagicMock()
    scalars_mock.all.return_value = []
    result.scalars.return_value = scalars_mock
    result.scalar_one_or_none.return_value = None
    result.all.return_value = []

    # group_by 반환도 빈 iterable
    result.__iter__ = lambda self: iter([])

    db.execute = AsyncMock(return_value=result)
    return db


# ── Fixtures ───────────────────────────────────────────
@pytest.fixture
def authed_client():
    """조직 인증 + 빈 DB가 주입된 클라이언트."""
    org_tuple = _make_org_tuple()
    fake_db = _make_empty_db()

    async def _override_org():
        return org_tuple

    async def _override_db():
        yield fake_db

    app.dependency_overrides[get_current_org_member] = _override_org
    app.dependency_overrides[get_db] = _override_db

    yield AsyncClient(transport=ASGITransport(app=app), base_url="http://test")

    app.dependency_overrides.clear()


@pytest.fixture
def unauth_client():
    """dependency override 없는 순수 클라이언트 — 401 검증용."""
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


# ── 인증·조직 통과 경로 ────────────────────────────────
@pytest.mark.asyncio
async def test_get_defects_empty_returns_200(authed_client):
    """목록 조회 — 빈 결과도 200 + items/total 구조."""
    async with authed_client as ac:
        res = await ac.get("/api/v1/defects")
    assert res.status_code == 200
    data = res.json()
    assert "items" in data
    assert "total" in data
    assert data["items"] == []


@pytest.mark.asyncio
async def test_get_defect_summary_structure(authed_client):
    """요약 통계 — total / by_severity / by_area 필드 보장."""
    async with authed_client as ac:
        res = await ac.get("/api/v1/defects/summary")
    assert res.status_code == 200
    data = res.json()
    assert "total" in data
    assert "by_severity" in data
    assert "by_area" in data


@pytest.mark.asyncio
async def test_filter_defects_by_severity(authed_client):
    async with authed_client as ac:
        res = await ac.get("/api/v1/defects?severity=HIGH")
    assert res.status_code == 200


@pytest.mark.asyncio
async def test_filter_defects_by_area(authed_client):
    async with authed_client as ac:
        res = await ac.get("/api/v1/defects?area=A")
    assert res.status_code == 200


@pytest.mark.asyncio
async def test_get_nonexistent_defect_returns_404(authed_client):
    """scalar_one_or_none → None이면 404."""
    fake_id = "00000000-0000-0000-0000-000000000000"
    async with authed_client as ac:
        res = await ac.get(f"/api/v1/defects/{fake_id}")
    assert res.status_code == 404


# ── 무인증 경로 ────────────────────────────────────────
@pytest.mark.asyncio
async def test_list_without_auth_returns_401(unauth_client):
    """토큰 없으면 401 (get_current_user 단계)."""
    async with unauth_client as ac:
        res = await ac.get("/api/v1/defects")
    assert res.status_code == 401


@pytest.mark.asyncio
async def test_summary_without_auth_returns_401(unauth_client):
    async with unauth_client as ac:
        res = await ac.get("/api/v1/defects/summary")
    assert res.status_code == 401


@pytest.mark.asyncio
async def test_delete_without_auth_returns_401(unauth_client):
    """오늘 추가한 조직 스코프가 DELETE에도 걸렸는지 확인."""
    fake_id = "00000000-0000-0000-0000-000000000000"
    async with unauth_client as ac:
        res = await ac.delete(f"/api/v1/defects/{fake_id}")
    assert res.status_code == 401
