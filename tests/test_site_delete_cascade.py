# =============================================
# tests/test_site_delete_cascade.py
# 역할: DELETE /sites/{id} 가 자식(defect/report/telemetry) 을 먼저 정리하는지 검증.
#   회귀 방지: site_id FK 에 ondelete 가 없어, 하자/보고서/텔레메트리가 달린 현장을
#   삭제하면 IntegrityError → 500 이 떴다. 이제 앱 레벨에서 자식을 bulk delete 한 뒤
#   site 를 삭제한다. (inspection_schedules 는 DB CASCADE)
# 실행: pytest tests/test_site_delete_cascade.py -v
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
def delete_client():
    """site 가 존재하고 자식이 있는 상황을 모사하며 실행된 statement 를 기록."""
    org_tuple = _make_org_tuple()
    site = SimpleNamespace(id=uuid4(), organization_id=org_tuple[2].id, name="현장")
    statements: list = []

    db = AsyncMock()

    # 첫 execute: site select → site, 두번째: crop 경로 select → 빈 목록,
    # 이후: 자식 bulk delete (반환값 무관)
    site_res = MagicMock()
    site_res.scalar_one_or_none.return_value = site
    crop_res = MagicMock()
    crop_res.all.return_value = []
    generic_res = MagicMock()

    seq = [site_res, crop_res, generic_res, generic_res, generic_res]

    async def _execute(stmt, *a, **k):
        statements.append(str(stmt).upper())
        return seq.pop(0) if seq else generic_res

    db.execute = _execute
    db.delete = AsyncMock()
    db.commit = AsyncMock()

    async def _override_org():
        return org_tuple

    async def _override_db():
        yield db

    app.dependency_overrides[get_current_org_member] = _override_org
    app.dependency_overrides[get_db] = _override_db

    yield AsyncClient(transport=ASGITransport(app=app), base_url="http://test"), statements, site, db

    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_delete_site_cleans_children_then_returns_204(delete_client):
    client, statements, site, db = delete_client
    async with client as ac:
        res = await ac.delete(f"/api/v1/sites/{site.id}")

    assert res.status_code == 204
    # 자식 3종 bulk delete 가 모두 발행됐는지 확인
    joined = " ".join(statements)
    assert "DELETE FROM DEFECT_LOGS" in joined
    assert "DELETE FROM REPORTS" in joined
    assert "DELETE FROM TELEMETRY_LOGS" in joined
    # site 자체 삭제 + commit
    db.delete.assert_awaited_once()
    db.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_delete_missing_site_returns_404(delete_client):
    client, statements, site, db = delete_client
    # site select 가 None 을 반환하도록 교체
    db.execute  # noqa
    async def _execute_none(stmt, *a, **k):
        statements.append(str(stmt).upper())
        r = MagicMock()
        r.scalar_one_or_none.return_value = None
        return r
    db.execute = _execute_none

    async with client as ac:
        res = await ac.delete(f"/api/v1/sites/{uuid4()}")
    assert res.status_code == 404
    db.delete.assert_not_awaited()
