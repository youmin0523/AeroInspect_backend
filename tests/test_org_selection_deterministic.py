"""
tests/test_org_selection_deterministic.py
역할: _get_user_org 의 조직 선택이 결정적이고 X-Organization-Id 를 존중하는지 검증.
  회귀 방지: ORDER BY 가 없어 다중 조직 사용자에게 임의 조직이 선택되고, 헤더도
  무시돼 멤버십/역할이 요청 의도와 다른 조직을 가리킬 수 있었다.
실행: pytest tests/test_org_selection_deterministic.py -v
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from app.api.organization import _get_user_org


def _db_capturing(row):
    captured = {}
    db = AsyncMock()
    result = MagicMock()
    result.first.return_value = row

    async def _execute(stmt, *a, **k):
        captured["sql"] = str(stmt)
        return result

    db.execute = _execute
    return db, captured


@pytest.mark.asyncio
async def test_no_header_uses_deterministic_order():
    member, org = SimpleNamespace(role="owner"), SimpleNamespace(id=uuid4())
    db, captured = _db_capturing((member, org))

    got_member, got_org = await _get_user_org(db, uuid4())
    assert (got_member, got_org) == (member, org)
    sql = captured["sql"].lower()
    assert "order by" in sql and "joined_at" in sql


@pytest.mark.asyncio
async def test_header_scopes_to_requested_org():
    member, org = SimpleNamespace(role="admin"), SimpleNamespace(id=uuid4())
    db, captured = _db_capturing((member, org))
    target = uuid4()

    await _get_user_org(db, uuid4(), str(target))
    sql = captured["sql"].lower()
    # 지정 조직으로 필터 (organizations.id = ...), 임의 최근순 정렬 아님
    assert "organizations.id" in sql
    assert "order by" not in sql


@pytest.mark.asyncio
async def test_invalid_org_id_returns_none_without_query():
    db, captured = _db_capturing(("x", "y"))
    member, org = await _get_user_org(db, uuid4(), "not-a-uuid")
    assert member is None and org is None
    assert "sql" not in captured  # 쿼리 자체를 실행하지 않음
