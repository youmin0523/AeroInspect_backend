# =============================================
# tests/test_oauth_email_verification.py
# 역할: OAuth 자동 계정연결의 verified-email 게이트 검증.
#   회귀 방지: _find_or_create_oauth_user 가 이메일만 일치하면 기존 로컬 계정에
#   소셜 ID 를 무조건 연결해, 공격자가 피해자 이메일로 미검증 소셜 계정을 만들면
#   피해자 계정을 탈취할 수 있었다. 이제 email_verified=True 일 때만 연결.
# 실행: pytest tests/test_oauth_email_verification.py -v
# =============================================

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from fastapi import HTTPException

from app.api.oauth import _find_or_create_oauth_user


def _result_returning(obj):
    r = MagicMock()
    r.scalar_one_or_none.return_value = obj
    return r


def _db_with_lookups(oauth_hit, email_hit):
    """db.execute 가 1) oauth_id 조회 2) email 조회 순으로 반환하도록 구성."""
    db = AsyncMock()
    db.execute = AsyncMock(side_effect=[
        _result_returning(oauth_hit),
        _result_returning(email_hit),
    ])
    db.flush = AsyncMock()
    db.add = lambda obj: None
    return db


@pytest.mark.asyncio
async def test_unverified_email_does_not_link_existing_account():
    """미검증 이메일이 기존 로컬 계정과 일치하면 403 (자동 연결 거부)."""
    existing = SimpleNamespace(
        id=uuid4(), email="victim@example.com",
        oauth_provider=None, oauth_id=None,
    )
    db = _db_with_lookups(oauth_hit=None, email_hit=existing)

    with pytest.raises(HTTPException) as exc:
        await _find_or_create_oauth_user(
            db, provider="google", oauth_id="attacker-sub",
            email="victim@example.com", name="Attacker",
            email_verified=False,
        )
    assert exc.value.status_code == 403
    # 피해자 계정에 공격자 소셜 ID 가 절대 연결되지 않아야 함
    assert existing.oauth_id is None
    assert existing.oauth_provider is None


@pytest.mark.asyncio
async def test_verified_email_links_existing_account():
    """검증된 이메일이면 기존 계정에 소셜 ID 를 연결."""
    existing = SimpleNamespace(
        id=uuid4(), email="user@example.com",
        oauth_provider=None, oauth_id=None,
    )
    db = _db_with_lookups(oauth_hit=None, email_hit=existing)

    user = await _find_or_create_oauth_user(
        db, provider="google", oauth_id="real-sub-123",
        email="user@example.com", name="User",
        email_verified=True,
    )
    assert user is existing
    assert existing.oauth_provider == "google"
    assert existing.oauth_id == "real-sub-123"


@pytest.mark.asyncio
async def test_existing_oauth_id_returns_immediately_without_email_check():
    """이미 oauth_id 로 연결된 계정은 이메일 검증 없이 즉시 반환."""
    linked = SimpleNamespace(
        id=uuid4(), email="x@example.com",
        oauth_provider="google", oauth_id="real-sub-123",
    )
    db = AsyncMock()
    db.execute = AsyncMock(side_effect=[_result_returning(linked)])
    db.flush = AsyncMock()

    user = await _find_or_create_oauth_user(
        db, provider="google", oauth_id="real-sub-123",
        email="x@example.com", name="X",
        email_verified=False,  # 무시되어야 함
    )
    assert user is linked
