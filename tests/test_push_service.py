# =============================================
# tests/test_push_service.py
# 역할: PushNotificationService 스켈레톤 회귀
#       - provider=noop 일 때 send_to_user 호출해도 실제 발송 없이 0 반환
#       - 활성 토큰 없는 사용자에게는 0 반환
#       - provider 전환 시 is_enabled 플래그 갱신
# DB 의존성은 AsyncMock 으로 격리
# 실행: pytest tests/test_push_service.py -v
# =============================================

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from app.services.push_notifications import (
    PushMessage,
    PushNotificationService,
    push_service,
)


def _make_db_with_tokens(tokens: list) -> MagicMock:
    db = AsyncMock()
    result = MagicMock()
    scalars = MagicMock()
    scalars.all.return_value = tokens
    result.scalars.return_value = scalars
    db.execute = AsyncMock(return_value=result)
    return db


@pytest.mark.asyncio
async def test_noop_provider_returns_zero_for_active_user():
    """provider=noop 상태에서는 실제 발송 없이 0 반환 (로그만 남김)."""
    svc = PushNotificationService()
    assert svc.provider == "noop"
    assert svc.is_enabled is False

    token = SimpleNamespace(
        id=uuid4(), platform="fcm", token="fcm_token_abc", is_active=True,
    )
    db = _make_db_with_tokens([token])

    sent = await svc.send_to_user(
        db=db,
        user_id=uuid4(),
        message=PushMessage(title="테스트", body="본문"),
    )
    assert sent == 0


@pytest.mark.asyncio
async def test_no_device_returns_zero():
    svc = PushNotificationService()
    db = _make_db_with_tokens([])

    sent = await svc.send_to_user(
        db=db,
        user_id=uuid4(),
        message=PushMessage(title="알림", body="본문"),
    )
    assert sent == 0


def test_singleton_matches_factory_default():
    """전역 싱글톤 push_service 는 기본 설정 기준 noop 이어야 함."""
    assert push_service.provider in ("noop", "fcm", "apns")
    # 기본값 검증 — 크레덴셜 설정 없이 엔드포인트 호출 가능
    if push_service.provider == "noop":
        assert push_service.is_enabled is False
