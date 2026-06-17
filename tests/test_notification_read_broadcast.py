"""
tests/test_notification_read_broadcast.py
역할: 알림 읽음 처리 시 notifications:{uid} 채널로 read/read_all 이 전파되는지 검증.
  회귀 방지: 읽음 처리가 DB 만 갱신하고 WS 로 전파하지 않아, 같은 사용자의 다른
  탭/기기 배지가 새 알림이 올 때까지 갱신되지 않는 드리프트가 있었다.
실행: pytest tests/test_notification_read_broadcast.py -v
"""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from httpx import AsyncClient, ASGITransport

import app.api.notifications as notif_api
from app.dependencies import get_current_user, get_db
from app.main import app
from app.services.notification_service import NotificationService


# ── 서비스 레벨: 브로드캐스트 페이로드 ──────────────────────
@pytest.mark.asyncio
async def test_broadcast_read_payload():
    ws = MagicMock()
    ws.broadcast = AsyncMock()
    svc = NotificationService(ws)
    uid, nid = uuid4(), uuid4()

    await svc.broadcast_read(uid, nid)

    ws.broadcast.assert_awaited_once_with(
        f"notifications:{uid}",
        {"type": "notification.read", "data": {"id": str(nid)}},
    )


@pytest.mark.asyncio
async def test_broadcast_read_all_payload():
    ws = MagicMock()
    ws.broadcast = AsyncMock()
    svc = NotificationService(ws)
    uid = uuid4()

    await svc.broadcast_read_all(uid)

    ws.broadcast.assert_awaited_once_with(
        f"notifications:{uid}",
        {"type": "notification.read_all", "data": {}},
    )


# ── API 와이어링: 핸들러가 브로드캐스트를 호출하는지 ──────────
@pytest.fixture
def user_and_overrides():
    user = SimpleNamespace(id=uuid4())

    async def _override_user():
        return user

    app.dependency_overrides[get_current_user] = _override_user
    yield user
    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_mark_as_read_broadcasts(user_and_overrides, monkeypatch):
    user = user_and_overrides
    nid = uuid4()
    notif = SimpleNamespace(
        id=nid, user_id=user.id, category="report", title="t",
        message=None, metadata_=None, is_read=False,
        created_at=datetime.now(timezone.utc),
    )

    db = AsyncMock()
    result = MagicMock()
    result.scalar_one_or_none.return_value = notif
    db.execute = AsyncMock(return_value=result)
    db.flush = AsyncMock()

    async def _override_db():
        yield db

    app.dependency_overrides[get_db] = _override_db
    monkeypatch.setattr(notif_api.notification_service, "broadcast_read", AsyncMock())

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        res = await ac.patch(f"/api/v1/notifications/{nid}/read")

    assert res.status_code == 200
    notif_api.notification_service.broadcast_read.assert_awaited_once_with(user.id, nid)


@pytest.mark.asyncio
async def test_mark_all_read_broadcasts(user_and_overrides, monkeypatch):
    user = user_and_overrides

    db = AsyncMock()
    db.execute = AsyncMock(return_value=SimpleNamespace(rowcount=3))
    db.flush = AsyncMock()

    async def _override_db():
        yield db

    app.dependency_overrides[get_db] = _override_db
    monkeypatch.setattr(notif_api.notification_service, "broadcast_read_all", AsyncMock())

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        res = await ac.patch("/api/v1/notifications/read-all")

    assert res.status_code == 200
    assert res.json()["updated"] == 3
    notif_api.notification_service.broadcast_read_all.assert_awaited_once_with(user.id)
