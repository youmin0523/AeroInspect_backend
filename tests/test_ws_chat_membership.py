"""
tests/test_ws_chat_membership.py
역할: WebSocket chat:{conversation_id} 구독이 대화방 멤버에게만 허용되는지 검증.
  회귀 방지: 멤버십을 발행 측에서만 검증해, 비멤버가 대화방 UUID 로 구독하면
  메시지를 실시간 수신할 수 있었다(구독 IDOR). 이제 구독 시점에 멤버십을 확인하고
  비멤버 채널은 거부(rejected)되며 공개 'defects' 로 폴백한다.
실행: pytest tests/test_ws_chat_membership.py -v
"""

from __future__ import annotations

from uuid import uuid4

import pytest
from starlette.testclient import TestClient

import app.api.websocket as ws_module
from app.dependencies import get_ws_manager
from app.main import app


class _FakeManager:
    """connection.established 만 받아볼 수 있으면 충분한 최소 WS 매니저."""

    async def connect(self, websocket, channel):
        await websocket.accept()

    def register(self, websocket, channel):
        pass

    async def send_personal(self, websocket, message):
        await websocket.send_json(message)

    def disconnect(self, websocket, channel):
        pass


@pytest.fixture
def fake_ws(monkeypatch):
    uid = str(uuid4())
    cid = str(uuid4())
    monkeypatch.setattr(ws_module, "decode_access_token", lambda t: uid)
    app.dependency_overrides[get_ws_manager] = lambda: _FakeManager()
    yield uid, cid
    app.dependency_overrides.clear()


def test_non_member_chat_channel_is_rejected(fake_ws, monkeypatch):
    uid, cid = fake_ws

    async def _not_member(conversation_id, user_id):
        return False

    monkeypatch.setattr(ws_module, "_is_conversation_member", _not_member)

    client = TestClient(app)
    with client.websocket_connect(f"/api/v1/ws?channels=chat:{cid}&token=x") as wsc:
        msg = wsc.receive_json()

    assert msg["type"] == "connection.established"
    assert f"chat:{cid}" in msg["data"]["rejected"]
    assert f"chat:{cid}" not in msg["data"]["channels"]
    # 유효 채널 0개 → 공개 'defects' 로 폴백
    assert msg["data"]["channels"] == ["defects"]


def test_member_chat_channel_is_allowed(fake_ws, monkeypatch):
    uid, cid = fake_ws

    async def _is_member(conversation_id, user_id):
        # 멤버십 확인이 올바른 인자로 호출되는지 함께 검증
        assert conversation_id == cid
        assert user_id == uid
        return True

    monkeypatch.setattr(ws_module, "_is_conversation_member", _is_member)

    client = TestClient(app)
    with client.websocket_connect(f"/api/v1/ws?channels=chat:{cid}&token=x") as wsc:
        msg = wsc.receive_json()

    assert msg["type"] == "connection.established"
    assert f"chat:{cid}" in msg["data"]["channels"]
    assert f"chat:{cid}" not in msg["data"]["rejected"]


def test_chat_channel_without_token_is_rejected(monkeypatch):
    """토큰이 없으면 token_sub=None → _authorize_channel 단계에서 이미 거부."""
    cid = str(uuid4())
    monkeypatch.setattr(ws_module, "decode_access_token", lambda t: None)
    app.dependency_overrides[get_ws_manager] = lambda: _FakeManager()
    try:
        client = TestClient(app)
        with client.websocket_connect(f"/api/v1/ws?channels=chat:{cid}") as wsc:
            msg = wsc.receive_json()
        assert f"chat:{cid}" in msg["data"]["rejected"]
        assert msg["data"]["channels"] == ["defects"]
    finally:
        app.dependency_overrides.clear()
