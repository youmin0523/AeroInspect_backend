"""
tests/test_ws_channel_whitelist.py
역할: WebSocket 채널 화이트리스트 검증.
  - 고정 채널 (defects/telemetry/thermal/camera) 허용
  - 동적 prefix (chat:, user:, notifications:) 허용
  - 알 수 없는 채널은 거부 (기본값으로 폴백)
실행: pytest tests/test_ws_channel_whitelist.py -v
"""

from uuid import uuid4

import pytest

from app.api.websocket import _is_valid_channel


@pytest.mark.parametrize("channel", ["defects", "telemetry", "thermal", "camera"])
def test_static_channels_pass(channel):
    assert _is_valid_channel(channel) is True


@pytest.mark.parametrize("prefix", ["chat:", "user:", "notifications:"])
def test_dynamic_channels_pass(prefix):
    assert _is_valid_channel(f"{prefix}{uuid4()}") is True


def test_notifications_channel_passes_for_notification_service():
    """notification_service.create() 가 broadcast 하는 채널 형식 검증."""
    user_id = uuid4()
    assert _is_valid_channel(f"notifications:{user_id}") is True


def test_unknown_channel_rejected():
    assert _is_valid_channel("admin") is False
    assert _is_valid_channel("") is False
    assert _is_valid_channel("random-channel") is False
