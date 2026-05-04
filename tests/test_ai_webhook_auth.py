"""
tests/test_ai_webhook_auth.py
역할: /api/v1/ai/* 웹훅 엔드포인트의 X-AI-Webhook-Secret 헤더 검증.

검증 범위:
  - 헤더 누락 → 401
  - 헤더 불일치 → 401
  - 헤더 일치 → 비즈니스 로직 도달 (200/201)
  - 서버 시크릿 미설정 → 401 (안전한 기본값)
실행: pytest tests/test_ai_webhook_auth.py -v
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from httpx import AsyncClient, ASGITransport

from app.config import settings
from app.dependencies import get_db, get_ws_manager
from app.main import app


_WEBHOOK_SECRET = "test-webhook-secret-abcdef123456"


@pytest.fixture
def configured_secret(monkeypatch):
    """서버에 시크릿이 설정된 상태."""
    monkeypatch.setattr(settings, "AI_WEBHOOK_SECRET", _WEBHOOK_SECRET)
    yield _WEBHOOK_SECRET


@pytest.fixture
def empty_secret(monkeypatch):
    """서버 시크릿이 빈 문자열(미설정) 상태."""
    monkeypatch.setattr(settings, "AI_WEBHOOK_SECRET", "")
    yield


@pytest.fixture
def fake_db():
    """DB 세션을 통과시키는 AsyncMock (실제 INSERT는 안 함)."""
    db = AsyncMock()
    db.add = lambda obj: None
    db.flush = AsyncMock()

    async def _gen():
        yield db

    app.dependency_overrides[get_db] = _gen
    yield db
    app.dependency_overrides.pop(get_db, None)


@pytest.fixture
def fake_ws_manager():
    manager = AsyncMock()
    manager.broadcast = AsyncMock()
    app.dependency_overrides[get_ws_manager] = lambda: manager
    yield manager
    app.dependency_overrides.pop(get_ws_manager, None)


def _thermal_payload():
    return {
        "zone": "wall_north",
        "max_temp": 28.5,
        "min_temp": 18.2,
        "avg_temp": 23.0,
    }


@pytest.mark.asyncio
async def test_thermal_without_header_returns_401(configured_secret, fake_ws_manager):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/api/v1/ai/thermal", json=_thermal_payload())
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_thermal_with_wrong_header_returns_401(configured_secret, fake_ws_manager):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/api/v1/ai/thermal",
            json=_thermal_payload(),
            headers={"X-AI-Webhook-Secret": "wrong-secret"},
        )
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_thermal_with_correct_header_passes(configured_secret, fake_ws_manager):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/api/v1/ai/thermal",
            json=_thermal_payload(),
            headers={"X-AI-Webhook-Secret": configured_secret},
        )
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
    fake_ws_manager.broadcast.assert_awaited_once()


@pytest.mark.asyncio
async def test_thermal_when_server_secret_unset_returns_401(empty_secret, fake_ws_manager):
    """서버 시크릿이 비어있으면 어떤 헤더도 통과하지 않아야 함."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/api/v1/ai/thermal",
            json=_thermal_payload(),
            headers={"X-AI-Webhook-Secret": ""},
        )
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_detection_without_header_returns_401(configured_secret, fake_db, fake_ws_manager):
    payload = {
        "area": "A",
        "category_code": "A-02",
        "defect_type": "구조 균열",
        "severity": "HIGH",
        "confidence": 0.87,
    }
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/api/v1/ai/detection", json=payload)
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_batch_without_header_returns_401(configured_secret, fake_db, fake_ws_manager):
    payload = {"detections": []}
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/api/v1/ai/batch", json=payload)
    assert resp.status_code == 401
