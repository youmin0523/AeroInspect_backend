# =============================================
# tests/test_thermal_screening_review.py
# 역할: 단열 스크리닝 검수 피드백 엔드포인트(POST /thermal-screening/review) 라우터 레벨 테스트
#       - test_defects_api.py 패턴 미러링: dependency_overrides + AsyncMock (운영 DB/JWT 미접촉)
#
# 검증 범위:
#   - confirmed: 200 + 응답 구조 + audit_logs 적재(db.add) + WS 브로드캐스트 호출
#   - flagged_false_positive: review_note 없으면 400, 있으면 200
#   - 잘못된 review_status: 422 (pydantic pattern)
#   - 무인증: 401
# 실행: pytest tests/test_thermal_screening_review.py -v
# =============================================

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from httpx import AsyncClient, ASGITransport

from app.dependencies import get_current_org_member, get_db, get_ws_manager
from app.main import app


def _make_org_tuple():
    user = SimpleNamespace(id=uuid4(), email="test@example.com", is_superadmin=False)
    org = SimpleNamespace(id=uuid4(), name="Test Org")
    member = SimpleNamespace(role="owner", user_id=user.id, organization_id=org.id)
    return user, member, org


@pytest.fixture
def authed():
    """조직 인증 + AsyncMock DB/WS 가 주입된 클라이언트 컨텍스트."""
    org_tuple = _make_org_tuple()
    fake_db = AsyncMock()
    fake_db.add = MagicMock()       # write_audit 의 db.add 동기 호출
    fake_ws = AsyncMock()           # broadcast 는 await 됨

    async def _override_org():
        return org_tuple

    async def _override_db():
        yield fake_db

    def _override_ws():
        return fake_ws

    app.dependency_overrides[get_current_org_member] = _override_org
    app.dependency_overrides[get_db] = _override_db
    app.dependency_overrides[get_ws_manager] = _override_ws

    client = AsyncClient(transport=ASGITransport(app=app), base_url="http://test")
    yield SimpleNamespace(client=client, db=fake_db, ws=fake_ws)

    app.dependency_overrides.clear()


@pytest.fixture
def unauth_client():
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


_BASE_PAYLOAD = {
    "video_timestamp_sec": 45.3,
    "filename": "thermal_test.mp4",
    "frame_w": 1920,
    "frame_h": 1080,
    "bbox": {"x1": 820, "y1": 410, "x2": 980, "y2": 560},
    "kind": "patch",
    "severity": "MED",
    "score": 7.3,
    "client_item_id": "45.300_1",
}

URL = "/api/v1/thermal-screening/review"


@pytest.mark.asyncio
async def test_confirm_returns_200_and_audits(authed):
    body = {**_BASE_PAYLOAD, "review_status": "confirmed"}
    async with authed.client as ac:
        res = await ac.post(URL, json=body)
    assert res.status_code == 200, res.text
    data = res.json()
    assert data["ok"] is True
    assert data["review_status"] == "confirmed"
    assert data["client_item_id"] == "45.300_1"
    assert "reviewed_at" in data and data["reviewed_by_user_id"]
    # audit_logs 적재(db.add) + 같은-세션 반영용 WS 브로드캐스트 호출 확인
    assert authed.db.add.called, "audit_logs 에 적재(db.add)되지 않음"
    assert authed.ws.broadcast.await_count == 1


@pytest.mark.asyncio
async def test_flag_without_note_returns_400(authed):
    body = {**_BASE_PAYLOAD, "review_status": "flagged_false_positive"}  # note 누락
    async with authed.client as ac:
        res = await ac.post(URL, json=body)
    assert res.status_code == 400, res.text


@pytest.mark.asyncio
async def test_flag_with_note_returns_200(authed):
    body = {**_BASE_PAYLOAD, "review_status": "flagged_false_positive", "review_note": "창틀 반사를 단열 의심으로 오인"}
    async with authed.client as ac:
        res = await ac.post(URL, json=body)
    assert res.status_code == 200, res.text
    assert res.json()["review_status"] == "flagged_false_positive"


@pytest.mark.asyncio
async def test_invalid_status_returns_422(authed):
    body = {**_BASE_PAYLOAD, "review_status": "bogus"}
    async with authed.client as ac:
        res = await ac.post(URL, json=body)
    assert res.status_code == 422, res.text


@pytest.mark.asyncio
async def test_without_auth_returns_401(unauth_client):
    body = {**_BASE_PAYLOAD, "review_status": "confirmed"}
    async with unauth_client as ac:
        res = await ac.post(URL, json=body)
    assert res.status_code == 401, res.text
