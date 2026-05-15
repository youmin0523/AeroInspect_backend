"""
tests/test_floorplan_http_integration.py
역할: FastAPI TestClient 로 실 HTTP 라우터 통합 검증.
      인증 의존성은 dependency_overrides 로 우회 (단위 테스트 표준 패턴).

검증 엔드포인트:
  - POST /api/v1/floorplan/analyze
  - POST /api/v1/floorplan/validate
  - POST /api/v1/missions/autonomous-scan/start
  - GET  /api/v1/missions/{id}
  - POST /api/v1/missions/{id}/cancel
"""
from __future__ import annotations

import asyncio
import time
from collections import Counter
from unittest.mock import patch
from uuid import uuid4

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient


# ── 인증 의존성 우회 ────────────────────────
class _MockUser:
    id = uuid4()
    email = "test@example.com"
    is_superadmin = False


class _MockMember:
    id = uuid4()
    role = "admin"


class _MockOrg:
    id = uuid4()
    name = "Test Org"


def _override_user():
    return _MockUser()


def _override_org_member():
    return (_MockUser(), _MockMember(), _MockOrg())


@pytest.fixture(scope="module")
def client():
    """app 부팅 + 인증 의존성 우회 + TestClient 반환."""
    from app.main import app
    from app.dependencies import get_current_user, get_current_org_member

    app.dependency_overrides[get_current_user] = _override_user
    app.dependency_overrides[get_current_org_member] = _override_org_member

    with TestClient(app) as c:
        yield c

    app.dependency_overrides.clear()


def _make_floorplan_image() -> bytes:
    """가구 포함 평면도 합성 (PNG bytes)."""
    W, H = 1600, 1200
    img = np.full((H, W, 3), 252, dtype=np.uint8)
    BLACK = (15, 15, 15); GRAY = (90, 90, 90); THK = 14
    cv2.rectangle(img, (100, 100), (1500, 1100), BLACK, THK)
    cv2.line(img, (800, 100), (800, 1100), BLACK, THK)
    cv2.rectangle(img, (200, 200), (550, 480), GRAY, -1)      # 침대
    cv2.circle(img, (1100, 700), 100, GRAY, -1)               # 식탁
    cv2.rectangle(img, (1100, 950), (1400, 1050), GRAY, -1)   # 책상
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


# ────────────────────────────────────────────
# /floorplan/analyze
# ────────────────────────────────────────────

def test_analyze_returns_walls_and_furniture(client):
    img_bytes = _make_floorplan_image()
    resp = client.post(
        "/api/v1/floorplan/analyze",
        files={"file": ("test.png", img_bytes, "image/png")},
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()

    assert data["wall_count"] > 0
    assert data["furniture_count"] > 0
    assert data["image_width"] == 1600
    assert data["image_height"] == 1200
    assert isinstance(data["walls"], list)
    assert isinstance(data["furniture"], list)
    assert all({"x1", "y1", "x2", "y2"} <= set(w.keys()) for w in data["walls"])
    assert all({"cx", "cy", "w", "h", "label"} <= set(f.keys()) for f in data["furniture"])


def test_analyze_rejects_non_image(client):
    """지원 안 되는 MIME 은 400."""
    resp = client.post(
        "/api/v1/floorplan/analyze",
        files={"file": ("doc.pdf", b"%PDF-1.4 fake", "application/pdf")},
    )
    assert resp.status_code == 400


def test_validate_quality_check(client):
    img_bytes = _make_floorplan_image()
    resp = client.post(
        "/api/v1/floorplan/validate",
        files={"file": ("test.png", img_bytes, "image/png")},
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["status"] in ("ok", "warning", "rejected")
    assert "score" in data
    assert "checks" in data


# ────────────────────────────────────────────
# /missions/autonomous-scan/start
# ────────────────────────────────────────────

@pytest.mark.asyncio
async def test_autonomous_scan_full_flow(client):
    """walls + furniture 페이로드 → 미션 시작 → status 폴링 → cancel 가능."""
    received = []

    async def fake_broadcast(channel, payload):
        received.append((channel, payload["type"]))

    walls = [
        {"x1": 0.02, "y1": 0.02, "x2": 0.98, "y2": 0.02},
        {"x1": 0.98, "y1": 0.02, "x2": 0.98, "y2": 0.98},
        {"x1": 0.98, "y1": 0.98, "x2": 0.02, "y2": 0.98},
        {"x1": 0.02, "y1": 0.98, "x2": 0.02, "y2": 0.02},
    ]
    furniture = [
        {"cx": 0.3, "cy": 0.3, "w": 0.2, "h": 0.15, "angle": 0, "label": "rectangular"},
        {"cx": 0.7, "cy": 0.7, "w": 0.15, "h": 0.15, "angle": 0, "label": "circular"},
    ]

    with patch("app.services.autonomous_flight_simulator.ws_manager.broadcast",
               side_effect=fake_broadcast):
        # START — 작은 환경 + 최대 속도로 테스트 시간 단축
        resp = client.post(
            "/api/v1/missions/autonomous-scan/start",
            json={
                "walls": walls,
                "outline": [],
                "furniture": furniture,
                "world_w": 4.0,
                "world_d": 3.0,
                "altitude": 1.5,
                "speed": 3.0,
            },
        )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        mid = data["mission_id"]
        assert data["walls_count"] == 4
        assert data["furniture_count"] == 2
        assert data["world_w"] == 4.0
        assert data["world_d"] == 3.0

        # 완주 대기 (충분한 시간)
        deadline = time.time() + 30
        while time.time() < deadline:
            await asyncio.sleep(0.3)
            r = client.get(f"/api/v1/missions/{mid}")
            if r.status_code == 200 and r.json()["status"] != "running":
                break

        final = client.get(f"/api/v1/missions/{mid}").json()
        assert final["status"] == "completed", f"status={final['status']}, progress={final.get('progress')}"
        assert final["points_emitted"] > 0
        assert final["progress"] == pytest.approx(1.0, abs=0.05)

        # WS 이벤트 확인
        types = Counter(t for _, t in received)
        assert types.get("telemetry.update", 0) > 0
        assert types.get("lidar.points", 0) > 0
        assert types.get("mission.completed", 0) == 1


@pytest.mark.asyncio
async def test_autonomous_scan_cancel(client):
    """미션 시작 → 즉시 cancel → status=cancelled."""
    walls = [
        {"x1": 0.02, "y1": 0.02, "x2": 0.98, "y2": 0.02},
        {"x1": 0.98, "y1": 0.02, "x2": 0.98, "y2": 0.98},
        {"x1": 0.98, "y1": 0.98, "x2": 0.02, "y2": 0.98},
        {"x1": 0.02, "y1": 0.98, "x2": 0.02, "y2": 0.02},
    ]
    with patch("app.services.autonomous_flight_simulator.ws_manager.broadcast",
               side_effect=lambda *a, **k: None):
        r = client.post(
            "/api/v1/missions/autonomous-scan/start",
            json={"walls": walls, "world_w": 10, "world_d": 10, "speed": 0.3},
        )
        assert r.status_code == 200
        mid = r.json()["mission_id"]

        await asyncio.sleep(0.4)
        cancel = client.post(f"/api/v1/missions/{mid}/cancel")
        assert cancel.status_code == 200, cancel.text
        await asyncio.sleep(0.3)
        final = client.get(f"/api/v1/missions/{mid}").json()
        assert final["status"] == "cancelled"


def test_autonomous_scan_requires_walls(client):
    """walls + floorplan_id 둘 다 없으면 400."""
    resp = client.post(
        "/api/v1/missions/autonomous-scan/start",
        json={"world_w": 8.0, "world_d": 6.0},
    )
    assert resp.status_code == 400


def test_missions_list(client):
    """활성 미션 목록 조회."""
    resp = client.get("/api/v1/missions")
    assert resp.status_code == 200
    data = resp.json()
    assert "items" in data
    assert isinstance(data["items"], list)
