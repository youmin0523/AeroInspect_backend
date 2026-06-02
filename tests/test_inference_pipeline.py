# =============================================
# tests/test_inference_pipeline.py
# 역할: 3-모델 추론 파이프라인 단위 + API 테스트
#       - xyxy → xywhn 변환 회귀 방지
#       - taxonomy 매핑 (good=Burst/터짐, Crack=균열 등)
#       - /health, /api/v1/detect, 에러 케이스
#       - 가중치 없을 때의 동작 (is_loaded=False → 503)
#
# 가중치 없이도 통과하도록 설계 — 실모델 추론은 별도 e2e 스모크 테스트에서.
# 실행: pytest tests/test_inference_pipeline.py -v
# =============================================

from __future__ import annotations

import io

import pytest
from httpx import AsyncClient, ASGITransport
from PIL import Image

from app.main import app
from app.services.defect_taxonomy import (
    CLASS_DISPLAY_MAP,
    WALLPAPER_CLASSES,
    WALLPAPER_SEVERE_CLASSES,
    YOLO_DISPLAY_MAP,
    get_display_names,
    map_to_legacy,
    xyxy_to_xywhn,
)
from app.services.inference_pipeline import pipeline


# ── xyxy → xywhn 변환 회귀 테스트 ─────────────
class TestXyxyToXywhn:
    def test_basic_center(self):
        """640x480 이미지 중앙의 100x100 박스 → cx=0.5, cy=0.5, w=0.156, h=0.208"""
        cx, cy, w, h = xyxy_to_xywhn([270, 190, 370, 290], 640, 480)
        assert cx == pytest.approx(320 / 640)
        assert cy == pytest.approx(240 / 480)
        assert w == pytest.approx(100 / 640)
        assert h == pytest.approx(100 / 480)

    def test_full_frame(self):
        """전체 프레임 박스 → cx=0.5, cy=0.5, w=1.0, h=1.0"""
        cx, cy, w, h = xyxy_to_xywhn([0, 0, 640, 480], 640, 480)
        assert (cx, cy, w, h) == (0.5, 0.5, 1.0, 1.0)

    def test_corner_box(self):
        """좌상단 0~50 박스 → cx=cy=0.05 (50/1000/2)... 실제로는 25/1000=0.025"""
        cx, cy, w, h = xyxy_to_xywhn([0, 0, 50, 50], 1000, 1000)
        assert cx == pytest.approx(0.025)
        assert cy == pytest.approx(0.025)
        assert w == pytest.approx(0.05)
        assert h == pytest.approx(0.05)

    def test_clips_out_of_range(self):
        """이미지 밖 좌표도 0~1 범위로 클리핑"""
        cx, cy, w, h = xyxy_to_xywhn([-10, -10, 700, 500], 640, 480)
        assert 0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0
        assert 0.0 <= w <= 1.0 and 0.0 <= h <= 1.0

    def test_zero_size_image(self):
        """W/H가 0이면 (0,0,0,0) 폴백"""
        assert xyxy_to_xywhn([10, 10, 20, 20], 0, 0) == (0.0, 0.0, 0.0, 0.0)


# ── Taxonomy 매핑 테스트 ──────────────────────
class TestTaxonomy:
    def test_wallpaper_19_classes(self):
        """WALLPAPER_CLASSES는 정확히 19개이고 체크포인트 순서 유지"""
        assert len(WALLPAPER_CLASSES) == 19
        assert WALLPAPER_CLASSES[0] == "Baseboard"
        assert WALLPAPER_CLASSES[-1] == "good"

    def test_good_is_burst_not_normal(self):
        """⚠️ 'good' 클래스는 '터짐(Burst)' — '정상' 아님"""
        en, ko = CLASS_DISPLAY_MAP["good"]
        assert en == "Burst"
        assert ko == "터짐"
        # severity 격상 대상에 포함돼야 함 (MED로 올라감)
        assert "good" in WALLPAPER_SEVERE_CLASSES

    def test_yolo_display_map(self):
        """YOLO 3 클래스 표시명 확인"""
        assert YOLO_DISPLAY_MAP["Crack"] == ("Crack", "균열")
        assert YOLO_DISPLAY_MAP["Moisture"] == ("Moisture", "습기")
        assert YOLO_DISPLAY_MAP["delamination"] == ("Delamination", "박리")

    def test_get_display_names_fallback(self):
        """매핑 없는 클래스는 내부명 그대로 반환"""
        en, ko = get_display_names("unknown_class_xyz")
        assert en == "unknown_class_xyz"
        assert ko == "unknown_class_xyz"

    def test_map_to_legacy_thermal(self):
        """YOLO thermal → A-E taxonomy 매핑"""
        area, code, dtype = map_to_legacy("yolo_thermal", "Crack")
        assert area == "A"
        assert code == "A-02"
        assert "균열" in dtype

    def test_map_to_legacy_delam(self):
        """delamination → B-02"""
        area, code, dtype = map_to_legacy("yolo_delam", "delamination")
        assert area == "B"
        assert code == "B-02"

    def test_map_to_legacy_good_is_C04(self):
        """good(=터짐)은 C-04에 편입"""
        area, code, _ = map_to_legacy("wallpaper", "good")
        assert area == "C"
        assert code == "C-04"

    def test_map_to_legacy_unmapped_fallback(self):
        """매핑 없는 벽지 클래스는 (None, None, 한글 표시명)"""
        area, code, dtype = map_to_legacy("wallpaper", "Kink")
        assert area is None
        assert code is None
        assert dtype == "꼬임"  # display_ko로 폴백


# ── /health 엔드포인트 ───────────────────────
@pytest.fixture
async def client():
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as ac:
        yield ac


@pytest.mark.asyncio
async def test_health_has_required_fields(client):
    """/health 응답 기본 필드 확인. (모델 미로드 환경에선 503 + status 필드 반환이 정상)"""
    response = await client.get("/health")
    # 모델 가중치 미로드 테스트 환경에서는 503(degraded)도 정상 — status 필드는 항상 존재
    assert response.status_code in (200, 503)
    data = response.json()
    assert "status" in data


# ── /api/v1/detect 엔드포인트 ─────────────────
def _make_dummy_jpeg() -> bytes:
    """테스트용 더미 JPEG (PIL로 생성)."""
    img = Image.new("RGB", (320, 240), color=(128, 128, 128))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return buf.getvalue()


@pytest.mark.asyncio
async def test_detect_without_models_returns_503(client):
    """/detect 는 인증(Bearer/webhook) 필요 → 무인증 401, 인증+모델미로드 503."""
    if pipeline.is_loaded:
        pytest.skip("파이프라인이 로드된 상태에서는 이 테스트 스킵")
    jpeg = _make_dummy_jpeg()
    response = await client.post(
        "/api/v1/detect",
        files={"image": ("test.jpg", jpeg, "image/jpeg")},
    )
    # 인증 의존성(verify_ai_webhook_or_user)이 먼저 평가되어 401, 모델까지 가면 503
    assert response.status_code in (401, 503)


@pytest.mark.asyncio
async def test_detect_rejects_non_image(client):
    """이미지 아닌 content-type → 인증(401)/모델(503)/형식(400) 중 하나로 거부."""
    response = await client.post(
        "/api/v1/detect",
        files={"image": ("test.txt", b"not an image", "text/plain")},
    )
    assert response.status_code in (400, 401, 503)


@pytest.mark.asyncio
async def test_detect_nonexistent_endpoint_404(client):
    """잘못된 경로는 404"""
    response = await client.get("/api/v1/detect/does-not-exist")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_detect_batch_rejects_empty(client):
    """배치 — 파일 없이 호출: 인증(401) 또는 검증(422) 으로 거부."""
    response = await client.post("/api/v1/detect/batch")
    assert response.status_code in (401, 422)
