# =============================================
# tests/test_image_storage.py
# 역할: ImageStorage 서비스 검증
#       - Base64 JPEG 저장 경로 생성
#       - data: URL prefix 허용
#       - 잘못된 Base64 graceful fallback
#       - get_url / delete
# 실행: pytest tests/test_image_storage.py -v
# =============================================

from __future__ import annotations

import base64
import io
import os

import pytest
from PIL import Image

from app.services.image_storage import ImageStorage


def _make_jpeg_b64() -> str:
    """테스트용 실제 JPEG 바이트 → base64 문자열."""
    img = Image.new("RGB", (16, 16), color="red")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


@pytest.fixture
def storage(tmp_path, monkeypatch):
    """tmp_path 기준 격리된 스토리지."""
    svc = ImageStorage()
    monkeypatch.setattr(svc, "UPLOAD_ROOT", str(tmp_path))
    # _ensure_base_dir 재호출로 디렉터리 생성
    os.makedirs(os.path.join(str(tmp_path), svc.DEFECT_SUBDIR), exist_ok=True)
    return svc


class TestImageStorage:
    async def test_save_base64_creates_file(self, storage, tmp_path):
        b64 = _make_jpeg_b64()
        rel_path = await storage.save_base64_jpeg(b64)
        assert rel_path is not None
        assert rel_path.startswith("defects/")
        assert rel_path.endswith(".jpg")
        abs_path = os.path.join(str(tmp_path), rel_path)
        assert os.path.exists(abs_path)
        assert os.path.getsize(abs_path) > 0

    async def test_save_base64_with_data_url_prefix(self, storage):
        b64 = _make_jpeg_b64()
        data_url = f"data:image/jpeg;base64,{b64}"
        rel_path = await storage.save_base64_jpeg(data_url)
        assert rel_path is not None
        assert rel_path.endswith(".jpg")

    async def test_save_none_returns_none(self, storage):
        assert await storage.save_base64_jpeg(None) is None
        assert await storage.save_base64_jpeg("") is None

    async def test_save_invalid_base64_returns_none(self, storage):
        assert await storage.save_base64_jpeg("not_valid_base64!!!") is None

    def test_get_url_builds_uploads_path(self, storage):
        assert storage.get_url("defects/2026-04-21/abc.jpg") == "/uploads/defects/2026-04-21/abc.jpg"
        assert storage.get_url(None) is None

    async def test_delete_removes_file(self, storage, tmp_path):
        rel_path = await storage.save_base64_jpeg(_make_jpeg_b64())
        abs_path = os.path.join(str(tmp_path), rel_path)
        assert os.path.exists(abs_path)
        assert storage.delete(rel_path) is True
        assert not os.path.exists(abs_path)

    def test_delete_missing_returns_false(self, storage):
        assert storage.delete("defects/2020-01-01/nonexistent.jpg") is False
        assert storage.delete(None) is False

    async def test_multiple_saves_unique_paths(self, storage):
        """같은 base64를 여러 번 저장해도 서로 다른 파일명."""
        b64 = _make_jpeg_b64()
        p1 = await storage.save_base64_jpeg(b64)
        p2 = await storage.save_base64_jpeg(b64)
        assert p1 != p2
