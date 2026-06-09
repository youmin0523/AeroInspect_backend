# =============================================
# app/services/image_storage.py
# 역할: 하자 크롭 이미지를 파일시스템에 저장하고 상대 경로를 반환
#       - DB에 Base64로 저장하던 방식을 파일 경로 저장으로 대체
#       - 저장 루트: ./uploads/defects/{YYYY-MM-DD}/{uuid}.jpg
#       - 클라이언트는 /uploads/defects/... URL로 직접 접근 (StaticFiles)
#
# 추후 S3/GCS로 이전 시 이 모듈의 save/get_url 시그니처만 유지하면 호환.
# =============================================

from __future__ import annotations

import base64
import logging
import os
import re
import uuid
from datetime import datetime, timezone
from typing import Optional

import aiofiles

from app.config import settings

logger = logging.getLogger(__name__)


class ImageStorage:
    """하자 크롭 이미지 파일 저장 서비스."""

    # /uploads는 main.py에서 StaticFiles로 마운트됨
    UPLOAD_ROOT = "./uploads"
    DEFECT_SUBDIR = "defects"
    DATA_URL_PREFIX_RE = re.compile(r"^data:image/[a-z]+;base64,", re.IGNORECASE)

    def __init__(self):
        self._ensure_base_dir()

    def _ensure_base_dir(self) -> None:
        os.makedirs(os.path.join(self.UPLOAD_ROOT, self.DEFECT_SUBDIR), exist_ok=True)

    def _today_dir(self) -> str:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        path = os.path.join(self.UPLOAD_ROOT, self.DEFECT_SUBDIR, today)
        os.makedirs(path, exist_ok=True)
        return path

    async def save_base64_jpeg(self, b64: Optional[str]) -> Optional[str]:
        """
        Base64 JPEG 문자열을 파일로 저장하고 상대 경로 반환.
        `data:image/jpeg;base64,...` prefix 있어도 처리.
        디스크 쓰기는 aiofiles 로 비동기 처리하여 이벤트 루프를 차단하지 않음
        (웹훅 배치 루프에서 다건 저장 시 중요).

        Returns:
            예: "defects/2026-04-21/3f4a...uuid.jpg" (UPLOAD_ROOT 기준 상대 경로)
            None 입력 시 None.
        """
        if not b64:
            return None

        clean = self.DATA_URL_PREFIX_RE.sub("", b64).strip()
        try:
            data = base64.b64decode(clean, validate=True)
        except (base64.binascii.Error, ValueError) as e:
            logger.warning("[ImageStorage] Base64 디코드 실패: %s", e)
            return None

        today_dir = self._today_dir()
        filename = f"{uuid.uuid4().hex}.jpg"
        abs_path = os.path.join(today_dir, filename)

        async with aiofiles.open(abs_path, "wb") as f:
            await f.write(data)

        # DB에는 UPLOAD_ROOT 기준 상대 경로만 저장 (예: "defects/2026-04-21/xxx.jpg")
        rel_path = os.path.relpath(abs_path, self.UPLOAD_ROOT).replace(os.sep, "/")
        return rel_path

    def get_url(self, rel_path: Optional[str]) -> Optional[str]:
        """저장된 상대 경로 → 클라이언트 접근 URL."""
        if not rel_path:
            return None
        # StaticFiles 마운트 경로와 동일 (main.py: app.mount("/uploads", ...))
        return f"/uploads/{rel_path.lstrip('/')}"

    def delete(self, rel_path: Optional[str]) -> bool:
        """파일 삭제 (하자 레코드 삭제 시 정리용)."""
        if not rel_path:
            return False
        abs_path = os.path.join(self.UPLOAD_ROOT, rel_path)
        if os.path.exists(abs_path):
            os.remove(abs_path)
            return True
        return False


# ── 모듈 레벨 싱글톤 ─────────────────────────
image_storage = ImageStorage()


__all__ = ["ImageStorage", "image_storage"]
