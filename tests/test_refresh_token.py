# =============================================
# tests/test_refresh_token.py
# 역할: 리프레시 토큰 발급/검증 로직 회귀
#       - access vs refresh 교차 사용 차단 (type 클레임)
#       - 레거시 토큰(type 필드 없음) 호환 — decode_access_token은 허용
#       - 만료 검증 및 디코드 실패 graceful
#
# 실행: pytest tests/test_refresh_token.py -v
# =============================================

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from jose import jwt

from app.config import settings
from app.core.jwt import (
    ALGORITHM,
    create_access_token,
    create_refresh_token,
    decode_access_token,
    decode_refresh_token,
)


class TestRoundtrip:
    def test_access_token_roundtrip(self):
        token = create_access_token("user-1")
        assert decode_access_token(token) == "user-1"

    def test_refresh_token_roundtrip(self):
        token = create_refresh_token("user-2")
        assert decode_refresh_token(token) == "user-2"


class TestTypeSeparation:
    def test_access_token_rejected_by_refresh_decoder(self):
        """access 토큰을 /auth/refresh에 그대로 넣으면 거절돼야 함."""
        access = create_access_token("user-x")
        assert decode_refresh_token(access) is None

    def test_refresh_token_rejected_by_access_decoder(self):
        """refresh 토큰을 Authorization 헤더로 넣으면 거절돼야 함."""
        refresh = create_refresh_token("user-y")
        assert decode_access_token(refresh) is None


class TestLegacyCompat:
    def test_legacy_token_without_type_accepted_as_access(self):
        """type 필드 없는 옛날 토큰도 access로 취급 — 기존 발급분 깨지지 않게."""
        expire = datetime.now(timezone.utc) + timedelta(minutes=30)
        legacy = jwt.encode(
            {"sub": "legacy-user", "exp": expire},
            settings.JWT_SECRET,
            algorithm=ALGORITHM,
        )
        assert decode_access_token(legacy) == "legacy-user"
        # legacy는 refresh로는 못 씀
        assert decode_refresh_token(legacy) is None


class TestInvalid:
    def test_tampered_token_returns_none(self):
        token = create_access_token("user-1") + "x"
        assert decode_access_token(token) is None

    def test_expired_access_token_returns_none(self):
        expire = datetime.now(timezone.utc) - timedelta(seconds=10)
        token = jwt.encode(
            {"sub": "u", "exp": expire, "type": "access"},
            settings.JWT_SECRET,
            algorithm=ALGORITHM,
        )
        assert decode_access_token(token) is None

    def test_expired_refresh_token_returns_none(self):
        expire = datetime.now(timezone.utc) - timedelta(seconds=10)
        token = jwt.encode(
            {"sub": "u", "exp": expire, "type": "refresh"},
            settings.JWT_SECRET,
            algorithm=ALGORITHM,
        )
        assert decode_refresh_token(token) is None

    def test_wrong_secret_returns_none(self):
        expire = datetime.now(timezone.utc) + timedelta(minutes=5)
        token = jwt.encode(
            {"sub": "u", "exp": expire, "type": "refresh"},
            "wrong-secret",
            algorithm=ALGORITHM,
        )
        assert decode_refresh_token(token) is None
