# =============================================
# app/core/jwt.py
# 역할: JWT 액세스/리프레시 토큰 생성 및 검증
#       - python-jose 기반 HS256 서명
#       - settings.JWT_SECRET으로 서명/검증
#       - access/refresh 토큰은 payload.type 필드로 구분
# 사용: from app.core.jwt import create_access_token, create_refresh_token,
#       decode_access_token, decode_refresh_token
# =============================================

from datetime import datetime, timedelta, timezone
from typing import Optional

from jose import JWTError, jwt

from app.config import settings

ALGORITHM = "HS256"

TOKEN_TYPE_ACCESS = "access"
TOKEN_TYPE_REFRESH = "refresh"


def create_access_token(user_id: str, expires_minutes: Optional[int] = None) -> str:
    """
    사용자 UUID를 sub 클레임에 담은 JWT 액세스 토큰 발급.
    type="access"로 리프레시 토큰과 혼용 방지.
    """
    expire = datetime.now(timezone.utc) + timedelta(
        minutes=expires_minutes or settings.JWT_EXPIRE_MINUTES
    )
    payload = {"sub": str(user_id), "exp": expire, "type": TOKEN_TYPE_ACCESS}
    return jwt.encode(payload, settings.JWT_SECRET, algorithm=ALGORITHM)


def create_refresh_token(user_id: str, expires_days: Optional[int] = None) -> str:
    """
    장기 유효 리프레시 토큰. /auth/refresh에서만 받아들임.
    access 토큰과 같은 비밀키로 서명하되 type 클레임으로 용도 구분.
    """
    expire = datetime.now(timezone.utc) + timedelta(
        days=expires_days or settings.JWT_REFRESH_EXPIRE_DAYS
    )
    payload = {"sub": str(user_id), "exp": expire, "type": TOKEN_TYPE_REFRESH}
    return jwt.encode(payload, settings.JWT_SECRET, algorithm=ALGORITHM)


def _decode(token: str, expected_type: str) -> Optional[str]:
    """공통 디코드. type 클레임이 기대값과 다르면 거절 → 교차 사용 방지."""
    try:
        payload = jwt.decode(token, settings.JWT_SECRET, algorithms=[ALGORITHM])
    except JWTError:
        return None
    if payload.get("type") not in (None, expected_type):
        # 레거시 토큰은 type 미포함 → access로 간주해서 깨지지 않게 허용.
        # 신규 토큰은 type 일치 필수.
        return None
    if payload.get("type") and payload.get("type") != expected_type:
        return None
    return payload.get("sub")


def decode_access_token(token: str) -> Optional[str]:
    """
    액세스 토큰 검증 후 sub(user_id) 반환. 만료·변조·type 불일치 시 None.
    레거시 토큰(type 필드 없음)도 호환.
    """
    return _decode(token, TOKEN_TYPE_ACCESS)


def decode_refresh_token(token: str) -> Optional[str]:
    """
    리프레시 토큰 전용 디코더. type="refresh"가 아니면 거절.
    access 토큰을 /auth/refresh에 그대로 쏘는 실수 차단.
    """
    try:
        payload = jwt.decode(token, settings.JWT_SECRET, algorithms=[ALGORITHM])
    except JWTError:
        return None
    if payload.get("type") != TOKEN_TYPE_REFRESH:
        return None
    return payload.get("sub")
