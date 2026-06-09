# =============================================
# app/core/redis_client.py
# 역할: 공유 Redis(asyncio) 클라이언트 — 레이트리밋/토큰 폐기/스트림모드 등에서 재사용.
#       - lazy 연결: 최초 사용 시 1회 연결, 이후 재사용(커넥션 풀)
#       - graceful degrade: redis 미설치/미가용이면 None 반환 → 호출부가 메모리 폴백
#       - 연결 실패는 캐시해 매 호출마다 재시도하지 않음(쿨다운)
#
# 사용:
#   r = await get_redis()
#   if r is not None:
#       await r.incr(...)
#   else:
#       ... 메모리 폴백 ...
# =============================================

from __future__ import annotations

import time
from typing import Optional

from app.config import settings
from app.core.logging import get_logger

logger = get_logger("redis")

_client = None                  # redis.asyncio.Redis | None
_unavailable_until: float = 0.0  # 연결 실패 후 재시도 쿨다운 (epoch sec)
_RETRY_COOLDOWN_SEC = 30.0
_DEFAULT_FALSE = object()


async def get_redis():
    """공유 Redis 클라이언트 반환. 미설치/미가용이면 None (호출부는 폴백)."""
    global _client, _unavailable_until

    if _client is not None:
        return _client

    # 최근 연결 실패 → 쿨다운 동안 재시도 안 함
    if time.time() < _unavailable_until:
        return None

    try:
        import redis.asyncio as redis  # lazy import — 미설치 환경에서도 모듈 import 가능
    except ImportError:
        logger.warning("redis_not_installed", hint="pip install redis — Redis 기능 비활성(메모리 폴백)")
        _unavailable_until = time.time() + _RETRY_COOLDOWN_SEC
        return None

    try:
        client = redis.from_url(
            settings.REDIS_URL,
            encoding="utf-8",
            decode_responses=True,
            socket_connect_timeout=2.0,
            socket_timeout=2.0,
        )
        # 연결 확인 (실패 시 즉시 폴백)
        await client.ping()
        _client = client
        logger.info("redis_connected", url=settings.REDIS_URL)
        return _client
    except Exception as e:
        logger.warning("redis_connect_failed", error=str(e), cooldown_sec=_RETRY_COOLDOWN_SEC)
        _unavailable_until = time.time() + _RETRY_COOLDOWN_SEC
        return None


async def close_redis() -> None:
    """앱 종료 시 호출 — 커넥션 정리."""
    global _client
    if _client is not None:
        try:
            await _client.aclose()
        except Exception:
            pass
        _client = None
