# =============================================
# app/core/token_denylist.py
# 역할: JWT 개별 폐기(denylist) — 로그아웃/리프레시 회전 시 jti 를 Redis 에 등록해
#       만료 전이라도 즉시 무효화한다.
#
# 설계:
#   - 저장소: Redis (key: denylist:{jti}, TTL = 토큰 잔여 수명). TTL 만료 시 자동 정리.
#   - graceful degrade: TOKEN_DENYLIST_ENABLED=False 이거나 Redis 미가용이면
#       revoke 는 no-op, is_revoked 는 False(fail-open) — Redis 장애로 전 사용자가
#       잠기는 것을 막는다(가용성 우선). 폐기 정확성보다 가용성을 택한 의도적 결정.
# =============================================

from __future__ import annotations

import time

from app.config import settings
from app.core.logging import get_logger
from app.core.redis_client import get_redis

logger = get_logger("token_denylist")

_PREFIX = "denylist:"


def _enabled() -> bool:
    return bool(settings.TOKEN_DENYLIST_ENABLED)


async def revoke_jti(jti: str, exp: int | float | None) -> bool:
    """jti 를 폐기 목록에 등록. exp(epoch sec)로 TTL 산정. 등록되면 True."""
    if not _enabled() or not jti:
        return False
    r = await get_redis()
    if r is None:
        return False
    # 잔여 수명 계산 — 이미 만료됐으면 등록 불필요
    ttl = int((exp or 0) - time.time()) if exp else 0
    if exp and ttl <= 0:
        return False
    try:
        # exp 없으면 보수적으로 7일 보관
        await r.setex(f"{_PREFIX}{jti}", ttl if ttl > 0 else 7 * 24 * 3600, "1")
        return True
    except Exception as e:
        logger.warning("denylist_revoke_failed", error=str(e))
        return False


async def is_revoked(jti: str | None) -> bool:
    """jti 가 폐기됐으면 True. 비활성/미가용/오류 시 False(fail-open)."""
    if not _enabled() or not jti:
        return False
    r = await get_redis()
    if r is None:
        return False
    try:
        return bool(await r.exists(f"{_PREFIX}{jti}"))
    except Exception:
        return False
