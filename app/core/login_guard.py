# =============================================
# app/core/login_guard.py
# 역할: 계정(username) 단위 로그인 실패 잠금.
#       IP 단위 rate-limit(rate_limit.py)만으로는 봇넷(다수 IP)의 분산 무차별 대입을
#       막지 못한다. 같은 계정에 대한 연속 실패를 세어 일정 횟수 초과 시 짧게 잠근다.
#
# 정책:
#   - 연속 실패 MAX_FAILS(기본 5)회 → LOCK_SECONDS(기본 300초) 동안 잠금.
#   - 로그인 성공 시 카운터 리셋.
#   - Redis 우선(멀티워커 정합), 미가용이면 메모리 폴백(워커 단위).
#   - 저장소 오류는 fail-open — 보안 기능이 로그인 자체를 막지 않는다.
# =============================================

from __future__ import annotations

import time
from typing import Dict, Optional, Tuple

from app.core.redis_client import get_redis

MAX_FAILS = 5
LOCK_SECONDS = 300
# 연속 실패 카운터 자체의 수명 — 이 시간 동안 추가 실패가 없으면 카운터가 만료(자연 리셋).
FAIL_WINDOW_SECONDS = 900

# 메모리 폴백: username → (fail_count, first_fail_ts, locked_until_ts)
_mem: Dict[str, Tuple[int, float, float]] = {}


def _norm(username: str) -> str:
    return (username or "").strip().lower()


async def check_locked(username: str) -> Optional[int]:
    """잠겨 있으면 남은 잠금 시간(초)을 반환, 아니면 None."""
    key = _norm(username)
    if not key:
        return None
    r = await get_redis()
    if r is not None:
        try:
            ttl = await r.ttl(f"login:lock:{key}")
            return ttl if ttl and ttl > 0 else None
        except Exception:
            pass  # Redis 오류 → 메모리 폴백
    entry = _mem.get(key)
    if entry:
        _, _, locked_until = entry
        remaining = int(locked_until - time.time())
        if remaining > 0:
            return remaining
    return None


async def record_failure(username: str) -> None:
    """로그인 실패 1회 기록. 임계치 도달 시 계정을 잠근다."""
    key = _norm(username)
    if not key:
        return
    r = await get_redis()
    if r is not None:
        try:
            cnt_key = f"login:fail:{key}"
            count = await r.incr(cnt_key)
            if count == 1:
                await r.expire(cnt_key, FAIL_WINDOW_SECONDS)
            if count >= MAX_FAILS:
                await r.set(f"login:lock:{key}", "1", ex=LOCK_SECONDS)
                await r.delete(cnt_key)
            return
        except Exception:
            pass  # 메모리 폴백
    now = time.time()
    count, first_ts, locked_until = _mem.get(key, (0, now, 0.0))
    # 실패 윈도우 만료 시 카운터 리셋
    if now - first_ts > FAIL_WINDOW_SECONDS:
        count, first_ts = 0, now
    count += 1
    if count >= MAX_FAILS:
        _mem[key] = (0, now, now + LOCK_SECONDS)
    else:
        _mem[key] = (count, first_ts, locked_until)


async def reset(username: str) -> None:
    """로그인 성공 — 실패 카운터/잠금 해제."""
    key = _norm(username)
    if not key:
        return
    r = await get_redis()
    if r is not None:
        try:
            await r.delete(f"login:fail:{key}", f"login:lock:{key}")
            return
        except Exception:
            pass
    _mem.pop(key, None)
