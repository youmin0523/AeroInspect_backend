# =============================================
# app/core/rate_limit.py
# 역할: IP + 엔드포인트 prefix 기반 분당 슬라이딩 윈도우 제한
#       - 외부 의존성 없이 메모리 deque 로 구현 (단일 워커 가정)
#       - 운영 멀티워커 환경에서는 Redis 백엔드로 교체 필요
#
# 차단 응답: 429 Too Many Requests + Retry-After: 60
# =============================================

from __future__ import annotations

import asyncio
import time
from collections import defaultdict, deque
from typing import Callable, Deque, Dict

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response


# 엔드포인트 prefix → 분당 허용 횟수
# 가장 긴 prefix 가 우선 적용되도록 길이 정렬로 평가한다.
PATH_LIMITS: Dict[str, int] = {
    "/api/v1/auth/login": 10,
    "/api/v1/auth/signup": 5,
    "/api/v1/auth/refresh": 30,
    "/api/v1/auth/check-email": 30,
    "/api/v1/auth/check-username": 30,
    "/api/v1/auth/find-id": 5,
    "/api/v1/auth/find-pw": 5,
    "/api/v1/oauth/": 20,
    "/api/v1/detect": 60,
    "/api/v1/ai/": 600,           # AI 워커 콜백 — 시크릿 인증되므로 여유
    "/api/v1/telemetry": 600,     # 드론 텔레메트리 — webhook secret 인증
    # AI 챗봇: CRUD 일반 한도. SSE 메시지 전송은 라우터 내부 사용자별 카운터로 추가 보호.
    "/api/v1/ai-chat": 120,
}
DEFAULT_LIMIT = 120  # 분당 기본 한도
WINDOW_SEC = 60

# 미들웨어가 건드리지 말아야 할 경로 (헬스체크/메트릭/정적)
EXEMPT_PATHS = ("/", "/health", "/metrics", "/uploads")


def _resolve_limit(path: str) -> tuple[str, int]:
    """경로에 매칭되는 (bucket_key, limit) 반환. 가장 긴 prefix 우선."""
    best_prefix: str | None = None
    for prefix in PATH_LIMITS:
        if path.startswith(prefix):
            if best_prefix is None or len(prefix) > len(best_prefix):
                best_prefix = prefix
    if best_prefix is not None:
        return best_prefix, PATH_LIMITS[best_prefix]
    return "_default", DEFAULT_LIMIT


class RateLimiter:
    """IP + bucket 단위 슬라이딩 윈도우 카운터."""

    def __init__(self) -> None:
        self._hits: Dict[str, Deque[float]] = defaultdict(deque)
        self._lock = asyncio.Lock()

    async def check(self, ip: str, path: str) -> tuple[bool, int, int]:
        bucket, limit = _resolve_limit(path)
        key = f"{ip}|{bucket}"
        now = time.monotonic()
        cutoff = now - WINDOW_SEC

        async with self._lock:
            q = self._hits[key]
            while q and q[0] < cutoff:
                q.popleft()
            if len(q) >= limit:
                return False, len(q), limit
            q.append(now)
            return True, len(q), limit


class RateLimitMiddleware(BaseHTTPMiddleware):
    """슬라이딩 윈도우 기반 IP rate-limit 미들웨어."""

    def __init__(self, app) -> None:  # type: ignore[no-untyped-def]
        super().__init__(app)
        self._limiter = RateLimiter()

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        path = request.url.path
        if any(path == ep or path.startswith(ep + "/") for ep in EXEMPT_PATHS):
            return await call_next(request)

        # IP 추출: 프록시 헤더 우선, 없으면 client.host
        xff = request.headers.get("x-forwarded-for", "")
        ip = xff.split(",")[0].strip() if xff else (request.client.host if request.client else "anonymous")

        allowed, used, limit = await self._limiter.check(ip, path)
        if not allowed:
            return JSONResponse(
                status_code=429,
                content={
                    "detail": "요청이 너무 잦습니다. 잠시 후 다시 시도하세요.",
                    "limit": limit,
                    "window_sec": WINDOW_SEC,
                },
                headers={"Retry-After": str(WINDOW_SEC)},
            )

        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(limit)
        response.headers["X-RateLimit-Remaining"] = str(max(limit - used, 0))
        return response
