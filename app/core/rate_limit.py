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

from app.config import settings
from app.core.redis_client import get_redis


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
    # 스트림/테스트모드 제어·폴링 — 프론트가 /test/active 를 1초마다 폴링하고
    # 대시보드가 state/defect 등을 자주 조회한다. 기본 120/분(_default 공유)에 묶이면
    # 폴링만으로 429("서버 연결 불가") → 영상 재생/검출 전부 막힘. 넉넉히 분리.
    "/api/v1/stream/": 1200,
}
DEFAULT_LIMIT = 120  # 분당 기본 한도
WINDOW_SEC = 60

# 미들웨어가 건드리지 말아야 할 경로 (헬스체크/메트릭/정적 + 미디어 스트림)
# 미디어 스트림은 rate-limit 대상에서 제외:
#   - MJPEG(/stream*/rgb·thermal·blend): 장기 연결이지만 <img> 재연결·소스전환 빈발
#   - 영상 직접재생(/test/upload/file/*): <video> 가 버퍼링/시킹 시 다수의 HTTP range
#     요청을 보내 금세 한도 초과 → 검은 화면. 정적 파일 서빙과 동급으로 예외 처리.
EXEMPT_PATHS = (
    "/", "/health", "/metrics", "/uploads",
    "/api/v1/stream/rgb", "/api/v1/stream/thermal", "/api/v1/stream/blend",
    "/api/v1/stream/test/rgb", "/api/v1/stream/test/thermal",
    "/api/v1/stream/test/upload/file",
)


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

        # 멀티워커 정합이 필요하면 Redis 고정 윈도우 카운터 사용.
        # Redis 미설정/미가용이면 _check_redis 가 None → 메모리 폴백.
        if settings.RATE_LIMIT_BACKEND.lower() == "redis":
            redis_result = await self._check_redis(ip, bucket, limit)
            if redis_result is not None:
                return redis_result

        return await self._check_memory(ip, bucket, limit)

    async def _check_memory(self, ip: str, bucket: str, limit: int) -> tuple[bool, int, int]:
        key = f"{ip}|{bucket}"
        now = time.monotonic()
        cutoff = now - WINDOW_SEC

        async with self._lock:
            q = self._hits[key]
            while q and q[0] < cutoff:
                q.popleft()
            if len(q) >= limit:
                # 만료 후 비었는데 한도 초과는 불가능 — 한도≥1이면 항상 append 경로로 빠짐.
                return False, len(q), limit
            q.append(now)
            return True, len(q), limit

    async def _check_redis(self, ip: str, bucket: str, limit: int):
        """Redis 고정 윈도우(분 단위) 카운터. 미가용이면 None 반환(→ 메모리 폴백)."""
        r = await get_redis()
        if r is None:
            return None
        try:
            window = int(time.time() // WINDOW_SEC)
            key = f"rl:{ip}:{bucket}:{window}"
            count = await r.incr(key)
            if count == 1:
                # 윈도우 첫 요청에만 TTL 설정 (자동 만료)
                await r.expire(key, WINDOW_SEC)
            if count > limit:
                return False, count, limit
            return True, count, limit
        except Exception:
            # Redis 일시 오류 → 메모리 폴백
            return None

    async def sweep(self) -> None:
        """
        만료되어 비어버린 deque 키를 제거해 무한 메모리 누적을 방지.
        check() 는 허용 시 항상 append 하므로 그 안에서 키가 비는 일이 없다.
        조용해진(더 이상 요청 없는) IP|bucket 키는 여기서만 회수 가능하므로
        미들웨어가 주기적으로 호출한다.
        """
        cutoff = time.monotonic() - WINDOW_SEC
        async with self._lock:
            for key in list(self._hits.keys()):
                q = self._hits[key]
                while q and q[0] < cutoff:
                    q.popleft()
                if not q:
                    del self._hits[key]


class RateLimitMiddleware(BaseHTTPMiddleware):
    """슬라이딩 윈도우 기반 IP rate-limit 미들웨어."""

    def __init__(self, app) -> None:  # type: ignore[no-untyped-def]
        super().__init__(app)
        self._limiter = RateLimiter()
        self._sweep_at = time.monotonic() + WINDOW_SEC  # 다음 sweep 예정 시각

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        path = request.url.path

        # 주기적으로 비어버린 rate-limit 키 회수 (무한 메모리 누적 방지)
        now = time.monotonic()
        if now >= self._sweep_at:
            self._sweep_at = now + WINDOW_SEC
            await self._limiter.sweep()

        if any(path == ep or path.startswith(ep + "/") for ep in EXEMPT_PATHS):
            return await call_next(request)

        # IP 추출: 신뢰 가능한 리버스 프록시(nginx 등)가 앞단에 있다는 가정 하에 XFF 사용.
        # 주의: XFF의 첫 홉은 클라이언트가 임의로 위조 가능하므로(per-IP 한도 우회),
        #       프록시 없이 직접 노출되는 환경에서는 반드시 직접 연결 IP를 사용해야 한다.
        # 최소·안전 개선: XFF가 비어 있거나 첫 토큰이 공백이면 직접 연결 IP(client.host)로 폴백.
        direct_ip = request.client.host if request.client else "anonymous"
        xff = request.headers.get("x-forwarded-for", "")
        xff_first = xff.split(",")[0].strip() if xff else ""
        ip = xff_first or direct_ip

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
