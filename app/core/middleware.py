# =============================================
# app/core/middleware.py
# 역할: 요청/응답 미들웨어
#       - RequestIDMiddleware: 모든 요청에 고유 ID 생성·전파
#         · X-Request-ID 헤더로 클라이언트와 공유 (요청·응답 양방향)
#         · structlog contextvars에 바인딩 → 모든 로그에 자동 포함
#         · 응답 시 요청 경로·상태·소요시간 INFO 로그 출력
# =============================================

from __future__ import annotations

import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from structlog.contextvars import bind_contextvars, clear_contextvars

from app.core.logging import get_logger

logger = get_logger("http")


class RequestIDMiddleware(BaseHTTPMiddleware):
    """모든 요청에 request_id를 부여하고 로그/응답 헤더에 포함."""

    HEADER_NAME = "X-Request-ID"

    async def dispatch(self, request: Request, call_next) -> Response:
        # 클라이언트가 헤더로 보낸 ID가 있으면 재사용 (분산 추적)
        incoming_id = request.headers.get(self.HEADER_NAME)
        request_id = incoming_id or uuid.uuid4().hex[:16]

        # structlog의 모든 후속 로그에 자동 포함됨
        clear_contextvars()
        bind_contextvars(
            request_id=request_id,
            method=request.method,
            path=request.url.path,
        )

        start = time.perf_counter()
        response: Response
        try:
            response = await call_next(request)
        except Exception:
            duration_ms = (time.perf_counter() - start) * 1000
            logger.exception(
                "http.request.failed",
                duration_ms=round(duration_ms, 2),
            )
            raise

        duration_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "http.request",
            status=response.status_code,
            duration_ms=round(duration_ms, 2),
        )
        response.headers[self.HEADER_NAME] = request_id
        return response


__all__ = ["RequestIDMiddleware"]
