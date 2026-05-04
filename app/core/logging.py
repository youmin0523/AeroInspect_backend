# =============================================
# app/core/logging.py
# 역할: 구조화된 로깅 설정 (structlog)
#       - JSON 출력 (운영) / 컬러 콘솔 출력 (개발)
#       - 요청별 고유 ID(request_id) 컨텍스트에 자동 바인딩
#       - print() 대체용 logger 제공
#
# 사용법:
#   from app.core.logging import get_logger
#   logger = get_logger(__name__)
#   logger.info("event.name", key=value)
# =============================================

from __future__ import annotations

import logging
import sys
from contextvars import ContextVar
from typing import Optional

import structlog
from structlog.contextvars import merge_contextvars


# 요청별 컨텍스트 (미들웨어에서 bind, 로거에서 자동 포함)
request_id_ctx: ContextVar[Optional[str]] = ContextVar("request_id", default=None)


def configure_logging(json_output: bool = False, level: str = "INFO") -> None:
    """
    앱 시작 시 한 번만 호출. main.py lifespan 진입 직후 권장.

    Args:
        json_output: True면 JSON 로그 (운영), False면 컬러 콘솔 (개발)
        level: DEBUG / INFO / WARNING / ERROR
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    # 표준 logging 설정 (uvicorn, sqlalchemy 등이 사용)
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
    )

    # structlog 프로세서 체인
    processors = [
        merge_contextvars,  # contextvars의 값을 이벤트에 자동 병합 (request_id 등)
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if json_output:
        processors.append(structlog.processors.JSONRenderer(ensure_ascii=False))
    else:
        processors.append(structlog.dev.ConsoleRenderer(colors=True))

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: Optional[str] = None):
    """호출자에게 바인딩된 structlog logger 반환."""
    return structlog.get_logger(name)


__all__ = ["configure_logging", "get_logger", "request_id_ctx"]
