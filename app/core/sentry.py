# =============================================
# app/core/sentry.py
# 역할: Sentry 에러 모니터링 통합
#       - init_sentry(settings) — FastAPI/Starlette/SQLAlchemy/Asyncio integration
#       - structlog 의 request_id contextvar 를 Sentry tag/context 로 자동 첨부
#       - before_send 훅에서 password/token/secret/authorization 키 redact
#       - SENTRY_DSN 비어 있으면 no-op (로컬 개발 / CI 가 막히지 않도록)
#
# 운영 사용:
#   from app.core.sentry import init_sentry
#   from app.config import settings
#   init_sentry(settings)  # main.py lifespan 시작 부분
# =============================================

from __future__ import annotations

from typing import Any, Dict, Optional

from app.core.logging import get_logger

logger = get_logger(__name__)


# redact 대상 키워드 (대소문자 무시 substring 매칭).
# request body / query / headers / extra 등 dict 전체에 재귀적으로 적용.
_REDACT_KEYS = (
    "password",
    "passwd",
    "token",
    "secret",
    "authorization",
    "api_key",
    "apikey",
    "client_secret",
    "refresh_token",
    "access_token",
    "session",
    "cookie",
    "set-cookie",
)
_REDACTED = "[REDACTED]"


def _redact_mapping(data: Any) -> Any:
    """dict/list 를 재귀 순회하며 민감 키 값을 [REDACTED] 로 치환한다."""
    if isinstance(data, dict):
        out: Dict[str, Any] = {}
        for k, v in data.items():
            key_lower = str(k).lower()
            if any(needle in key_lower for needle in _REDACT_KEYS):
                out[k] = _REDACTED
            else:
                out[k] = _redact_mapping(v)
        return out
    if isinstance(data, list):
        return [_redact_mapping(item) for item in data]
    return data


def _before_send(event: Dict[str, Any], hint: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Sentry 전송 직전 훅. 민감 데이터 redact + request_id 태깅.

    - request.data / request.query_string / request.headers / extra / contexts 전부 sanitize.
    - structlog contextvars 의 request_id 를 tags['request_id'] 로 승격.
    """
    try:
        # request 영역
        request = event.get("request")
        if isinstance(request, dict):
            for field in ("data", "headers", "cookies", "query_string", "env"):
                if field in request:
                    request[field] = _redact_mapping(request[field])
            event["request"] = request

        # extra / contexts
        if "extra" in event:
            event["extra"] = _redact_mapping(event["extra"])
        if "contexts" in event:
            event["contexts"] = _redact_mapping(event["contexts"])

        # structlog contextvars → Sentry tag (RequestIDMiddleware 가 이미 set_tag 했지만,
        # 다른 경로로 발생한 이벤트도 안전망 차원에서 한 번 더 시도).
        try:
            from structlog.contextvars import get_contextvars

            ctx = get_contextvars() or {}
            req_id = ctx.get("request_id")
            if req_id:
                tags = event.setdefault("tags", {})
                tags.setdefault("request_id", req_id)
        except Exception:
            pass  # contextvars 조회 실패는 절대 이벤트 전송을 막지 않음
    except Exception as e:  # noqa: BLE001
        # 훅 자체가 실패해도 원본 이벤트는 보내야 한다 (관측 가능성 우선)
        logger.warning("sentry.before_send.error", error=str(e))
    return event


def init_sentry(settings) -> bool:
    """
    Sentry 초기화. 호출 안전 (DSN 미설정 / 패키지 미설치 시 no-op).

    Returns:
        True  — 실제로 초기화됨
        False — DSN 미설정 또는 sentry-sdk 미설치 (정상 no-op)
    """
    dsn = getattr(settings, "SENTRY_DSN", None)
    env = getattr(settings, "SENTRY_ENVIRONMENT", "development")

    # 운영 환경에서 DSN 누락은 운영 갭 — 경고 로그 (기동 차단 X)
    app_env = (
        getattr(settings, "APP_ENV", None)
        or __import__("os").environ.get("APP_ENV", "")
    ).strip().lower()
    if not dsn:
        if app_env in {"production", "prod", "live"}:
            logger.warning(
                "sentry.dsn.missing_in_production",
                hint="운영에서 SENTRY_DSN 이 비어 있습니다. Fly secrets 에 등록하세요.",
            )
        else:
            logger.info("sentry.disabled", reason="SENTRY_DSN not set")
        return False

    try:
        import sentry_sdk
        from sentry_sdk.integrations.fastapi import FastApiIntegration
        from sentry_sdk.integrations.starlette import StarletteIntegration
        from sentry_sdk.integrations.sqlalchemy import SqlalchemyIntegration
        from sentry_sdk.integrations.asyncio import AsyncioIntegration
    except Exception as e:  # noqa: BLE001
        logger.warning("sentry.sdk.import_failed", error=str(e))
        return False

    try:
        sentry_sdk.init(
            dsn=dsn,
            environment=env,
            traces_sample_rate=float(getattr(settings, "SENTRY_TRACES_SAMPLE_RATE", 0.1)),
            profiles_sample_rate=float(getattr(settings, "SENTRY_PROFILES_SAMPLE_RATE", 0.0)),
            integrations=[
                StarletteIntegration(),
                FastApiIntegration(),
                SqlalchemyIntegration(),
                AsyncioIntegration(),
            ],
            send_default_pii=False,  # 민감정보 자동 첨부 차단 (이메일/IP 등)
            attach_stacktrace=True,
            before_send=_before_send,
            release=_resolve_release(),
        )
        logger.info("sentry.initialized", environment=env)
        return True
    except Exception as e:  # noqa: BLE001
        logger.warning("sentry.init.failed", error=str(e))
        return False


def _resolve_release() -> Optional[str]:
    """
    릴리즈 식별자. Fly.io 는 FLY_MACHINE_VERSION / FLY_RELEASE_VERSION 등을 제공.
    못 찾으면 None — Sentry SDK 가 자동 탐지 시도.
    """
    import os

    for key in ("SENTRY_RELEASE", "FLY_RELEASE_VERSION", "FLY_MACHINE_VERSION", "GIT_SHA"):
        v = os.environ.get(key)
        if v:
            return v
    return None


__all__ = ["init_sentry"]
