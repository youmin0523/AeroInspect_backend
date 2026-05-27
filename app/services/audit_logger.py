# =============================================
# app/services/audit_logger.py
# 역할: 감사 로그 기록 헬퍼
#       - write_audit() 단일 진입점으로 모든 책임 추적 사건 기록
#       - 민감 키(password/token/secret/api_key) 자동 redact
#       - structlog request_id 자동 첨부 (RequestIDMiddleware 와 연결)
#       - 실패 시 silent (감사 로그 자체가 메인 트랜잭션을 막으면 안 됨)
# 사용 예: 하자 검수 / 리포트 발행 / 현장 수정 / 권한 변경 / 인증
# =============================================

from __future__ import annotations

import re
from typing import Any, Optional
from uuid import UUID

import structlog
from fastapi import Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.audit_log import AuditLog
from app.core.logging import request_id_ctx

logger = structlog.get_logger(__name__)

# ── 민감 키 패턴 ──────────────────────────────
# 키 이름에 이 패턴이 부분 일치하면 값을 "[REDACTED]" 로 치환.
# 대소문자 무시. JSONB 저장 시 비밀이 새지 않도록 보장.
_REDACT_KEY_PATTERN = re.compile(
    r"(password|passwd|pwd|token|secret|api[_-]?key|authorization|cookie|session|"
    r"private[_-]?key|access[_-]?key|client[_-]?secret|webhook[_-]?secret)",
    re.IGNORECASE,
)


def _redact(obj: Any) -> Any:
    """민감 키를 재귀적으로 [REDACTED] 치환. JSON 직렬화 가능한 자료형 가정."""
    if isinstance(obj, dict):
        return {
            k: ("[REDACTED]" if _REDACT_KEY_PATTERN.search(str(k)) else _redact(v))
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [_redact(v) for v in obj]
    return obj


async def write_audit(
    db: AsyncSession,
    *,
    action: str,
    resource_type: str,
    user_id: Optional[UUID] = None,
    organization_id: Optional[UUID] = None,
    resource_id: Optional[UUID] = None,
    before: Optional[dict] = None,
    after: Optional[dict] = None,
    note: Optional[str] = None,
    request: Optional[Request] = None,
) -> Optional[AuditLog]:
    """
    감사 로그 1건 기록.
    실패해도 예외 던지지 않음 — 메인 트랜잭션이 망가지면 안 됨.
    db.add 만 호출 (commit 은 호출자 책임 — get_db dependency 가 처리).

    Args:
        action: 점 구분 doted-name (예: "defect.review.approve")
        resource_type: 대상 자원 종류 (예: "defect", "report", "site")
        user_id: 행위 주체. None=시스템 동작.
        organization_id: 조직 컨텍스트. 다조직 격리용.
        resource_id: 대상 자원 ID. CREATE/DELETE/UPDATE 모두 가능.
        before/after: 변경 전·후 스냅샷. 민감 키 자동 redact.
        note: 자유 형식 사유.
        request: FastAPI Request — IP/UA/request_id 자동 추출.

    Returns:
        AuditLog 인스턴스 (DB flush 전). 실패 시 None.
    """
    try:
        ip = None
        user_agent = None
        if request is not None:
            # X-Forwarded-For 우선 (Fly.io / Cloudflare 프록시 환경)
            ip = (
                request.headers.get("x-forwarded-for", "").split(",")[0].strip()
                or (request.client.host if request.client else None)
            )
            user_agent = request.headers.get("user-agent")
            if user_agent and len(user_agent) > 500:
                user_agent = user_agent[:500]

        # structlog ContextVar 에서 request_id 가져옴 (RequestIDMiddleware 설정)
        try:
            request_id = request_id_ctx.get(None)
        except LookupError:
            request_id = None

        entry = AuditLog(
            user_id=user_id,
            organization_id=organization_id,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            before=_redact(before) if before is not None else None,
            after=_redact(after) if after is not None else None,
            ip=ip,
            user_agent=user_agent,
            request_id=request_id,
            note=note,
        )
        db.add(entry)
        return entry
    except Exception as exc:
        # 감사 로그 실패가 메인 비즈니스 트랜잭션을 깨면 안 됨.
        logger.warning(
            "audit_log_write_failed",
            action=action,
            resource_type=resource_type,
            error=str(exc),
        )
        return None
