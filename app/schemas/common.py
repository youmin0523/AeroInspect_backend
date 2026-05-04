# =============================================
# app/schemas/common.py
# 역할: 모든 라우터에서 공유하는 OpenAPI 응답 스키마/responses 딕셔너리
#       - ErrorResponse: FastAPI HTTPException 표준 직렬화 형태({"detail": "..."})와 1:1
#       - PROTECTED_RESPONSES: 인증 필요 라우터 공통 401/403/422 응답
#       - PUBLIC_RESPONSES: 비인증 라우터 공통 422 응답
# 사용: router.include_router(..., responses=PROTECTED_RESPONSES)
# =============================================

from pydantic import BaseModel, Field


class ErrorResponse(BaseModel):
    """FastAPI HTTPException 직렬화 표준 — `{"detail": "..."}`."""
    detail: str = Field(..., description="사람이 읽는 오류 메시지")

    model_config = {
        "json_schema_extra": {
            "example": {"detail": "유효하지 않거나 만료된 토큰입니다."},
        },
    }


# 인증이 필요한 라우터(/sites, /defects, /telemetry, /report 등)에 적용.
# 422 는 Pydantic validation 실패(FastAPI 기본 동작)라 모든 POST/PATCH 에 사실상 자동.
PROTECTED_RESPONSES = {
    401: {
        "model": ErrorResponse,
        "description": "인증 토큰 누락 / 만료 / 위변조",
    },
    403: {
        "model": ErrorResponse,
        "description": "권한 부족 (조직 미소속·역할 부족·슈퍼어드민 전용 등)",
    },
}


# 비인증 라우터(/auth, /oauth)에는 401/403을 default로 박지 않음 — 거짓 노출 방지.
# Validation 422는 FastAPI 가 자동 노출하므로 별도 명시 불필요.
PUBLIC_RESPONSES: dict = {}


# AI Webhook 라우터 — 자체 시크릿 헤더(X-AI-Webhook-Secret) 인증.
WEBHOOK_RESPONSES = {
    401: {
        "model": ErrorResponse,
        "description": "X-AI-Webhook-Secret 헤더 누락 또는 불일치",
    },
}
