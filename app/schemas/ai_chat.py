# =============================================
# app/schemas/ai_chat.py
# 역할: OpenAI 챗봇 Pydantic 스키마
#       - ThreadCreate/Update/Response: 대화방 CRUD 요청/응답
#       - MessageCreate/Response: 메시지 전송/조회
#       - role=system 은 응답에서 제외 (보안: 시스템 프롬프트 노출 방지)
# =============================================

from datetime import datetime
from typing import Literal, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


# ── 대화방(Thread) 스키마 ─────────────────────


class ThreadCreate(BaseModel):
    """대화방 생성 요청. 제목 없으면 첫 메시지로 자동 추출 가능."""
    title: Optional[str] = Field(None, max_length=200, description="대화방 제목 (선택)")


class ThreadUpdate(BaseModel):
    """대화방 제목 수정."""
    title: str = Field(..., min_length=1, max_length=200)


class ThreadResponse(BaseModel):
    """대화방 응답. 목록·단건 모두 동일 형태."""
    id: UUID
    title: Optional[str]
    last_message_at: datetime
    created_at: datetime
    has_summary: bool = Field(..., description="이전 대화 요약 보유 여부")

    model_config = ConfigDict(from_attributes=True)


class ThreadListResponse(BaseModel):
    """대화방 목록 응답."""
    threads: list[ThreadResponse]
    has_more: bool = Field(..., description="다음 페이지 존재 여부 (커서 페이지네이션)")


# ── 메시지(Message) 스키마 ───────────────────


class MessageCreate(BaseModel):
    """메시지 전송 요청. 입력 길이 가드 4000자."""
    content: str = Field(..., min_length=1, max_length=4000)


class MessageResponse(BaseModel):
    """메시지 응답. system role 은 노출 안 함 (보안)."""
    id: UUID
    role: Literal["user", "assistant"]
    content: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class MessageHistoryResponse(BaseModel):
    """메시지 히스토리 응답 (커서 페이지네이션)."""
    messages: list[MessageResponse]
    has_more: bool
