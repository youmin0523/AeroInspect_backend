# =============================================
# app/schemas/chat.py
# 역할: 메신저 Pydantic 입출력 스키마 정의
#       - ConversationCreate / ConversationResponse
#       - MessageCreate / MessageResponse / MessageListResponse
# =============================================

from datetime import datetime
from typing import Optional, List
from uuid import UUID

from pydantic import BaseModel, Field


# ── 멤버 요약 ────────────────────────────────
class MemberBrief(BaseModel):
    user_id: UUID
    name: str
    initials: str


# ── 마지막 메시지 요약 ───────────────────────
class LastMessageBrief(BaseModel):
    text: Optional[str] = None
    file_name: Optional[str] = None
    sender_name: str
    created_at: datetime


# ── 대화방 ────────────────────────────────────
class ConversationCreate(BaseModel):
    """대화방 생성 요청"""
    type: str = Field(..., pattern="^(dm|group|channel)$")
    name: Optional[str] = Field(None, max_length=200)
    participant_ids: List[UUID] = Field(..., min_length=1)


class ConversationResponse(BaseModel):
    """대화방 응답"""
    id: UUID
    type: str
    name: Optional[str]
    participants: List[MemberBrief]
    created_at: datetime
    updated_at: datetime
    last_message: Optional[LastMessageBrief] = None

    class Config:
        from_attributes = True


class ConversationListResponse(BaseModel):
    """대화방 목록 응답"""
    items: List[ConversationResponse]
    total: int


# ── 메시지 ────────────────────────────────────
class MessageCreate(BaseModel):
    """메시지 전송 요청"""
    text: str = Field(..., min_length=1, max_length=5000)


class MessageResponse(BaseModel):
    """메시지 응답"""
    id: UUID
    conversation_id: UUID
    sender_id: UUID
    sender_name: str
    sender_initials: str
    sender_profile_image_url: Optional[str] = None
    text: Optional[str] = None
    file_url: Optional[str] = None
    file_name: Optional[str] = None
    file_content_type: Optional[str] = None
    read_by_count: int = 0
    created_at: datetime

    class Config:
        from_attributes = True


class MessageListResponse(BaseModel):
    """메시지 목록 응답"""
    items: List[MessageResponse]
    total: int
    has_more: bool


# ── 미읽음 카운트 ────────────────────────────
class UnreadCountResponse(BaseModel):
    """미읽음 카운트 응답"""
    total: int
    per_conversation: dict  # { conversation_id: count }
