# =============================================
# app/models/ai_chat.py
# 역할: OpenAI 기반 건축물·하자 도메인 챗봇 ORM 모델
#       - AiChatThread: 사용자별 대화방 (ChatGPT 스타일, 영속화)
#       - AiChatMessage: 대화 내 메시지 (user/assistant/system)
#       - thread.summary 로 오래된 턴을 LLM 압축 저장 → context window 보호
#       - 멀티테넌트: user_id + organization_id 이중 격리
# 테이블명: ai_chat_threads, ai_chat_messages
# =============================================

import uuid

from sqlalchemy import (
    Column, String, Text, Integer,
    DateTime, Enum as SAEnum, Index, func, ForeignKey,
)
from sqlalchemy.dialects.postgresql import UUID, JSONB

from app.db.base import Base


class AiChatThread(Base):
    """챗봇 대화방. 사용자가 ChatGPT 처럼 세션별로 수동 생성/이어가기."""
    __tablename__ = "ai_chat_threads"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)

    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        comment="대화 소유자",
    )
    organization_id = Column(
        UUID(as_uuid=True),
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=False,
        comment="대화가 속한 조직 — RAG 데이터 격리 기준",
    )

    title = Column(
        String(200),
        nullable=True,
        comment="대화방 제목 (첫 메시지로 자동 생성 가능)",
    )

    last_message_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        comment="목록 정렬 기준 — 메시지 INSERT 시 갱신",
    )

    # ── 컨텍스트 압축 ─────────────────────────
    # 오래된 메시지(SUMMARY_KEEP_RECENT 이전)는 LLM 으로 요약하여 summary 1 레코드로 압축.
    # build_context_messages 호출 시 summary + 최근 N개 메시지를 OpenAI 에 전달.
    summary = Column(
        Text,
        nullable=True,
        comment="요약본 (이전 대화 압축)",
    )
    summary_until_message_id = Column(
        UUID(as_uuid=True),
        ForeignKey("ai_chat_messages.id", ondelete="SET NULL"),
        nullable=True,
        comment="요약이 어디까지 커버하는지 watermark",
    )

    archived_at = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="soft delete 시각 (NULL = 활성)",
    )

    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    __table_args__ = (
        Index("idx_ai_threads_user_last", "user_id", last_message_at.desc()),
        Index("idx_ai_threads_org", "organization_id"),
    )

    def __repr__(self) -> str:
        return f"<AiChatThread id={self.id} user={self.user_id} title={self.title}>"


class AiChatMessage(Base):
    """챗봇 대화 메시지."""
    __tablename__ = "ai_chat_messages"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)

    thread_id = Column(
        UUID(as_uuid=True),
        ForeignKey("ai_chat_threads.id", ondelete="CASCADE"),
        nullable=False,
    )

    role = Column(
        SAEnum("user", "assistant", "system", name="ai_chat_role_enum"),
        nullable=False,
        comment="메시지 발신 역할 (OpenAI Chat Completions 호환)",
    )

    content = Column(Text, nullable=False, comment="메시지 본문")

    tokens = Column(
        Integer,
        nullable=True,
        comment="OpenAI usage.completion_tokens / prompt_tokens (있을 때만)",
    )

    meta = Column(
        JSONB,
        nullable=True,
        comment="디버깅용 메타 — 사용된 RAG 키, 모델명, finish_reason 등",
    )

    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    __table_args__ = (
        Index("idx_ai_messages_thread_created", "thread_id", created_at.asc()),
    )

    def __repr__(self) -> str:
        return f"<AiChatMessage id={self.id} thread={self.thread_id} role={self.role}>"
