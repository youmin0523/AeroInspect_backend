# =============================================
# app/models/conversation_member.py
# 역할: 대화방 ↔ 사용자 M:N 관계 테이블
#       - 참여 시각 + 마지막 읽은 시각으로 읽음 상태 추적
# 테이블명: conversation_members
# =============================================

import uuid
from sqlalchemy import (
    Column, DateTime, Index, func, ForeignKey, UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import UUID

from app.db.base import Base


class ConversationMember(Base):
    """
    대화방 참여자 매핑 테이블.
    conversation_id + user_id 유니크 제약.
    last_read_at 으로 미읽음 메시지 수 계산.
    """
    __tablename__ = "conversation_members"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)

    conversation_id = Column(
        UUID(as_uuid=True),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
    )

    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id"),
        nullable=False,
    )

    joined_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    last_read_at = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="마지막 읽은 시각 (null = 읽은 적 없음)",
    )

    __table_args__ = (
        UniqueConstraint("conversation_id", "user_id", name="uq_conv_member"),
        Index("idx_conv_member_user", "user_id"),
        Index("idx_conv_member_conv", "conversation_id"),
    )

    def __repr__(self):
        return f"<ConversationMember conv={self.conversation_id} user={self.user_id}>"
