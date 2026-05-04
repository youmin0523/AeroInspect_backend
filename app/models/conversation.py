# =============================================
# app/models/conversation.py
# 역할: 대화방 ORM 모델 정의
#       - 1:1 DM, 그룹 채팅, 팀 채널 지원
#       - 참여자 관리는 conversation_members M:N 테이블로 분리
# 테이블명: conversations
# =============================================

import uuid
from sqlalchemy import (
    Column, String, DateTime, Enum as SAEnum, Index, func, ForeignKey,
)
from sqlalchemy.dialects.postgresql import UUID

from app.db.base import Base


class Conversation(Base):
    """
    대화방 테이블.
    dm(1:1) / group(그룹 채팅) / channel(팀 채널) 세 유형.
    """
    __tablename__ = "conversations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)

    type = Column(
        SAEnum("dm", "group", "channel", name="conversation_type_enum"),
        nullable=False,
        comment="대화 유형 (dm / group / channel)",
    )

    name = Column(String(200), nullable=True, comment="대화방 이름 (DM은 null)")

    organization_id = Column(
        UUID(as_uuid=True),
        ForeignKey("organizations.id"),
        nullable=True,
        comment="소속 조직 ID (같은 조직 멤버만 참여 가능)",
    )

    created_by = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id"),
        nullable=True,
        comment="생성자 ID (system 채널은 null 가능)",
    )

    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
        comment="마지막 메시지 시각 기준 정렬용",
    )

    __table_args__ = (
        Index("idx_conversations_type", "type"),
        Index("idx_conversations_updated_at", updated_at.desc()),
    )

    def __repr__(self):
        return f"<Conversation id={self.id} type={self.type} name={self.name}>"
