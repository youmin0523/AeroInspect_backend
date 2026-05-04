# =============================================
# app/models/message.py
# 역할: 메시지 ORM 모델 정의
#       - 대화방 내 개별 메시지 1건 = 1 레코드
#       - conversation_id + created_at 복합 인덱스로 메시지 조회 최적화
# 테이블명: messages
# =============================================

import uuid
from sqlalchemy import (
    Column, String, Text, DateTime, Index, func, ForeignKey,
)
from sqlalchemy.dialects.postgresql import UUID

from app.db.base import Base


class Message(Base):
    """
    메시지 테이블.
    대화방 내 텍스트/파일 메시지 1건.
    """
    __tablename__ = "messages"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)

    conversation_id = Column(
        UUID(as_uuid=True),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
        comment="소속 대화방 ID",
    )

    sender_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id"),
        nullable=False,
        comment="발신자 ID",
    )

    text = Column(Text, nullable=True, comment="메시지 본문 (파일만 보낼 경우 null)")

    # 첨부파일
    file_url = Column(String(500), nullable=True, comment="첨부파일 URL 경로 (/uploads/chat/uuid.ext)")
    file_name = Column(String(300), nullable=True, comment="원본 파일명")
    file_content_type = Column(String(100), nullable=True, comment="MIME 타입 (image/png, application/pdf 등)")

    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    __table_args__ = (
        Index("idx_messages_conv_created", "conversation_id", created_at.asc()),
        Index("idx_messages_sender", "sender_id"),
    )

    def __repr__(self):
        return f"<Message id={self.id} conv={self.conversation_id} sender={self.sender_id}>"
