# =============================================
# app/models/device_token.py
# 역할: 푸시 알림용 디바이스 토큰 저장
#       - 사용자별 여러 기기 허용 (폰/태블릿/웹 푸시)
#       - FCM(Android/Web) / APNs(iOS) 구분
#       - 로그아웃/앱 제거 시 soft disable
# 테이블명: device_tokens
# =============================================

import uuid

from sqlalchemy import (
    Boolean, Column, DateTime, ForeignKey, Index, String, UniqueConstraint, func,
)
from sqlalchemy.dialects.postgresql import UUID

from app.db.base import Base


class DeviceToken(Base):
    """
    푸시 알림 대상 디바이스 토큰.
    (user_id, token) 복합 UNIQUE — 같은 기기 재등록 시 upsert 대상.
    """
    __tablename__ = "device_tokens"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # FCM / APNs / WEB
    platform = Column(String(10), nullable=False, comment="fcm | apns | web")
    token = Column(String(512), nullable=False, comment="디바이스 토큰 문자열")

    # 기기 라벨 (디버깅·사용자 UI용)
    device_label = Column(String(100), comment="사용자가 식별하기 위한 이름 (예: iPhone 15)")

    # 활성 여부. 푸시 실패 누적 or 사용자 로그아웃 시 False.
    is_active = Column(Boolean, default=True, nullable=False)

    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    last_used_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    __table_args__ = (
        UniqueConstraint("user_id", "token", name="uq_device_tokens_user_token"),
        Index("idx_device_tokens_active", "user_id", "is_active"),
    )

    def __repr__(self):
        return f"<DeviceToken user={self.user_id} platform={self.platform} active={self.is_active}>"
