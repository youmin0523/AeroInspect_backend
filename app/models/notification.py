# =============================================
# app/models/notification.py
# 역할: 알림 ORM 모델 정의
#       - 사용자별 알림을 저장하는 테이블
#       - 10종 카테고리: schedule, site, blueprint, work, defect,
#         report, drone, team, system, compliance
#       - JSONB metadata로 관련 엔티티 참조 (site_id, report_id, link 등)
# 테이블명: notifications
# =============================================

import uuid
from sqlalchemy import (
    Column, String, Text, Boolean,
    DateTime, Enum as SAEnum, Index, func, ForeignKey,
)
from sqlalchemy.dialects.postgresql import UUID, JSONB

from app.db.base import Base


class Notification(Base):
    """
    알림 테이블.
    시스템 이벤트(하자 탐지, 보고서 생성, 일정 변경 등)에 의해 생성되어
    해당 사용자에게 전달되는 알림 1건 = 1 레코드.
    """
    __tablename__ = "notifications"

    # ── 기본 키 ──────────────────────────────
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)

    # ── 수신자 ───────────────────────────────
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id"),
        nullable=False,
        comment="알림 수신 사용자 ID",
    )

    # ── 알림 분류 ─────────────────────────────
    category = Column(
        SAEnum(
            "schedule", "site", "blueprint", "work", "defect",
            "report", "drone", "team", "system", "compliance",
            name="notification_category_enum",
        ),
        nullable=False,
        comment="알림 카테고리",
    )

    # ── 내용 ─────────────────────────────────
    title = Column(String(300), nullable=False, comment="알림 제목")
    message = Column(Text, nullable=True, comment="알림 상세 메시지")

    # ── 메타데이터 ────────────────────────────
    # 관련 엔티티 참조: {site_id, report_id, defect_id, drone_id, link, severity, ...}
    metadata_ = Column("metadata", JSONB, nullable=True, comment="알림 부가 정보 JSON")

    # ── 읽음 상태 ─────────────────────────────
    is_read = Column(Boolean, default=False, nullable=False, comment="읽음 여부")

    # ── 생성 시각 ─────────────────────────────
    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        comment="알림 생성 시각 (UTC)",
    )

    # ── 인덱스 ───────────────────────────────
    __table_args__ = (
        Index("idx_notification_user_created", "user_id", created_at.desc()),
        Index("idx_notification_user_read", "user_id", "is_read"),
    )

    def __repr__(self):
        return (
            f"<Notification id={self.id} "
            f"category={self.category} "
            f"is_read={self.is_read}>"
        )
