# =============================================
# app/models/inspection_schedule.py
# 역할: 점검 일정 ORM 모델
#       - "오늘 09:00 잠실 리센츠 — 백승희" 같은 일정 1건 = 1 레코드
#       - sites/users 와 FK 로 연결, 시각·상태·운영자 보유
#       - EmployeeLanding 의 "오늘 일정" 위젯 데이터 출처
# 테이블명: inspection_schedules
# =============================================

import uuid
from sqlalchemy import (
    Column, String, Text, DateTime, Enum as SAEnum, Index, func, ForeignKey,
)
from sqlalchemy.dialects.postgresql import UUID

from app.db.base import Base


class InspectionSchedule(Base):
    """드론 점검 일정 (예: 2026-04-22 14:00 잠실 리센츠 — 백승희 담당)."""

    __tablename__ = "inspection_schedules"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)

    # ── FK ──────────────────────────────────
    site_id = Column(
        UUID(as_uuid=True),
        ForeignKey("sites.id", ondelete="CASCADE"),
        nullable=False,
        comment="대상 현장 ID",
    )
    operator_user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        comment="담당 운영자 (배정 안 됐으면 null)",
    )
    organization_id = Column(
        UUID(as_uuid=True),
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=True,
        comment="멀티테넌트 격리용 조직 ID",
        index=True,
    )

    # ── 일정 정보 ───────────────────────────
    scheduled_at = Column(
        DateTime(timezone=True),
        nullable=False,
        comment="점검 예정 시각 (UTC)",
    )
    status = Column(
        SAEnum(
            "upcoming", "in_progress", "completed", "cancelled",
            name="schedule_status_enum",
        ),
        nullable=False,
        default="upcoming",
        comment="일정 상태",
    )

    # ── 비고 ─────────────────────────────────
    note = Column(Text, nullable=True, comment="비고 / 특이사항")

    # ── 감사 ─────────────────────────────────
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    __table_args__ = (
        Index("idx_schedule_org_time", "organization_id", scheduled_at.asc()),
        Index("idx_schedule_site_time", "site_id", scheduled_at.asc()),
    )

    def __repr__(self):
        return f"<InspectionSchedule id={self.id} site={self.site_id} at={self.scheduled_at}>"
