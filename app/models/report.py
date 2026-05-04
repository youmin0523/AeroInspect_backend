# =============================================
# app/models/report.py
# 역할: 생성된 하자 점검 보고서 ORM 모델
#       - LLM이 생성한 보고서를 DB에 저장하여 조회/다운로드 가능
# 테이블명: reports
# =============================================

import uuid
from datetime import datetime

from sqlalchemy import Column, String, Integer, Text, DateTime, func, ForeignKey
from sqlalchemy.dialects.postgresql import UUID

from app.db.base import Base


class Report(Base):
    """생성된 하자 점검 보고서 테이블"""
    __tablename__ = "reports"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)

    # ── 현장 연결 (경향보고서 연계) ───────────
    site_id = Column(UUID(as_uuid=True), ForeignKey("sites.id"), nullable=True, comment="연결 현장 ID")

    # ── 보고서 정보 ──────────────────────────
    title = Column(String(200), comment="보고서 제목")
    building_name = Column(String(200), comment="점검 건물명")
    inspector_name = Column(String(100), comment="점검자명")
    provider = Column(String(20), comment="LLM 제공자 (claude / gemini)")

    # ── 보고서 본문 ──────────────────────────
    content = Column(Text, nullable=False, comment="마크다운 보고서 본문")

    # ── 하자 통계 ────────────────────────────
    defect_count = Column(Integer, default=0, comment="총 하자 건수")
    high_count = Column(Integer, default=0)
    med_count = Column(Integer, default=0)
    low_count = Column(Integer, default=0)

    # ── 타임스탬프 ───────────────────────────
    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    def __repr__(self):
        return f"<Report id={self.id} title={self.title}>"
