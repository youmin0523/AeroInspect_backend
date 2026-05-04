# =============================================
# app/models/site.py
# 역할: 현장(Site) ORM 모델 정의
#       - 드론 하자점검이 수행되는 건설 현장 / 건물 단위
#       - B2B(기업 발주) / B2C(개인 의뢰) 모두 포괄
#       - 경향보고서 연계를 위해 DefectLog, Report 에서 FK 참조
# 테이블명: sites
# =============================================

import uuid
from datetime import datetime

from sqlalchemy import (
    Column, String, Float, Integer, Text, Date,
    DateTime, Enum as SAEnum, Index, func, ForeignKey,
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship

from app.db.base import Base


class Site(Base):
    """현장 관리 테이블"""
    __tablename__ = "sites"

    # ── 기본 키 ──────────────────────────────
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)

    # ── 등록 순번 ─────────────────────────────
    seq = Column(Integer, nullable=False, comment="등록 순번 (자동 증가)")

    # ── 현장 정보 ─────────────────────────────
    name = Column(String(200), nullable=False, comment="현장명 (건물명+동호수)")
    inspection_type = Column(String(50), nullable=True, default="사전점검", comment="점검 구분 (사전/입주/정기/하자/특별/기타)")
    address = Column(String(500), nullable=True, comment="주소")
    building_type = Column(
        SAEnum(
            "아파트", "오피스텔", "상가", "주상복합", "오피스", "단독주택", "기타",
            name="building_type_enum",
        ),
        nullable=False,
        default="아파트",
        comment="건물 유형",
    )
    total_area = Column(Float, nullable=True, comment="공급면적 (㎡)")
    building_count = Column(Integer, nullable=True, comment="동 수")
    unit_count = Column(Integer, nullable=True, comment="세대(호) 수")

    # ── 의뢰 정보 (B2B / B2C) ─────────────────
    client_type = Column(
        SAEnum("B2B", "B2C", name="client_type_enum"),
        nullable=False,
        default="B2B",
        comment="의뢰 유형",
    )
    client_name = Column(String(200), nullable=True, comment="B2B: 발주처/시행사명, B2C: 의뢰인 성명")
    client_contact = Column(String(100), nullable=True, comment="의뢰인/담당자 연락처")

    # ── 일정 ─────────────────────────────────
    contract_start = Column(Date, nullable=True, comment="계약/점검 시작일")
    contract_end = Column(Date, nullable=True, comment="계약/점검 종료일")

    # ── 상태 ─────────────────────────────────
    status = Column(
        SAEnum("active", "pending", "completed", "cancelled", name="site_status_enum"),
        nullable=False,
        default="pending",
        comment="현장 상태",
    )

    # ── 팀 & 운영 ────────────────────────────
    assigned_members = Column(JSONB, nullable=True, default=list, comment="배정 팀원 [{id, name, role}]")
    memo = Column(Text, nullable=True, comment="비고/메모")
    inspection_count = Column(Integer, nullable=False, default=0, comment="누적 점검 횟수")
    last_inspection_date = Column(Date, nullable=True, comment="최근 점검일")

    # ── 촬영 영상 메타 ────────────────────────
    recordings = Column(JSONB, nullable=True, default=list, comment="촬영 이력 [{id, date, type, duration_sec, url}]")

    # ── 소속 조직 ────────────────────────────
    organization_id = Column(
        UUID(as_uuid=True),
        ForeignKey("organizations.id"),
        nullable=True,
        comment="소속 조직 ID (멀티테넌트 격리 기준)",
        index=True,
    )

    # ── 등록 정보 ────────────────────────────
    created_by = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True, comment="등록자")
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    # ── 인덱스 ───────────────────────────────
    __table_args__ = (
        Index("idx_sites_status", "status"),
        Index("idx_sites_name", "name"),
        Index("idx_sites_created_at", created_at.desc()),
        Index("idx_sites_org_id", "organization_id"),
    )

    def __repr__(self):
        return f"<Site id={self.id} name={self.name} status={self.status}>"
