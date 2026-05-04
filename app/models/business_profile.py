# =============================================
# app/models/business_profile.py
# 역할: 사업자 회원 전용 속성 ORM 모델
#       - users 테이블과 1:1 관계 (account_type='business'일 때만 행 존재)
#       - 개인 회원 레코드에 NULL 컬럼이 쌓이는 현상을 방지 (3NF)
# 테이블명: business_profiles
# =============================================

from sqlalchemy import (
    Column, String, DateTime, ForeignKey, Index
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from app.db.base import Base


class BusinessProfile(Base):
    """
    사업자 프로파일 테이블.
    users.id 를 기본키로 공유 (1:1 상속 관계).
    """
    __tablename__ = "business_profiles"

    # ── 기본 키 = 외래 키 (users.id) ──────────
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        primary_key=True,
        comment="사용자 FK (users.id)",
    )

    # ── 사업자 정보 ───────────────────────────
    biz_number = Column(
        String(10),
        nullable=False,
        unique=True,
        comment="사업자등록번호 ('-' 제외 10자리)",
    )
    ceo_name = Column(String(100), nullable=False, comment="대표자 성명")

    # ── 국세청 진위확인 결과 ──────────────────
    # 확인 성공 시 그 시각을 기록. 미확인 계정은 NULL 유지.
    verified_at = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="국세청 진위확인 성공 시각",
    )

    # ── 관계 ─────────────────────────────────
    user = relationship("User", back_populates="business_profile")

    __table_args__ = (
        Index("idx_biz_profiles_biz_number", "biz_number"),
    )

    def __repr__(self) -> str:
        return (
            f"<BusinessProfile user_id={self.user_id} "
            f"biz_number={self.biz_number}>"
        )
