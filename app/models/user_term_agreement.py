# =============================================
# app/models/user_term_agreement.py
# 역할: 사용자-약관 동의 이력 ORM 모델 (M:N 연결 테이블)
#       - users ↔ terms 관계를 원자화해 1NF/3NF 충족
#       - version 스냅샷으로 약관 개정 후에도 동의 당시 버전 추적 가능
# 테이블명: user_term_agreements
# =============================================

from sqlalchemy import (
    Column, Integer, String, DateTime, ForeignKey,
    PrimaryKeyConstraint, Index, func
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from app.db.base import Base


class UserTermAgreement(Base):
    """
    사용자 약관 동의 레코드.
    (user_id, term_id) 복합 PK → 사용자당 약관 1건만 최신 동의 보존.
    약관 재동의(버전업)가 필요하면 version 컬럼만 갱신.
    """
    __tablename__ = "user_term_agreements"

    # ── 외래 키 (복합 PK 구성) ────────────────
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        comment="사용자 FK",
    )
    term_id = Column(
        Integer,
        ForeignKey("terms.id"),
        nullable=False,
        comment="약관 FK",
    )

    # ── 동의 메타데이터 ───────────────────────
    # 동의 당시 약관 버전 스냅샷 (terms.version과 일치 or 과거 버전)
    version = Column(String(20), nullable=False, comment="동의 당시 약관 버전")
    agreed_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        comment="동의 시각",
    )

    # ── 관계 ─────────────────────────────────
    user = relationship("User", back_populates="term_agreements")
    term = relationship("Term", back_populates="agreements")

    # ── 제약 / 인덱스 ─────────────────────────
    __table_args__ = (
        PrimaryKeyConstraint("user_id", "term_id", name="pk_user_term"),
        Index("idx_uta_user", "user_id"),
        Index("idx_uta_term", "term_id"),
    )

    def __repr__(self) -> str:
        return (
            f"<UserTermAgreement user_id={self.user_id} "
            f"term_id={self.term_id} version={self.version}>"
        )
