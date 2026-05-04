# =============================================
# app/models/term.py
# 역할: 약관 마스터 ORM 모델
#       - 서비스/개인정보/마케팅 등 동의 항목을 개별 행으로 관리
#       - 약관 본문은 현재 프론트 상수에서 관리 (DB 저장은 과설계 → 제외)
#       - 버전 관리로 개정 시 동의 재수집 가능
# 테이블명: terms
# =============================================

from sqlalchemy import (
    Column, Integer, String, Boolean, DateTime, func
)
from sqlalchemy.orm import relationship

from app.db.base import Base


class Term(Base):
    """
    약관 마스터 테이블.
    초기 seed: service (필수), privacy (필수), marketing (선택).
    """
    __tablename__ = "terms"

    # ── 기본 키 ──────────────────────────────
    id = Column(Integer, primary_key=True, autoincrement=True)

    # ── 약관 식별자 ───────────────────────────
    # code 는 프론트·백엔드가 공유하는 안정적 키
    code = Column(
        String(30),
        nullable=False,
        unique=True,
        comment="약관 코드 (service/privacy/marketing)",
    )
    title = Column(String(200), nullable=False, comment="약관 제목")

    # ── 동의 정책 ─────────────────────────────
    is_required = Column(
        Boolean,
        nullable=False,
        comment="필수 동의 여부 (False=선택 동의)",
    )
    version = Column(
        String(20),
        nullable=False,
        comment="현재 활성 버전 (예: 1.0)",
    )
    effective_from = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        comment="이 버전이 효력을 가지는 시작 시각",
    )

    # ── 관계 ─────────────────────────────────
    agreements = relationship("UserTermAgreement", back_populates="term")

    def __repr__(self) -> str:
        return (
            f"<Term id={self.id} code={self.code} "
            f"version={self.version} required={self.is_required}>"
        )
