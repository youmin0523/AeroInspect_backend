# =============================================
# app/models/user.py
# 역할: 회원 공통 정보 ORM 모델 (개인/사업자 공용)
#       - 3NF 기준으로 공통 속성만 보유
#       - 사업자 전용 필드는 business_profiles 테이블에 1:1 분리
#       - 약관 동의 이력은 user_term_agreements 테이블로 분리 (M:N)
# 테이블명: users
# =============================================

import uuid
from sqlalchemy import (
    Boolean, Column, String, DateTime, Enum as SAEnum, Index, func
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from app.db.base import Base


class User(Base):
    """
    회원 공통 정보 테이블.
    1 레코드 = 1 가입 사용자 (개인 또는 사업자 담당자).
    """
    __tablename__ = "users"

    # ── 기본 키 ──────────────────────────────
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # ── 계정 유형 ─────────────────────────────
    # personal: 개인 회원 / business: 사업자(개인/법인) 회원
    account_type = Column(
        SAEnum("personal", "business", name="account_type_enum"),
        nullable=False,
        comment="회원 유형",
    )

    # ── 로그인 식별자 ─────────────────────────
    # 프론트에서는 local@domain 분리 입력 → 서버에서 조합해 저장
    email = Column(String(255), nullable=False, unique=True, comment="이메일 (조합된 전체)")
    username = Column(String(50), nullable=False, unique=True, comment="로그인 아이디")

    # ── 인증 ─────────────────────────────────
    # bcrypt 해시 (~60자) / argon2 여유 대비 255 확보
    # OAuth 가입 사용자는 비밀번호 없이 가입 가능 → nullable
    password_hash = Column(String(255), nullable=True, comment="비밀번호 해시 (OAuth 사용자는 NULL)")

    # ── OAuth 소셜 로그인 ────────────────────
    oauth_provider = Column(
        String(20), nullable=True,
        comment="소셜 로그인 제공자 (google / kakao / naver / NULL=일반가입)",
    )
    oauth_id = Column(
        String(255), nullable=True, unique=True,
        comment="소셜 제공자 고유 사용자 ID",
    )

    # ── 플랫폼 관리자 ────────────────────────
    is_superadmin = Column(
        Boolean, nullable=False, default=False, server_default="false",
        comment="플랫폼 전체 관리자 여부 (전체 사용자/조직 관리 가능)",
    )

    # ── 개인정보 ─────────────────────────────
    name = Column(String(100), nullable=False, comment="이름 (사업자는 담당자 성명)")
    phone = Column(String(20), nullable=False, comment="휴대폰 번호 (010-0000-0000 포맷)")

    # ── 프로필 이미지 ────────────────────────
    profile_image_url = Column(
        String(500), nullable=True,
        comment="프로필 이미지 경로 (uploads/profiles/UUID.ext)",
    )

    # ── 감사 타임스탬프 ───────────────────────
    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        comment="가입 시각",
    )
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
        comment="최종 수정 시각",
    )

    # ── 관계 ─────────────────────────────────
    # 사업자 회원만 business_profile 행 보유 (1:1)
    business_profile = relationship(
        "BusinessProfile",
        back_populates="user",
        uselist=False,
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    # 약관 동의 이력 (1:N)
    term_agreements = relationship(
        "UserTermAgreement",
        back_populates="user",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    # ── 인덱스 ───────────────────────────────
    # email/username 은 UNIQUE 제약으로 인덱스 자동 생성됨
    # account_type 은 범주형이나 추후 필터 조회 대비
    __table_args__ = (
        Index("idx_users_account_type", "account_type"),
    )

    def __repr__(self) -> str:
        return (
            f"<User id={self.id} "
            f"username={self.username} "
            f"type={self.account_type}>"
        )
