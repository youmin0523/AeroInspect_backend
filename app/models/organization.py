# =============================================
# app/models/organization.py
# 역할: 조직(회사) + 조직 멤버 ORM 모델
#
# 설계 근거 (대기업/중견기업 혼합 방식):
#   방식 1 — 자동 매칭: B2B 회원가입 시 biz_number(사업자등록번호) 가
#            이미 존재하는 Organization 의 biz_number 와 일치하면 자동 소속
#   방식 2 — 관리자 초대: 조직 admin 이 이메일/아이디로 멤버를 초대 →
#            초대받은 사용자가 수락 시 소속 확정
#
#   두 방식 모두 OrganizationMember 레코드 생성으로 귀결됨.
#
# 테이블명: organizations, organization_members
# =============================================

import secrets
import uuid
from datetime import datetime, timedelta, timezone

from sqlalchemy import (
    Column, String, DateTime, Enum as SAEnum, Index, func, ForeignKey,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import UUID

from app.db.base import Base

# 초대코드 기본 유효기간 (일)
INVITE_CODE_VALIDITY_DAYS = 30


def _generate_invite_code(length: int = 8) -> str:
    """8자리 영숫자 초대 코드 생성 (대문자+숫자, 혼동 문자 제외)"""
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"  # 0/O, 1/I/L 제외
    return "".join(secrets.choice(alphabet) for _ in range(length))


def _default_invite_code_expires_at() -> datetime:
    """초대코드 기본 만료 시각 (현재 + 30일)"""
    return datetime.now(timezone.utc) + timedelta(days=INVITE_CODE_VALIDITY_DAYS)


class Organization(Base):
    """
    조직(회사) 테이블.
    사업자등록번호 기준으로 유일성을 보장하여 같은 회사 자동 그룹핑.
    """
    __tablename__ = "organizations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)

    name = Column(String(200), nullable=False, comment="회사/조직명")

    # 사업자등록번호 — 같은 번호로 가입한 B2B 사용자는 자동 소속
    biz_number = Column(
        String(10),
        nullable=True,
        unique=True,
        comment="사업자등록번호 (10자리, '-' 제외). 개인 조직은 null",
    )

    # 초대 코드 — 미소속 사용자가 이 코드로 조직 가입
    invite_code = Column(
        String(8),
        nullable=False,
        unique=True,
        default=_generate_invite_code,
        comment="8자리 초대 코드 (조직 가입용)",
    )

    invite_code_expires_at = Column(
        DateTime(timezone=True),
        nullable=True,
        default=_default_invite_code_expires_at,
        comment="초대 코드 만료 시각 (null=무제한, 기본 30일)",
    )

    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    __table_args__ = (
        Index("idx_org_biz_number", "biz_number"),
        Index("idx_org_invite_code", "invite_code"),
    )

    def regenerate_invite_code(self):
        """초대코드 재생성 + 만료일 갱신"""
        self.invite_code = _generate_invite_code()
        self.invite_code_expires_at = _default_invite_code_expires_at()

    def is_invite_code_expired(self) -> bool:
        """초대 코드가 만료되었는지 확인"""
        if self.invite_code_expires_at is None:
            return False
        return datetime.now(timezone.utc) >= self.invite_code_expires_at

    def __repr__(self):
        return f"<Organization id={self.id} name={self.name} biz={self.biz_number}>"


class OrganizationMember(Base):
    """
    조직 멤버 매핑 테이블.
    role: owner(최초 생성자), admin(관리자), member(일반 멤버)
    status: active(활성), invited(초대 수락 대기), deactivated(비활성)
    """
    __tablename__ = "organization_members"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)

    organization_id = Column(
        UUID(as_uuid=True),
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=False,
    )

    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )

    role = Column(
        SAEnum("owner", "admin", "member", name="org_member_role_enum"),
        nullable=False,
        default="member",
        comment="조직 내 권한 (owner/admin/member)",
    )

    department = Column(String(100), nullable=True, comment="부서명 (안전진단 1팀, 드론운용팀 등)")
    position = Column(String(50), nullable=True, comment="직위 (과장, 대리, 사원 등)")

    status = Column(
        SAEnum("active", "invited", "deactivated", name="org_member_status_enum"),
        nullable=False,
        default="active",
        comment="멤버 상태",
    )

    joined_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    # ── 계약 기간 관리 ───────────────────────
    started_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        comment="입사/계약 시작일",
    )
    ended_at = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="퇴사/계약 만료일 (null=재직 중, 과거 날짜=접근 차단)",
    )

    __table_args__ = (
        UniqueConstraint("organization_id", "user_id", name="uq_org_member"),
        Index("idx_org_member_org", "organization_id"),
        Index("idx_org_member_user", "user_id"),
    )

    def __repr__(self):
        return f"<OrganizationMember org={self.organization_id} user={self.user_id} role={self.role}>"
