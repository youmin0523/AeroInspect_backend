# =============================================
# app/schemas/organization.py
# 역할: 조직(회사) 관련 Pydantic 스키마
#       - OrganizationCreate: 조직 생성 요청
#       - OrganizationResponse: 조직 정보 응답
#       - OrgMemberResponse: 조직 멤버 정보 응답 (채팅 팀원 목록용)
#       - InviteMemberRequest: 멤버 초대 요청
# =============================================

from datetime import datetime
from typing import Optional, List
from uuid import UUID

from pydantic import BaseModel, Field


# ── 조직 ──────────────────────────────────────
class OrganizationCreate(BaseModel):
    """조직 생성 요청"""
    name: str = Field(..., max_length=200, description="회사/조직명")
    biz_number: Optional[str] = Field(None, max_length=10, description="사업자등록번호 (10자리)")


class OrganizationResponse(BaseModel):
    """조직 정보 응답"""
    id: UUID
    name: str
    biz_number: Optional[str]
    invite_code: Optional[str] = None
    invite_code_expires_at: Optional[datetime] = None
    member_count: int = 0
    created_at: datetime

    class Config:
        from_attributes = True


# ── 조직 멤버 ────────────────────────────────
class OrgMemberResponse(BaseModel):
    """
    조직 멤버 정보 응답 — 메신저 팀원 목록에서 사용.
    User 정보 + 조직 내 부서/직위/권한 결합.
    """
    user_id: UUID
    name: str
    email: str
    phone: Optional[str] = None
    initials: str
    role: str             # owner / admin / member
    department: Optional[str]
    position: Optional[str]
    status: str           # active / invited / deactivated
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class OrgMemberListResponse(BaseModel):
    """조직 멤버 목록 응답"""
    organization: OrganizationResponse
    members: List[OrgMemberResponse]
    total: int


# ── 멤버 초대 ────────────────────────────────
class InviteMemberRequest(BaseModel):
    """멤버 초대 요청"""
    email: str = Field(..., description="초대할 사용자 이메일")
    role: str = Field("member", pattern="^(admin|member)$")
    department: Optional[str] = None
    position: Optional[str] = None


# ── 멤버 정보 수정 ───────────────────────────
class UpdateMemberRequest(BaseModel):
    """멤버 부서/직위/권한 수정"""
    role: Optional[str] = Field(None, pattern="^(admin|member)$")
    department: Optional[str] = None
    position: Optional[str] = None
    status: Optional[str] = Field(None, pattern="^(active|deactivated)$")
    started_at: Optional[datetime] = Field(None, description="입사/계약 시작일")
    ended_at: Optional[datetime] = Field(None, description="퇴사/계약 만료일")


# ── 초대코드 가입 ────────────────────────────
class JoinByCodeRequest(BaseModel):
    """초대 코드로 조직 가입"""
    invite_code: str = Field(..., min_length=8, max_length=8, description="8자리 초대 코드")


# ── 사용자 배정 ──────────────────────────────
class AssignMemberRequest(BaseModel):
    """미소속 사용자를 조직에 배정"""
    user_id: UUID = Field(..., description="배정할 사용자 ID")
    organization_id: Optional[UUID] = Field(None, description="배정할 조직 ID (슈퍼어드민 전용, 미지정 시 현재 조직)")
    role: str = Field("member", pattern="^(admin|member)$")
    department: Optional[str] = None
    position: Optional[str] = None


# ── 미소속 사용자 응답 ───────────────────────
class UnaffiliatedUserResponse(BaseModel):
    """미소속 사용자 정보"""
    id: UUID
    name: str
    email: str
    account_type: str
    created_at: datetime

    class Config:
        from_attributes = True
