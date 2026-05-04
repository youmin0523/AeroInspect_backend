# =============================================
# app/schemas/user.py
# 역할: 회원가입 및 사용자 조회 관련 Pydantic 입출력 스키마 정의
#       - UserSignupRequest: 회원가입 POST 요청 body (프론트 폼과 1:1 대응)
#       - UserResponse: 가입 완료/조회 시 직렬화 형식 (password_hash 절대 포함 X)
#       - UsernameCheck / EmailCheck: 중복확인 응답
# 사용: app/api/auth.py 라우터의 request body / response_model
# =============================================

from datetime import datetime
from typing import Optional, Literal
from uuid import UUID

from pydantic import BaseModel, EmailStr, Field, field_validator


# ── 조직 정보 (UserResponse 중첩용) ─────────
class OrgBriefResponse(BaseModel):
    """사용자 응답에 포함되는 조직 요약 정보"""
    id: UUID
    name: str
    role: str
    department: Optional[str] = None
    position: Optional[str] = None


# ── 중첩 입력 스키마 ──────────────────────────
class BusinessInfoInput(BaseModel):
    """사업자 회원 가입 시 추가 제출 필드"""
    biz_number: str = Field(
        ...,
        pattern=r"^\d{10}$",
        description="사업자등록번호 '-' 제외 10자리",
    )
    ceo_name: str = Field(..., min_length=1, max_length=100, description="대표자 성명")


class TermsAgreementInput(BaseModel):
    """약관 동의 상태 (프론트 3개 고정 항목)"""
    service: bool = Field(..., description="서비스 이용약관 (필수)")
    privacy: bool = Field(..., description="개인정보 수집·이용 (필수)")
    marketing: bool = Field(False, description="마케팅 수신 동의 (선택)")


# ── 회원가입 요청 스키마 ──────────────────────
class UserSignupRequest(BaseModel):
    """
    회원가입 POST 요청.
    프론트의 Signup.jsx form 필드를 백엔드 규약으로 정규화한 형태.
    - emailLocal/emailDomain 은 프론트에서 조합해 email 단일 필드로 전달
    - userId → username 으로 명명 통일
    """
    account_type: Literal["personal", "business"]
    email: EmailStr = Field(..., description="전체 이메일 (local@domain)")
    username: str = Field(
        ...,
        min_length=4,
        max_length=12,
        pattern=r"^[A-Za-z0-9]+$",
        description="로그인 아이디 (영문/숫자 4~12자)",
    )
    password: str = Field(
        ...,
        min_length=8,
        max_length=72,  # bcrypt 입력 한계 (72 bytes)
        description="비밀번호 (평문, 서버에서 해싱 후 저장)",
    )
    name: str = Field(..., min_length=1, max_length=100)
    phone: str = Field(
        ...,
        pattern=r"^\d{3}-\d{3,4}-\d{4}$",
        description="휴대폰 번호 (010-0000-0000 포맷)",
    )
    business: Optional[BusinessInfoInput] = Field(
        None,
        description="account_type='business' 일 때 필수",
    )
    terms: TermsAgreementInput

    # ── 교차 필드 검증 ────────────────────────
    @field_validator("terms")
    @classmethod
    def _required_terms_must_be_agreed(cls, v: TermsAgreementInput) -> TermsAgreementInput:
        if not (v.service and v.privacy):
            raise ValueError("필수 약관(서비스 이용약관, 개인정보)에 동의해야 합니다.")
        return v

    # account_type='business' 인데 business 정보가 비어있으면 거부
    # (Pydantic v2 model_validator는 schemas/defect.py 스타일과 맞추려고 field_validator만 사용)


# ── 응답 스키마 ──────────────────────────────
class UserResponse(BaseModel):
    """
    사용자 조회/가입 완료 시 반환 형식.
    ⚠ password_hash 는 절대 포함하지 않는다.
    """
    id: UUID
    account_type: str
    email: EmailStr
    username: str
    name: str
    phone: str
    profile_image_url: Optional[str] = None
    is_superadmin: bool = False
    created_at: datetime
    business: Optional["BusinessInfoResponse"] = None
    organizations: list["OrgBriefResponse"] = []

    class Config:
        from_attributes = True


class BusinessInfoResponse(BaseModel):
    """사업자 프로파일 응답 부분 (UserResponse에 중첩)"""
    biz_number: str
    ceo_name: str
    verified_at: Optional[datetime] = None

    class Config:
        from_attributes = True


# 전방 참조 해결 (BusinessInfoResponse 가 UserResponse 보다 아래 정의됨)
UserResponse.model_rebuild()


# ── 로그인 / 토큰 스키마 ─────────────────────
class LoginRequest(BaseModel):
    """일반 로그인 (아이디 + 비밀번호)"""
    username: str = Field(..., description="로그인 아이디")
    password: str = Field(..., description="비밀번호")

    model_config = {
        "json_schema_extra": {
            "example": {"username": "admin", "password": "admin"},
        },
    }


class OAuthCallbackRequest(BaseModel):
    """프론트에서 전달하는 OAuth 인가 코드"""
    code: str = Field(..., description="OAuth authorization code")
    redirect_uri: str = Field(..., description="프론트에서 사용한 redirect_uri")


class TokenResponse(BaseModel):
    """로그인 성공 시 반환하는 JWT 토큰 + 사용자 정보"""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    user: "UserResponse"

    model_config = {
        "json_schema_extra": {
            "example": {
                "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                "token_type": "bearer",
                "user": {
                    "id": "550e8400-e29b-41d4-a716-446655440000",
                    "account_type": "personal",
                    "email": "user@example.com",
                    "username": "tester01",
                    "name": "홍길동",
                    "phone": "010-1234-5678",
                    "is_superadmin": False,
                    "created_at": "2026-05-03T10:00:00Z",
                    "organizations": [],
                },
            },
        },
    }


class RefreshTokenRequest(BaseModel):
    """/auth/refresh 요청 바디"""
    refresh_token: str


class RefreshTokenResponse(BaseModel):
    """새 access_token 반환 (refresh_token은 재사용 권장)"""
    access_token: str
    token_type: str = "bearer"


# 전방 참조 해결
TokenResponse.model_rebuild()


# ── 중복확인 응답 스키마 ──────────────────────
class AvailabilityResponse(BaseModel):
    """이메일/아이디 중복 확인 응답"""
    available: bool = Field(..., description="사용 가능 여부 (True=미사용)")
    message: str


# ── 계정 찾기 요청/응답 스키마 ────────────────
class FindIdRequest(BaseModel):
    """아이디 찾기 요청 (이름 + 이메일로 조회 → 결과를 이메일로 발송)"""
    type: Literal["personal", "business"] = Field("personal", description="사용자 유형")
    name: str = Field(..., min_length=1, description="이름 (사업자는 담당자명)")
    email: EmailStr = Field(..., description="가입 시 등록한 이메일")
    bizNumber: Optional[str] = Field(None, description="사업자등록번호 (사업자 유형일 때)")


class FindPasswordRequest(BaseModel):
    """비밀번호 찾기 요청 (아이디 + 이메일 확인 → 임시 비밀번호 발급)"""
    type: Literal["personal", "business"] = Field("personal", description="사용자 유형")
    userId: str = Field(..., min_length=1, description="아이디")
    email: EmailStr = Field(..., description="가입 시 등록한 이메일")
    bizNumber: Optional[str] = Field(None, description="사업자등록번호 (사업자 유형일 때)")


class AccountRecoveryResponse(BaseModel):
    """아이디/비밀번호 찾기 공통 응답"""
    success: bool
    message: str
