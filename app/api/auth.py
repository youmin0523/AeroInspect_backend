# =============================================
# app/api/auth.py
# 역할: 회원가입, 로그인, 계정 중복확인 엔드포인트
#       - POST /auth/signup           → 신규 회원 생성 (개인/사업자 공용)
#       - POST /auth/login            → 일반 로그인 (아이디+비밀번호 → JWT)
#       - GET  /auth/me               → 현재 로그인 사용자 조회
#       - GET  /auth/check-email      → 이메일 중복 확인
#       - GET  /auth/check-username   → 아이디 중복 확인
# =============================================

import asyncio
import os
import secrets
import string
import uuid as uuid_mod
from typing import Optional

import aiofiles
from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import hash_password, verify_password
from app.core.jwt import create_access_token, create_refresh_token, decode_refresh_token
from app.services.email_service import send_found_username_email, send_temp_password_email
from app.dependencies import get_db, get_current_user_with_org
from app.models.user import User
from app.models.business_profile import BusinessProfile
from app.models.term import Term
from app.models.user_term_agreement import UserTermAgreement
from app.schemas.user import (
    AccountRecoveryResponse,
    AvailabilityResponse,
    BusinessInfoResponse,
    FindIdRequest,
    FindPasswordRequest,
    LoginRequest,
    OrgBriefResponse,
    RefreshTokenRequest,
    RefreshTokenResponse,
    TokenResponse,
    UserSignupRequest,
    UserResponse,
)

PROFILE_UPLOAD_DIR = "./uploads/profiles"
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp", "image/gif"}
MAX_PROFILE_IMAGE_SIZE = 5 * 1024 * 1024  # 5 MB

router = APIRouter()


# ── 내부 헬퍼: 사용자 조직 목록 조회 ─────────
async def _get_user_orgs(db, user_id):
    """사용자의 활성 조직 멤버십 → OrgBriefResponse 리스트"""
    from datetime import datetime, timezone
    from app.models.organization import Organization, OrganizationMember
    result = await db.execute(
        select(OrganizationMember, Organization)
        .join(Organization, Organization.id == OrganizationMember.organization_id)
        .where(OrganizationMember.user_id == user_id)
        .where(OrganizationMember.status == "active")
        .where(
            (OrganizationMember.ended_at.is_(None))
            | (OrganizationMember.ended_at > datetime.now(timezone.utc))
        )
    )
    return [
        OrgBriefResponse(
            id=org.id, name=org.name, role=m.role,
            department=m.department, position=m.position,
        )
        for m, org in result
    ]


# ── 내부 헬퍼 ────────────────────────────────
async def _email_exists(db: AsyncSession, email: str) -> bool:
    result = await db.execute(select(User.id).where(User.email == email))
    return result.scalar_one_or_none() is not None


async def _username_exists(db: AsyncSession, username: str) -> bool:
    result = await db.execute(select(User.id).where(User.username == username))
    return result.scalar_one_or_none() is not None


async def _biz_number_exists(db: AsyncSession, biz_number: str) -> bool:
    result = await db.execute(
        select(BusinessProfile.user_id).where(BusinessProfile.biz_number == biz_number)
    )
    return result.scalar_one_or_none() is not None


# ── 중복 확인 ────────────────────────────────
@router.get("/check-email", response_model=AvailabilityResponse)
async def check_email(
    email: str = Query(..., description="확인할 이메일 주소"),
    db: AsyncSession = Depends(get_db),
):
    """이메일 사용 가능 여부 확인 (회원가입 버튼 옆 '중복 확인')."""
    taken = await _email_exists(db, email)
    return AvailabilityResponse(
        available=not taken,
        message="이미 사용 중인 이메일입니다." if taken else "사용 가능한 이메일입니다.",
    )


@router.get("/check-username", response_model=AvailabilityResponse)
async def check_username(
    username: str = Query(..., description="확인할 아이디"),
    db: AsyncSession = Depends(get_db),
):
    """아이디 사용 가능 여부 확인."""
    taken = await _username_exists(db, username)
    return AvailabilityResponse(
        available=not taken,
        message="이미 사용 중인 아이디입니다." if taken else "사용 가능한 아이디입니다.",
    )


# ── 회원가입 ─────────────────────────────────
@router.post(
    "/signup",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
)
async def signup(
    payload: UserSignupRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    신규 회원 등록.
    개인/사업자 공용 — account_type='business' 시 BusinessProfile 1:1 동시 생성.
    선택한 약관에 대해 UserTermAgreement 레코드도 함께 생성.
    """
    # ── 1) account_type 별 필수값 교차 검증 ──
    if payload.account_type == "business" and payload.business is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="사업자 회원은 business 정보(biz_number, ceo_name)가 필요합니다.",
        )

    # ── 2) 유니크 필드 사전 검증 ─────────────
    # (DB UNIQUE 제약과 이중 체크: 친절한 메시지 + 경쟁 조건 보호)
    if await _email_exists(db, payload.email):
        raise HTTPException(status_code=409, detail="이미 사용 중인 이메일입니다.")
    if await _username_exists(db, payload.username):
        raise HTTPException(status_code=409, detail="이미 사용 중인 아이디입니다.")
    if payload.business and await _biz_number_exists(db, payload.business.biz_number):
        raise HTTPException(status_code=409, detail="이미 등록된 사업자등록번호입니다.")

    # ── 3) User 레코드 생성 ──────────────────
    user = User(
        account_type=payload.account_type,
        email=payload.email,
        username=payload.username,
        password_hash=hash_password(payload.password),
        name=payload.name,
        phone=payload.phone,
    )
    db.add(user)
    await db.flush()  # user.id 확보 (자식 테이블 FK 연결용)

    # ── 4) 사업자 프로파일 (해당 시) ──────────
    if payload.business:
        db.add(
            BusinessProfile(
                user_id=user.id,
                biz_number=payload.business.biz_number,
                ceo_name=payload.business.ceo_name,
                # verified_at은 별도 진위확인 엔드포인트에서 갱신
            )
        )

    # ── 5) 약관 동의 이력 기록 ───────────────
    # 프론트의 3개 고정 체크 → DB terms.code 로 매핑
    agreement_map = {
        "service": payload.terms.service,
        "privacy": payload.terms.privacy,
        "marketing": payload.terms.marketing,
    }
    agreed_codes = [code for code, checked in agreement_map.items() if checked]

    if agreed_codes:
        term_rows = await db.execute(
            select(Term).where(Term.code.in_(agreed_codes))
        )
        for term in term_rows.scalars().all():
            db.add(
                UserTermAgreement(
                    user_id=user.id,
                    term_id=term.id,
                    version=term.version,  # 동의 당시 버전 스냅샷
                )
            )

    # ── 6) 응답 생성 ────────────────────────
    # (get_db 의존성에서 commit 수행됨)
    await db.flush()

    business_resp = None
    if payload.business:
        business_resp = BusinessInfoResponse(
            biz_number=payload.business.biz_number,
            ceo_name=payload.business.ceo_name,
            verified_at=None,
        )

    return UserResponse(
        id=user.id,
        account_type=user.account_type,
        email=user.email,
        username=user.username,
        name=user.name,
        phone=user.phone,
        profile_image_url=user.profile_image_url,
        created_at=user.created_at,
        business=business_resp,
    )


# ── 로그인 ──────────────────────────────────
@router.post("/login", response_model=TokenResponse)
async def login(
    payload: LoginRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    일반 로그인: 아이디 + 비밀번호 → JWT 액세스 토큰 발급.
    """
    result = await db.execute(select(User).where(User.username == payload.username))
    user = result.scalar_one_or_none()

    if user is None or user.password_hash is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="아이디 또는 비밀번호가 올바르지 않습니다.")

    # bcrypt(rounds=12) 검증은 ~250ms 동기 CPU 작업 → 스레드로 오프로드해 이벤트 루프 블로킹 제거.
    if not await asyncio.to_thread(verify_password, payload.password, user.password_hash):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="아이디 또는 비밀번호가 올바르지 않습니다.")

    token = create_access_token(user.id)
    refresh = create_refresh_token(user.id)
    orgs = await _get_user_orgs(db, user.id)

    return TokenResponse(
        access_token=token,
        refresh_token=refresh,
        user=UserResponse(
            id=user.id,
            account_type=user.account_type,
            email=user.email,
            username=user.username,
            name=user.name,
            phone=user.phone,
            profile_image_url=user.profile_image_url,
            is_superadmin=user.is_superadmin,
            created_at=user.created_at,
            organizations=orgs,
        ),
    )


# ── 리프레시 토큰으로 access 재발급 ──────────
@router.post("/refresh", response_model=RefreshTokenResponse)
async def refresh_access_token(
    payload: RefreshTokenRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    유효한 refresh_token → 새 access_token + 새 refresh_token 재발급 (회전).

    R-v1.1.17: refresh token rotation 도입. 매 refresh 시 새 refresh도 발급.
    탈취된 refresh로 무제한 access 갱신을 차단 (P0 보안).
    클라이언트는 응답의 refresh_token으로 localStorage 덮어써야 함.
    """
    user_id = decode_refresh_token(payload.refresh_token)
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="유효하지 않거나 만료된 리프레시 토큰입니다.",
        )

    # 사용자 존재·활성 여부 재확인 (refresh 발급 이후 계정 삭제된 케이스 대비)
    user = await db.scalar(select(User).where(User.id == uuid_mod.UUID(user_id)))
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="사용자를 찾을 수 없습니다.",
        )

    new_access = create_access_token(user.id)
    new_refresh = create_refresh_token(user.id)  # 회전: 새 refresh도 발급
    return RefreshTokenResponse(access_token=new_access, refresh_token=new_refresh)


# ── 현재 사용자 조회 ────────────────────────
@router.get("/me", response_model=UserResponse)
async def get_me(
    result=Depends(get_current_user_with_org),
):
    """Bearer 토큰으로 현재 로그인 사용자 정보 반환 (조직 정보 포함)."""
    user, memberships = result
    orgs = [
        OrgBriefResponse(
            id=org.id, name=org.name, role=m.role,
            department=m.department, position=m.position,
        )
        for m, org in memberships
    ] if memberships else []

    return UserResponse(
        id=user.id,
        account_type=user.account_type,
        email=user.email,
        username=user.username,
        name=user.name,
        phone=user.phone,
        profile_image_url=user.profile_image_url,
        is_superadmin=user.is_superadmin,
        created_at=user.created_at,
        organizations=orgs,
    )


# ── 내 정보 수정 ────────────────────────────
class UpdateMeRequest(BaseModel):
    """내 정보 수정 요청"""
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    phone: Optional[str] = Field(None, pattern=r"^\d{3}-\d{3,4}-\d{4}$")


@router.patch("/me", response_model=UserResponse)
async def update_me(
    payload: UpdateMeRequest,
    result=Depends(get_current_user_with_org),
    db: AsyncSession = Depends(get_db),
):
    """현재 사용자 이름/전화번호 수정."""
    user, memberships = result

    if payload.name is not None:
        user.name = payload.name
    if payload.phone is not None:
        user.phone = payload.phone
    await db.flush()

    orgs = [
        OrgBriefResponse(
            id=org.id, name=org.name, role=m.role,
            department=m.department, position=m.position,
        )
        for m, org in memberships
    ] if memberships else []

    return UserResponse(
        id=user.id,
        account_type=user.account_type,
        email=user.email,
        username=user.username,
        name=user.name,
        phone=user.phone,
        profile_image_url=user.profile_image_url,
        is_superadmin=user.is_superadmin,
        created_at=user.created_at,
        organizations=orgs,
    )


# ── 프로필 이미지 업로드 ────────────────────
@router.put("/me/profile-image", response_model=UserResponse)
async def upload_profile_image(
    file: UploadFile = File(...),
    result=Depends(get_current_user_with_org),
    db: AsyncSession = Depends(get_db),
):
    """프로필 이미지 업로드 (기존 이미지가 있으면 교체)."""
    user, memberships = result

    if file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="허용되지 않는 이미지 형식입니다. (JPEG, PNG, WebP, GIF만 가능)",
        )

    content = await file.read()
    if len(content) > MAX_PROFILE_IMAGE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="이미지 크기는 5MB 이하만 가능합니다.",
        )

    # 기존 파일 삭제
    if user.profile_image_url:
        old_path = user.profile_image_url.lstrip("/")
        if os.path.exists(old_path):
            os.remove(old_path)

    # 새 파일 저장
    os.makedirs(PROFILE_UPLOAD_DIR, exist_ok=True)
    ext = os.path.splitext(file.filename or "unknown")[1].lower()
    if ext not in {".jpg", ".jpeg", ".png", ".webp", ".gif"}:
        ext = ".jpg"
    saved_filename = f"{uuid_mod.uuid4()}{ext}"
    file_path = os.path.join(PROFILE_UPLOAD_DIR, saved_filename)

    async with aiofiles.open(file_path, "wb") as f:
        await f.write(content)

    # DB 업데이트 — URL 경로로 저장
    user.profile_image_url = f"/uploads/profiles/{saved_filename}"
    await db.flush()

    orgs = [
        OrgBriefResponse(
            id=org.id, name=org.name, role=m.role,
            department=m.department, position=m.position,
        )
        for m, org in memberships
    ] if memberships else []

    return UserResponse(
        id=user.id,
        account_type=user.account_type,
        email=user.email,
        username=user.username,
        name=user.name,
        phone=user.phone,
        profile_image_url=user.profile_image_url,
        is_superadmin=user.is_superadmin,
        created_at=user.created_at,
        organizations=orgs,
    )


# ── 프로필 이미지 삭제 ────────────────────
@router.delete("/me/profile-image", response_model=UserResponse)
async def delete_profile_image(
    result=Depends(get_current_user_with_org),
    db: AsyncSession = Depends(get_db),
):
    """프로필 이미지 삭제 (이니셜 표시로 복귀)."""
    user, memberships = result

    if user.profile_image_url:
        old_path = user.profile_image_url.lstrip("/")
        if os.path.exists(old_path):
            os.remove(old_path)
        user.profile_image_url = None
        await db.flush()

    orgs = [
        OrgBriefResponse(
            id=org.id, name=org.name, role=m.role,
            department=m.department, position=m.position,
        )
        for m, org in memberships
    ] if memberships else []

    return UserResponse(
        id=user.id,
        account_type=user.account_type,
        email=user.email,
        username=user.username,
        name=user.name,
        phone=user.phone,
        profile_image_url=user.profile_image_url,
        is_superadmin=user.is_superadmin,
        created_at=user.created_at,
        organizations=orgs,
    )


# ── 아이디 찾기 ────────────────────────────
@router.post("/find-id", response_model=AccountRecoveryResponse)
async def find_id(
    payload: FindIdRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    이름 + ��메일로 사용자 조회 → 아이디를 이메일로 발송.
    사업자 유형 시 사업자등록번호 추가 검증.
    보안: 사용자 존재 여부와 무관하게 동일한 응답을 반환하여 계정 열거 방지.
    """
    # 이�� + 이메일로 사용자 검색
    query = select(User).where(User.email == payload.email, User.name == payload.name)
    result = await db.execute(query)
    user = result.scalar_one_or_none()

    # 사업자 유형: 사업자등록번호 추가 검증
    if user and payload.type == "business" and payload.bizNumber:
        biz_result = await db.execute(
            select(BusinessProfile).where(
                BusinessProfile.user_id == user.id,
                BusinessProfile.biz_number == payload.bizNumber,
            )
        )
        if biz_result.scalar_one_or_none() is None:
            user = None  # 사업자번호 불일치 시 미발견 처리

    # 사용자를 찾았으면 이메일 발송
    if user:
        send_found_username_email(to=user.email, name=user.name, username=user.username)

    # 보안: 결과와 무관하게 동일한 성공 메시지 반환 (계정 ���거 방지)
    return AccountRecoveryResponse(
        success=True,
        message="입력하신 이메일�� 아이디 정보를 발송했습니다. 메일함을 ��인해주세요.",
    )


# ── 비밀번호 찾기 (임시 비밀번호 발급) ────────
def _generate_temp_password(length: int = 12) -> str:
    """영문 대소문자 + 숫자 + 특수문자 혼합 임시 비밀번호 생성."""
    alphabet = string.ascii_letters + string.digits + "!@#$%^&*"
    while True:
        pw = ''.join(secrets.choice(alphabet) for _ in range(length))
        # 최소 1개씩 포함 보장
        if (any(c.islower() for c in pw)
                and any(c.isupper() for c in pw)
                and any(c.isdigit() for c in pw)
                and any(c in "!@#$%^&*" for c in pw)):
            return pw


@router.post("/find-pw", response_model=AccountRecoveryResponse)
async def find_password(
    payload: FindPasswordRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    아이디 + 이메일로 사용자 확인 → 임시 비밀번호 생성 후 이메�� 발송.
    DB 비밀번호를 임시 비밀번��� 해시로 교체.
    """
    # 아이�� + 이메일로 사용자 검색
    query = select(User).where(User.username == payload.userId, User.email == payload.email)
    result = await db.execute(query)
    user = result.scalar_one_or_none()

    # 사업자 유형: 사업자등록번호 추가 검증
    if user and payload.type == "business" and payload.bizNumber:
        biz_result = await db.execute(
            select(BusinessProfile).where(
                BusinessProfile.user_id == user.id,
                BusinessProfile.biz_number == payload.bizNumber,
            )
        )
        if biz_result.scalar_one_or_none() is None:
            user = None

    # 사용자를 찾았으면 임시 비밀번호 발급
    if user:
        temp_pw = _generate_temp_password()
        user.password_hash = hash_password(temp_pw)
        await db.flush()
        send_temp_password_email(to=user.email, name=user.name, temp_password=temp_pw)

    return AccountRecoveryResponse(
        success=True,
        message="입력하신 이메일로 임시 비밀번호를 발송했습니다. 메일함을 확인해주세요.",
    )
