# =============================================
# app/api/oauth.py
# 역할: SNS 소셜 로그인 OAuth 콜백 처리
#       - POST /oauth/google  → Google 인가 코드 교환 + 사용자 조회/생성
#       - POST /oauth/kakao   → Kakao 인가 코드 교환
#       - POST /oauth/naver   → Naver 인가 코드 교환
#       공통 플로우:
#         1) 프론트에서 받은 authorization code를 provider에 전달해 access_token 획득
#         2) access_token으로 사용자 프로필(이메일, 이름) 조회
#         3) DB에 해당 oauth_id 사용자 존재 → 로그인 / 미존재 → 자동 회원가입
#         4) JWT 발급 후 TokenResponse 반환
# =============================================

import uuid

import httpx
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.core.jwt import create_access_token, create_refresh_token
from app.core.logging import get_logger
from app.dependencies import get_db
from app.models.user import User
from app.schemas.user import OAuthCallbackRequest, OrgBriefResponse, TokenResponse, UserResponse

router = APIRouter()
logger = get_logger(__name__)


# ── 공통 헬퍼 ────────────────────────────────
async def _get_user_orgs(db: AsyncSession, user_id):
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


def _build_user_response(user: User, orgs=None) -> UserResponse:
    return UserResponse(
        id=user.id,
        account_type=user.account_type,
        email=user.email,
        username=user.username,
        name=user.name,
        phone=user.phone,
        is_superadmin=user.is_superadmin,
        created_at=user.created_at,
        organizations=orgs or [],
    )


def _email_link_forbidden() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail=(
            "이미 가입된 이메일입니다. 소셜 계정의 이메일이 검증되지 않아 자동 연결할 수 없습니다. "
            "비밀번호로 로그인한 뒤 소셜 계정을 연결해 주세요."
        ),
    )


async def _find_or_create_oauth_user(
    db: AsyncSession,
    provider: str,
    oauth_id: str,
    email: str,
    name: str,
    email_verified: bool = False,
) -> User:
    """
    OAuth 사용자를 조회하거나 신규 생성.
    - oauth_id 기준으로 먼저 검색
    - 없으면 email 기준 검색 (기존 일반 가입 계정에 OAuth 연결)
    - 둘 다 없으면 새 계정 생성

    ⚠️ 보안: 기존 로컬 계정에 이메일로 자동 연결하는 것은 provider 가 그 이메일을
    '검증됨'으로 보고한 경우(email_verified=True)에만 허용한다. 미검증 이메일 연결을
    허용하면 공격자가 피해자 이메일로 소셜 계정을 만들어 피해자 계정을 탈취할 수 있다.
    """
    # 1) oauth_id로 조회 (이미 연결된 소셜 계정 — 가장 신뢰 가능)
    result = await db.execute(
        select(User).where(User.oauth_provider == provider, User.oauth_id == oauth_id)
    )
    user = result.scalar_one_or_none()
    if user:
        return user

    # 2) 동일 이메일 계정 존재 시 OAuth 연결 (대소문자 무시) — 검증된 이메일에 한함
    if email:
        result = await db.execute(
            select(User).where(func.lower(User.email) == func.lower(email))
        )
        user = result.scalar_one_or_none()
        if user:
            if not email_verified:
                raise _email_link_forbidden()
            user.oauth_provider = provider
            user.oauth_id = oauth_id
            await db.flush()
            return user

    # 3) 신규 생성 (OAuth 전용: 비밀번호 없음)
    try:
        user = User(
            account_type="personal",
            email=email,
            username=f"{provider}_{uuid.uuid4().hex[:8]}",
            password_hash=None,
            name=name or (email.split("@")[0] if email else provider),
            phone="000-0000-0000",
            oauth_provider=provider,
            oauth_id=oauth_id,
        )
        db.add(user)
        await db.flush()
        return user
    except IntegrityError:
        # 동시 가입 경합 등으로 unique 제약 충돌 → 기존 계정 재조회.
        # 이 경로의 연결도 검증된 이메일에 한해 허용(2)와 동일 정책).
        await db.rollback()
        if email:
            result = await db.execute(
                select(User).where(func.lower(User.email) == func.lower(email))
            )
            user = result.scalar_one_or_none()
            if user:
                if not email_verified:
                    raise _email_link_forbidden()
                user.oauth_provider = provider
                user.oauth_id = oauth_id
                await db.flush()
                return user
        raise HTTPException(status_code=409, detail="이메일 충돌이 발생했습니다. 다시 시도해 주세요.")


# ── Google OAuth ─────────────────────────────
@router.post("/google", response_model=TokenResponse)
async def google_callback(
    payload: OAuthCallbackRequest,
    db: AsyncSession = Depends(get_db),
):
    """Google authorization code → 토큰 교환 → 사용자 정보 → JWT 발급"""
    # 1) code → access_token
    async with httpx.AsyncClient() as client:
        token_resp = await client.post(
            "https://oauth2.googleapis.com/token",
            data={
                "code": payload.code,
                "client_id": settings.GOOGLE_CLIENT_ID,
                "client_secret": settings.GOOGLE_CLIENT_SECRET,
                "redirect_uri": payload.redirect_uri,
                "grant_type": "authorization_code",
            },
        )
    if token_resp.status_code != 200:
        error_detail = token_resp.json() if token_resp.headers.get("content-type", "").startswith("application/json") else token_resp.text
        # 시크릿은 로그에 남기지 않고 상태/비민감 컨텍스트만 기록.
        logger.warning(
            "oauth.google.token_exchange_failed",
            status_code=token_resp.status_code,
            error_detail=error_detail,
            redirect_uri=payload.redirect_uri,
        )
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Google 토큰 교환 실패: {error_detail}")
    token_data = token_resp.json()
    access_token = token_data.get("access_token")

    # 2) access_token → 사용자 프로필
    async with httpx.AsyncClient() as client:
        profile_resp = await client.get(
            "https://www.googleapis.com/oauth2/v2/userinfo",
            headers={"Authorization": f"Bearer {access_token}"},
        )
    if profile_resp.status_code != 200:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Google 사용자 정보 조회 실패")
    profile = profile_resp.json()

    # 3) DB 조회/생성 — Google 은 verified_email/email_verified 로 검증 여부 보고
    google_verified = bool(profile.get("verified_email") or profile.get("email_verified"))
    user = await _find_or_create_oauth_user(
        db,
        provider="google",
        oauth_id=profile["id"],
        email=profile.get("email", ""),
        name=profile.get("name", ""),
        email_verified=google_verified,
    )

    # 4) JWT 발급
    token = create_access_token(user.id)
    refresh = create_refresh_token(user.id)
    orgs = await _get_user_orgs(db, user.id)
    return TokenResponse(
        access_token=token, refresh_token=refresh,
        user=_build_user_response(user, orgs),
    )


# ── Kakao OAuth ──────────────────────────────
@router.post("/kakao", response_model=TokenResponse)
async def kakao_callback(
    payload: OAuthCallbackRequest,
    db: AsyncSession = Depends(get_db),
):
    """Kakao authorization code → 토큰 교환 → 사용자 정보 → JWT 발급"""
    # 1) code → access_token
    async with httpx.AsyncClient() as client:
        token_resp = await client.post(
            "https://kauth.kakao.com/oauth/token",
            data={
                "grant_type": "authorization_code",
                "client_id": settings.KAKAO_CLIENT_ID,
                "client_secret": settings.KAKAO_CLIENT_SECRET,
                "redirect_uri": payload.redirect_uri,
                "code": payload.code,
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
    if token_resp.status_code != 200:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Kakao 토큰 교환 실패")
    access_token = token_resp.json().get("access_token")

    # 2) access_token → 사용자 프로필
    async with httpx.AsyncClient() as client:
        profile_resp = await client.get(
            "https://kapi.kakao.com/v2/user/me",
            headers={"Authorization": f"Bearer {access_token}"},
        )
    if profile_resp.status_code != 200:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Kakao 사용자 정보 조회 실패")
    profile = profile_resp.json()
    kakao_account = profile.get("kakao_account", {})

    # 3) DB 조회/생성 — Kakao 는 kakao_account.is_email_verified 로 검증 여부 보고
    kakao_verified = bool(kakao_account.get("is_email_verified"))
    user = await _find_or_create_oauth_user(
        db,
        provider="kakao",
        oauth_id=str(profile["id"]),
        email=kakao_account.get("email", ""),
        name=kakao_account.get("profile", {}).get("nickname", ""),
        email_verified=kakao_verified,
    )

    # 4) JWT 발급
    token = create_access_token(user.id)
    refresh = create_refresh_token(user.id)
    orgs = await _get_user_orgs(db, user.id)
    return TokenResponse(
        access_token=token, refresh_token=refresh,
        user=_build_user_response(user, orgs),
    )


# ── Naver OAuth ──────────────────────────────
@router.post("/naver", response_model=TokenResponse)
async def naver_callback(
    payload: OAuthCallbackRequest,
    db: AsyncSession = Depends(get_db),
):
    """Naver authorization code → 토큰 교환 → 사용자 정보 → JWT 발급"""
    # 1) code → access_token
    async with httpx.AsyncClient() as client:
        token_resp = await client.post(
            "https://nid.naver.com/oauth2.0/token",
            data={
                "grant_type": "authorization_code",
                "client_id": settings.NAVER_CLIENT_ID,
                "client_secret": settings.NAVER_CLIENT_SECRET,
                "redirect_uri": payload.redirect_uri,
                "code": payload.code,
            },
        )
    if token_resp.status_code != 200:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Naver 토큰 교환 실패")
    access_token = token_resp.json().get("access_token")

    # 2) access_token → 사용자 프로필
    async with httpx.AsyncClient() as client:
        profile_resp = await client.get(
            "https://openapi.naver.com/v1/nid/me",
            headers={"Authorization": f"Bearer {access_token}"},
        )
    if profile_resp.status_code != 200:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Naver 사용자 정보 조회 실패")
    response_data = profile_resp.json().get("response", {})

    # 3) DB 조회/생성 — Naver 는 가입 시 검증된 이메일만 /nid/me 로 반환(별도 플래그 없음).
    # 따라서 이메일이 존재하면 검증된 것으로 간주한다.
    naver_verified = bool(response_data.get("email"))
    user = await _find_or_create_oauth_user(
        db,
        provider="naver",
        oauth_id=response_data["id"],
        email=response_data.get("email", ""),
        name=response_data.get("name", ""),
        email_verified=naver_verified,
    )

    # 4) JWT 발급
    token = create_access_token(user.id)
    refresh = create_refresh_token(user.id)
    orgs = await _get_user_orgs(db, user.id)
    return TokenResponse(
        access_token=token, refresh_token=refresh,
        user=_build_user_response(user, orgs),
    )
