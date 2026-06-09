# =============================================
# app/dependencies.py
# 역할: FastAPI 공유 의존성(Depends) 팩토리 모음
#       - DB 세션, 서비스 싱글톤을 라우터에 주입하는 함수 제공
#       - 모든 라우터에서 동일한 인스턴스를 재사용하도록 보장
# 사용: router 함수 파라미터에 Depends(get_db) 등으로 주입
# =============================================

from datetime import datetime, timezone
from hmac import compare_digest
from typing import AsyncGenerator, Optional
from uuid import UUID

from fastapi import Depends, Header, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db.session import async_session_factory
from app.core.jwt import decode_access_token
from app.services.camera import rgb_camera_service, thermal_camera_service
from app.services.recording import recording_service
from app.services.yolo_inference import yolo_service

_bearer_scheme = HTTPBearer(auto_error=False)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    비동기 DB 세션 의존성.
    요청마다 새 세션을 생성하고, 응답 후 자동으로 닫는다.
    """
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def get_current_user(
    cred: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
    db: AsyncSession = Depends(get_db),
):
    """
    Bearer 토큰에서 현재 사용자를 추출하는 인증 의존성.
    보호가 필요한 라우터에 Depends(get_current_user)로 주입.
    """
    if cred is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="인증 토큰이 필요합니다.")

    user_id = decode_access_token(cred.credentials)
    if user_id is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="유효하지 않거나 만료된 토큰입니다.")

    from app.models.user import User  # 순환 임포트 방지
    result = await db.execute(select(User).where(User.id == UUID(user_id)))
    user = result.scalar_one_or_none()
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="사용자를 찾을 수 없습니다.")
    return user


async def get_current_org_member(
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    x_organization_id: Optional[str] = Header(None),
):
    """
    현재 사용자의 활성 조직 멤버십을 반환.
    다중 조직 시 X-Organization-Id 헤더로 선택, 없으면 가장 최근 활성 조직.
    미소속 또는 계약 만료 시 403 반환.
    반환: (user, org_member, organization) 튜플
    """
    from app.models.organization import Organization, OrganizationMember

    query = (
        select(OrganizationMember, Organization)
        .join(Organization, Organization.id == OrganizationMember.organization_id)
        .where(OrganizationMember.user_id == current_user.id)
        .where(OrganizationMember.status == "active")
        .where(
            (OrganizationMember.ended_at.is_(None))
            | (OrganizationMember.ended_at > datetime.now(timezone.utc))
        )
    )
    if x_organization_id:
        query = query.where(Organization.id == UUID(x_organization_id))
    else:
        query = query.order_by(OrganizationMember.joined_at.desc())

    result = await db.execute(query)
    row = result.first()
    if not row:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="조직에 소속되지 않은 사용자입니다. 관리자에게 문의하세요.",
        )
    member, org = row
    return current_user, member, org


async def get_current_user_with_org(
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    현재 사용자 + 조직 정보 (nullable) 반환.
    /auth/me 등 조직 미소속도 허용하는 엔드포인트용.
    """
    from app.models.organization import Organization, OrganizationMember

    result = await db.execute(
        select(OrganizationMember, Organization)
        .join(Organization, Organization.id == OrganizationMember.organization_id)
        .where(OrganizationMember.user_id == current_user.id)
        .where(OrganizationMember.status == "active")
        .where(
            (OrganizationMember.ended_at.is_(None))
            | (OrganizationMember.ended_at > datetime.now(timezone.utc))
        )
        .order_by(OrganizationMember.joined_at.desc())
    )
    rows = result.all()
    if not rows:
        return current_user, []
    return current_user, [(m, o) for m, o in rows]


def require_role(*allowed_roles: str):
    """
    특정 조직 역할이 필요한 엔드포인트용 의존성 팩토리.
    사용: Depends(require_role("owner", "admin"))
    """
    async def _check(org_tuple=Depends(get_current_org_member)):
        user, member, org = org_tuple
        if member.role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"이 작업은 {', '.join(allowed_roles)} 권한이 필요합니다.",
            )
        return user, member, org
    return _check


async def require_superadmin(
    current_user=Depends(get_current_user),
):
    """플랫폼 슈퍼어드민 전용 의존성. is_superadmin=False 시 403."""
    if not current_user.is_superadmin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="플랫폼 관리자 권한이 필요합니다.",
        )
    return current_user


async def require_admin_or_superadmin(
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    x_organization_id: Optional[str] = Header(None),
):
    """플랫폼 슈퍼어드민 OR 활성 조직의 owner/admin 일 때만 통과.

    super_admin 은 조직 미소속이어도 통과 (분기 순서 중요).
    그 외에는 활성 조직 멤버십에서 role ∈ {owner, admin} 확인.
    """
    if current_user.is_superadmin:
        return current_user

    from app.models.organization import OrganizationMember

    query = (
        select(OrganizationMember)
        .where(OrganizationMember.user_id == current_user.id)
        .where(OrganizationMember.status == "active")
        .where(
            (OrganizationMember.ended_at.is_(None))
            | (OrganizationMember.ended_at > datetime.now(timezone.utc))
        )
    )
    if x_organization_id:
        query = query.where(OrganizationMember.organization_id == UUID(x_organization_id))
    else:
        query = query.order_by(OrganizationMember.joined_at.desc())

    result = await db.execute(query)
    member = result.scalars().first()
    if member is None or member.role not in ("owner", "admin"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="조직 관리자 권한이 필요합니다.",
        )
    return current_user


async def verify_ai_webhook(
    x_ai_webhook_secret: Optional[str] = Header(None, alias="X-AI-Webhook-Secret"),
):
    """
    AI 추론 서버 → 백엔드 콜백(/api/v1/ai/*) 인증 의존성.
    settings.AI_WEBHOOK_SECRET 와 X-AI-Webhook-Secret 헤더를 timing-safe 비교.
    시크릿이 미설정(빈 문자열)이거나 헤더가 없거나 불일치 시 401.
    """
    expected = settings.AI_WEBHOOK_SECRET
    if not expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="AI 웹훅 시크릿이 서버에 설정되지 않았습니다.",
        )
    if not x_ai_webhook_secret or not compare_digest(x_ai_webhook_secret, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="AI 웹훅 인증에 실패했습니다.",
        )


async def verify_ai_webhook_or_user(
    x_ai_webhook_secret: Optional[str] = Header(None, alias="X-AI-Webhook-Secret"),
    cred: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
    db: AsyncSession = Depends(get_db),
):
    """
    내부 워커(X-AI-Webhook-Secret) 또는 인증된 사용자(Bearer JWT) 둘 중 하나면 통과.
    /detect 처럼 외부 AI 서버 콜백과 웹 UI 사용자 호출이 모두 들어오는 엔드포인트용.

    반환:
      - webhook 인증 시: None
      - user 인증 시: User 모델
    """
    expected = settings.AI_WEBHOOK_SECRET
    if x_ai_webhook_secret and expected and compare_digest(x_ai_webhook_secret, expected):
        return None  # webhook 측 통과 (사용자 컨텍스트 없음)

    if cred is not None:
        user_id = decode_access_token(cred.credentials)
        if user_id is not None:
            from app.models.user import User  # 순환 임포트 방지
            result = await db.execute(select(User).where(User.id == UUID(user_id)))
            user = result.scalar_one_or_none()
            if user is not None:
                return user

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Bearer 토큰 또는 X-AI-Webhook-Secret 헤더가 필요합니다.",
    )


def get_ws_manager():
    """WebSocket 연결 매니저 싱글톤 반환.

    lifespan 에서 RedisConnectionManager 로 교체될 수 있어 매번 모듈 어트리뷰트를
    lazy 참조한다. top-level `from app.core.ws_manager import ws_manager` 로
    캡처하면 교체 후에도 옛 인스턴스를 가리킨다.
    """
    from app.core import ws_manager as wsmod
    return wsmod.get_active_manager()


def get_rgb_camera():
    """RGB 카메라 서비스 싱글톤 반환"""
    return rgb_camera_service


def get_thermal_camera():
    """열화상 카메라 서비스 싱글톤 반환"""
    return thermal_camera_service


def get_recording_service():
    """녹화 서비스 싱글톤 반환"""
    return recording_service


def get_yolo_service():
    """YOLOv8 추론 서비스 싱글톤 반환"""
    return yolo_service
