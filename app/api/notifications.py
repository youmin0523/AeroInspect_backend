# =============================================
# app/api/notifications.py
# 역할: 알림 REST API 엔드포인트
#       - GET    /notifications              → 목록 조회 (필터, 페이지네이션)
#       - GET    /notifications/unread-count  → 미읽음 수
#       - PATCH  /notifications/{id}/read     → 단건 읽음 처리
#       - PATCH  /notifications/read-all      → 전체 읽음 처리
#       - DELETE /notifications/{id}          → 단건 삭제
#       - DELETE /notifications               → 전체 삭제 (현재 사용자)
# =============================================

from uuid import UUID
from typing import Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import delete, select, func, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import get_db, get_current_user
from app.models.device_token import DeviceToken
from app.models.notification import Notification
from app.schemas.notification import (
    NotificationResponse,
    NotificationListResponse,
    NotificationUnreadCount,
)
from app.services.push_notifications import push_service

router = APIRouter()


@router.get("", response_model=NotificationListResponse)
async def list_notifications(
    category: Optional[str] = Query(None, description="카테고리 필터"),
    is_read: Optional[bool] = Query(None, description="읽음 상태 필터"),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    현재 사용자의 알림 목록 조회.
    카테고리·읽음 상태 필터 + 페이지네이션 지원.
    """
    base = select(Notification).where(Notification.user_id == current_user.id)
    count_q = select(func.count(Notification.id)).where(Notification.user_id == current_user.id)

    if category is not None:
        base = base.where(Notification.category == category)
        count_q = count_q.where(Notification.category == category)
    if is_read is not None:
        base = base.where(Notification.is_read == is_read)
        count_q = count_q.where(Notification.is_read == is_read)

    total = await db.scalar(count_q)

    result = await db.execute(
        base.order_by(Notification.created_at.desc())
        .limit(limit)
        .offset(offset)
    )
    items = [NotificationResponse.model_validate(r) for r in result.scalars().all()]

    return NotificationListResponse(items=items, total=total, limit=limit, offset=offset)


@router.get("/unread-count", response_model=NotificationUnreadCount)
async def get_unread_count(
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """현재 사용자의 미읽음 알림 수 반환"""
    count = await db.scalar(
        select(func.count(Notification.id))
        .where(Notification.user_id == current_user.id)
        .where(Notification.is_read == False)  # noqa: E712
    )
    return NotificationUnreadCount(count=count or 0)


@router.patch("/{notification_id}/read", response_model=NotificationResponse)
async def mark_as_read(
    notification_id: UUID,
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """단건 알림 읽음 처리"""
    result = await db.execute(
        select(Notification)
        .where(Notification.id == notification_id)
        .where(Notification.user_id == current_user.id)
    )
    notif = result.scalar_one_or_none()
    if notif is None:
        raise HTTPException(status_code=404, detail="알림을 찾을 수 없습니다.")

    notif.is_read = True
    await db.flush()
    return NotificationResponse.model_validate(notif)


@router.patch("/read-all")
async def mark_all_as_read(
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """현재 사용자의 미읽음 알림 전체 읽음 처리"""
    result = await db.execute(
        update(Notification)
        .where(Notification.user_id == current_user.id)
        .where(Notification.is_read == False)  # noqa: E712
        .values(is_read=True)
    )
    return {"updated": result.rowcount}


@router.delete("/{notification_id}", status_code=204)
async def delete_notification(
    notification_id: UUID,
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """단건 알림 삭제"""
    result = await db.execute(
        select(Notification)
        .where(Notification.id == notification_id)
        .where(Notification.user_id == current_user.id)
    )
    notif = result.scalar_one_or_none()
    if notif is None:
        raise HTTPException(status_code=404, detail="알림을 찾을 수 없습니다.")

    await db.delete(notif)
    await db.flush()


@router.delete("")
async def delete_all_notifications(
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """현재 사용자의 알림 전체 삭제"""
    result = await db.execute(
        delete(Notification).where(Notification.user_id == current_user.id)
    )
    return {"deleted": result.rowcount}


# ── 푸시 알림 (FCM/APNs) ────────────────────────────
class DeviceTokenRegister(BaseModel):
    """디바이스 토큰 등록/갱신 요청"""
    platform: Literal["fcm", "apns", "web"] = Field(..., description="fcm | apns | web")
    token: str = Field(..., min_length=10, max_length=512)
    device_label: Optional[str] = Field(None, max_length=100, description="사용자 식별용 라벨")


class DeviceTokenResponse(BaseModel):
    id: UUID
    platform: str
    device_label: Optional[str] = None
    is_active: bool


class PushTestRequest(BaseModel):
    """테스트 푸시 발송 요청 (provider=noop 시 로그만 남김)"""
    title: str = Field(..., min_length=1, max_length=100)
    body: str = Field(..., min_length=1, max_length=500)


@router.post("/tokens", response_model=DeviceTokenResponse, status_code=201)
async def register_device_token(
    payload: DeviceTokenRegister,
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    푸시 알림 디바이스 토큰 등록/갱신.
    같은 (user_id, token) 조합이 이미 있으면 is_active=True 로 되살림.
    """
    existing = await db.scalar(
        select(DeviceToken).where(
            DeviceToken.user_id == current_user.id,
            DeviceToken.token == payload.token,
        )
    )
    if existing is not None:
        existing.platform = payload.platform
        existing.device_label = payload.device_label
        existing.is_active = True
        await db.flush()
        return DeviceTokenResponse(
            id=existing.id,
            platform=existing.platform,
            device_label=existing.device_label,
            is_active=existing.is_active,
        )

    token_row = DeviceToken(
        user_id=current_user.id,
        platform=payload.platform,
        token=payload.token,
        device_label=payload.device_label,
        is_active=True,
    )
    db.add(token_row)
    await db.flush()

    return DeviceTokenResponse(
        id=token_row.id,
        platform=token_row.platform,
        device_label=token_row.device_label,
        is_active=token_row.is_active,
    )


@router.delete("/tokens/{token_id}", status_code=204)
async def deregister_device_token(
    token_id: UUID,
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """디바이스 토큰 제거 (로그아웃/앱 삭제 시). 소유자만 가능."""
    result = await db.execute(
        delete(DeviceToken).where(
            DeviceToken.id == token_id,
            DeviceToken.user_id == current_user.id,
        )
    )
    if result.rowcount == 0:
        raise HTTPException(status_code=404, detail="디바이스 토큰을 찾을 수 없습니다.")


@router.post("/push/test")
async def send_test_push(
    payload: PushTestRequest,
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    본인 등록 디바이스에 테스트 푸시 발송.
    PUSH_PROVIDER=noop (기본) 이면 실제 전송은 안 되고 로그만 남음.
    """
    from app.services.push_notifications import PushMessage

    sent = await push_service.send_to_user(
        db=db,
        user_id=current_user.id,
        message=PushMessage(title=payload.title, body=payload.body),
    )
    return {
        "provider": push_service.provider,
        "enabled": push_service.is_enabled,
        "devices_sent": sent,
    }
