# =============================================
# app/services/notification_service.py
# 역할: 알림 생성 및 실시간 전송 서비스
#       - DB 저장 + WebSocket 푸시를 한 번에 처리
#       - 기존 API 핸들러(defects, floorplan, report 등)에서 호출
#       - 다수 사용자 대상 일괄 알림 지원
# =============================================

import asyncio
from uuid import UUID
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.notification import Notification
from app.schemas.notification import NotificationResponse
from app.core.ws_manager import ConnectionManager


class NotificationService:
    """
    알림 서비스.
    DB 저장 후 WebSocket `notifications:{user_id}` 채널로 즉시 푸시.
    """

    def __init__(self, ws_manager: ConnectionManager):
        self._ws = ws_manager

    async def create(
        self,
        db: AsyncSession,
        user_id: UUID,
        category: str,
        title: str,
        message: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> Notification:
        """
        알림 1건 생성.
        1) DB 에 Notification 레코드 삽입
        2) WebSocket `notifications:{user_id}` 채널로 브로드캐스트
        """
        notif = Notification(
            user_id=user_id,
            category=category,
            title=title,
            message=message,
            metadata_=metadata,
        )
        db.add(notif)
        await db.flush()

        # WebSocket 실시간 푸시
        response = NotificationResponse.model_validate(notif)
        await self._ws.broadcast(f"notifications:{user_id}", {
            "type": "notification.new",
            "data": response.model_dump(mode="json"),
        })

        return notif

    async def create_for_many(
        self,
        db: AsyncSession,
        user_ids: list[UUID],
        category: str,
        title: str,
        message: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> list[Notification]:
        """동일 알림을 다수 사용자에게 전송.

        - DB INSERT 는 한 번의 flush 로 일괄 처리 (사용자 수만큼 round-trip 하지 않음).
        - WebSocket 푸시는 asyncio.gather 로 동시 fan-out (순차 await 제거).
        반환되는 Notification 목록과 브로드캐스트 payload 는 create() 와 동일하다.
        """
        if not user_ids:
            return []

        # 1) 모든 Notification 을 추가한 뒤 단일 flush — id/created_at 등 DB 기본값 채움
        notifs = [
            Notification(
                user_id=uid,
                category=category,
                title=title,
                message=message,
                metadata_=metadata,
            )
            for uid in user_ids
        ]
        db.add_all(notifs)
        await db.flush()

        # 2) WebSocket 실시간 푸시 — 동시 fan-out
        async def _broadcast(notif: Notification) -> None:
            response = NotificationResponse.model_validate(notif)
            await self._ws.broadcast(f"notifications:{notif.user_id}", {
                "type": "notification.new",
                "data": response.model_dump(mode="json"),
            })

        await asyncio.gather(*(_broadcast(n) for n in notifs))

        return notifs


# ── 모듈 레벨 싱글톤 ─────────────────────────
from app.core.ws_manager import ws_manager  # noqa: E402
notification_service = NotificationService(ws_manager)
