# =============================================
# app/services/push_notifications.py
# 역할: FCM/APNs 푸시 알림 발송 서비스 (스켈레톤)
#       - 현재는 Firebase 크레덴셜 미확보 → no-op 모드로 동작
#       - PUSH_PROVIDER 설정으로 fcm|apns|noop 선택
#       - 실 발송 연결 시 firebase-admin / apns2 라이브러리 추가 후
#         _send_fcm / _send_apns 메서드만 구현하면 됨
#
# 사용:
#   from app.services.push_notifications import push_service
#   await push_service.send_to_user(db, user_id, title="결함 발견", body="A-02 HIGH")
# =============================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
from uuid import UUID

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.core.logging import get_logger
from app.models.device_token import DeviceToken


logger = get_logger("push_notifications")


@dataclass
class PushMessage:
    title: str
    body: str
    data: Optional[dict] = None  # 커스텀 key/value (예: defect_id, severity)


class PushNotificationService:
    """
    푸시 발송 조율자.
    provider는 config.PUSH_PROVIDER 로 결정 (기본 "noop").
    """

    def __init__(self):
        self._provider = getattr(settings, "PUSH_PROVIDER", "noop").lower()
        self._enabled = self._provider in ("fcm", "apns")

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    @property
    def provider(self) -> str:
        return self._provider

    # ── 외부 API ────────────────────────────────
    async def send_to_user(
        self,
        db: AsyncSession,
        user_id: UUID,
        message: PushMessage,
    ) -> int:
        """
        사용자의 활성 디바이스 전부에 발송. 성공한 건수 반환.
        provider=noop 이면 로그만 찍고 0 반환.
        """
        result = await db.execute(
            select(DeviceToken).where(
                DeviceToken.user_id == user_id,
                DeviceToken.is_active.is_(True),
            )
        )
        tokens = result.scalars().all()

        if not tokens:
            logger.info("push.no_device", user_id=str(user_id))
            return 0

        if not self._enabled:
            logger.info(
                "push.noop",
                user_id=str(user_id),
                device_count=len(tokens),
                title=message.title,
            )
            return 0

        sent = 0
        for t in tokens:
            try:
                if t.platform == "fcm" or t.platform == "web":
                    ok = await self._send_fcm(t.token, message)
                elif t.platform == "apns":
                    ok = await self._send_apns(t.token, message)
                else:
                    logger.warning("push.unknown_platform", platform=t.platform)
                    ok = False
                if ok:
                    sent += 1
                else:
                    await self._mark_inactive(db, t.id)
            except Exception as e:
                logger.exception("push.send_failed", token_id=str(t.id), error=str(e))
                await self._mark_inactive(db, t.id)
        return sent

    # ── 내부: 프로바이더 어댑터 ────────────────
    async def _send_fcm(self, token: str, message: PushMessage) -> bool:
        """
        TODO: firebase-admin 연결 후 구현.
        현재는 스켈레톤 → always False (실제 발송 안 함).
        """
        logger.info("push.fcm.skeleton", token_preview=token[:12])
        return False

    async def _send_apns(self, token: str, message: PushMessage) -> bool:
        """
        TODO: apns2 또는 aioapns 연결 후 구현.
        """
        logger.info("push.apns.skeleton", token_preview=token[:12])
        return False

    async def _mark_inactive(self, db: AsyncSession, token_id: UUID) -> None:
        """발송 실패 누적되면 토큰 비활성화 (갈라지지 않은 데이터 유지)."""
        await db.execute(
            update(DeviceToken)
            .where(DeviceToken.id == token_id)
            .values(is_active=False)
        )


# ── 싱글톤 ──────────────────────────────────
push_service = PushNotificationService()


__all__ = ["PushNotificationService", "PushMessage", "push_service"]
