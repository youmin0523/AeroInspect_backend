# =============================================
# app/services/push_notifications.py
# 역할: FCM/APNs 푸시 알림 발송 서비스 (실 발송 구현)
#       - PUSH_PROVIDER 설정으로 fcm|apns|noop 선택
#       - FCM: 서비스 계정 JSON → RS256 JWT → OAuth2 access_token (캐시) →
#              HTTP v1 messages:send 호출 (gcp_compute.py 패턴 미러링)
#       - APNs: .p8 키 → ES256 JWT (캐시) → HTTP/2 /3/device/{token} 호출
#
# 사용:
#   from app.services.push_notifications import push_service
#   await push_service.send_to_user(db, user_id, title="결함 발견", body="A-02 HIGH")
#
# NOTE: 실 크레덴셜이 없어 실제 전송 단(end-to-end)은 검증 불가.
#       코드는 correct-by-construction 으로 작성됨 — 실 FCM/APNs 크레덴셜로
#       최종 전송 검증 필요.
# =============================================

from __future__ import annotations

import asyncio
import base64
import json
import time
from dataclasses import dataclass
from typing import Optional
from uuid import UUID

import httpx
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.core.logging import get_logger
from app.models.device_token import DeviceToken


logger = get_logger("push_notifications")


# ── FCM HTTP v1 상수 ────────────────────────
FCM_TOKEN_URL = "https://oauth2.googleapis.com/token"
FCM_SCOPE = "https://www.googleapis.com/auth/firebase.messaging"
FCM_SEND_URL = "https://fcm.googleapis.com/v1/projects/{project_id}/messages:send"

# ── APNs 상수 ───────────────────────────────
APNS_HOST_PROD = "https://api.push.apple.com"
APNS_HOST_SANDBOX = "https://api.sandbox.push.apple.com"
APNS_JWT_TTL = 50 * 60  # 50분 (Apple 권장: 토큰 갱신 주기 20~60분)


@dataclass
class PushMessage:
    title: str
    body: str
    data: Optional[dict] = None  # 커스텀 key/value (예: defect_id, severity)


@dataclass
class _CachedToken:
    value: str
    expires_at: float  # epoch seconds


class PushNotificationService:
    """
    푸시 발송 조율자.
    provider는 config.PUSH_PROVIDER 로 결정 (기본 "noop").
    """

    def __init__(self):
        self._provider = getattr(settings, "PUSH_PROVIDER", "noop").lower()
        self._enabled = self._provider in ("fcm", "apns")

        # FCM 상태
        self._fcm_sa_cache: Optional[dict] = None
        self._fcm_token_cache: Optional[_CachedToken] = None
        self._fcm_token_lock = asyncio.Lock()

        # APNs 상태
        self._apns_jwt_cache: Optional[_CachedToken] = None
        self._apns_jwt_lock = asyncio.Lock()

        # 호출마다 새 TLS 커넥션을 맺지 않도록 httpx 클라이언트를 재사용.
        self._http: Optional[httpx.AsyncClient] = None       # FCM (HTTP/1.1)
        self._http2: Optional[httpx.AsyncClient] = None       # APNs (HTTP/2)
        self._http2_unavailable = False  # h2 미설치 등으로 HTTP/2 불가 플래그

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

    # ── httpx 클라이언트 (재사용) ──────────────
    def _get_http(self) -> httpx.AsyncClient:
        """FCM/일반 HTTP/1.1 클라이언트."""
        if self._http is None or self._http.is_closed:
            self._http = httpx.AsyncClient(timeout=20.0)
        return self._http

    def _get_http2(self) -> Optional[httpx.AsyncClient]:
        """APNs 용 HTTP/2 클라이언트. h2 미설치 시 None 반환."""
        if self._http2_unavailable:
            return None
        if self._http2 is None or self._http2.is_closed:
            try:
                # http2=True 는 내부적으로 h2 패키지를 요구한다.
                self._http2 = httpx.AsyncClient(http2=True, timeout=20.0)
            except Exception as e:
                # h2 미설치/초기화 실패 → 크래시 대신 비활성화하고 로깅.
                self._http2_unavailable = True
                logger.error(
                    "push.apns.http2_unavailable",
                    error=str(e),
                    hint="pip install 'httpx[http2]' (h2 패키지) 필요",
                )
                return None
        return self._http2

    async def aclose(self) -> None:
        """앱 종료 시 호출 — 커넥션 정리."""
        if self._http is not None and not self._http.is_closed:
            await self._http.aclose()
            self._http = None
        if self._http2 is not None and not self._http2.is_closed:
            await self._http2.aclose()
            self._http2 = None

    # ── 내부: FCM (HTTP v1) ────────────────────
    def _load_fcm_service_account(self) -> dict:
        """FCM_CREDENTIALS_JSON 파싱 (gcp_compute 패턴: base64-or-raw)."""
        if self._fcm_sa_cache is not None:
            return self._fcm_sa_cache

        raw = (getattr(settings, "FCM_CREDENTIALS_JSON", "") or "").strip()
        if not raw:
            raise RuntimeError("FCM_CREDENTIALS_JSON 환경변수가 비어있습니다.")

        # base64 인코딩이면 디코드, 아니면 JSON 원문으로 간주
        if not raw.startswith("{"):
            try:
                raw = base64.b64decode(raw).decode("utf-8")
            except Exception as e:
                raise RuntimeError(f"FCM_CREDENTIALS_JSON base64 디코딩 실패: {e}")

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"FCM_CREDENTIALS_JSON 파싱 실패: {e}")

        for key in ("client_email", "private_key"):
            if key not in data:
                raise RuntimeError(f"FCM 서비스 계정 JSON 에 '{key}' 누락")

        self._fcm_sa_cache = data
        return data

    async def _get_fcm_access_token(self) -> str:
        """RS256 JWT → OAuth2 access_token. 만료 30초 전까지 캐시.

        gcp_compute._get_access_token 의 JWT-bearer 흐름을 미러링.
        """
        now = time.time()
        cached = self._fcm_token_cache
        if cached and cached.expires_at > now + 30:
            return cached.value

        async with self._fcm_token_lock:
            cached = self._fcm_token_cache
            if cached and cached.expires_at > now + 30:
                return cached.value

            # jose 는 선택적 의존성 → lazy import (없으면 명확한 에러).
            try:
                from jose import jwt as jose_jwt
            except Exception as e:  # pragma: no cover - 환경 의존
                raise RuntimeError(
                    f"FCM JWT 서명을 위한 'python-jose' 가져오기 실패: {e}"
                )

            sa = self._load_fcm_service_account()
            iat = int(time.time())
            exp = iat + 3600
            assertion = jose_jwt.encode(
                {
                    "iss": sa["client_email"],
                    "scope": FCM_SCOPE,
                    "aud": FCM_TOKEN_URL,
                    "iat": iat,
                    "exp": exp,
                },
                sa["private_key"],
                algorithm="RS256",
            )

            resp = await self._get_http().post(
                FCM_TOKEN_URL,
                data={
                    "grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer",
                    "assertion": assertion,
                },
                timeout=15.0,
            )
            if resp.status_code != 200:
                # 시크릿 노출 방지: status 만 로깅.
                raise RuntimeError(f"FCM 토큰 발급 실패 (status={resp.status_code})")
            payload = resp.json()
            token = payload["access_token"]
            ttl = int(payload.get("expires_in", 3600))
            self._fcm_token_cache = _CachedToken(
                value=token, expires_at=time.time() + ttl
            )
            return token

    async def _send_fcm(self, token: str, message: PushMessage) -> bool:
        """FCM HTTP v1 messages:send 로 단일 디바이스 발송. 200 이면 True."""
        project_id = getattr(settings, "FCM_PROJECT_ID", "") or ""
        if not project_id:
            logger.error("push.fcm.no_project_id")
            return False

        try:
            access_token = await self._get_fcm_access_token()
        except Exception as e:
            logger.error("push.fcm.token_error", error=str(e))
            return False

        # data 페이로드는 FCM 규약상 모든 값이 문자열이어야 함.
        data_field = None
        if message.data:
            data_field = {k: str(v) for k, v in message.data.items()}

        msg: dict = {
            "token": token,
            "notification": {"title": message.title, "body": message.body},
        }
        if data_field:
            msg["data"] = data_field

        url = FCM_SEND_URL.format(project_id=project_id)
        try:
            resp = await self._get_http().post(
                url,
                headers={
                    "Authorization": f"Bearer {access_token}",
                    "Content-Type": "application/json",
                },
                json={"message": msg},
            )
        except Exception as e:
            logger.error("push.fcm.request_error", error=str(e))
            return False

        if resp.status_code == 200:
            logger.info("push.fcm.sent", token_preview=token[:12])
            return True

        # 시크릿 미포함 — status + 응답 본문(에러 코드 진단용)만 로깅.
        logger.warning(
            "push.fcm.failed",
            status=resp.status_code,
            token_preview=token[:12],
            detail=resp.text[:500],
        )
        return False

    # ── 내부: APNs (token-based, HTTP/2) ────────
    def _load_apns_key(self) -> str:
        """APNS_AUTH_KEY 를 PEM 문자열로 반환 (raw PEM 또는 base64)."""
        raw = (getattr(settings, "APNS_AUTH_KEY", "") or "").strip()
        if not raw:
            raise RuntimeError("APNS_AUTH_KEY 환경변수가 비어있습니다.")
        # raw PEM 이면 헤더(-----BEGIN)로 시작. 아니면 base64 로 간주해 디코드.
        if "BEGIN" not in raw:
            try:
                raw = base64.b64decode(raw).decode("utf-8").strip()
            except Exception as e:
                raise RuntimeError(f"APNS_AUTH_KEY base64 디코딩 실패: {e}")
        return raw

    async def _get_apns_jwt(self) -> str:
        """ES256 JWT 서명. ~50분 캐시."""
        now = time.time()
        cached = self._apns_jwt_cache
        if cached and cached.expires_at > now + 60:
            return cached.value

        async with self._apns_jwt_lock:
            cached = self._apns_jwt_cache
            if cached and cached.expires_at > now + 60:
                return cached.value

            try:
                from jose import jwt as jose_jwt
            except Exception as e:  # pragma: no cover - 환경 의존
                raise RuntimeError(
                    f"APNs JWT 서명을 위한 'python-jose' 가져오기 실패: {e}"
                )

            key_id = getattr(settings, "APNS_KEY_ID", "") or ""
            team_id = getattr(settings, "APNS_TEAM_ID", "") or ""
            if not key_id or not team_id:
                raise RuntimeError("APNS_KEY_ID / APNS_TEAM_ID 가 비어있습니다.")

            pem = self._load_apns_key()
            iat = int(time.time())
            token = jose_jwt.encode(
                {"iss": team_id, "iat": iat},
                pem,
                algorithm="ES256",
                headers={"kid": key_id, "alg": "ES256"},
            )
            self._apns_jwt_cache = _CachedToken(
                value=token, expires_at=time.time() + APNS_JWT_TTL
            )
            return token

    async def _send_apns(self, token: str, message: PushMessage) -> bool:
        """APNs token-based HTTP/2 발송. 200 이면 True."""
        topic = getattr(settings, "APNS_TOPIC", "") or ""
        if not topic:
            logger.error("push.apns.no_topic")
            return False

        client = self._get_http2()
        if client is None:
            # h2 미설치 → 크래시 대신 실패 반환 (위 _get_http2 에서 로깅 완료).
            return False

        try:
            jwt_token = await self._get_apns_jwt()
        except Exception as e:
            logger.error("push.apns.jwt_error", error=str(e))
            return False

        use_sandbox = bool(getattr(settings, "APNS_USE_SANDBOX", True))
        host = APNS_HOST_SANDBOX if use_sandbox else APNS_HOST_PROD
        url = f"{host}/3/device/{token}"

        aps: dict = {"alert": {"title": message.title, "body": message.body}}
        payload: dict = {"aps": aps}
        if message.data:
            # 커스텀 키는 최상위에 병합 (APNs 규약). aps 키와 충돌 방지.
            for k, v in message.data.items():
                if k != "aps":
                    payload[k] = v

        try:
            resp = await client.post(
                url,
                headers={
                    "authorization": f"bearer {jwt_token}",
                    "apns-topic": topic,
                    "apns-push-type": "alert",
                },
                json=payload,
            )
        except Exception as e:
            logger.error("push.apns.request_error", error=str(e))
            return False

        if resp.status_code == 200:
            logger.info("push.apns.sent", token_preview=token[:12])
            return True

        logger.warning(
            "push.apns.failed",
            status=resp.status_code,
            token_preview=token[:12],
            detail=resp.text[:500],
        )
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
