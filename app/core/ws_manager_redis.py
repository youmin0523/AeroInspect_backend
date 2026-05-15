# =============================================
# app/core/ws_manager_redis.py
# 역할: 멀티 워커/다중 서버 환경용 Redis pub/sub 기반 WebSocket 매니저
#       - 기존 ConnectionManager(메모리 전용) 를 상속
#       - broadcast() 호출 시 Redis 채널에 publish
#       - 각 워커는 구독자 태스크를 돌려서 들어오는 메시지를 로컬 연결로 다시 뿌림
#       - WS_BACKEND=redis 일 때 main.py lifespan 이 start()/stop() 호출
#
# 구동 요구:
#   - pip install redis
#   - REDIS_URL=redis://localhost:6379/0 환경변수
#
# 설계 노트:
#   - 원본 broadcast 로직은 그대로 두고 "외부 채널을 타고 자기 자신에게도 되돌아오는"
#     경로로 로컬 연결에 전달. 덕분에 기존 라우터 코드는 수정 불필요.
#   - 단일 워커로 기동해도 (publish + subscribe 자기 자신) 정상 동작.
# =============================================

from __future__ import annotations

import asyncio
import json
from typing import Iterable, Optional

from app.core.logging import get_logger
from app.core.ws_manager import ConnectionManager


logger = get_logger("ws.redis")


class RedisConnectionManager(ConnectionManager):
    """
    pub/sub 백엔드 포함 WebSocket 매니저.
    브로드캐스트가 오면 → Redis publish → 모든 워커 수신 → 각자 로컬 연결에 전달.
    """

    def __init__(
        self,
        redis_url: str,
        channels: Iterable[str] = ("defects", "telemetry", "thermal", "camera", "stream"),
        patterns: Iterable[str] = ("notifications:*", "user:*", "chat:*"),
    ):
        super().__init__()
        self._redis_url = redis_url
        self._channels = list(channels)
        # 동적 채널 prefix 패턴 (notifications:{uid}, user:{uid}, chat:{uuid}) —
        # 각 워커가 사용자 채널 메시지를 모두 수신할 수 있도록 psubscribe 로 구독.
        self._patterns = list(patterns)
        self._redis = None            # redis.asyncio.Redis
        self._pubsub = None           # PubSub 객체
        self._subscriber_task: Optional[asyncio.Task] = None
        self._running = False

    # ── 생명주기 ─────────────────────────────────
    async def start(self) -> None:
        """구독자 태스크 기동. main.py lifespan 에서 호출."""
        try:
            import redis.asyncio as redis  # lazy import — 라이브러리 없으면 에러를 시점에 알림
        except ImportError as e:
            raise RuntimeError(
                "WS_BACKEND=redis 이지만 `redis` 패키지 미설치. `pip install redis` 필요."
            ) from e

        self._redis = redis.from_url(self._redis_url, decode_responses=True)
        self._pubsub = self._redis.pubsub()
        if self._channels:
            await self._pubsub.subscribe(*self._channels)
        if self._patterns:
            await self._pubsub.psubscribe(*self._patterns)

        self._running = True
        self._subscriber_task = asyncio.create_task(
            self._subscriber_loop(), name="ws_redis_subscriber"
        )
        logger.info(
            "ws.redis.started",
            channels=self._channels,
            patterns=self._patterns,
            url=self._redis_url,
        )

    async def stop(self) -> None:
        self._running = False
        if self._subscriber_task is not None:
            self._subscriber_task.cancel()
            try:
                await self._subscriber_task
            except (asyncio.CancelledError, Exception):
                pass
            self._subscriber_task = None
        if self._pubsub is not None:
            try:
                await self._pubsub.unsubscribe()
            except Exception:
                pass
            try:
                if self._patterns:
                    await self._pubsub.punsubscribe()
            except Exception:
                pass
            await self._pubsub.aclose()
            self._pubsub = None
        if self._redis is not None:
            await self._redis.aclose()
            self._redis = None
        logger.info("ws.redis.stopped")

    # ── 브로드캐스트 override ────────────────────
    async def broadcast(self, channel: str, payload: dict) -> None:
        """
        로컬 직접 전송 대신 Redis publish. 구독자 루프에서 되돌아와 로컬 연결로 간다.
        Redis 미기동 상태라면 그냥 로컬 전송으로 폴백 (개발 편의).
        """
        if self._redis is None:
            await super().broadcast(channel, payload)
            return

        message = json.dumps(payload, ensure_ascii=False, default=str)
        try:
            await self._redis.publish(channel, message)
        except Exception as e:
            logger.warning("ws.redis.publish_failed", error=str(e), channel=channel)
            # 폴백: 연결 로컬로라도
            await super().broadcast(channel, payload)

    # ── 내부: 구독 루프 ──────────────────────────
    async def _subscriber_loop(self) -> None:
        """Redis 에서 받은 메시지(subscribe + psubscribe 양쪽)를 로컬 연결 풀에 분배."""
        assert self._pubsub is not None
        while self._running:
            try:
                # ignore_subscribe_messages=True 는 subscribe 확인 메시지만 무시.
                # pmessage 는 정상적으로 반환되므로 type 필드로 구분 처리.
                msg = await self._pubsub.get_message(
                    ignore_subscribe_messages=True, timeout=1.0
                )
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning("ws.redis.subscriber_error", error=str(e))
                await asyncio.sleep(0.5)
                continue

            if msg is None:
                continue
            mtype = msg.get("type") if isinstance(msg, dict) else None
            if mtype not in ("message", "pmessage"):
                continue
            try:
                channel = msg["channel"]
                payload = json.loads(msg["data"])
            except (KeyError, json.JSONDecodeError) as e:
                logger.warning("ws.redis.bad_message", error=str(e))
                continue

            # 부모의 broadcast 는 로컬 연결에만 전송
            await super().broadcast(channel, payload)


def create_ws_manager(
    backend: str,
    redis_url: Optional[str] = None,
) -> ConnectionManager:
    """
    설정값 기반 WS 매니저 팩토리.

    backend: "memory" (기본) | "redis"
    redis_url: backend="redis" 일 때 필수
    """
    if backend == "redis":
        if not redis_url:
            raise ValueError("WS_BACKEND=redis 인데 REDIS_URL 이 없습니다.")
        return RedisConnectionManager(redis_url=redis_url)
    if backend == "memory":
        return ConnectionManager()
    raise ValueError(f"알 수 없는 WS_BACKEND: {backend!r} (memory|redis)")


__all__ = ["RedisConnectionManager", "create_ws_manager"]
