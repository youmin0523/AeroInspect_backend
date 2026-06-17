# =============================================
# app/api/websocket.py
# 역할: WebSocket 실시간 이벤트 엔드포인트
#       - WS /ws?channels={a,b,c}&token={jwt} → 다중 채널 구독
#       - 채널: defects / telemetry / thermal / camera (공개)
#                / chat:{uuid} / user:{uuid} / notifications:{uuid} (본인 한정)
#       - 본인 채널은 JWT 토큰 sub 와 일치해야만 구독 허용
#       - WS_HEARTBEAT_INTERVAL 초마다 ping 전송으로 죽은 연결 감지
# =============================================

import asyncio
from uuid import UUID

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from sqlalchemy import select

from app.config import settings
from app.core.jwt import decode_access_token
from app.core.ws_manager import ConnectionManager
from app.db.session import async_session_factory
from app.dependencies import get_ws_manager
from app.models.conversation_member import ConversationMember

router = APIRouter()

STATIC_CHANNELS = {"defects", "telemetry", "thermal", "camera"}

# 동적 채널 prefix — 사용자/대화방/알림별 개인 채널
# notifications: / user: 는 본인 user_id 와 일치하는 토큰을 가진 경우에만 구독 허용
# chat: 는 토큰 보유 + 대화방 멤버십(DB)까지 검증해야 구독 허용 (구독 IDOR 방지)
_DYNAMIC_CHANNEL_PREFIXES = ("chat:", "user:", "notifications:")
_USER_SCOPED_PREFIXES = ("user:", "notifications:")
_AUTH_REQUIRED_PREFIXES = ("chat:", "user:", "notifications:")


def _is_valid_channel(channel: str) -> bool:
    """드론 모니터링 고정 채널 또는 동적 채널(chat/user/notifications) 허용."""
    if channel in STATIC_CHANNELS:
        return True
    return any(channel.startswith(p) for p in _DYNAMIC_CHANNEL_PREFIXES)


def _channel_uid(channel: str) -> str | None:
    """동적 채널의 식별자 부분만 추출. 'user:abc' → 'abc'."""
    for p in _USER_SCOPED_PREFIXES:
        if channel.startswith(p):
            return channel[len(p):]
    return None


async def _is_conversation_member(conversation_id: str, user_id: str | None) -> bool:
    """token 사용자가 해당 대화방의 멤버인지 DB 로 확인 (chat 구독 IDOR 방지).

    멤버십은 발행 측에서만 검증됐었어서, 비멤버가 대화방 UUID 로 구독하면 메시지를
    실시간 수신할 수 있었다. 구독 시점에 짧은 세션으로 멤버십을 직접 검증한다.
    """
    if not user_id:
        return False
    try:
        cid = UUID(conversation_id)
        uid = UUID(user_id)
    except (ValueError, TypeError, AttributeError):
        return False
    async with async_session_factory() as session:
        found = await session.scalar(
            select(ConversationMember.id).where(
                ConversationMember.conversation_id == cid,
                ConversationMember.user_id == uid,
            )
        )
    return found is not None


def _authorize_channel(channel: str, token_sub: str | None) -> bool:
    """채널 구독 권한 검증.

    - 정적 채널(defects/telemetry/camera/thermal): 누구나 OK
    - chat:*  : 토큰만 있으면 OK (멤버십은 발행 측에서 검증)
    - user:{uid} / notifications:{uid}: 토큰 sub 와 uid 일치 필수
    """
    if channel in STATIC_CHANNELS:
        return True
    if not any(channel.startswith(p) for p in _AUTH_REQUIRED_PREFIXES):
        return False
    if token_sub is None:
        return False
    uid = _channel_uid(channel)
    if uid is None:
        # chat: 채널 — 토큰만 있으면 통과
        return True
    return uid == token_sub


@router.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    channel: str = "defects",
    channels: str | None = None,
    token: str | None = None,
    manager: ConnectionManager = Depends(get_ws_manager),
):
    """
    WebSocket 구독 엔드포인트 (단일/다중 채널 + JWT 인증 지원).

    연결 방법:
        ws://.../ws?channel=defects                                       (단일)
        ws://.../ws?channels=defects,telemetry,camera,thermal              (다중)
        ws://.../ws?channels=defects,notifications:{uid}&token={jwt}       (본인 채널)
        ws://.../ws?channel=user:{uid}&token={jwt}
        ws://.../ws?channel=chat:{conversation_id}&token={jwt}

    채널 인증 규칙:
        - defects / telemetry / camera / thermal : 공개 (token 불필요)
        - notifications:{uid}, user:{uid}        : token sub 와 uid 일치 필수
        - chat:{conversation_id}                  : token 보유 필수 (멤버십은 발행 측 검증)

    인증 실패한 채널은 묵시적으로 제거되고, 남은 유효 채널이 0개면
    'defects' 채널로 폴백.
    """
    # 토큰 디코드 (있으면 sub=user_id, 없거나 무효면 None)
    token_sub = decode_access_token(token) if token else None

    # 다중 채널 파라미터 (있으면 우선), 없으면 단일 channel 사용
    raw = channels if channels else channel
    requested = [c.strip() for c in raw.split(",") if c.strip()]

    # 형식 + 권한 검증 둘 다 통과한 것만 채택
    authorized: list[str] = []
    rejected: list[str] = []
    for c in requested:
        if not _is_valid_channel(c):
            rejected.append(c)
            continue
        if not _authorize_channel(c, token_sub):
            rejected.append(c)
            continue
        # chat: 채널은 토큰 보유뿐 아니라 대화방 멤버십까지 확인 (구독 IDOR 방지)
        if c.startswith("chat:"):
            conversation_id = c[len("chat:"):]
            if not await _is_conversation_member(conversation_id, token_sub):
                rejected.append(c)
                continue
        authorized.append(c)

    # 폴백: 유효 채널 0개면 공개 채널 'defects' 부여
    if not authorized:
        authorized = ["defects"]

    # 중복 제거 (순서 보존)
    seen: set[str] = set()
    subscribed: list[str] = []
    for c in authorized:
        if c not in seen:
            seen.add(c)
            subscribed.append(c)

    # 첫 채널은 connect() 로 accept + 등록, 이후는 register() 로 추가만
    first, rest = subscribed[0], subscribed[1:]
    await manager.connect(websocket, first)
    for c in rest:
        manager.register(websocket, c)

    # 연결 즉시 현재 상태 전송 (거부된 채널이 있으면 함께 알림)
    await manager.send_personal(websocket, {
        "type": "connection.established",
        "data": {
            "channels": subscribed,
            "rejected": rejected,
            "authenticated": token_sub is not None,
            "message": (
                f"{len(subscribed)}개 채널에 연결되었습니다: {', '.join(subscribed)}"
                + (f" / 거부됨: {', '.join(rejected)}" if rejected else "")
            ),
        },
    })

    try:
        # 하트비트 태스크: 주기적으로 ping을 보내 연결 유지
        async def heartbeat():
            while True:
                await asyncio.sleep(settings.WS_HEARTBEAT_INTERVAL)
                try:
                    await manager.send_personal(websocket, {"type": "ping"})
                except Exception:
                    break

        heartbeat_task = asyncio.create_task(heartbeat())

        try:
            # 클라이언트 메시지 수신 루프
            while True:
                data = await websocket.receive_json()
                # pong 응답 처리
                if data.get("type") == "pong":
                    continue
                # 기타 클라이언트 메시지 처리 (필요 시 확장)
        finally:
            heartbeat_task.cancel()

    except WebSocketDisconnect:
        pass
    finally:
        for c in subscribed:
            manager.disconnect(websocket, c)
