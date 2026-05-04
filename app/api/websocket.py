# =============================================
# app/api/websocket.py
# 역할: WebSocket 실시간 이벤트 엔드포인트
#       - WS /ws?channel={channel} → 채널 구독
#       - 채널: defects / telemetry / thermal / camera
#                / chat:{uuid} / user:{uuid} / notifications:{uuid}
#       - 연결 시 현재 카메라 모드 즉시 전송
#       - WS_HEARTBEAT_INTERVAL 초마다 ping 전송으로 죽은 연결 감지
# =============================================

import asyncio

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends

from app.config import settings
from app.core.ws_manager import ConnectionManager
from app.dependencies import get_ws_manager

router = APIRouter()

STATIC_CHANNELS = {"defects", "telemetry", "thermal", "camera"}

# 동적 채널 prefix — 사용자/대화방/알림별 개인 채널
# notifications: → notification_service 가 사용 (멤버 배정/보고서 알림 등)
# TODO: JWT 토큰으로 본인 채널만 구독 가능하게 인증 추가 필요
_DYNAMIC_CHANNEL_PREFIXES = ("chat:", "user:", "notifications:")


def _is_valid_channel(channel: str) -> bool:
    """드론 모니터링 고정 채널 또는 동적 채널(chat/user/notifications) 허용."""
    if channel in STATIC_CHANNELS:
        return True
    return any(channel.startswith(p) for p in _DYNAMIC_CHANNEL_PREFIXES)


@router.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    channel: str = "defects",
    manager: ConnectionManager = Depends(get_ws_manager),
):
    """
    WebSocket 구독 엔드포인트.

    연결 방법:
        ws://localhost:8000/api/v1/ws?channel=defects
        ws://localhost:8000/api/v1/ws?channel=chat:{conversation_id}
        ws://localhost:8000/api/v1/ws?channel=user:{user_id}

    수신 이벤트 타입:
        defects           채널: {"type": "defect.new", "data": {...}}
        telemetry         채널: {"type": "telemetry.update", "data": {...}}
        thermal           채널: {"type": "thermal.frame", "data": {...}}
        camera            채널: {"type": "camera.mode_changed", "data": {"mode": "..."}}
        chat:uuid         채널: {"type": "chat.new_message", "data": {...}}
        user:uuid         채널: {"type": "chat.new_message", "data": {...}}
        notifications:uuid 채널: {"type": "notification.new", "data": {...}}
    """
    if not _is_valid_channel(channel):
        channel = "defects"

    await manager.connect(websocket, channel)

    # 연결 즉시 현재 상태 전송
    await manager.send_personal(websocket, {
        "type": "connection.established",
        "data": {
            "channel": channel,
            "message": f"'{channel}' 채널에 연결되었습니다.",
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
        manager.disconnect(websocket, channel)
