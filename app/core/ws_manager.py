# =============================================
# app/core/ws_manager.py
# 역할: WebSocket 연결 매니저 (모듈 레벨 싱글톤)
#       - 채널별 연결된 클라이언트 목록 관리
#       - 채널 브로드캐스트: 하자 탐지, 드론 텔레메트리, 열화상 데이터
#       - 죽은 연결 자동 정리
#       - asyncio.gather로 병렬 브로드캐스트 (HOL 블로킹 방지)
#
# 채널 목록:
#   "defects"    → 새 하자 탐지 이벤트
#   "telemetry"  → 드론 위치/자세/배터리 텔레메트리
#   "thermal"    → 열화상 온도 데이터 (Recharts용)
#   "camera"     → 카메라 모드 전환 이벤트
# =============================================

import asyncio
import json
from collections import defaultdict
from typing import Dict, List

from fastapi import WebSocket
from starlette.websockets import WebSocketDisconnect

# 브로드캐스트 전송별 타임아웃(초). 느린 클라이언트가 전체 브로드캐스트를 막는 것 방지.
WS_SEND_TIMEOUT = 5.0


class ConnectionManager:
    """
    WebSocket 연결 매니저.
    채널별로 클라이언트 목록을 관리하고 브로드캐스트한다.
    단일 인스턴스(싱글톤)로 앱 전체에서 공유.
    """

    def __init__(self):
        # 채널 → 연결된 WebSocket 목록
        self._connections: Dict[str, List[WebSocket]] = defaultdict(list)

    async def connect(self, websocket: WebSocket, channel: str) -> None:
        """새 클라이언트를 채널에 등록 (accept + register).

        다중 채널 구독 시에는 첫 채널만 connect() 로 받고 이후 채널은
        register() 로 추가하면 accept 중복 호출을 피할 수 있다.
        """
        await websocket.accept()
        self._add(websocket, channel)

    def register(self, websocket: WebSocket, channel: str) -> None:
        """이미 accept 된 연결을 추가 채널에 등록만 한다 (accept 호출 안 함)."""
        self._add(websocket, channel)

    def _add(self, websocket: WebSocket, channel: str) -> None:
        if websocket not in self._connections[channel]:
            self._connections[channel].append(websocket)
        print(f"[WS] 연결됨: channel={channel}, 총={len(self._connections[channel])}개")

    def disconnect(self, websocket: WebSocket, channel: str) -> None:
        """클라이언트를 채널에서 제거. 마지막 연결이면 채널 항목 자체를 제거(메모리 누수 방지)."""
        sockets = self._connections.get(channel)
        if sockets and websocket in sockets:
            sockets.remove(websocket)
        remaining = len(self._connections.get(channel, []))
        # 동적 채널(chat:{uuid}, user:{uid})이 빈 리스트로 영구 누적되는 것 방지
        if channel in self._connections and not self._connections[channel]:
            del self._connections[channel]
        print(f"[WS] 연결 해제: channel={channel}, 남은={remaining}개")

    async def broadcast(self, channel: str, payload: dict) -> None:
        """
        채널의 모든 클라이언트에게 메시지 브로드캐스트.
        - 느린 클라이언트 1명이 전체 브로드캐스트를 막지 않도록 전송별 타임아웃 적용.
        - 죽은/타임아웃 연결은 즉시 정리.
        """
        if channel not in self._connections:
            return

        message = json.dumps(payload, ensure_ascii=False, default=str)
        dead_sockets = []

        # 동시 전송 (asyncio.gather) — 전송별 타임아웃으로 slow-consumer 격리
        async def send_to(ws: WebSocket):
            try:
                await asyncio.wait_for(ws.send_text(message), timeout=WS_SEND_TIMEOUT)
            except Exception:
                # WebSocketDisconnect / RuntimeError / TimeoutError / ConnectionReset 등
                # 어떤 실패든 해당 소켓을 죽은 것으로 간주하고 정리한다.
                dead_sockets.append(ws)

        # 목록 복사본으로 순회 (순회 중 수정 방지)
        connections_snapshot = list(self._connections[channel])
        # return_exceptions=True — 한 소켓 실패가 나머지 전송을 중단시키지 않도록
        await asyncio.gather(
            *[send_to(ws) for ws in connections_snapshot],
            return_exceptions=True,
        )

        # 끊긴 소켓 정리
        for ws in dead_sockets:
            self.disconnect(ws, channel)

    async def send_personal(self, websocket: WebSocket, payload: dict) -> None:
        """특정 클라이언트에게만 메시지 전송"""
        try:
            message = json.dumps(payload, ensure_ascii=False, default=str)
            await websocket.send_text(message)
        except (WebSocketDisconnect, RuntimeError):
            pass

    def get_connection_count(self, channel: str) -> int:
        """채널의 현재 연결 수 반환"""
        return len(self._connections.get(channel, []))

    def get_all_channels(self) -> Dict[str, int]:
        """전체 채널별 연결 수 반환"""
        return {ch: len(sockets) for ch, sockets in self._connections.items()}


# ── 모듈 레벨 싱글톤 ─────────────────────────
# 앱 전체에서 단 하나의 인스턴스를 공유 (멀티 프로세스 불가)
#
# 주의(R-fix): Redis 백엔드 모드에서 main.py lifespan 이 활성 매니저를 교체한다.
# 과거에는 `ws_manager` 모듈 어트리뷰트를 재바인딩했는데, 핫패스 모듈
# (ws_stream.py / stream_inference.py)이 `from ... import ws_manager` 로 *값*을
# 캡처해버려 교체가 반영되지 않았다(= Redis 브로드캐스트가 죽는 치명 버그).
# 해결: `ws_manager` 를 항상 현재 활성 매니저로 위임하는 프록시로 만들어, import
# 시점에 캡처돼도 호출 시점의 활성 매니저로 forwarding 되게 한다. 교체는 반드시
# set_active_manager() 로 수행한다(모듈 어트리뷰트 재바인딩 금지).
_active_manager = ConnectionManager()


def set_active_manager(manager) -> None:
    """활성 WebSocket 매니저 교체 (Redis 모드 전환 등). main.py lifespan 에서 호출."""
    global _active_manager
    _active_manager = manager


def get_active_manager():
    """현재 활성 WebSocket 매니저 반환."""
    return _active_manager


class _ActiveManagerProxy:
    """현재 활성 매니저로 모든 속성/메서드 접근을 위임하는 프록시.

    `from app.core.ws_manager import ws_manager` 로 import 시점에 캡처돼도
    호출 시점에 set_active_manager() 로 바뀐 매니저를 정확히 사용하게 한다.
    """

    def __getattr__(self, name):
        return getattr(_active_manager, name)

    def __repr__(self) -> str:  # 디버깅 편의
        return f"<ActiveManagerProxy -> {_active_manager!r}>"


ws_manager = _ActiveManagerProxy()
