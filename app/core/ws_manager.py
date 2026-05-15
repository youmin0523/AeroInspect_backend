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
        """클라이언트를 채널에서 제거"""
        if websocket in self._connections[channel]:
            self._connections[channel].remove(websocket)
        print(f"[WS] 연결 해제: channel={channel}, 남은={len(self._connections[channel])}개")

    async def broadcast(self, channel: str, payload: dict) -> None:
        """
        채널의 모든 클라이언트에게 메시지 브로드캐스트.
        죽은 연결은 즉시 정리.
        """
        if channel not in self._connections:
            return

        message = json.dumps(payload, ensure_ascii=False, default=str)
        dead_sockets = []

        # 동시 전송 (asyncio.gather)
        async def send_to(ws: WebSocket):
            try:
                await ws.send_text(message)
            except (WebSocketDisconnect, RuntimeError):
                dead_sockets.append(ws)

        # 목록 복사본으로 순회 (순회 중 수정 방지)
        connections_snapshot = list(self._connections[channel])
        await asyncio.gather(*[send_to(ws) for ws in connections_snapshot])

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
ws_manager = ConnectionManager()
