# =============================================
# tests/test_ws_manager.py
# 역할: WebSocket 연결 매니저 단위 테스트
#       - 연결/해제 기능
#       - 채널 브로드캐스트
#       - 죽은 연결 자동 정리
# 실행: pytest tests/test_ws_manager.py -v
# =============================================

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock

from app.core.ws_manager import ConnectionManager


@pytest.fixture
def manager():
    """테스트용 ConnectionManager 인스턴스"""
    return ConnectionManager()


@pytest.fixture
def mock_websocket():
    """모킹된 WebSocket 객체"""
    ws = AsyncMock()
    ws.accept = AsyncMock()
    ws.send_text = AsyncMock()
    return ws


@pytest.mark.asyncio
async def test_connect(manager, mock_websocket):
    """연결 등록 테스트"""
    await manager.connect(mock_websocket, "defects")
    assert manager.get_connection_count("defects") == 1
    mock_websocket.accept.assert_called_once()


@pytest.mark.asyncio
async def test_disconnect(manager, mock_websocket):
    """연결 해제 테스트"""
    await manager.connect(mock_websocket, "defects")
    manager.disconnect(mock_websocket, "defects")
    assert manager.get_connection_count("defects") == 0


@pytest.mark.asyncio
async def test_broadcast(manager, mock_websocket):
    """브로드캐스트 테스트"""
    await manager.connect(mock_websocket, "defects")
    await manager.broadcast("defects", {"type": "defect.new", "data": {"id": "123"}})
    mock_websocket.send_text.assert_called_once()


@pytest.mark.asyncio
async def test_broadcast_empty_channel(manager):
    """연결 없는 채널 브로드캐스트 → 오류 없이 처리"""
    await manager.broadcast("empty_channel", {"type": "test"})  # 예외 발생하지 않아야 함


@pytest.mark.asyncio
async def test_multiple_clients(manager):
    """다중 클라이언트 브로드캐스트"""
    clients = [AsyncMock() for _ in range(3)]
    for c in clients:
        c.accept = AsyncMock()
        c.send_text = AsyncMock()
        await manager.connect(c, "defects")

    assert manager.get_connection_count("defects") == 3

    await manager.broadcast("defects", {"type": "test"})
    for c in clients:
        c.send_text.assert_called_once()


@pytest.mark.asyncio
async def test_get_all_channels(manager, mock_websocket):
    """전체 채널 목록 조회"""
    await manager.connect(mock_websocket, "defects")
    channels = manager.get_all_channels()
    assert "defects" in channels
    assert channels["defects"] == 1
