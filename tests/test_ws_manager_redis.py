# =============================================
# tests/test_ws_manager_redis.py
# 역할: Redis 백엔드 추상화 회귀
#       - create_ws_manager 팩토리가 올바른 타입 반환
#       - Redis 미기동 상태에서 broadcast 가 로컬 폴백
#       - 잘못된 backend 문자열은 ValueError
#
# 실제 Redis 서버 없이도 돌 수 있도록 start() 호출하지 않는 케이스만 검증.
# 실 Redis 연동은 통합 테스트로 별도.
# 실행: pytest tests/test_ws_manager_redis.py -v
# =============================================

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, patch

from app.core.ws_manager import ConnectionManager
from app.core.ws_manager_redis import RedisConnectionManager, create_ws_manager


class TestFactory:
    def test_memory_backend_returns_plain_manager(self):
        mgr = create_ws_manager(backend="memory")
        assert isinstance(mgr, ConnectionManager)
        assert not isinstance(mgr, RedisConnectionManager)

    def test_redis_backend_returns_redis_manager(self):
        mgr = create_ws_manager(backend="redis", redis_url="redis://localhost:6379/0")
        assert isinstance(mgr, RedisConnectionManager)
        assert isinstance(mgr, ConnectionManager)  # 상속 관계 유지

    def test_redis_without_url_raises(self):
        with pytest.raises(ValueError, match="REDIS_URL"):
            create_ws_manager(backend="redis")

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="WS_BACKEND"):
            create_ws_manager(backend="rabbitmq")


class TestRedisFallback:
    @pytest.mark.asyncio
    async def test_broadcast_before_start_falls_back_to_local(self):
        """start() 호출 전에는 Redis 클라이언트 None → 부모 ConnectionManager 로직으로 폴백."""
        mgr = RedisConnectionManager(redis_url="redis://fake/0")
        # _connections 비어있으니 broadcast 가 에러 없이 완료돼야 함
        await mgr.broadcast("defects", {"type": "defect.new", "data": {}})

    @pytest.mark.asyncio
    async def test_broadcast_publish_error_falls_back(self):
        """publish 실패 시 예외 삼키고 로컬 브로드캐스트로 대체."""
        mgr = RedisConnectionManager(redis_url="redis://fake/0")
        # redis 속성을 가짜로 채우고 publish 가 실패하게
        fake_redis = AsyncMock()
        fake_redis.publish.side_effect = RuntimeError("Connection refused")
        mgr._redis = fake_redis

        # 폴백 경로가 호출되는지 확인 — 부모 broadcast spy
        with patch.object(
            ConnectionManager, "broadcast", new=AsyncMock(return_value=None)
        ) as parent_broadcast:
            await mgr.broadcast("defects", {"hello": 1})
            parent_broadcast.assert_awaited_once()
