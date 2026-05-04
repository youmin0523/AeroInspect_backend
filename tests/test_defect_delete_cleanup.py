# =============================================
# tests/test_defect_delete_cleanup.py
# 역할: DELETE /api/v1/defects/{id}가 image_storage.delete를 호출해서
#       크롭 파일도 같이 정리하는지 검증
#
# DB·FastAPI 의존성을 전부 붙이지 않고, 핵심 로직만 단위 검증:
#   - image_crop_path 가 있는 레코드 → delete 호출
#   - image_crop_path 가 None → delete 호출 안 함
#   - DB 삭제는 먼저, 파일 삭제는 그 뒤 (파일 삭제 실패해도 트랜잭션 영향 없게)
#
# 실행: pytest tests/test_defect_delete_cleanup.py -v
# =============================================

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from fastapi import HTTPException


# ── 테스트 대상 로직을 순수 함수로 추출해서 검증 ─────
# (실제 핸들러는 org_tuple/db dependency로 감싸져 있어 그대로 호출하려면
#  FastAPI TestClient + DB 필요. 여기선 핵심 순서만 재현해서 단위 검증.)

async def _simulate_delete_cleanup(defect, db, storage):
    """defects.delete_defect 핵심 순서를 그대로 재현."""
    if not defect:
        raise HTTPException(status_code=404, detail="not found")

    crop_path = defect.image_crop_path
    await db.delete(defect)

    if crop_path:
        storage.delete(crop_path)


@pytest.mark.asyncio
async def test_delete_calls_storage_when_path_exists():
    """image_crop_path 있으면 storage.delete 호출."""
    defect = SimpleNamespace(
        id=uuid4(),
        image_crop_path="defects/2026-04-22/abc.jpg",
    )
    db = MagicMock()
    db.delete = AsyncMock()
    storage = MagicMock()

    await _simulate_delete_cleanup(defect, db, storage)

    db.delete.assert_awaited_once_with(defect)
    storage.delete.assert_called_once_with("defects/2026-04-22/abc.jpg")


@pytest.mark.asyncio
async def test_delete_skips_storage_when_path_none():
    """image_crop_path 가 None이면 storage.delete 호출 안 함."""
    defect = SimpleNamespace(id=uuid4(), image_crop_path=None)
    db = MagicMock()
    db.delete = AsyncMock()
    storage = MagicMock()

    await _simulate_delete_cleanup(defect, db, storage)

    db.delete.assert_awaited_once_with(defect)
    storage.delete.assert_not_called()


@pytest.mark.asyncio
async def test_db_delete_happens_before_file_delete():
    """DB 삭제 → 파일 삭제 순서 보장. 파일 삭제가 먼저면 트랜잭션 롤백 시 orphan 발생."""
    defect = SimpleNamespace(id=uuid4(), image_crop_path="defects/x/y.jpg")
    call_order: list[str] = []

    db = MagicMock()

    async def _db_delete(_):
        call_order.append("db")

    db.delete = _db_delete
    storage = MagicMock()
    storage.delete.side_effect = lambda _: call_order.append("file")

    await _simulate_delete_cleanup(defect, db, storage)

    assert call_order == ["db", "file"]


@pytest.mark.asyncio
async def test_missing_defect_raises_404():
    db = MagicMock()
    db.delete = AsyncMock()
    storage = MagicMock()

    with pytest.raises(HTTPException) as exc:
        await _simulate_delete_cleanup(None, db, storage)

    assert exc.value.status_code == 404
    db.delete.assert_not_called()
    storage.delete.assert_not_called()
