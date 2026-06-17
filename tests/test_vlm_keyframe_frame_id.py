"""
tests/test_vlm_keyframe_frame_id.py
역할: VLM 키프레임 루프가 DB 저장 시 '캡처된 frame_id' 를 쓰는지 검증.
  회귀 방지: save_batch 에 증가 중인 self._submitted_count 를 넘겨, DB 레코드가
  같은 검출의 broadcast(frame_id=캡처값)와 다른 frame_id 로 저장되던 버그.
실행: pytest tests/test_vlm_keyframe_frame_id.py -v
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import numpy as np
import pytest

import app.core.stream_inference as si


@pytest.mark.asyncio
async def test_vlm_keyframe_persists_captured_frame_id(monkeypatch):
    worker = si.StreamInferenceWorker()
    worker._running = True
    worker._last_frame_bgr = np.zeros((2, 2, 3), dtype=np.uint8)
    worker._last_frame_id = 42            # 캡처된 frame_id (브로드캐스트가 쓰는 값)
    worker._submitted_count = 999         # 그동안 계속 증가한 전역 카운터 (≠ 42)

    # 한 번 처리 후 루프 종료
    async def _sleep_once(_interval):
        worker._running = False

    monkeypatch.setattr(si.asyncio, "sleep", _sleep_once)
    monkeypatch.setattr(si.settings, "VLM_HYBRID_ENABLED", False)
    monkeypatch.setattr(si.settings, "VLM_KEYFRAME_INTERVAL_SEC", 1)
    monkeypatch.setattr(si.cv2, "imencode", lambda ext, frame: (True, np.zeros((4,), dtype=np.uint8)))
    monkeypatch.setattr(si, "detect_vlm_async", AsyncMock(return_value=SimpleNamespace(model_dump_json=lambda: "{}")))
    monkeypatch.setattr(si.ws_manager, "broadcast", AsyncMock())
    monkeypatch.setattr(si.defect_persistence, "save_batch", AsyncMock(return_value=1))

    # 검출 1건 + lidar 없음 (정적/스태틱 메서드를 인스턴스에서 덮어씀)
    worker._vlm_to_dicts = lambda vr: [{
        "code": "A-02", "conf": 0.8, "class_display_ko": "균열",
        "severity": "HIGH", "class": "crack", "defect_source": "vlm",
    }]
    worker._compute_lidar_xyz = lambda: None

    await worker._vlm_keyframe_loop()

    si.defect_persistence.save_batch.assert_awaited_once()
    kwargs = si.defect_persistence.save_batch.call_args.kwargs
    assert kwargs["frame_id"] == 42, "DB 저장은 캡처된 frame_id 를 써야 한다 (증가 카운터 아님)"
    assert kwargs["frame_id"] != 999
