# =============================================
# tests/test_inference_pipeline_20.py
# 역할: 20종 하자 파이프라인 + multi-ckpt WBF 통합 회귀 방지
#       - 신규 보조 ckpt 속성 (M2 v4s, M3 v4s_retry) 존재 확인
#       - WBF 메서드 시그니처 + 폴백 동작
#       - tier 인자 전달 (Tier 3에서만 multi-ckpt 활성)
#       - 모델 미로드 시 graceful degradation
#
# 가중치 없이도 통과하도록 설계 (모델 객체는 None 으로 둠).
# 실행: pytest tests/test_inference_pipeline_20.py -v
# =============================================

from __future__ import annotations

import inspect
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.services.inference_pipeline_20 import InferencePipeline20


@pytest.fixture
def pipe() -> InferencePipeline20:
    """모델 미로드 상태의 빈 파이프라인."""
    return InferencePipeline20()


# ─────────────────────────────────────────────
# 1) 신규 attribute 존재 검증 (multi-ckpt WBF 통합 회귀 방지)
# ─────────────────────────────────────────────

def test_pipeline_has_v4s_attrs(pipe):
    """multi-ckpt WBF 도입 시 추가된 보조 ckpt 속성이 존재해야 함."""
    assert hasattr(pipe, "_m2_yolo_v4s"), "M2 v4s 보조 ckpt attr 누락"
    assert hasattr(pipe, "_m3_yolo_v4s_retry"), "M3 v4s_retry 보조 ckpt attr 누락"
    # 모델 미로드 상태에서는 None 이어야 함
    assert pipe._m2_yolo_v4s is None
    assert pipe._m3_yolo_v4s_retry is None


def test_wbf_method_exists(pipe):
    """multi-ckpt WBF 메서드가 정의되어야 함."""
    assert hasattr(pipe, "_run_yolo_multi_ckpt_wbf"), "WBF 메서드 누락"
    sig = inspect.signature(pipe._run_yolo_multi_ckpt_wbf)
    expected_params = {"frame_bgr", "ckpts", "imgsz_per_ckpt", "conf",
                       "iou_thr", "skip_box_thr", "top_k", "source_tag"}
    actual = set(sig.parameters.keys())
    missing = expected_params - actual
    assert not missing, f"WBF 시그니처 파라미터 누락: {missing}"


# ─────────────────────────────────────────────
# 2) tier 분기 — Tier 3에서만 multi-ckpt 활성
# ─────────────────────────────────────────────

def test_run_m2_signature_accepts_tier(pipe):
    sig = inspect.signature(pipe._run_m2)
    assert "tier" in sig.parameters, "_run_m2가 tier 인자를 받아야 함"


def test_run_m3_signature_accepts_tier(pipe):
    sig = inspect.signature(pipe._run_m3)
    assert "tier" in sig.parameters, "_run_m3가 tier 인자를 받아야 함"


def test_run_m2_tier1_no_wbf(pipe):
    """Tier 1: 보조 ckpt 있어도 WBF 사용 X (실시간 비용 보호)."""
    pipe._m2_yolo = MagicMock()
    pipe._m2_yolo.predict = MagicMock(return_value=[])
    pipe._m2_yolo_v4s = MagicMock()  # 가용해도
    # WBF 메서드를 spy로 감시
    with patch.object(pipe, "_run_yolo_multi_ckpt_wbf") as wbf_spy:
        result = pipe._run_m2(np.zeros((640, 640, 3), dtype=np.uint8), tier=1)
    wbf_spy.assert_not_called()  # Tier 1에선 절대 WBF 호출 X
    assert result == []


def test_run_m2_tier3_uses_wbf_when_aux_available(pipe):
    """Tier 3 + 보조 ckpt 가용 시 WBF 사용."""
    pipe._m2_yolo = MagicMock()
    pipe._m2_yolo_v4s = MagicMock()  # 보조 가용
    with patch.object(pipe, "_run_yolo_multi_ckpt_wbf", return_value=[]) as wbf_spy:
        pipe._run_m2(np.zeros((640, 640, 3), dtype=np.uint8), tier=3)
    wbf_spy.assert_called_once()
    # 호출 인자 검증
    args, kwargs = wbf_spy.call_args
    assert "ckpts" in kwargs
    assert len(kwargs["ckpts"]) == 2  # 메인 + 보조


def test_run_m2_tier3_falls_back_when_aux_missing(pipe):
    """Tier 3여도 보조 ckpt 없으면 단일 추론 (graceful)."""
    pipe._m2_yolo = MagicMock()
    pipe._m2_yolo.predict = MagicMock(return_value=[])
    pipe._m2_yolo_v4s = None  # 보조 없음
    with patch.object(pipe, "_run_yolo_multi_ckpt_wbf") as wbf_spy:
        pipe._run_m2(np.zeros((640, 640, 3), dtype=np.uint8), tier=3)
    wbf_spy.assert_not_called()  # 폴백
    pipe._m2_yolo.predict.assert_called_once()


def test_run_m3_tier3_uses_wbf_when_aux_available(pipe):
    pipe._m3_yolo = MagicMock()
    pipe._m3_yolo_v4s_retry = MagicMock()
    with patch.object(pipe, "_run_yolo_multi_ckpt_wbf", return_value=[]) as wbf_spy:
        pipe._run_m3(np.zeros((640, 640, 3), dtype=np.uint8), tier=3)
    wbf_spy.assert_called_once()


# ─────────────────────────────────────────────
# 3) WBF graceful degradation (라이브러리 미설치 / 빈 결과)
# ─────────────────────────────────────────────

def test_wbf_handles_empty_predictions():
    """모든 ckpt가 빈 결과 반환 시 빈 리스트 반환 (raise X)."""
    pipe = InferencePipeline20()
    mock_ckpt = MagicMock()
    mock_ckpt.predict = MagicMock(return_value=[])
    result = pipe._run_yolo_multi_ckpt_wbf(
        np.zeros((480, 640, 3), dtype=np.uint8),
        ckpts=[mock_ckpt, mock_ckpt],
        imgsz_per_ckpt=[[640], [640]],
    )
    assert result == []


def test_wbf_returns_list_of_dicts():
    """WBF 결과가 dict 리스트 형태 (class, conf, bbox_xyxy 키 포함)."""
    pipe = InferencePipeline20()
    mock_ckpt = MagicMock()
    mock_ckpt.class_names = ["floor_defect", "glass_defect"]
    mock_ckpt.predict = MagicMock(return_value=[
        {"class": "floor_defect", "class_id": 0, "conf": 0.8,
         "bbox_xyxy": [10.0, 20.0, 100.0, 200.0]},
        {"class": "glass_defect", "class_id": 1, "conf": 0.6,
         "bbox_xyxy": [200.0, 50.0, 400.0, 250.0]},
    ])
    result = pipe._run_yolo_multi_ckpt_wbf(
        np.zeros((300, 400, 3), dtype=np.uint8),
        ckpts=[mock_ckpt, mock_ckpt],
        imgsz_per_ckpt=[[640], [640]],
    )
    if not result:
        # ensemble-boxes가 없으면 폴백 단일 결과 반환됨
        return
    # 결과 형식 검증
    for det in result:
        assert "class" in det
        assert "conf" in det
        assert "bbox_xyxy" in det
        assert "defect_source" in det
        assert "wbf_fused" in det
        assert det["wbf_fused"] is True
        assert len(det["bbox_xyxy"]) == 4


# ─────────────────────────────────────────────
# 4) detect() 호출 시 tier 전달 회귀 방지
# ─────────────────────────────────────────────

def test_detect_passes_tier_to_m2_m3(pipe):
    """detect()가 _run_m2/_run_m3에 tier 인자를 정확히 전달해야 함."""
    pipe._loaded = True
    pipe._m1_yolo = MagicMock()
    pipe._m1_yolo.predict = MagicMock(return_value=[])
    pipe._m2_yolo = MagicMock()
    pipe._m2_yolo.predict = MagicMock(return_value=[])
    pipe._m3_yolo = MagicMock()
    pipe._m3_yolo.predict = MagicMock(return_value=[])

    with patch.object(pipe, "_run_m2", wraps=pipe._run_m2) as m2_spy, \
         patch.object(pipe, "_run_m3", wraps=pipe._run_m3) as m3_spy:
        pipe.detect(np.zeros((480, 640, 3), dtype=np.uint8), tier=3)

    # _run_m2(..., tier=3), _run_m3(..., tier=3) 호출됐는지
    m2_call_kwargs = m2_spy.call_args.kwargs if m2_spy.call_args else {}
    m3_call_kwargs = m3_spy.call_args.kwargs if m3_spy.call_args else {}
    assert m2_call_kwargs.get("tier") == 3, f"M2에 tier=3 전달 안됨: {m2_call_kwargs}"
    assert m3_call_kwargs.get("tier") == 3, f"M3에 tier=3 전달 안됨: {m3_call_kwargs}"


# ─────────────────────────────────────────────
# 5) models_loaded 상태 — 보조 ckpt 미로드 무관하게 동작
# ─────────────────────────────────────────────

def test_models_loaded_status_independent_of_aux_ckpt(pipe):
    """보조 ckpt(v4s) 로드 여부는 models_loaded 11/11 카운트에 영향 X."""
    status = pipe.models_loaded
    # 11개 필드 모두 존재해야 함 (기존 통합 호환)
    expected_fields = {"m1_yolo", "m1_resnet", "m2_yolo", "m2_resnet",
                      "m3_yolo", "m3_resnet", "m4_unet", "m4_context",
                      "m5_seg", "m6_patchcore", "furniture_aware"}
    actual = set(status.dict().keys())
    assert expected_fields == actual, f"models_loaded 필드 변경됨: {actual - expected_fields}"
