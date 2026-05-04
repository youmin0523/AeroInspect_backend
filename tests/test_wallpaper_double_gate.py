# =============================================
# tests/test_wallpaper_double_gate.py
# 역할: 벽지 분류 이중 게이트 회귀 방지
#       - top1_conf >= WALLPAPER_CONF_THRESHOLD
#       - AND (top1_conf - top2_conf) >= WALLPAPER_MARGIN_THRESHOLD
#
# wallpaper_classifier.classify를 monkeypatch해서 실제 모델 없이 로직만 검증.
# 실행: pytest tests/test_wallpaper_double_gate.py -v
# =============================================

from __future__ import annotations

import numpy as np
import pytest

from app.services import wallpaper_classifier as wc_mod
from app.services.inference_pipeline import InferencePipeline


def _make_pipeline(conf_thr: float = 0.35, margin_thr: float = 0.15) -> InferencePipeline:
    """모델 로드 없이 임계값만 세팅한 파이프라인."""
    p = InferencePipeline()
    p._wallpaper_conf_threshold = conf_thr
    p._wallpaper_margin_threshold = margin_thr
    return p


class TestWallpaperDoubleGate:
    def _patch_classify(self, monkeypatch, top3):
        """wallpaper_classifier.classify를 고정 응답으로 대체."""
        def fake_classify(image_rgb):
            return top3[0][0], top3[0][1], top3
        monkeypatch.setattr(wc_mod.wallpaper_classifier, "classify", fake_classify)

    def test_confident_when_both_gates_pass(self, monkeypatch):
        # top1=0.60, top2=0.30 → margin=0.30 >= 0.15, conf 0.60 >= 0.35 → confident
        self._patch_classify(monkeypatch, [("Mold", 0.60), ("Damage", 0.30), ("good", 0.05)])
        p = _make_pipeline()
        pred = p._run_wallpaper(np.zeros((224, 224, 3), dtype=np.uint8))
        assert pred.is_confident is True
        assert pred.top1_class == "Mold"

    def test_not_confident_when_top1_below_threshold(self, monkeypatch):
        # top1=0.30 < 0.35 → 첫번째 게이트 실패
        self._patch_classify(monkeypatch, [("Mold", 0.30), ("Damage", 0.10), ("good", 0.05)])
        p = _make_pipeline()
        pred = p._run_wallpaper(np.zeros((224, 224, 3), dtype=np.uint8))
        assert pred.is_confident is False

    def test_not_confident_when_margin_too_small(self, monkeypatch):
        # top1=0.41, top2=0.35 → margin=0.06 < 0.15 → 두번째 게이트 실패
        # (top1 자체는 0.35 넘음 → 기존 단일 게이트였다면 confident였을 케이스)
        self._patch_classify(monkeypatch, [("Mold", 0.41), ("Damage", 0.35), ("good", 0.10)])
        p = _make_pipeline()
        pred = p._run_wallpaper(np.zeros((224, 224, 3), dtype=np.uint8))
        assert pred.is_confident is False

    def test_edge_just_above_thresholds(self, monkeypatch):
        # 경계값 바로 위: top1=0.36 (> 0.35), margin≈0.16 (> 0.15) → 두 게이트 모두 통과
        # 정확한 0.35-0.20=0.15는 float 표현상 0.14999...로 평가되어 불안정 — 살짝 위로.
        self._patch_classify(monkeypatch, [("Mold", 0.36), ("Damage", 0.20), ("good", 0.10)])
        p = _make_pipeline()
        pred = p._run_wallpaper(np.zeros((224, 224, 3), dtype=np.uint8))
        assert pred.is_confident is True

    def test_top3_returned_in_order(self, monkeypatch):
        self._patch_classify(monkeypatch, [("Mold", 0.60), ("Damage", 0.30), ("good", 0.05)])
        p = _make_pipeline()
        pred = p._run_wallpaper(np.zeros((224, 224, 3), dtype=np.uint8))
        assert len(pred.top3) == 3
        assert pred.top3[0].conf >= pred.top3[1].conf >= pred.top3[2].conf
