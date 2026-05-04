# =============================================
# tests/test_ensemble.py
# 앙상블 함수 단위 테스트
# - compute_combined_confidence (Noisy-OR)
# - cross_model_nms
# - cross_model_spatial_boost
# =============================================

import pytest

from app.services.ensemble import (
    compute_combined_confidence,
    cross_model_nms,
    cross_model_spatial_boost,
)


class TestNoisyOR:
    def test_basic(self):
        # 1 - (1-0.3)*(1-0.6) = 1 - 0.28 = 0.72
        assert compute_combined_confidence(0.3, 0.6) == pytest.approx(0.72, abs=0.01)

    def test_zero(self):
        assert compute_combined_confidence(0.0, 0.5) == pytest.approx(0.5, abs=0.01)

    def test_both_high(self):
        result = compute_combined_confidence(0.9, 0.9)
        assert result == pytest.approx(0.99, abs=0.01)

    def test_symmetry(self):
        assert compute_combined_confidence(0.3, 0.7) == pytest.approx(
            compute_combined_confidence(0.7, 0.3), abs=1e-6
        )


class TestCrossModelNMS:
    def test_same_class_dedup(self):
        dets = [
            {"class": "crack", "conf": 0.9, "bbox_xyxy": [10, 10, 50, 50]},
            {"class": "crack", "conf": 0.7, "bbox_xyxy": [12, 12, 52, 52]},
        ]
        result = cross_model_nms(dets, iou_threshold=0.3)
        assert len(result) == 1
        assert result[0]["conf"] == 0.9

    def test_different_class_kept(self):
        dets = [
            {"class": "crack", "conf": 0.9, "bbox_xyxy": [10, 10, 50, 50]},
            {"class": "moisture", "conf": 0.7, "bbox_xyxy": [10, 10, 50, 50]},
        ]
        result = cross_model_nms(dets)
        assert len(result) == 2

    def test_empty(self):
        assert cross_model_nms([]) == []

    def test_single(self):
        dets = [{"class": "crack", "conf": 0.5, "bbox_xyxy": [0, 0, 50, 50]}]
        assert cross_model_nms(dets) == dets


class TestCrossModelSpatialBoost:
    def test_different_source_boosted(self):
        dets = [
            {"class": "crack", "conf": 0.4, "bbox_xyxy": [10, 10, 50, 50], "defect_source": "yolo_structural"},
            {"class": "moisture", "conf": 0.3, "bbox_xyxy": [15, 15, 55, 55], "defect_source": "yolo_surface"},
        ]
        result = cross_model_spatial_boost(dets, iou_threshold=0.3, boost_factor=0.15)
        assert result[0]["conf"] == pytest.approx(0.55, abs=0.01)
        assert result[1]["conf"] == pytest.approx(0.45, abs=0.01)
        assert result[0].get("cross_model_boosted") is True

    def test_same_source_not_boosted(self):
        dets = [
            {"class": "crack", "conf": 0.4, "bbox_xyxy": [10, 10, 50, 50], "defect_source": "yolo_structural"},
            {"class": "moisture", "conf": 0.3, "bbox_xyxy": [15, 15, 55, 55], "defect_source": "yolo_structural"},
        ]
        result = cross_model_spatial_boost(dets)
        assert result[0]["conf"] == 0.4  # 변경 없음
        assert result[1]["conf"] == 0.3

    def test_no_overlap_not_boosted(self):
        dets = [
            {"class": "crack", "conf": 0.4, "bbox_xyxy": [0, 0, 50, 50], "defect_source": "yolo_structural"},
            {"class": "moisture", "conf": 0.3, "bbox_xyxy": [200, 200, 300, 300], "defect_source": "yolo_surface"},
        ]
        result = cross_model_spatial_boost(dets)
        assert result[0]["conf"] == 0.4
        assert result[1]["conf"] == 0.3

    def test_conf_capped_at_1(self):
        dets = [
            {"class": "crack", "conf": 0.95, "bbox_xyxy": [10, 10, 50, 50], "defect_source": "m1"},
            {"class": "moisture", "conf": 0.95, "bbox_xyxy": [10, 10, 50, 50], "defect_source": "m2"},
        ]
        result = cross_model_spatial_boost(dets, boost_factor=0.2)
        assert result[0]["conf"] <= 1.0
        assert result[1]["conf"] <= 1.0
