# =============================================
# tests/test_tta.py
# 역할: TTAEnsemble 단위 테스트
# =============================================

import numpy as np
import pytest

from app.services.tta import (
    TTAEnsemble,
    _flip_horizontal,
    _flip_horizontal_inverse_bbox,
    _scale,
    _scale_inverse_bbox,
    _merge_max_conf,
    _merge_wbf,
    load_tta_from_config,
)


# ── Augmentation primitives ──

class TestAugmentationPrimitives:
    def test_hflip_inverse_round_trip(self):
        """hflip 후 inverse 좌표 변환이 원래대로 돌아오는지."""
        img = np.ones((100, 200, 3), dtype=np.uint8)
        flipped, meta = _flip_horizontal(img)
        assert flipped.shape == img.shape
        assert meta["img_w"] == 200

        # Original bbox [10, 20, 50, 60] → hflip 좌표 [150, 20, 190, 60] → inverse → [10, 20, 50, 60]
        flipped_bbox = [150.0, 20.0, 190.0, 60.0]
        inv = _flip_horizontal_inverse_bbox(flipped_bbox, meta)
        assert inv == [10.0, 20.0, 50.0, 60.0]

    def test_scale_inverse(self):
        """scale 후 inverse 좌표 변환."""
        img = np.ones((100, 100, 3), dtype=np.uint8)
        scaled, meta = _scale(img, 0.8)
        assert scaled.shape[0] == 80
        assert scaled.shape[1] == 80
        assert meta["factor"] == 0.8

        # 0.8배 축소된 이미지에서 [0, 0, 80, 80] 검출 → 원본 좌표 [0, 0, 100, 100]
        scaled_bbox = [0.0, 0.0, 80.0, 80.0]
        inv = _scale_inverse_bbox(scaled_bbox, meta)
        assert inv[2] == pytest.approx(100.0)
        assert inv[3] == pytest.approx(100.0)


# ── 병합 메서드 ──

class TestMergeMethods:
    def test_max_conf_keeps_highest(self):
        dets = [
            {"class": "crack", "conf": 0.5, "bbox_xyxy": [10, 10, 50, 50]},
            {"class": "crack", "conf": 0.9, "bbox_xyxy": [12, 12, 48, 48]},  # IoU > 0.5
            {"class": "crack", "conf": 0.7, "bbox_xyxy": [11, 11, 49, 49]},  # IoU > 0.5
        ]
        merged = _merge_max_conf(dets, iou_threshold=0.5)
        assert len(merged) == 1
        assert merged[0]["conf"] == 0.9

    def test_max_conf_different_classes_keep_separate(self):
        dets = [
            {"class": "crack", "conf": 0.5, "bbox_xyxy": [10, 10, 50, 50]},
            {"class": "moisture", "conf": 0.6, "bbox_xyxy": [10, 10, 50, 50]},
        ]
        merged = _merge_max_conf(dets, iou_threshold=0.5)
        assert len(merged) == 2

    def test_wbf_weighted_average(self):
        dets = [
            {"class": "crack", "conf": 0.6, "bbox_xyxy": [40.0, 40.0, 60.0, 60.0]},
            {"class": "crack", "conf": 0.4, "bbox_xyxy": [40.0, 40.0, 60.0, 60.0]},
        ]
        merged = _merge_wbf(dets, iou_threshold=0.5)
        assert len(merged) == 1
        # avg conf = (0.6 + 0.4) / 2 = 0.5, boost = 1.0 + 0.1*(2-1) = 1.1 → 0.55
        assert merged[0]["conf"] == pytest.approx(0.55)
        assert merged[0]["tta_cluster_size"] == 2

    def test_wbf_separate_clusters_different_locations(self):
        dets = [
            {"class": "crack", "conf": 0.6, "bbox_xyxy": [10, 10, 50, 50]},
            {"class": "crack", "conf": 0.5, "bbox_xyxy": [200, 200, 250, 250]},
        ]
        merged = _merge_wbf(dets, iou_threshold=0.5)
        assert len(merged) == 2


# ── TTAEnsemble ──

class TestTTAEnsemble:
    def test_known_augmentation_init(self):
        tta = TTAEnsemble(augmentations=["horizontal_flip"])
        assert tta.augmentations == ["horizontal_flip"]

    def test_unknown_augmentation_raises(self):
        with pytest.raises(ValueError, match="Unknown augmentation"):
            TTAEnsemble(augmentations=["fake_aug"])

    def test_unknown_merge_method_raises(self):
        with pytest.raises(ValueError, match="Unknown merge method"):
            TTAEnsemble(merge_method="fake_method")

    def test_predict_with_hflip(self):
        """hflip + original → 좌표 다른 두 검출."""
        def fake(_img):
            return [{"class": "crack", "conf": 0.7, "bbox_xyxy": [10.0, 10.0, 50.0, 50.0]}]
        tta = TTAEnsemble(augmentations=["horizontal_flip"], merge_method="max_conf", include_original=True)
        img = np.ones((100, 200, 3), dtype=np.uint8)
        out = tta.predict(fake, img)
        # 같은 위치 검출인데 hflip 좌표 역변환 → 다른 위치 → 둘 다 유지
        assert len(out) == 2

    def test_predict_with_overlapping_results(self):
        """모든 augmentation 결과가 겹치면 max_conf로 1개만."""
        def fake_with_overlap(img):
            # 항상 이미지 중앙
            h, w = img.shape[:2]
            cx, cy = w // 2, h // 2
            return [{"class": "crack", "conf": 0.7, "bbox_xyxy": [cx - 10.0, cy - 10.0, cx + 10.0, cy + 10.0]}]
        tta = TTAEnsemble(augmentations=["horizontal_flip"], merge_method="max_conf", include_original=True)
        img = np.ones((100, 100, 3), dtype=np.uint8)
        out = tta.predict(fake_with_overlap, img)
        # hflip 후에도 중앙 → 같은 위치 → 1개
        assert len(out) == 1

    def test_predict_classifier_no_bbox_merged(self):
        """bbox 없는 검출(분류기)은 class 단위로 1개만 유지 (max conf)."""
        def fake_classifier(_img):
            return [{"class": "wallpaper_seam", "conf": 0.8, "bbox_xyxy": None}]
        tta = TTAEnsemble(augmentations=["horizontal_flip"], merge_method="max_conf", include_original=True)
        img = np.ones((100, 100, 3), dtype=np.uint8)
        out = tta.predict(fake_classifier, img)
        # 같은 클래스 + bbox 없음 → max conf 1개
        assert len(out) == 1
        assert out[0]["class"] == "wallpaper_seam"
        assert out[0]["conf"] == 0.8

    def test_predict_classifier_different_classes_kept(self):
        """bbox 없는 검출이라도 다른 class면 둘 다 유지."""
        call_count = [0]
        def fake_classifier(_img):
            call_count[0] += 1
            # 첫 호출(original)은 class A, 두 번째(hflip)는 class B
            if call_count[0] == 1:
                return [{"class": "class_a", "conf": 0.8, "bbox_xyxy": None}]
            else:
                return [{"class": "class_b", "conf": 0.7, "bbox_xyxy": None}]
        tta = TTAEnsemble(augmentations=["horizontal_flip"], merge_method="max_conf", include_original=True)
        img = np.ones((100, 100, 3), dtype=np.uint8)
        out = tta.predict(fake_classifier, img)
        assert len(out) == 2
        classes = sorted([d["class"] for d in out])
        assert classes == ["class_a", "class_b"]


# ── config 로딩 ──

class TestConfigLoading:
    def test_disabled_for_returns_none(self):
        cfg = {"enabled_for": ["M4_CONTEXT"], "augmentations": ["horizontal_flip"], "merge_method": "max_conf"}
        # M2_YOLO는 enabled_for에 없음
        tta = load_tta_from_config(cfg, "M2_YOLO")
        assert tta is None

    def test_enabled_for_returns_instance(self):
        cfg = {
            "enabled_for": ["M4_CONTEXT", "M5_SEG"],
            "augmentations": ["horizontal_flip", "scale_0_8"],
            "merge_method": "wbf",
            "iou_merge_threshold": 0.5,
        }
        tta = load_tta_from_config(cfg, "M4_CONTEXT")
        assert tta is not None
        assert tta.merge_method == "wbf"
        assert "horizontal_flip" in tta.augmentations
