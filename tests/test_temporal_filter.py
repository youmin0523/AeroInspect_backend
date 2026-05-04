# =============================================
# tests/test_temporal_filter.py
# TemporalFilter 단위 테스트
# - IoU SpatialBucket 매칭
# - Noisy-OR 누적 신뢰도
# - 투표 로직 (min_detections)
# - 즉시 보고 (instant_threshold)
# - LiDAR 공간 중복 억제
# =============================================

import pytest

from app.services.temporal_filter import TemporalFilter, _iou


class TestIoU:
    """IoU 계산 유틸 테스트."""

    def test_identical_boxes(self):
        assert _iou([0, 0, 100, 100], [0, 0, 100, 100]) == pytest.approx(1.0, abs=1e-4)

    def test_no_overlap(self):
        assert _iou([0, 0, 50, 50], [100, 100, 200, 200]) == pytest.approx(0.0, abs=1e-4)

    def test_partial_overlap(self):
        # 50x50 box, 25x25 overlap
        iou = _iou([0, 0, 50, 50], [25, 25, 75, 75])
        expected = 625.0 / (2500 + 2500 - 625)  # ~0.143
        assert iou == pytest.approx(expected, abs=1e-3)

    def test_contained_box(self):
        iou = _iou([0, 0, 100, 100], [25, 25, 75, 75])
        expected = 2500.0 / (10000 + 2500 - 2500)  # 0.25
        assert iou == pytest.approx(expected, abs=1e-3)


class TestTemporalFilterVoting:
    """투표 로직 테스트."""

    def setup_method(self):
        self.tf = TemporalFilter(
            window_size=5,
            min_detections=2,
            instant_threshold=0.85,
            iou_threshold=0.3,
        )

    def test_single_detection_not_reported(self):
        """단일 프레임 저신뢰 검출은 보고되지 않음."""
        dets = [{"class": "crack", "conf": 0.5, "bbox_xyxy": [10, 10, 50, 50]}]
        result = self.tf.update(dets, frame_id=1)
        assert len(result) == 0

    def test_two_detections_same_location_reported(self):
        """동일 위치에서 2프레임 탐지 → 보고됨."""
        det = {"class": "crack", "conf": 0.5, "bbox_xyxy": [10, 10, 50, 50]}
        self.tf.update([det], frame_id=1)
        result = self.tf.update([det], frame_id=2)
        assert len(result) == 1
        assert result[0]["class"] == "crack"

    def test_different_location_not_merged(self):
        """다른 위치의 동일 클래스 검출은 별개 취급."""
        det1 = {"class": "crack", "conf": 0.5, "bbox_xyxy": [0, 0, 50, 50]}
        det2 = {"class": "crack", "conf": 0.5, "bbox_xyxy": [200, 200, 300, 300]}
        self.tf.update([det1], frame_id=1)
        result = self.tf.update([det2], frame_id=2)
        # 위치가 다르므로 투표 통과 안 됨
        assert len(result) == 0

    def test_window_expiry(self):
        """윈도우 밖 오래된 검출은 만료됨."""
        det = {"class": "crack", "conf": 0.5, "bbox_xyxy": [10, 10, 50, 50]}
        self.tf.update([det], frame_id=1)
        # frame_id=7 → frame_id=1은 윈도우(5) 밖
        result = self.tf.update([det], frame_id=7)
        assert len(result) == 0  # 첫 검출이 만료되어 1개만 있음


class TestInstantReport:
    """고신뢰 즉시 보고 테스트."""

    def setup_method(self):
        self.tf = TemporalFilter(instant_threshold=0.85)

    def test_high_confidence_instant(self):
        """conf >= 0.85는 즉시 보고."""
        dets = [{"class": "crack", "conf": 0.90, "bbox_xyxy": [10, 10, 50, 50]}]
        result = self.tf.update(dets, frame_id=1)
        assert len(result) == 1
        assert result[0]["accumulated_conf"] == 0.90

    def test_below_threshold_not_instant(self):
        """conf < 0.85는 즉시 보고 안 됨."""
        dets = [{"class": "crack", "conf": 0.84, "bbox_xyxy": [10, 10, 50, 50]}]
        result = self.tf.update(dets, frame_id=1)
        assert len(result) == 0


class TestNoisyOR:
    """Noisy-OR 누적 신뢰도 테스트."""

    def setup_method(self):
        self.tf = TemporalFilter(
            window_size=5,
            min_detections=2,
            instant_threshold=0.99,  # 즉시 보고 비활성화
        )

    def test_accumulated_conf_increases(self):
        """반복 탐지 시 누적 conf가 개별 conf보다 높음."""
        det = {"class": "crack", "conf": 0.4, "bbox_xyxy": [10, 10, 50, 50]}
        self.tf.update([det], frame_id=1)
        result = self.tf.update([det], frame_id=2)
        assert len(result) == 1
        # Noisy-OR: 1 - (1-0.4)*(1-0.4) = 1 - 0.36 = 0.64
        assert result[0]["accumulated_conf"] == pytest.approx(0.64, abs=0.01)


class TestSpatialDedup:
    """LiDAR 공간 중복 억제 테스트."""

    def setup_method(self):
        self.tf = TemporalFilter(
            min_detections=1,
            instant_threshold=0.5,
            spatial_dedup_radius=0.3,
        )

    def test_duplicate_lidar_suppressed(self):
        """같은 LiDAR 좌표에서 반복 보고 차단."""
        det = {"class": "crack", "conf": 0.9, "bbox_xyxy": [10, 10, 50, 50]}
        pos = {"x": 1.0, "y": 2.0, "z": 3.0}
        r1 = self.tf.update([det], frame_id=1, lidar_pos=pos)
        r2 = self.tf.update([det], frame_id=2, lidar_pos=pos)
        assert len(r1) == 1
        assert len(r2) == 0  # 중복 차단

    def test_different_location_not_suppressed(self):
        """다른 LiDAR 좌표는 별개."""
        det = {"class": "crack", "conf": 0.9, "bbox_xyxy": [10, 10, 50, 50]}
        r1 = self.tf.update([det], frame_id=1, lidar_pos={"x": 0, "y": 0, "z": 0})
        r2 = self.tf.update([det], frame_id=2, lidar_pos={"x": 10, "y": 10, "z": 10})
        assert len(r1) == 1
        assert len(r2) == 1


class TestReset:
    """리셋 테스트."""

    def test_reset_clears_state(self):
        tf = TemporalFilter(min_detections=2, instant_threshold=0.99)
        det = {"class": "crack", "conf": 0.5, "bbox_xyxy": [10, 10, 50, 50]}
        tf.update([det], frame_id=1)
        tf.reset()
        result = tf.update([det], frame_id=2)
        assert len(result) == 0  # 리셋 후 첫 프레임이므로 1개만
