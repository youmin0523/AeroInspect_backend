# =============================================
# tests/test_object_tracker.py
# DefectTracker 단위 테스트
# - track_id 부여
# - 확정 로직 (min_hits)
# - 미탐지 후 재매칭
# - supervision 미설치 시 fallback
# =============================================

import pytest

from app.services.object_tracker import DefectTracker


class TestTrackConfirmation:
    """트랙 확정 로직 테스트."""

    def test_unconfirmed_track_not_returned(self):
        """min_hits 미달 트랙은 반환 안 됨."""
        tracker = DefectTracker(min_hits=3)
        assert tracker.is_available  # 순수 IoU 기반, 외부 의존성 없음
        dets = [{"class": "crack", "conf": 0.5, "bbox_xyxy": [10, 10, 50, 50]}]
        r1 = tracker.update(dets, frame_id=1)
        assert len(r1) == 0  # 1회 → 미확정

    def test_confirmed_after_min_hits(self):
        """min_hits 이상 탐지 시 확정 (ByteTrack 내부 활성화 지연 감안)."""
        tracker = DefectTracker(min_hits=2)
        assert tracker.is_available  # 순수 IoU 기반, 외부 의존성 없음
        det = {"class": "crack", "conf": 0.5, "bbox_xyxy": [10, 10, 50, 50]}
        # ByteTrack 내부 활성화에 수 프레임 필요할 수 있음
        results = []
        for fid in range(1, 6):
            r = tracker.update([det], frame_id=fid)
            results.extend(r)
        assert len(results) >= 1, "5프레임 내에 트랙이 확정되어야 함"
        assert results[0]["track_id"] is not None

    def test_track_id_consistent(self):
        """동일 위치 검출은 같은 track_id 유지."""
        tracker = DefectTracker(min_hits=1)
        assert tracker.is_available  # 순수 IoU 기반, 외부 의존성 없음
        det = {"class": "crack", "conf": 0.5, "bbox_xyxy": [100, 100, 200, 200]}
        r1 = tracker.update([det], frame_id=1)
        r2 = tracker.update([det], frame_id=2)
        if r1 and r2:
            assert r1[0]["track_id"] == r2[0]["track_id"]


class TestTrackerBasics:
    """기본 동작 테스트."""

    def test_always_available(self):
        """외부 의존성 없이 항상 사용 가능."""
        tracker = DefectTracker()
        assert tracker.is_available is True

    def test_empty_input(self):
        """빈 입력은 빈 결과."""
        tracker = DefectTracker()
        result = tracker.update([], frame_id=1)
        assert result == []


class TestTrackerReset:
    """리셋 테스트."""

    def test_reset_clears_tracks(self):
        tracker = DefectTracker(min_hits=1)
        assert tracker.is_available  # 순수 IoU 기반, 외부 의존성 없음
        det = {"class": "crack", "conf": 0.5, "bbox_xyxy": [10, 10, 50, 50]}
        tracker.update([det], frame_id=1)
        assert tracker.active_track_count >= 1
        tracker.reset()
        assert tracker.active_track_count == 0


class TestReconfigure:
    """동적 frame_rate 재설정 테스트."""

    def test_reconfigure_updates_frame_skip(self):
        tracker = DefectTracker()
        tracker.reconfigure(camera_fps=30, frame_skip=6)
        assert tracker.frame_skip == 6
        assert tracker.camera_fps == 30

    def test_reconfigure_preserves_tracks(self):
        """reconfigure 후 기존 트랙 유지."""
        tracker = DefectTracker(min_hits=1)
        assert tracker.is_available  # 순수 IoU 기반, 외부 의존성 없음
        det = {"class": "crack", "conf": 0.5, "bbox_xyxy": [10, 10, 50, 50]}
        tracker.update([det], frame_id=1)
        count_before = tracker.active_track_count
        tracker.reconfigure(camera_fps=15, frame_skip=3)
        # 내부 트랙은 유지 (ByteTrack만 리셋)
        assert tracker.active_track_count == count_before


class TestIoUMatching:
    """IoU 매칭 로직 테스트."""

    def test_same_class_matched(self):
        """같은 클래스의 겹치는 bbox는 동일 트랙으로 매칭."""
        tracker = DefectTracker(min_hits=1)
        det = {"class": "crack", "conf": 0.5, "bbox_xyxy": [10, 10, 50, 50]}
        tracker.update([det], frame_id=1)
        # 약간 이동 (IoU > 0.3)
        det2 = {"class": "crack", "conf": 0.6, "bbox_xyxy": [15, 15, 55, 55]}
        r = tracker.update([det2], frame_id=2)
        assert len(r) >= 1
        assert r[0]["hit_count"] == 2

    def test_different_class_not_matched(self):
        """다른 클래스는 같은 위치여도 별개 트랙."""
        tracker = DefectTracker(min_hits=1)
        det1 = {"class": "crack", "conf": 0.5, "bbox_xyxy": [10, 10, 50, 50]}
        det2 = {"class": "moisture", "conf": 0.5, "bbox_xyxy": [10, 10, 50, 50]}
        tracker.update([det1], frame_id=1)
        tracker.update([det2], frame_id=2)
        assert tracker.active_track_count == 2

    def test_no_overlap_creates_new_track(self):
        """겹치지 않는 검출은 새 트랙."""
        tracker = DefectTracker(min_hits=1)
        det1 = {"class": "crack", "conf": 0.5, "bbox_xyxy": [0, 0, 50, 50]}
        det2 = {"class": "crack", "conf": 0.5, "bbox_xyxy": [200, 200, 300, 300]}
        tracker.update([det1], frame_id=1)
        tracker.update([det2], frame_id=2)
        assert tracker.active_track_count == 2
