# =============================================
# app/services/object_tracker.py
# 역할: IoU 기반 프레임 간 객체 추적 (드론 환경 특화)
#       - 검출 결과에 track_id 부여 (프레임 간 동일 하자 식별)
#       - 일시적 미탐지(블러·흔들림)에도 트랙 유지 (max_age 허용)
#       - 확정 기준: min_hits 이상 탐지된 트랙만 보고
#       - Observation-Centric: 재매칭 시 관측값으로 상태 직접 교정
#       - 신뢰도 누적은 하지 않음 → TemporalFilter에 위임
#
# 순수 IoU 매칭 방식 (Kalman filter 없음):
#   - 드론 환경에서 하자는 이미지 좌표 상 비선형 동적 객체
#   - Kalman의 등속 가정이 풍압/틸트/접근 등에서 부적합
#   - 단순 IoU 매칭이 비선형 환경에서 더 강건
#   - supervision ByteTrack 정적 객체 매칭 불안정 대체
# =============================================

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class TrackedDefect:
    """추적 중인 단일 하자."""
    track_id: int
    class_: str
    bbox_xyxy: List[float]
    conf: float
    hit_count: int = 1           # 탐지된 총 프레임 수
    miss_count: int = 0          # 연속 미탐지 프레임 수
    confirmed: bool = False      # min_hits 충족 여부
    last_frame_id: int = 0       # 마지막 탐지 프레임
    extra: dict = field(default_factory=dict)


def _iou(box_a: List[float], box_b: List[float]) -> float:
    """두 xyxy bbox의 IoU."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    return inter / (area_a + area_b - inter + 1e-6)


class DefectTracker:
    """
    IoU 기반 하자 추적기 (드론 환경 특화).

    프레임마다 YOLO 검출 결과(List[dict])를 받아
    track_id를 부여하고, 확정된 트랙만 반환한다.

    Args:
        min_hits: 트랙 확정에 필요한 최소 탐지 횟수 (기본 3)
        max_age: 미탐지 허용 프레임 수 — 초과 시 트랙 삭제 (기본 15)
        iou_threshold: IoU 매칭 임계값 (기본 0.3)
        camera_fps: 카메라 원본 FPS (기본 30)
        frame_skip: 프레임 스킵 값 (기본 3)
    """

    def __init__(
        self,
        min_hits: int = 3,
        max_age: int = 15,
        iou_threshold: float = 0.3,
        camera_fps: float = 30.0,
        frame_skip: int = 3,
    ):
        self.min_hits = min_hits
        self.max_age = max_age
        self.iou_threshold = iou_threshold
        self.camera_fps = camera_fps
        self.frame_skip = frame_skip

        self._tracks: Dict[int, TrackedDefect] = {}
        self._next_id: int = 1

    @property
    def is_available(self) -> bool:
        """항상 True (외부 라이브러리 의존 없음)."""
        return True

    def update(
        self, detections: List[dict], frame_id: int = 0,
    ) -> List[dict]:
        """
        새 프레임 검출 결과로 트랙 갱신.

        1단계: 기존 트랙과 새 검출 간 IoU 매칭 (greedy)
        2단계: 매칭된 트랙 갱신, 미매칭 검출은 새 트랙 생성
        3단계: 미탐지 트랙 miss_count 증가, max_age 초과 시 삭제
        4단계: 확정된 트랙만 반환

        Returns:
            확정된(confirmed) 트랙의 검출 리스트.
            각 dict에 'track_id', 'hit_count' 필드 추가됨.
        """
        if len(detections) == 0:
            # 검출 없음 → 모든 기존 트랙 miss 처리
            self._age_tracks()
            return []

        # ── 1단계: IoU 매칭 (greedy best-match) ──
        matched_det_idx = set()
        matched_track_ids = set()

        # 모든 (track, detection) 쌍의 IoU 계산
        pairs = []
        for tid, track in self._tracks.items():
            for di, det in enumerate(detections):
                if det.get("class") != track.class_:
                    continue  # 같은 클래스만 매칭
                iou_val = _iou(track.bbox_xyxy, det["bbox_xyxy"])
                if iou_val >= self.iou_threshold:
                    pairs.append((iou_val, tid, di))

        # IoU 내림차순 greedy 매칭
        pairs.sort(key=lambda x: x[0], reverse=True)
        for iou_val, tid, di in pairs:
            if tid in matched_track_ids or di in matched_det_idx:
                continue
            matched_track_ids.add(tid)
            matched_det_idx.add(di)

            # ── 2단계: 매칭된 트랙 갱신 ──
            t = self._tracks[tid]
            det = detections[di]
            t.bbox_xyxy = det["bbox_xyxy"]
            t.conf = det["conf"]
            t.hit_count += 1
            t.miss_count = 0
            t.last_frame_id = frame_id
            t.extra = {
                k: v for k, v in det.items()
                if k not in ("class", "conf", "bbox_xyxy", "class_id")
            }
            if t.hit_count >= self.min_hits:
                t.confirmed = True

        # ── 미매칭 검출 → 새 트랙 생성 ──
        for di, det in enumerate(detections):
            if di in matched_det_idx:
                continue
            tid = self._next_id
            self._next_id += 1
            self._tracks[tid] = TrackedDefect(
                track_id=tid,
                class_=det.get("class", "unknown"),
                bbox_xyxy=det["bbox_xyxy"],
                conf=det["conf"],
                last_frame_id=frame_id,
                extra={
                    k: v for k, v in det.items()
                    if k not in ("class", "conf", "bbox_xyxy", "class_id")
                },
            )

        # ── 3단계: 미탐지 트랙 처리 ──
        self._age_tracks(exclude=matched_track_ids)

        # ── 4단계: 확정된 트랙만 반환 ──
        results = []
        for tid, t in self._tracks.items():
            if t.confirmed:
                results.append({
                    "class": t.class_,
                    "conf": t.conf,
                    "bbox_xyxy": t.bbox_xyxy,
                    "track_id": tid,
                    "hit_count": t.hit_count,
                    **t.extra,
                })

        return results

    def _age_tracks(self, exclude: set = None) -> None:
        """미탐지 트랙의 miss_count 증가 + max_age 초과 삭제."""
        exclude = exclude or set()
        lost = []
        for tid, t in self._tracks.items():
            if tid not in exclude:
                t.miss_count += 1
                if t.miss_count > self.max_age:
                    lost.append(tid)
        for tid in lost:
            del self._tracks[tid]

    def reconfigure(
        self,
        camera_fps: float = 30.0,
        frame_skip: int = 3,
    ) -> None:
        """FRAME_SKIP 또는 카메라 FPS 변경 시 설정 갱신."""
        self.camera_fps = camera_fps
        self.frame_skip = frame_skip

    def reset(self) -> None:
        """트랙 전체 초기화 (새 세션 시작 시)."""
        self._tracks.clear()
        self._next_id = 1

    @property
    def active_track_count(self) -> int:
        return len(self._tracks)

    @property
    def confirmed_track_count(self) -> int:
        return sum(1 for t in self._tracks.values() if t.confirmed)


# ── 모듈 레벨 싱글톤 ─────────────────────────
defect_tracker = DefectTracker()


__all__ = ["DefectTracker", "TrackedDefect", "defect_tracker"]
