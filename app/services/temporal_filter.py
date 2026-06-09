# =============================================
# app/services/temporal_filter.py
# 역할: 프레임 간 검출 추적 + 일시적 오탐 제거
#       - IoU 기반 공간 매칭: 동일 위치·동일 클래스만 집계
#       - Noisy-OR 신뢰도 누적: 반복 탐지 시 conf 상승
#       - 시간 일관성: 연속 N프레임 중 M회 이상 검출 시 보고
#       - 즉시 보고: 고신뢰(conf > threshold) 검출은 즉시
#       - 공간 중복 억제: 동일 LiDAR 좌표 중복 보고 방지
# =============================================

from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# 보고된 위치 기록(_reported)의 최대 보관 개수. 세션 전체 동안 무한정
# 누적되어 메모리/스캔 비용이 커지는 것을 막기 위해 최근 N개만 유지(오래된 항목 제거).
MAX_REPORTED_POSITIONS = 500
# 보고된 위치를 유지하는 시간 창(초). 이보다 오래된 기록은 중복 판정에서 제외.
REPORTED_RETENTION_SEC = 60.0


@dataclass
class BufferedDetection:
    """윈도우 내 버퍼링된 단일 검출."""
    frame_id: int
    timestamp: float
    conf: float
    bbox_xyxy: List[float]
    det: dict  # 원본 검출 dict 전체


@dataclass
class SpatialBucket:
    """IoU 기반 공간 매칭 버킷. 동일 위치의 검출을 모은다."""
    class_: str
    anchor_bbox: List[float]           # 최초 검출 bbox (매칭 기준)
    entries: List[BufferedDetection] = field(default_factory=list)

    @property
    def hit_count(self) -> int:
        return len(self.entries)

    @property
    def best_entry(self) -> BufferedDetection:
        return max(self.entries, key=lambda e: e.conf)

    @property
    def accumulated_conf(self) -> float:
        """Noisy-OR: 1 - Π(1 - conf_i)."""
        product = 1.0
        for e in self.entries:
            product *= (1.0 - e.conf)
        return 1.0 - product


def _iou(box_a: List[float], box_b: List[float]) -> float:
    """두 xyxy bbox의 IoU 계산."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    return inter / (area_a + area_b - inter + 1e-6)


class TemporalFilter:
    """
    스트리밍 환경에서 시간 일관성 + 공간 일관성 기반 필터링.

    규칙:
    - IoU 기반 공간 매칭으로 동일 위치·동일 클래스만 집계
    - 연속 window_size 프레임 중 min_detections 이상 검출 시 보고
    - Noisy-OR로 누적 신뢰도 계산 (반복 탐지 → conf 상승)
    - 단일 프레임이라도 conf > instant_threshold면 즉시 보고
    - 동일 3D 위치(LiDAR) 중복 보고 억제
    """

    def __init__(
        self,
        window_size: int = 5,
        min_detections: int = 2,
        instant_threshold: float = 0.85,
        spatial_dedup_radius: float = 0.3,
        iou_threshold: float = 0.3,
        window_time_sec: float = 2.0,
    ):
        self.window_size = window_size
        self.min_detections = min_detections
        self.instant_threshold = instant_threshold
        self.spatial_dedup_radius = spatial_dedup_radius
        self.iou_threshold = iou_threshold
        self.window_time_sec = window_time_sec

        # class → [SpatialBucket, ...] — 클래스별 공간 버킷
        self._buckets: Dict[str, List[SpatialBucket]] = defaultdict(list)
        # 보고된 하자 위치: [(class, x, y, z, timestamp)]
        # 무한 누적/선형 스캔 비용 방지를 위해 최근 MAX_REPORTED_POSITIONS개만 유지(오래된 항목 자동 제거).
        self._reported: deque = deque(maxlen=MAX_REPORTED_POSITIONS)

    # ── 공개 API ─────────────────────────────
    def update(
        self,
        detections: List[dict],
        frame_id: int,
        lidar_pos: Optional[dict] = None,
    ) -> List[dict]:
        """
        새 프레임 검출 결과 → IoU 매칭 → 투표 → 보고 대상만 반환.

        Args:
            detections: [{class, conf, bbox_xyxy, ...}]
            frame_id: 현재 비디오 프레임 번호
            lidar_pos: {x, y, z} 드론 LiDAR 3D 좌표 (선택)

        Returns:
            보고할 검출 리스트. 각 dict에 'accumulated_conf' 필드 추가됨.
        """
        now = time.time()
        approved: List[dict] = []

        # 윈도우 밖 오래된 항목 제거 (시간 기반)
        self._prune_stale(frame_id, now)

        for det in detections:
            cls = det["class"]
            bbox = det.get("bbox_xyxy")
            conf = det["conf"]

            # bbox 없는 검출(예: 벽지 분류)은 클래스명으로만 매칭
            if bbox is None:
                if conf >= self.instant_threshold:
                    if not self._is_spatial_duplicate(cls, lidar_pos):
                        approved.append(det)
                        self._record_position(cls, lidar_pos)
                continue

            # 즉시 보고 (고신뢰)
            if conf >= self.instant_threshold:
                if not self._is_spatial_duplicate(cls, lidar_pos):
                    result = dict(det)
                    result["accumulated_conf"] = conf
                    approved.append(result)
                    self._record_position(cls, lidar_pos)
                # 버킷에도 추가 (이후 프레임 매칭용)
                self._add_to_bucket(cls, bbox, frame_id, now, conf, det)
                continue

            # IoU 기반 공간 매칭 → 버킷에 추가
            bucket = self._find_or_create_bucket(cls, bbox, frame_id, now, conf, det)

            # 윈도우 내 충분한 검출 → 보고
            if bucket.hit_count >= self.min_detections:
                if not self._is_spatial_duplicate(cls, lidar_pos):
                    best = bucket.best_entry
                    result = dict(best.det)
                    result["conf"] = best.conf
                    result["accumulated_conf"] = round(bucket.accumulated_conf, 4)
                    result["temporal_hits"] = bucket.hit_count
                    approved.append(result)
                    self._record_position(cls, lidar_pos)
                # 버킷 초기화 (중복 보고 방지)
                self._buckets[cls] = [
                    b for b in self._buckets[cls] if b is not bucket
                ]

        return approved

    def reset(self):
        """필터 상태 초기화 (새 세션 시작 시)."""
        self._buckets.clear()
        self._reported.clear()

    # ── 내부: 공간 매칭 ──────────────────────────

    def _find_or_create_bucket(
        self,
        cls: str,
        bbox: List[float],
        frame_id: int,
        timestamp: float,
        conf: float,
        det: dict,
    ) -> SpatialBucket:
        """기존 버킷에 IoU 매칭하거나, 새 버킷 생성."""
        best_bucket = None
        best_iou = 0.0

        for bucket in self._buckets[cls]:
            iou_val = _iou(bbox, bucket.anchor_bbox)
            if iou_val > best_iou:
                best_iou = iou_val
                best_bucket = bucket

        if best_iou >= self.iou_threshold and best_bucket is not None:
            best_bucket.entries.append(BufferedDetection(
                frame_id=frame_id, timestamp=timestamp,
                conf=conf, bbox_xyxy=bbox, det=det,
            ))
            # anchor를 최신 bbox로 갱신 (카메라 이동 추적)
            best_bucket.anchor_bbox = bbox
            return best_bucket

        # 새 버킷
        new_bucket = SpatialBucket(
            class_=cls,
            anchor_bbox=bbox,
            entries=[BufferedDetection(
                frame_id=frame_id, timestamp=timestamp,
                conf=conf, bbox_xyxy=bbox, det=det,
            )],
        )
        self._buckets[cls].append(new_bucket)
        return new_bucket

    def _add_to_bucket(
        self,
        cls: str,
        bbox: List[float],
        frame_id: int,
        timestamp: float,
        conf: float,
        det: dict,
    ) -> None:
        """즉시 보고 검출도 버킷에 기록 (이후 프레임 매칭용)."""
        self._find_or_create_bucket(cls, bbox, frame_id, timestamp, conf, det)

    def _prune_stale(self, frame_id: int, now: float) -> None:
        """윈도우 밖 오래된 항목 제거. frame_id 기반 + 시간 기반 이중 기준."""
        for cls in list(self._buckets.keys()):
            for bucket in self._buckets[cls]:
                bucket.entries = [
                    e for e in bucket.entries
                    if (frame_id - e.frame_id) < self.window_size
                    and (now - e.timestamp) < self.window_time_sec
                ]
            # 빈 버킷 정리
            self._buckets[cls] = [
                b for b in self._buckets[cls] if len(b.entries) > 0
            ]
            if not self._buckets[cls]:
                del self._buckets[cls]

    # ── 내부: LiDAR 공간 중복 ────────────────────

    def _is_spatial_duplicate(self, cls: str, pos: Optional[dict]) -> bool:
        if pos is None:
            return False
        px = pos.get("x", 0.0)
        py = pos.get("y", 0.0)
        pz = pos.get("z", 0.0)
        # 시간 창 밖(오래된) 기록은 중복 판정에서 제외.
        cutoff = time.time() - REPORTED_RETENTION_SEC
        for rcls, rx, ry, rz, rts in self._reported:
            if rts < cutoff:
                continue
            if rcls != cls:
                continue
            dist = ((px - rx) ** 2 + (py - ry) ** 2 + (pz - rz) ** 2) ** 0.5
            if dist < self.spatial_dedup_radius:
                return True
        return False

    def _record_position(self, cls: str, pos: Optional[dict]):
        if pos is not None:
            # deque(maxlen) 가 초과분(가장 오래된 항목)을 자동 제거 → 메모리 상한 보장.
            self._reported.append(
                (cls, pos.get("x", 0.0), pos.get("y", 0.0), pos.get("z", 0.0), time.time())
            )
