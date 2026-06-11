# =============================================
# app/services/room_segmenter.py
# 역할: occupancy 2D 맵 → 룸 토폴로지 그래프
#
# 알고리즘 (업계 검사용 SLAM 점유격자 분리에 광범위하게 쓰이는 패턴):
#   1) 자유공간(free=255, occupied/unknown=0) 이진화
#   2) erode (drone_radius_m / resolution_m_per_px) → 좁은 통로 끊기
#   3) connectedComponentsWithStats → 룸 후보 (면적 필터)
#   4) 각 룸 컴포넌트 외곽선 → 폴리곤(simplify ε=2px)
#   5) 도어웨이: (원본 자유공간) - (erode된 자유공간) 차분영역 → CC →
#       각 차분 영역의 distanceTransform 평균 × 2 = 폭(m). 0.7~1.0m 필터
#       두 룸 컴포넌트와 모두 인접하는 차분영역만 정식 도어웨이로 채택
#   6) networkx 그래프 (옵셔널, 미설치 시 dict 폴백)
#
# 좌표계: occupancy 픽셀 (col, row) → 월드 (x, y) = (col*res, row*res). origin offset 인자.
# =============================================
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import cv2
import numpy as np

from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RoomNode:
    idx: int
    name: str
    polygon: List[Tuple[float, float]] = field(default_factory=list)
    area_m2: float = 0.0
    centroid: Tuple[float, float] = (0.0, 0.0)


@dataclass
class DoorwayEdge:
    src: int
    dst: int
    center: Tuple[float, float]
    width_m: float
    approach_yaw_rad: float


@dataclass
class RoomTopologyGraph:
    nodes: List[RoomNode] = field(default_factory=list)
    edges: List[DoorwayEdge] = field(default_factory=list)

    def to_jsonable(self) -> dict:
        return {
            "nodes": [
                {
                    "idx": n.idx, "name": n.name,
                    "polygon": [list(p) for p in n.polygon],
                    "area": n.area_m2,
                    "centroid": list(n.centroid),
                }
                for n in self.nodes
            ],
            "edges": [
                {
                    "from": e.src, "to": e.dst,
                    "doorway_center": list(e.center),
                    "doorway_width": e.width_m,
                    "approach_yaw": e.approach_yaw_rad,
                }
                for e in self.edges
            ],
        }


@dataclass
class SegmenterParams:
    resolution_m_per_px: float = 0.05
    drone_radius_m: float = 0.185
    doorway_min_m: float = 0.70
    doorway_max_m: float = 1.00
    min_room_area_m2: float = 1.5
    poly_simplify_px: float = 2.0
    origin_xy: Tuple[float, float] = (0.0, 0.0)   # 월드 좌표 원점 (occupancy[0,0]이 가리키는 월드 위치)


class RoomSegmenter:
    def __init__(self, params: SegmenterParams | None = None) -> None:
        self.params = params or SegmenterParams()

    # ── 좌표 변환 ─────────────────────────
    def _px_to_world(self, col: float, row: float) -> Tuple[float, float]:
        res = self.params.resolution_m_per_px
        ox, oy = self.params.origin_xy
        return (col * res + ox, row * res + oy)

    # ── 메인 ──────────────────────────────
    def segment(self, occupancy_grid: np.ndarray) -> RoomTopologyGraph:
        """
        Args:
            occupancy_grid: 2D ndarray (0=free, 1=occupied, -1=unknown).
                            shape=(H, W), int8/uint8 호환.
        """
        if occupancy_grid is None or occupancy_grid.size == 0:
            logger.warning("room_segmenter.empty_occupancy")
            return RoomTopologyGraph()

        # 이진화 — 자유공간만 255
        free = np.where(occupancy_grid == 0, 255, 0).astype(np.uint8)

        # erosion 커널: 드론 안전반경 픽셀 (홀수 강제)
        radius_px = max(1, int(round(self.params.drone_radius_m / self.params.resolution_m_per_px)))
        ksize = 2 * radius_px + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        eroded = cv2.erode(free, kernel)

        # 연결성분 — 자유공간 룸 후보
        n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(eroded, connectivity=8)
        nodes: List[RoomNode] = []
        room_label_to_idx: Dict[int, int] = {}
        idx_counter = 0
        min_room_px = max(
            1,
            int(round(self.params.min_room_area_m2 / (self.params.resolution_m_per_px ** 2))),
        )

        for lbl in range(1, n_labels):  # 0 = 배경
            area_px = int(stats[lbl, cv2.CC_STAT_AREA])
            if area_px < min_room_px:
                continue
            mask = (labels == lbl).astype(np.uint8) * 255
            polygon = self._extract_polygon(mask)
            if len(polygon) < 3:
                continue
            cx_px, cy_px = centroids[lbl]
            world_polygon = [self._px_to_world(p[0], p[1]) for p in polygon]
            world_centroid = self._px_to_world(cx_px, cy_px)
            area_m2 = area_px * (self.params.resolution_m_per_px ** 2)
            node = RoomNode(
                idx=idx_counter,
                name=f"room_{idx_counter}",
                polygon=world_polygon,
                area_m2=area_m2,
                centroid=world_centroid,
            )
            nodes.append(node)
            room_label_to_idx[lbl] = idx_counter
            idx_counter += 1

        # 도어웨이 — 차분영역(원본 free - eroded free)
        diff = cv2.subtract(free, eroded)
        n_diff_labels, diff_labels, diff_stats, _ = cv2.connectedComponentsWithStats(diff, connectivity=8)

        # 거리변환 — 자유공간 내부에서 가장 가까운 occupied 까지의 거리
        dist = cv2.distanceTransform(free, cv2.DIST_L2, 5)

        edges: List[DoorwayEdge] = []
        seen_pairs: set = set()
        for dl in range(1, n_diff_labels):
            diff_mask = (diff_labels == dl).astype(np.uint8)
            if diff_mask.sum() < 4:
                continue

            # 이 차분영역과 인접한 룸 라벨 찾기
            dilated = cv2.dilate(diff_mask, np.ones((3, 3), np.uint8))
            adjacent_room_labels = np.unique(labels[(dilated == 1) & (labels > 0)])
            adjacent_room_labels = [int(l) for l in adjacent_room_labels if int(l) in room_label_to_idx]
            if len(adjacent_room_labels) < 2:
                continue

            # 폭(m) = 차분영역의 distanceTransform 최대값 × 2
            d_in_diff = dist[diff_mask == 1]
            if d_in_diff.size == 0:
                continue
            width_m = float(d_in_diff.max()) * 2.0 * self.params.resolution_m_per_px
            if not (self.params.doorway_min_m <= width_m <= self.params.doorway_max_m):
                continue

            # 도어웨이 중심: 차분영역 중심 픽셀
            ys, xs = np.where(diff_mask == 1)
            cx, cy = float(xs.mean()), float(ys.mean())
            world_center = self._px_to_world(cx, cy)

            # 인접 룸 쌍별 엣지 생성
            for i in range(len(adjacent_room_labels)):
                for j in range(i + 1, len(adjacent_room_labels)):
                    a = room_label_to_idx[adjacent_room_labels[i]]
                    b = room_label_to_idx[adjacent_room_labels[j]]
                    if a == b:
                        continue
                    pair = (min(a, b), max(a, b))
                    if pair in seen_pairs:
                        continue
                    seen_pairs.add(pair)
                    # 통과 yaw — src centroid → dst centroid
                    sc = nodes[a].centroid
                    dc = nodes[b].centroid
                    yaw = float(np.arctan2(dc[1] - sc[1], dc[0] - sc[0]))
                    edges.append(DoorwayEdge(
                        src=a, dst=b,
                        center=world_center,
                        width_m=round(width_m, 3),
                        approach_yaw_rad=yaw,
                    ))

        logger.info(
            "room_segmenter.done",
            rooms=len(nodes), doorways=len(edges),
            occupancy_shape=tuple(occupancy_grid.shape),
        )
        return RoomTopologyGraph(nodes=nodes, edges=edges)

    # ── 보조: 마스크 → 폴리곤 ──────────────
    def _extract_polygon(self, mask: np.ndarray) -> List[Tuple[float, float]]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return []
        # 가장 큰 외곽선
        cnt = max(contours, key=cv2.contourArea)
        if len(cnt) < 3:
            return []
        eps = self.params.poly_simplify_px
        approx = cv2.approxPolyDP(cnt, eps, closed=True)
        pts = approx.reshape(-1, 2)
        return [(float(p[0]), float(p[1])) for p in pts]
