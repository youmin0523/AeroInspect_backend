# =============================================
# app/services/path_planner.py
# 역할: 3D 보스트로페돈(왕복) 그리드 경로 생성기
#       - 룸 폴리곤(occupancy 분리 결과) + 카메라 FoV → 셀 spacing 자동 산출
#       - 수직 3 레이어(천장 2.4m / 중층 1.5m / 바닥 0.6m, 사용자 설정 가능)
#       - VERIFICATION 단계 차이영역(discrepancies)은 spacing 1/2 가중하여 더 촘촘 커버
#       - 출력: 룸별 WP 리스트 + 셀 메타(coverage_grids 시드용)
#
# 설계 결정:
#   - 폴리곤은 shapely Polygon 사용 (라인-폴리곤 교집합, 면적, 클리핑 1차원 검증)
#     shapely 미설치 환경 대비 경량 fallback 보유 (직사각형 bounding box 근사)
#   - PCA: numpy SVD 직접 (scipy 미의존)
#   - WP yaw: 다음 WP 방향으로 자동 정렬
# =============================================
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from app.core.logging import get_logger

logger = get_logger(__name__)

# ── shapely 옵셔널 import ──────────────────
try:
    from shapely.geometry import LineString, Point, Polygon
    _SHAPELY_OK = True
except ImportError:  # pragma: no cover
    _SHAPELY_OK = False
    LineString = Point = Polygon = None   # type: ignore[assignment]


# ── 타입 ────────────────────────────────────
Polygon2D = Sequence[Tuple[float, float]]


@dataclass
class PlanParams:
    fov_h_deg: float = 80.0
    d_inspect_m: float = 1.5
    overlap: float = 0.30
    # 룸 수직 정보 — 천장/바닥 z 좌표(미터)
    floor_z_m: float = 0.0
    ceiling_z_m: float = 2.4
    # 벽 그리드 시 수직 라인 spacing 산출에 쓰는 천장-바닥 사이 z 라인 수 (자동 계산도 가능)
    wall_z_overlap: float = 0.30
    speed_room_mps: float = 0.5
    speed_doorway_mps: float = 0.25
    speed_window_mps: float = 0.25     # 창호 정밀 스캔은 더 천천히
    discrepancy_density_factor: float = 0.5
    window_density_factor: float = 0.5  # 창호 spacing 배수(촘촘)
    edge_margin_m: float = 0.30
    # 면별 스캔 토글
    scan_walls: bool = True
    scan_ceiling: bool = True
    scan_floor: bool = True
    scan_windows: bool = True
    # 천장/바닥 d_inspect 보정(좁은 실내 안전을 위해 약간 작게)
    ceiling_clearance_m: float = 0.40   # 천장에서 떨어진 거리 (드론 위쪽 안전)
    floor_clearance_m: float = 0.40     # 바닥에서 떨어진 거리


# 면 종류 — 카메라가 어느 방향을 향해 캡처해야 하는지
FACE_WALL = "wall"          # 벽 (yaw 자동 정렬, 면 normal 의 반대 방향)
FACE_CEILING = "ceiling"    # 천장 (드론 nose-up tilt 메타. fixed-forward FPV 한계는 카메라 수준에서 흡수)
FACE_FLOOR = "floor"        # 바닥 (드론 nose-down tilt 메타)
FACE_WINDOW = "window"      # 창호 (벽의 정밀 부분집합)


@dataclass
class Waypoint:
    x: float
    y: float
    z: float
    yaw_rad: float = 0.0
    speed_mps: float = 0.5
    cell_idx: Tuple[int, int, int] = (0, 0, 0)
    purpose: str = "coverage"   # coverage | doorway_approach | transition | discrepancy
    in_discrepancy: bool = False
    # 4면 정밀 스캔 메타 — 카메라/드론 자세 정렬에 사용
    face_kind: str = FACE_WALL  # FACE_WALL | FACE_CEILING | FACE_FLOOR | FACE_WINDOW
    face_idx: int = 0           # 같은 룸 내 면 인덱스(벽 0..N-1, 천장=0, 바닥=0, 창호 0..M-1)
    cam_pitch_rad: float = 0.0  # 카메라 틸트 메타 (천장 +pi/2, 바닥 -pi/2 가정. fixed-forward 면 0)


@dataclass
class RoomPlan:
    room_idx: int
    waypoints: List[Waypoint] = field(default_factory=list)
    spacing_m: float = 0.0
    cell_count: int = 0


@dataclass
class MissionGridPlan:
    rooms: Dict[int, RoomPlan] = field(default_factory=dict)
    params: PlanParams = field(default_factory=PlanParams)


# ── 그리드 spacing 산식 ─────────────────────
def grid_spacing(params: PlanParams) -> float:
    fov_rad = math.radians(params.fov_h_deg)
    s = 2.0 * params.d_inspect_m * math.tan(fov_rad / 2.0) * (1.0 - params.overlap)
    return max(s, 0.4)   # 최소 spacing 가드 (너무 좁으면 비행시간 폭증)


# ── PCA 1축 (numpy SVD) ────────────────────
def _pca_axis(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Args:
        points: (N, 2) 폴리곤 정점
    Returns:
        (centroid (2,), axis_u (2,)) — axis_u 는 분산 큰 방향 단위벡터.
    """
    centroid = points.mean(axis=0)
    centered = points - centroid
    # SVD: V[0] 이 최대 분산 방향
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    axis_u = vt[0]
    # 부호 정규화 (deterministic)
    if axis_u[0] < 0 or (axis_u[0] == 0 and axis_u[1] < 0):
        axis_u = -axis_u
    return centroid, axis_u


def _rotate_to_axis(points: np.ndarray, centroid: np.ndarray, axis_u: np.ndarray) -> Tuple[np.ndarray, float]:
    """폴리곤을 axis_u 가 +x 가 되도록 회전(중심 평행이동 포함). 회전각 yaw 반환."""
    yaw = math.atan2(axis_u[1], axis_u[0])
    c, s = math.cos(-yaw), math.sin(-yaw)
    R = np.array([[c, -s], [s, c]], dtype=np.float64)
    return (points - centroid) @ R.T, yaw


def _unrotate(point: Tuple[float, float], centroid: np.ndarray, yaw: float) -> Tuple[float, float]:
    c, s = math.cos(yaw), math.sin(yaw)
    R = np.array([[c, -s], [s, c]], dtype=np.float64)
    p = np.array(point, dtype=np.float64) @ R.T + centroid
    return float(p[0]), float(p[1])


# ── 라인-폴리곤 교집합 ──────────────────────
def _line_polygon_segments(
    polygon_xy: np.ndarray, y_const: float, x_min: float, x_max: float
) -> List[Tuple[float, float]]:
    """
    수평선 y=y_const 가 폴리곤과 만드는 각 교차구간 [(x_lo, x_hi), ...] 반환.
    - shapely 가용 시: LineString ∩ Polygon 직접
    - 미가용 시: Even-odd ray casting + 정렬
    """
    if _SHAPELY_OK:
        try:
            poly = Polygon(polygon_xy.tolist())
            line = LineString([(x_min - 1.0, y_const), (x_max + 1.0, y_const)])
            inter = poly.intersection(line)
            if inter.is_empty:
                return []
            segments: List[Tuple[float, float]] = []
            geoms = getattr(inter, "geoms", [inter])
            for g in geoms:
                if g.geom_type != "LineString":
                    continue
                xs = [pt[0] for pt in g.coords]
                if len(xs) >= 2:
                    segments.append((min(xs), max(xs)))
            return segments
        except Exception:  # pragma: no cover
            pass
    # Fallback — ray casting on edges
    xs: List[float] = []
    n = len(polygon_xy)
    for i in range(n):
        x1, y1 = polygon_xy[i]
        x2, y2 = polygon_xy[(i + 1) % n]
        if (y1 <= y_const < y2) or (y2 <= y_const < y1):
            t = (y_const - y1) / (y2 - y1) if y2 != y1 else 0.0
            xs.append(x1 + t * (x2 - x1))
    xs.sort()
    return [(xs[i], xs[i + 1]) for i in range(0, len(xs) - 1, 2)]


def _shrink_polygon(polygon_xy: np.ndarray, margin_m: float) -> np.ndarray:
    """폴리곤 안쪽 마진 적용. shapely 가용 시 buffer(-margin)."""
    if _SHAPELY_OK and margin_m > 0:
        try:
            poly = Polygon(polygon_xy.tolist())
            shrunk = poly.buffer(-margin_m, join_style=2)  # mitre
            if shrunk.is_empty:
                return polygon_xy
            if shrunk.geom_type == "Polygon":
                return np.array(list(shrunk.exterior.coords)[:-1], dtype=np.float64)
            # MultiPolygon — 가장 큰 컴포넌트 사용
            biggest = max(shrunk.geoms, key=lambda g: g.area)
            return np.array(list(biggest.exterior.coords)[:-1], dtype=np.float64)
        except Exception:  # pragma: no cover
            pass
    return polygon_xy


# ── 차이영역 마스크 ─────────────────────────
def _is_in_discrepancy(
    point_xy: Tuple[float, float], discrepancy_polygons: List[Polygon2D]
) -> bool:
    if not discrepancy_polygons:
        return False
    if _SHAPELY_OK:
        try:
            p = Point(point_xy)
            return any(Polygon(poly).contains(p) for poly in discrepancy_polygons)
        except Exception:  # pragma: no cover
            return False
    # Fallback ray casting
    x, y = point_xy
    for poly in discrepancy_polygons:
        n = len(poly)
        inside = False
        for i in range(n):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % n]
            if (y1 > y) != (y2 > y):
                xinters = (y - y1) * (x2 - x1) / (y2 - y1 + 1e-12) + x1
                if x < xinters:
                    inside = not inside
        if inside:
            return True
    return False


# ── 폴리곤 안쪽 normal 결정 ────────────────
def _polygon_signed_area(polygon_xy: np.ndarray) -> float:
    n = len(polygon_xy)
    s = 0.0
    for i in range(n):
        x1, y1 = polygon_xy[i]
        x2, y2 = polygon_xy[(i + 1) % n]
        s += (x2 - x1) * (y2 + y1)
    return s / 2.0  # CW: positive, CCW: negative (by convention here)


def _inward_normal(p0: np.ndarray, p1: np.ndarray, ccw: bool) -> np.ndarray:
    """벽 segment p0→p1 의 안쪽 normal 단위벡터."""
    edge = p1 - p0
    n = np.array([-edge[1], edge[0]], dtype=np.float64)
    norm = np.linalg.norm(n)
    if norm < 1e-9:
        return np.array([0.0, 0.0])
    n /= norm
    if not ccw:
        n = -n
    return n


# ── 메인 클래스 ────────────────────────────
class PathPlanner:
    def __init__(self, params: PlanParams | None = None) -> None:
        self.params = params or PlanParams()

    # ── 천장/바닥 평면 보스트로페돈 ────────
    def _plan_horizontal_plane(
        self,
        polygon_xy: np.ndarray,
        z_layer: float,
        spacing: float,
        discrepancy_polygons: List[Polygon2D],
        face_kind: str,
        cam_pitch_rad: float,
    ) -> List[Waypoint]:
        if len(polygon_xy) < 3:
            return []
        centroid, axis_u = _pca_axis(polygon_xy)
        rotated, yaw_rot = _rotate_to_axis(polygon_xy, centroid, axis_u)
        y_min, y_max = float(rotated[:, 1].min()), float(rotated[:, 1].max())
        x_min, x_max = float(rotated[:, 0].min()), float(rotated[:, 0].max())

        wps: List[Waypoint] = []
        n_lines = max(1, int(math.ceil((y_max - y_min) / spacing)))
        line_idx = 0
        for li in range(n_lines + 1):
            y = min(y_min + li * spacing, y_max)
            segs = _line_polygon_segments(rotated, y, x_min, x_max)
            if not segs:
                continue
            for seg_lo, seg_hi in segs:
                if line_idx % 2 == 0:
                    seg_pts = np.arange(seg_lo, seg_hi + 1e-6, spacing)
                else:
                    seg_pts = np.arange(seg_hi, seg_lo - 1e-6, -spacing)
                line_idx += 1
                for cell_x_idx, x in enumerate(seg_pts):
                    world_xy = _unrotate((float(x), float(y)), centroid, yaw_rot)
                    in_disc = _is_in_discrepancy(world_xy, discrepancy_polygons)
                    if in_disc:
                        # 차이영역 추가 통과 WP — spacing 1/2 효과(같은 위치 두 번 캡처)
                        wps.append(Waypoint(
                            x=world_xy[0], y=world_xy[1], z=z_layer,
                            yaw_rad=0.0,
                            speed_mps=self.params.speed_room_mps * 0.7,
                            cell_idx=(cell_x_idx, li, 0),
                            purpose="discrepancy", in_discrepancy=True,
                            face_kind=face_kind, face_idx=0, cam_pitch_rad=cam_pitch_rad,
                        ))
                    wps.append(Waypoint(
                        x=world_xy[0], y=world_xy[1], z=z_layer,
                        yaw_rad=0.0, speed_mps=self.params.speed_room_mps,
                        cell_idx=(cell_x_idx, li, 0),
                        purpose="coverage", in_discrepancy=in_disc,
                        face_kind=face_kind, face_idx=0, cam_pitch_rad=cam_pitch_rad,
                    ))
        # yaw 자동 정렬 — 다음 WP 방향
        for i in range(len(wps) - 1):
            dx = wps[i + 1].x - wps[i].x
            dy = wps[i + 1].y - wps[i].y
            if dx == 0 and dy == 0:
                continue
            wps[i].yaw_rad = math.atan2(dy, dx)
        if wps:
            wps[-1].yaw_rad = wps[-2].yaw_rad if len(wps) >= 2 else 0.0
        return wps

    # ── 벽 면 그리드 (벽 정면 d_inspect 거리에서 수평×수직 그리드) ──
    def _plan_walls(
        self,
        polygon_xy: np.ndarray,
        spacing_h: float,
        discrepancy_polygons: List[Polygon2D],
    ) -> List[Waypoint]:
        n_pts = len(polygon_xy)
        if n_pts < 3:
            return []
        ccw = _polygon_signed_area(polygon_xy) > 0
        d_inspect = self.params.d_inspect_m
        floor_z = self.params.floor_z_m + self.params.floor_clearance_m
        ceil_z = self.params.ceiling_z_m - self.params.ceiling_clearance_m
        if ceil_z <= floor_z:
            ceil_z = floor_z + 0.6   # 안전 가드
        spacing_v = max(0.4, spacing_h * (1.0 - self.params.wall_z_overlap))
        n_v = max(1, int(math.ceil((ceil_z - floor_z) / spacing_v)))
        z_layers = [floor_z + i * (ceil_z - floor_z) / n_v for i in range(n_v + 1)]

        wps: List[Waypoint] = []
        for face_idx in range(n_pts):
            p0 = polygon_xy[face_idx]
            p1 = polygon_xy[(face_idx + 1) % n_pts]
            edge_vec = p1 - p0
            edge_len = float(np.linalg.norm(edge_vec))
            if edge_len < 0.3:
                continue
            edge_t = edge_vec / edge_len
            normal = _inward_normal(p0, p1, ccw=ccw)
            yaw_to_wall = math.atan2(-normal[1], -normal[0])  # 카메라는 벽을 봄(=normal 반대)
            n_h = max(1, int(math.ceil(edge_len / spacing_h)))
            h_offsets = np.linspace(0.0, edge_len, n_h + 1)

            for vi, z in enumerate(z_layers):
                # 보스트로페돈: 짝수 z 정방향, 홀수 역방향
                offsets = h_offsets if vi % 2 == 0 else h_offsets[::-1]
                for hi, off in enumerate(offsets):
                    foot = p0 + edge_t * off
                    cam_pos = foot + normal * d_inspect
                    in_disc = _is_in_discrepancy((float(cam_pos[0]), float(cam_pos[1])), discrepancy_polygons)
                    if in_disc:
                        # 차이영역 — 같은 위치 보조 통과(spacing 1/2 효과)
                        wps.append(Waypoint(
                            x=float(cam_pos[0]), y=float(cam_pos[1]), z=float(z),
                            yaw_rad=yaw_to_wall,
                            speed_mps=self.params.speed_room_mps * 0.7,
                            cell_idx=(hi, vi, face_idx),
                            purpose="discrepancy", in_discrepancy=True,
                            face_kind=FACE_WALL, face_idx=face_idx, cam_pitch_rad=0.0,
                        ))
                    wps.append(Waypoint(
                        x=float(cam_pos[0]), y=float(cam_pos[1]), z=float(z),
                        yaw_rad=yaw_to_wall,
                        speed_mps=self.params.speed_room_mps,
                        cell_idx=(hi, vi, face_idx),
                        purpose="coverage", in_discrepancy=in_disc,
                        face_kind=FACE_WALL, face_idx=face_idx, cam_pitch_rad=0.0,
                    ))
        return wps

    # ── 천장 ───────────────────────────────
    # 천장/바닥은 fixed-forward 카메라라 각 WP 에서 mission_orchestrator 가
    # nose-up/down tilt + yaw 360° 회전으로 보강 캡처. 따라서 WP 자체는
    # 보스트로페돈 spacing 의 1.5× 로 살짝 듬성하게 둠.
    CEILING_FLOOR_SPACING_FACTOR = 1.5

    def _plan_ceiling(
        self, polygon_xy: np.ndarray, spacing: float, disc: List[Polygon2D],
    ) -> List[Waypoint]:
        z = self.params.ceiling_z_m - self.params.ceiling_clearance_m
        return self._plan_horizontal_plane(
            polygon_xy, z_layer=z,
            spacing=spacing * self.CEILING_FLOOR_SPACING_FACTOR,
            discrepancy_polygons=disc,
            face_kind=FACE_CEILING, cam_pitch_rad=math.pi / 2.0,
        )

    # ── 바닥 ───────────────────────────────
    def _plan_floor(
        self, polygon_xy: np.ndarray, spacing: float, disc: List[Polygon2D],
    ) -> List[Waypoint]:
        z = self.params.floor_z_m + self.params.floor_clearance_m
        return self._plan_horizontal_plane(
            polygon_xy, z_layer=z,
            spacing=spacing * self.CEILING_FLOOR_SPACING_FACTOR,
            discrepancy_polygons=disc,
            face_kind=FACE_FLOOR, cam_pitch_rad=-math.pi / 2.0,
        )

    # ── 창호 (정밀 spacing) ─────────────────
    def _plan_windows(
        self,
        polygon_xy: np.ndarray,
        window_polygons: List[Polygon2D],
    ) -> List[Waypoint]:
        if not window_polygons:
            return []
        ccw = _polygon_signed_area(polygon_xy) > 0
        d_inspect = self.params.d_inspect_m
        base_spacing = grid_spacing(self.params)
        spacing = max(0.25, base_spacing * self.params.window_density_factor)

        wps: List[Waypoint] = []
        for w_idx, win_poly in enumerate(window_polygons):
            if len(win_poly) < 3:
                continue
            # 창호는 벽의 부분이라 가정 — 창호 폴리곤의 가장 긴 edge 를 향함
            arr = np.asarray(win_poly, dtype=np.float64)
            best_len, best_i = 0.0, 0
            for i in range(len(arr)):
                seg = arr[(i + 1) % len(arr)] - arr[i]
                L = float(np.linalg.norm(seg))
                if L > best_len:
                    best_len, best_i = L, i
            p0 = arr[best_i]; p1 = arr[(best_i + 1) % len(arr)]
            edge_vec = p1 - p0; edge_len = float(np.linalg.norm(edge_vec))
            if edge_len < 0.3:
                continue
            edge_t = edge_vec / edge_len
            normal = _inward_normal(p0, p1, ccw=ccw)
            yaw_face = math.atan2(-normal[1], -normal[0])
            # 창호 z 범위 추정: 창호 polygon 이 평면(2D) 라 z 정보가 없으므로
            # 창호 중심 z = (floor_z + ceiling_z) / 2, 창호 vertical extent = 1.2m 가정.
            cz = (self.params.floor_z_m + self.params.ceiling_z_m) / 2.0
            z_extent = 1.2
            z_layers = [cz - z_extent / 2 + i * spacing for i in range(int(math.ceil(z_extent / spacing)) + 1)]
            n_h = max(1, int(math.ceil(edge_len / spacing)))
            h_offsets = np.linspace(0.0, edge_len, n_h + 1)
            for vi, z in enumerate(z_layers):
                offsets = h_offsets if vi % 2 == 0 else h_offsets[::-1]
                for hi, off in enumerate(offsets):
                    foot = p0 + edge_t * off
                    cam_pos = foot + normal * d_inspect
                    wps.append(Waypoint(
                        x=float(cam_pos[0]), y=float(cam_pos[1]), z=float(z),
                        yaw_rad=yaw_face,
                        speed_mps=self.params.speed_window_mps,
                        cell_idx=(hi, vi, w_idx),
                        purpose="coverage", in_discrepancy=False,
                        face_kind=FACE_WINDOW, face_idx=w_idx, cam_pitch_rad=0.0,
                    ))
        return wps

    # ── 룸 통합 ─────────────────────────────
    def plan_room(
        self,
        room_idx: int,
        polygon: Polygon2D,
        discrepancy_polygons: Optional[List[Polygon2D]] = None,
        window_polygons: Optional[List[Polygon2D]] = None,
    ) -> RoomPlan:
        polygon_xy = np.asarray(polygon, dtype=np.float64)
        if polygon_xy.ndim != 2 or polygon_xy.shape[0] < 3:
            logger.warning("path_planner.invalid_polygon", room_idx=room_idx, n=len(polygon_xy))
            return RoomPlan(room_idx=room_idx)
        polygon_xy = _shrink_polygon(polygon_xy, self.params.edge_margin_m)
        spacing = grid_spacing(self.params)
        plan = RoomPlan(room_idx=room_idx, spacing_m=spacing)
        disc = discrepancy_polygons or []

        # 면별로 추가. 각 면 wp 의 cell_idx 의 z 자리에 face 레이블을 두지 않고,
        # face_kind/face_idx 별도 필드로 식별. cell 유니크 키는 (mission, room, cx, cy, cz) 라서
        # cell_z 를 face 코드(0=floor, 1..N=wall, N+1=ceiling, ...) 로 인코딩.
        face_z_offset = 0
        # 1) 바닥
        if self.params.scan_floor:
            floor_wps = self._plan_floor(polygon_xy, spacing, disc)
            for w in floor_wps:
                w.cell_idx = (w.cell_idx[0], w.cell_idx[1], face_z_offset)  # cz=0
            plan.waypoints.extend(floor_wps)
            face_z_offset += 1
        # 2) 벽 4면 (혹은 N면)
        if self.params.scan_walls:
            wall_wps = self._plan_walls(polygon_xy, spacing, disc)
            for w in wall_wps:
                # cz 는 face_z_offset + face_idx 로 인코딩 → 셀 유니크 보장
                w.cell_idx = (w.cell_idx[0], w.cell_idx[1], face_z_offset + w.face_idx)
            plan.waypoints.extend(wall_wps)
            face_z_offset += max(1, len(polygon_xy))
        # 3) 천장
        if self.params.scan_ceiling:
            ceil_wps = self._plan_ceiling(polygon_xy, spacing, disc)
            for w in ceil_wps:
                w.cell_idx = (w.cell_idx[0], w.cell_idx[1], face_z_offset)
            plan.waypoints.extend(ceil_wps)
            face_z_offset += 1
        # 4) 창호 (있을 때만)
        if self.params.scan_windows and window_polygons:
            win_wps = self._plan_windows(polygon_xy, window_polygons)
            for w in win_wps:
                # 창호는 face_z_offset + window_idx 로 인코딩
                w.cell_idx = (w.cell_idx[0], w.cell_idx[1], face_z_offset + w.face_idx)
            plan.waypoints.extend(win_wps)

        plan.cell_count = len(plan.waypoints)
        logger.info(
            "path_planner.plan_room",
            room_idx=room_idx, spacing_m=round(spacing, 3),
            cell_count=plan.cell_count,
            walls=self.params.scan_walls, ceiling=self.params.scan_ceiling,
            floor=self.params.scan_floor, windows=bool(window_polygons),
            discrepancy_count=len(disc),
        )
        return plan

    def plan_mission(
        self,
        topology_nodes: List[Dict],
        discrepancy_regions: Optional[List[dict]] = None,
        window_polygons_per_room: Optional[Dict[int, List[Polygon2D]]] = None,
        params: PlanParams | None = None,
    ) -> MissionGridPlan:
        if params is not None:
            self.params = params
        disc_polys: List[Polygon2D] = []
        if discrepancy_regions:
            for d in discrepancy_regions:
                poly = d.get("polygon") or []
                if len(poly) >= 3:
                    disc_polys.append([tuple(p) for p in poly])

        mp = MissionGridPlan(params=self.params)
        for node in topology_nodes:
            idx = int(node["idx"])
            polygon: Polygon2D = [tuple(p) for p in node["polygon"]]
            wins = (window_polygons_per_room or {}).get(idx)
            mp.rooms[idx] = self.plan_room(
                idx, polygon, disc_polys, window_polygons=wins,
            )
        return mp

    def replan_uncaptured(
        self,
        room_plan: RoomPlan,
        captured_cells: set[Tuple[int, int, int]],
    ) -> RoomPlan:
        new = RoomPlan(room_idx=room_plan.room_idx, spacing_m=room_plan.spacing_m)
        new.waypoints = [w for w in room_plan.waypoints if w.cell_idx not in captured_cells]
        new.cell_count = len(new.waypoints)
        return new
