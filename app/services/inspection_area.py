# =============================================
# app/services/inspection_area.py
# 역할: 3D 모델 기반 점검 면적 산출 + 면별/룸별 커버리지율
#
# 검사 면적(업계 표준):
#   total = floor_area + ceiling_area + walls_area + windows_area
#   - floor / ceiling : 룸 폴리곤 면적 (Shoelace)
#   - walls           : 룸 둘레 × (ceiling_z - floor_z) - 창호 면적 합 (창호는 벽 일부)
#   - windows         : 창호 폴리곤 면적 합
#
# 커버리지율(업계 검사 표준):
#   - 룸/면별 captured_cells / total_cells
#   - 글로벌: ∑ captured_cells / ∑ total_cells (단순 평균이 아니라 셀 가중)
#   - 미점검 셀들의 룸/면별 분포 + 셀 좌표 → 보고서·UI 시각화 입력
#
# 산출 결과는 mission_orchestrator 가 mission_plans.plan_json["area_summary"] 에 저장하고
# WS coverage.summary 채널로 broadcast.
# =============================================
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from app.core.logging import get_logger
from app.services.path_planner import (
    FACE_CEILING, FACE_FLOOR, FACE_WALL, FACE_WINDOW,
    MissionGridPlan, PlanParams, Waypoint,
)

logger = get_logger(__name__)


Polygon2D = Sequence[Tuple[float, float]]


# ── Shoelace 폴리곤 면적 ────────────────────
def polygon_area_m2(polygon: Polygon2D) -> float:
    n = len(polygon)
    if n < 3:
        return 0.0
    s = 0.0
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return abs(s) / 2.0


def polygon_perimeter_m(polygon: Polygon2D) -> float:
    n = len(polygon)
    if n < 2:
        return 0.0
    p = 0.0
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        p += ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
    return p


# ── 결과 타입 ───────────────────────────────
@dataclass
class FaceArea:
    floor_m2: float = 0.0
    ceiling_m2: float = 0.0
    walls_m2: float = 0.0
    windows_m2: float = 0.0
    total_m2: float = 0.0


@dataclass
class FaceCoverage:
    captured: int = 0
    total: int = 0

    @property
    def ratio(self) -> float:
        return float(self.captured) / float(self.total) if self.total > 0 else 0.0


@dataclass
class RoomAreaSummary:
    room_idx: int
    name: str = ""
    area: FaceArea = field(default_factory=FaceArea)
    coverage_by_face: Dict[str, FaceCoverage] = field(default_factory=dict)
    overall_coverage: FaceCoverage = field(default_factory=FaceCoverage)

    def to_jsonable(self) -> dict:
        return {
            "room_idx": self.room_idx,
            "name": self.name,
            "area": {
                "floor_m2": round(self.area.floor_m2, 3),
                "ceiling_m2": round(self.area.ceiling_m2, 3),
                "walls_m2": round(self.area.walls_m2, 3),
                "windows_m2": round(self.area.windows_m2, 3),
                "total_m2": round(self.area.total_m2, 3),
            },
            "coverage_by_face": {
                face: {"captured": c.captured, "total": c.total, "ratio": round(c.ratio, 4)}
                for face, c in self.coverage_by_face.items()
            },
            "overall_coverage": {
                "captured": self.overall_coverage.captured,
                "total": self.overall_coverage.total,
                "ratio": round(self.overall_coverage.ratio, 4),
            },
        }


@dataclass
class MissionAreaSummary:
    rooms: List[RoomAreaSummary] = field(default_factory=list)
    grand_area: FaceArea = field(default_factory=FaceArea)
    grand_coverage: FaceCoverage = field(default_factory=FaceCoverage)
    supplied_area_m2: Optional[float] = None        # 사이트 등록 시 입력된 분양/공급 면적
    supplied_coverage_ratio: Optional[float] = None  # 분양면적 대비 검사면적 비율(SLAM 실측 ÷ 분양)

    def to_jsonable(self) -> dict:
        return {
            "rooms": [r.to_jsonable() for r in self.rooms],
            "grand_area": {
                "floor_m2": round(self.grand_area.floor_m2, 3),
                "ceiling_m2": round(self.grand_area.ceiling_m2, 3),
                "walls_m2": round(self.grand_area.walls_m2, 3),
                "windows_m2": round(self.grand_area.windows_m2, 3),
                "total_m2": round(self.grand_area.total_m2, 3),
            },
            "grand_coverage": {
                "captured": self.grand_coverage.captured,
                "total": self.grand_coverage.total,
                "ratio": round(self.grand_coverage.ratio, 4),
            },
            "supplied_area_m2": self.supplied_area_m2,
            "supplied_coverage_ratio": (
                round(self.supplied_coverage_ratio, 4)
                if self.supplied_coverage_ratio is not None else None
            ),
        }


# ── 메인 계산 ──────────────────────────────
class InspectionAreaCalculator:
    """
    SLAM 후 산출된 토폴로지 + 미션 플랜 + captured 셀 집합 기반 면적/커버리지 산출.
    """

    def __init__(self, params: PlanParams) -> None:
        self.params = params

    # 룸 단일
    def compute_room(
        self,
        room_idx: int,
        polygon: Polygon2D,
        room_waypoints: Iterable[Waypoint],
        captured_cells: set,
        window_polygons: Optional[List[Polygon2D]] = None,
        room_name: str = "",
    ) -> RoomAreaSummary:
        floor = polygon_area_m2(polygon)
        ceiling = floor   # 평평한 천장 가정 (BIM 표준 — 천장은 바닥 면적과 동일)
        windows = sum(polygon_area_m2(w) for w in (window_polygons or []))
        height = max(0.0, self.params.ceiling_z_m - self.params.floor_z_m)
        # 벽 총 면적 = 둘레 × 층높 - 창호 면적 (창호가 벽 일부라 가정)
        walls_gross = polygon_perimeter_m(polygon) * height
        walls = max(0.0, walls_gross - windows)
        total = floor + ceiling + walls + windows
        area = FaceArea(
            floor_m2=floor, ceiling_m2=ceiling, walls_m2=walls,
            windows_m2=windows, total_m2=total,
        )

        # 커버리지: WP 의 face_kind 별 captured/total
        face_count: Dict[str, FaceCoverage] = {
            FACE_FLOOR: FaceCoverage(),
            FACE_WALL: FaceCoverage(),
            FACE_CEILING: FaceCoverage(),
            FACE_WINDOW: FaceCoverage(),
        }
        overall = FaceCoverage()
        for w in room_waypoints:
            if w.purpose == "discrepancy":
                continue
            fc = face_count.setdefault(w.face_kind, FaceCoverage())
            fc.total += 1
            overall.total += 1
            if w.cell_idx in captured_cells:
                fc.captured += 1
                overall.captured += 1

        return RoomAreaSummary(
            room_idx=room_idx, name=room_name,
            area=area,
            coverage_by_face=face_count,
            overall_coverage=overall,
        )

    # 미션 전체
    def compute_mission(
        self,
        plan: MissionGridPlan,
        topology_nodes: List[dict],
        captured_cells: set,
        window_polygons_per_room: Optional[Dict[int, List[Polygon2D]]] = None,
        supplied_area_m2: Optional[float] = None,
    ) -> MissionAreaSummary:
        summary = MissionAreaSummary(supplied_area_m2=supplied_area_m2)
        node_by_idx = {int(n["idx"]): n for n in topology_nodes}

        for room_idx, room_plan in plan.rooms.items():
            node = node_by_idx.get(room_idx)
            if node is None:
                continue
            polygon = [tuple(p) for p in node["polygon"]]
            wins = (window_polygons_per_room or {}).get(room_idx)
            r = self.compute_room(
                room_idx=room_idx,
                polygon=polygon,
                room_waypoints=room_plan.waypoints,
                captured_cells=captured_cells,
                window_polygons=wins,
                room_name=node.get("name", f"room_{room_idx}"),
            )
            summary.rooms.append(r)
            # 누적
            summary.grand_area.floor_m2 += r.area.floor_m2
            summary.grand_area.ceiling_m2 += r.area.ceiling_m2
            summary.grand_area.walls_m2 += r.area.walls_m2
            summary.grand_area.windows_m2 += r.area.windows_m2
            summary.grand_area.total_m2 += r.area.total_m2
            summary.grand_coverage.captured += r.overall_coverage.captured
            summary.grand_coverage.total += r.overall_coverage.total

        if supplied_area_m2 and supplied_area_m2 > 0:
            # 분양/공급 면적은 통상 바닥 면적 기준. SLAM 실측 바닥 / 공급
            summary.supplied_coverage_ratio = (
                summary.grand_area.floor_m2 / supplied_area_m2
            )
        logger.info(
            "inspection_area.computed",
            rooms=len(summary.rooms),
            total_inspection_m2=round(summary.grand_area.total_m2, 2),
            coverage_ratio=round(summary.grand_coverage.ratio, 4),
            supplied_ratio=(
                round(summary.supplied_coverage_ratio, 4)
                if summary.supplied_coverage_ratio is not None else None
            ),
        )
        return summary
