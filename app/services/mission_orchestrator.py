# =============================================
# app/services/mission_orchestrator.py
# 역할: 자율비행 미션 FSM 오케스트레이터 (단일 인스턴스)
#       플랜 §6 상태머신:
#         IDLE → ARM → TAKEOFF → MAPPING → VERIFICATION → PATH_PLAN
#              → COVERAGE_FLY → ROOM_TRANSITION ↺ → COMPLETE → LAND
#         실패: FAILSAFE → 즉시 hover → LAND
#
# 본 파일은 FSM 본체 — 실제 비행 명령 흐름 + DB 영속화 + WS 브로드캐스트.
#
# 안전 가드 Task (별도 asyncio Task) 가 매 100ms safety_monitor.check 호출,
# 이상 감지 시 _on_safety_decision 콜백으로 즉시 인터럽트.
#
# 의존:
#   - fc_bridge: MSP 명령 송수신
#   - slam_runner: SLAM pose / occupancy / pointcloud
#   - path_planner: 보스트로페돈 그리드 WP
#   - room_segmenter: occupancy → 토폴로지
#   - obstacle_avoider: DWA 풍 회피 평가
#   - floorplan_verifier: 사전모델 ↔ SLAM 정합 검증
#   - safety_monitor: 안전 단일 진입점
# =============================================
from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional, Set, Tuple

from sqlalchemy import update

from app.core.logging import get_logger
from app.core.ws_manager import ConnectionManager
from app.db.session import async_session_factory
from app.models.coverage_grid import CoverageGrid
from app.models.mission_plan import MissionPlan
from app.models.room_topology import RoomTopology
from app.services.fc_bridge import MspCommand, TelemetryFrame, fc_bridge
from app.services.floorplan_verifier import (
    FloorplanVerifier, VerificationResult, VerificationVerdict,
)
from app.services.inspection_area import InspectionAreaCalculator, MissionAreaSummary
from app.services.obstacle_avoider import ObstacleAvoider, VelocityCommand
from app.services.path_planner import (
    FACE_CEILING, FACE_FLOOR, MissionGridPlan, PathPlanner, PlanParams, Waypoint,
)
from app.services.room_segmenter import RoomSegmenter, RoomTopologyGraph
from app.services.safety_monitor import (
    SafetyAction, SafetyDecision, TelemetrySnapshot, safety_monitor,
)
from app.services.slam_runner import SlamPose, SlamRunner

logger = get_logger(__name__)


class MissionPhase(str, Enum):
    IDLE = "idle"
    ARM = "arm"
    TAKEOFF = "takeoff"
    MAPPING = "mapping"
    VERIFICATION = "verification"
    PATH_PLAN = "path_plan"
    COVERAGE_FLY = "coverage_fly"
    ROOM_TRANSITION = "room_transition"
    COMPLETE = "complete"
    LAND = "land"
    FAILSAFE = "failsafe"


# ── 단일 WS 매니저 인스턴스 ───────────────────────
# main.py 에서 같은 매니저를 사용하도록 의존성 주입 가능. 본 파일 내 폴백 인스턴스.
_ws_manager: Optional[ConnectionManager] = None


def set_ws_manager(manager: ConnectionManager) -> None:
    """main.py lifespan 에서 시스템 단일 매니저 인스턴스를 주입."""
    global _ws_manager
    _ws_manager = manager


@dataclass
class MissionState:
    mission_id: Optional[str] = None
    phase: MissionPhase = MissionPhase.IDLE
    current_room_idx: Optional[int] = None
    last_telemetry: Optional[TelemetryFrame] = None
    last_pose: Optional[SlamPose] = None
    plan: Optional[MissionGridPlan] = None
    topology: Optional[RoomTopologyGraph] = None
    verification: Optional[VerificationResult] = None
    prior_polygons: Optional[list] = None
    window_polygons_per_room: Optional[dict] = None    # {room_idx: [Polygon2D, ...]}
    supplied_area_m2: Optional[float] = None           # sites.total_area 등 분양/공급 면적
    captured_cells: Set[Tuple[int, int, int]] = None  # type: ignore[assignment]
    area_summary: Optional[MissionAreaSummary] = None
    failure_reason: Optional[str] = None
    geofence_polygon: Optional[List[Tuple[float, float]]] = None

    def __post_init__(self) -> None:
        if self.captured_cells is None:
            self.captured_cells = set()


class MissionOrchestrator:
    """
    싱글톤. main.py lifespan 에서 1회 생성, lifecycle 동안 유지.
    """
    GUARD_INTERVAL_SEC = 0.1
    TAKEOFF_ALTITUDE_M = 1.0
    TAKEOFF_TIMEOUT_SEC = 10.0
    MAPPING_SPIN_SEC = 30.0
    WP_REACH_TOLERANCE_M = 0.4
    WP_REACH_TIMEOUT_SEC = 15.0
    CELL_DOUBLE_CAPTURE_SEC = 0.6   # RGB/Thermal 동기화 허용 윈도우
    POSE_VARIANCE_WINDOW = 10

    def __init__(self) -> None:
        self.state = MissionState()
        self.path_planner = PathPlanner()
        self.room_segmenter = RoomSegmenter()
        self.obstacle_avoider = ObstacleAvoider()
        self.slam_runner = SlamRunner()
        self.floorplan_verifier = FloorplanVerifier()
        self.area_calculator = InspectionAreaCalculator(self.path_planner.params)
        self._task: Optional[asyncio.Task] = None
        self._guard_task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()
        self._lock = asyncio.Lock()
        self._pose_history: List[Tuple[float, float, float]] = []
        self._latest_occupancy = None    # numpy ndarray 또는 None
        self._occupancy_origin_xy: Tuple[float, float] = (0.0, 0.0)
        self._occupancy_resolution = 0.05

        # 콜백 결선
        safety_monitor.register(self._on_safety_decision)
        fc_bridge.subscribe_telemetry(self._on_telemetry)
        self.slam_runner.subscribe_pose(self._on_pose)
        self.slam_runner.subscribe_pointcloud(self._on_pointcloud)

    # ── 외부 API ─────────────────────────
    async def start(
        self,
        mission_id: str,
        params: PlanParams | None = None,
        prior_polygons: Optional[list] = None,
        window_polygons_per_room: Optional[dict] = None,
        supplied_area_m2: Optional[float] = None,
    ) -> None:
        async with self._lock:
            if self.state.phase is not MissionPhase.IDLE:
                raise RuntimeError(f"mission already in phase={self.state.phase.value}")
            if not fc_bridge.is_attached:
                raise RuntimeError("fc_bridge_not_attached")
            self.state = MissionState(
                mission_id=mission_id,
                prior_polygons=prior_polygons,
                window_polygons_per_room=window_polygons_per_room,
                supplied_area_m2=supplied_area_m2,
            )
            if params:
                self.path_planner.params = params
                self.area_calculator = InspectionAreaCalculator(params)
            self._stop_event.clear()
            self._task = asyncio.create_task(self._run())
            self._guard_task = asyncio.create_task(self._safety_guard_loop())
        logger.info(
            "mission.start", mission_id=mission_id,
            has_prior_model=bool(prior_polygons),
        )

    async def abort(self, reason: str = "user_abort") -> None:
        await self._fail(reason)

    async def estop(self) -> None:
        safety_monitor.trigger_estop("user_estop")
        await fc_bridge.send(MspCommand.LAND)
        await self._fail("user_estop")

    async def rtl(self) -> None:
        await fc_bridge.send(MspCommand.RTH)
        await self._transition(MissionPhase.LAND)

    async def pause(self) -> None:
        """INAV PositionHold 강제 + DB status='paused' (FSM phase 는 그대로 유지)."""
        await fc_bridge.send(MspCommand.SET_NAV_MODE, {"mode": "POSHOLD"})
        logger.info("mission.pause")
        # DB 상태 갱신 — mission_plans.status='paused' (UI 가시성)
        if self.state.mission_id is None:
            return
        try:
            async with async_session_factory() as db:
                stmt = (
                    update(MissionPlan)
                    .where(MissionPlan.id == self.state.mission_id)
                    .values(status="paused")
                )
                await db.execute(stmt)
                await db.commit()
        except Exception as e:
            logger.error("mission.persist_pause_failed", error=str(e))

    def get_state(self) -> dict:
        v = self.state.verification
        return {
            "mission_id": self.state.mission_id,
            "phase": self.state.phase.value,
            "current_room_idx": self.state.current_room_idx,
            "failure_reason": self.state.failure_reason,
            "fc_attached": fc_bridge.is_attached,
            "verification": (v.to_jsonable() if v else None),
            "captured_cells": len(self.state.captured_cells),
        }

    # ── FSM 메인 루프 ──────────────────────
    async def _run(self) -> None:
        try:
            # 1) ARM
            await self._transition(MissionPhase.ARM)
            await fc_bridge.send(MspCommand.ARM)
            await asyncio.sleep(2.0)

            # 2) TAKEOFF
            await self._transition(MissionPhase.TAKEOFF)
            await self._do_takeoff()

            # 3) MAPPING — 슬로우 360° 스핀 + 짧은 직선 왕복
            await self._transition(MissionPhase.MAPPING)
            assert self.state.mission_id is not None
            await self.slam_runner.start(self.state.mission_id)
            await self._do_mapping_round()

            # 4) VERIFICATION — 사전모델 ↔ SLAM 정합
            await self._transition(MissionPhase.VERIFICATION)
            await self._run_verification()

            # 5) PATH_PLAN — room segmentation → 3D 보스트로페돈
            await self._transition(MissionPhase.PATH_PLAN)
            await self._do_path_plan()

            # 6) COVERAGE_FLY → ROOM_TRANSITION
            if self.state.plan is None or not self.state.plan.rooms:
                logger.warning("mission.no_plan_skip")
            else:
                # 룸 순회 — 토폴로지 그래프 BFS (centroid 가까운 순)
                room_order = self._room_traversal_order()
                for i, room_idx in enumerate(room_order):
                    self.state.current_room_idx = room_idx
                    await self._transition(MissionPhase.COVERAGE_FLY)
                    await self._do_coverage_fly(room_idx)

                    if i < len(room_order) - 1:
                        next_idx = room_order[i + 1]
                        await self._transition(MissionPhase.ROOM_TRANSITION)
                        await self._do_room_transition(room_idx, next_idx)

            # 7) COMPLETE → LAND
            await self._transition(MissionPhase.COMPLETE)
            await self._transition(MissionPhase.LAND)
            await fc_bridge.send(MspCommand.LAND)
            await asyncio.sleep(3.0)

        except asyncio.CancelledError:
            logger.warning("mission.cancelled")
        except Exception as e:
            logger.error("mission.run_failed", error=str(e), exc_info=True)
            await self._fail(f"exception:{e!s}")
        finally:
            await self.slam_runner.stop()
            if self._guard_task:
                self._guard_task.cancel()
                self._guard_task = None
            await self._transition(MissionPhase.IDLE)

    # ── 단계별 동작 ───────────────────────
    async def _do_takeoff(self) -> None:
        """INAV NAV WP_TAKEOFF 또는 RAW_RC 스로틀 상승. 고도 임계 도달까지 대기."""
        await fc_bridge.send(MspCommand.SET_NAV_MODE, {"mode": "POSHOLD"})
        await asyncio.sleep(0.5)
        # INAV 의 takeoff 는 LOAD_MISSION_WP 의 WP_TAKEOFF type 또는 별도 RAW_RC 상승.
        # 본 구현은 단일 takeoff 명령(추후 todo#7 Pi 측에서 MSP 매핑) + 고도 폴링.
        await fc_bridge.send(MspCommand.LOAD_MISSION_WP, {
            "waypoints": [{"type": "TAKEOFF", "alt": self.TAKEOFF_ALTITUDE_M}],
        })
        await fc_bridge.send(MspCommand.SET_NAV_MODE, {"mode": "WP"})

        deadline = time.time() + self.TAKEOFF_TIMEOUT_SEC
        while time.time() < deadline:
            t = self.state.last_telemetry
            if t and t.pos_z is not None and t.pos_z >= self.TAKEOFF_ALTITUDE_M - 0.1:
                logger.info("mission.takeoff_reached", z=t.pos_z)
                return
            await asyncio.sleep(0.2)
        raise RuntimeError("takeoff_timeout")

    async def _do_mapping_round(self) -> None:
        """슬로우 360° yaw 스핀 + 짧은 직선 왕복으로 SLAM 초기 안정화."""
        # 360° 스핀 — 1.5°/s, 240초가 너무 길어 30초 안에 마감
        spin_rate = (math.tau) / max(1.0, self.MAPPING_SPIN_SEC)  # rad/s
        for _ in range(int(self.MAPPING_SPIN_SEC * 5)):  # 0.2s 간격
            await fc_bridge.send(MspCommand.SET_RAW_RC, {
                "yaw_rate": float(spin_rate),
                "vx": 0.0, "vy": 0.0, "vz": 0.0,
            })
            await asyncio.sleep(0.2)
        # 짧은 전진 1m + 후진 1m
        for vx in (0.3, 0.0, -0.3, 0.0):
            await fc_bridge.send(MspCommand.SET_RAW_RC, {
                "yaw_rate": 0.0, "vx": float(vx), "vy": 0.0, "vz": 0.0,
            })
            await asyncio.sleep(2.0)

    async def _run_verification(self) -> None:
        """
        사전모델이 있으면 SLAM occupancy 와 정합 비교.
        verdict 에 따라:
          OK / NO_PRIOR_MODEL → 즉시 통과
          MARGINAL            → 추가 MAPPING 라운드 1회 후 재검증 1회
          DIVERGENT           → WS 알림 후 진행 (사용자 확인은 UI 의 PAUSE 사용)
        """
        prior = self.state.prior_polygons
        # SLAM 백엔드에서 직접 fetch — _do_path_plan 이전에 호출되므로 여기서도 미리 갱신
        grid, res, origin = self.slam_runner.get_latest_occupancy()
        if grid is not None:
            self._latest_occupancy = grid
            self._occupancy_resolution = res
            self._occupancy_origin_xy = origin

        result = self.floorplan_verifier.verify(
            occupancy_grid=self._latest_occupancy,
            resolution_m_per_px=self._occupancy_resolution,
            prior_polygons=prior, origin_xy=self._occupancy_origin_xy,
        )
        self.state.verification = result
        await self._broadcast("mission.verification_result", result.to_jsonable())

        if result.verdict is VerificationVerdict.MARGINAL:
            logger.info("mission.verification.retry_mapping")
            await self._do_mapping_round()
            grid2, res2, origin2 = self.slam_runner.get_latest_occupancy()
            if grid2 is not None:
                self._latest_occupancy = grid2
                self._occupancy_resolution = res2
                self._occupancy_origin_xy = origin2
            result2 = self.floorplan_verifier.verify(
                occupancy_grid=self._latest_occupancy,
                resolution_m_per_px=self._occupancy_resolution,
                prior_polygons=prior, origin_xy=self._occupancy_origin_xy,
            )
            self.state.verification = result2
            await self._broadcast("mission.verification_result", result2.to_jsonable())
            if result2.verdict is VerificationVerdict.DIVERGENT:
                await self._broadcast("mission.verification_alert", {
                    "iou": result2.iou,
                    "discrepancies": len(result2.discrepancies),
                })
        elif result.verdict is VerificationVerdict.DIVERGENT:
            await self._broadcast("mission.verification_alert", {
                "iou": result.iou,
                "discrepancies": len(result.discrepancies),
            })

    async def _do_path_plan(self) -> None:
        # 1) SLAM 백엔드에서 occupancy 가져오기 (없으면 prior_polygons 폴백)
        grid, res, origin = self.slam_runner.get_latest_occupancy()
        if grid is not None:
            self._latest_occupancy = grid
            self._occupancy_resolution = res
            self._occupancy_origin_xy = origin

        if self._latest_occupancy is None and not self.state.prior_polygons:
            logger.warning("mission.path_plan_skip", reason="no_occupancy_no_prior")
            return

        # 2) room segmentation 또는 prior_polygons 직접 토폴로지
        if self._latest_occupancy is not None:
            self.room_segmenter.params.resolution_m_per_px = self._occupancy_resolution
            self.room_segmenter.params.origin_xy = self._occupancy_origin_xy
            topology = self.room_segmenter.segment(self._latest_occupancy)
        else:
            # SLAM occupancy 없음 → 사전모델 폴리곤을 그대로 토폴로지로 (도어웨이 미상정)
            from app.services.room_segmenter import RoomNode, RoomTopologyGraph
            topology = RoomTopologyGraph()
            for i, poly in enumerate(self.state.prior_polygons or []):
                pts = [tuple(p) for p in poly]
                if len(pts) < 3:
                    continue
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
                topology.nodes.append(RoomNode(
                    idx=i, name=f"prior_room_{i}",
                    polygon=pts, area_m2=0.0, centroid=(cx, cy),
                ))
            logger.info("mission.path_plan.prior_fallback", rooms=len(topology.nodes))
        self.state.topology = topology

        # 2) topology DB 저장
        await self._persist_topology(topology)

        # 3) 차이영역 추출 → path_planner 입력
        v = self.state.verification
        disc_regions: List[dict] = []
        if v and v.discrepancies:
            disc_regions = [
                {"polygon": [list(p) for p in d.polygon], "kind": d.kind, "area_m2": d.area_m2}
                for d in v.discrepancies
            ]

        # 4) plan 생성
        plan = self.path_planner.plan_mission(
            topology_nodes=[
                {"idx": n.idx, "polygon": [list(p) for p in n.polygon]}
                for n in topology.nodes
            ],
            discrepancy_regions=disc_regions,
            window_polygons_per_room=self.state.window_polygons_per_room,
        )
        self.state.plan = plan

        # 5) coverage_grids 시드 + WS 알림
        await self._seed_coverage_grids(plan)

        # 6) 검사 면적 산출(초기 — captured 0) → plan_json 저장 + WS broadcast
        topo_nodes_jsonable = [
            {"idx": n.idx, "name": n.name,
             "polygon": [list(p) for p in n.polygon]}
            for n in topology.nodes
        ]
        summary = self.area_calculator.compute_mission(
            plan=plan,
            topology_nodes=topo_nodes_jsonable,
            captured_cells=set(),
            window_polygons_per_room=self.state.window_polygons_per_room,
            supplied_area_m2=self.state.supplied_area_m2,
        )
        self.state.area_summary = summary
        await self._persist_area_summary(summary)
        await self._broadcast("coverage.summary", summary.to_jsonable())

        await self._broadcast("mission.path", {
            "rooms": [
                {
                    "room_idx": r.room_idx, "spacing_m": r.spacing_m,
                    "cell_count": r.cell_count,
                    "waypoints": [
                        {
                            "x": w.x, "y": w.y, "z": w.z, "yaw": w.yaw_rad,
                            "purpose": w.purpose, "in_disc": w.in_discrepancy,
                            # 4면 정밀 스캔 메타 — 프론트 R3F 레이어가 face별 색/회전 결정에 사용
                            "face_kind": w.face_kind, "face_idx": w.face_idx,
                            "cam_pitch_rad": w.cam_pitch_rad,
                            # 셀 cell_idx — 프론트 cellsByKey 매칭에 필수
                            "cell_idx": list(w.cell_idx),
                        }
                        for w in r.waypoints
                    ],
                }
                for r in plan.rooms.values()
            ],
        })

    def _room_traversal_order(self) -> List[int]:
        """토폴로지 노드를 centroid 가까운 순(현 위치 기준)으로 정렬."""
        topo = self.state.topology
        if not topo or not topo.nodes:
            return []
        if self.state.last_pose is None:
            return [n.idx for n in topo.nodes]
        cx, cy = self.state.last_pose.x, self.state.last_pose.y
        order = sorted(
            topo.nodes,
            key=lambda n: (n.centroid[0] - cx) ** 2 + (n.centroid[1] - cy) ** 2,
        )
        return [n.idx for n in order]

    async def _do_coverage_fly(self, room_idx: int) -> None:
        plan = self.state.plan
        if plan is None or room_idx not in plan.rooms:
            return
        room_plan = plan.rooms[room_idx]
        # 미점검 셀만 남기기 (resume 지원)
        room_plan = self.path_planner.replan_uncaptured(room_plan, self.state.captured_cells)
        for wp in room_plan.waypoints:
            await self._fly_to_waypoint(wp, in_doorway=False)
            await self._capture_cell(room_idx, wp)

    async def _do_room_transition(self, src_idx: int, dst_idx: int) -> None:
        """도어웨이 통과 — 속도 50%, 안전 임계 +5cm 강화."""
        topo = self.state.topology
        if not topo:
            return
        edge = next(
            (e for e in topo.edges
             if (e.src == src_idx and e.dst == dst_idx) or (e.src == dst_idx and e.dst == src_idx)),
            None,
        )
        if edge is None:
            logger.warning("mission.no_doorway", src=src_idx, dst=dst_idx)
            return
        # 도어웨이 진입 전 yaw 정렬 — 룸 중간 높이로 통과
        p = self.path_planner.params
        mid_z = (p.floor_z_m + p.ceiling_z_m) / 2.0
        approach_wp = Waypoint(
            x=edge.center[0], y=edge.center[1],
            z=mid_z,
            yaw_rad=edge.approach_yaw_rad,
            speed_mps=p.speed_doorway_mps,
            purpose="doorway_approach",
        )
        await self._fly_to_waypoint(approach_wp, in_doorway=True)

    async def _fly_to_waypoint(self, wp: Waypoint, in_doorway: bool) -> None:
        """단일 WP 도달까지 직진. obstacle_avoider 매 스텝 평가."""
        deadline = time.time() + self.WP_REACH_TIMEOUT_SEC
        while time.time() < deadline:
            pose = self.state.last_pose
            if pose is None:
                await asyncio.sleep(0.1)
                continue
            dx, dy, dz = wp.x - pose.x, wp.y - pose.y, wp.z - pose.z
            dist = math.sqrt(dx * dx + dy * dy + dz * dz)
            if dist <= self.WP_REACH_TOLERANCE_M:
                return
            # 단순 P 제어 — 속도는 거리에 비례, 상한 적용
            v_max = (
                self.path_planner.params.speed_doorway_mps if in_doorway
                else self.path_planner.params.speed_room_mps
            )
            scale = min(1.0, dist) * v_max / max(dist, 1e-6)
            target = VelocityCommand(
                vx=dx * scale, vy=dy * scale, vz=dz * scale,
                yaw_rate=0.0,
            )
            # 회피 평가
            ev = self.obstacle_avoider.evaluate(
                target=target,
                current_pose_xyz=(pose.x, pose.y, pose.z),
                slam_confidence=pose.confidence,
                in_doorway=in_doorway,
            )
            if not ev.safe:
                # hover — 속도 0
                await fc_bridge.send(MspCommand.SET_RAW_RC, {
                    "vx": 0.0, "vy": 0.0, "vz": 0.0, "yaw_rate": 0.0,
                })
                await asyncio.sleep(self.obstacle_avoider.params.no_feature_hover_sec)
                continue
            await fc_bridge.send(MspCommand.SET_RAW_RC, {
                "vx": ev.chosen.vx, "vy": ev.chosen.vy,
                "vz": ev.chosen.vz, "yaw_rate": ev.chosen.yaw_rate,
            })
            await asyncio.sleep(0.1)
        logger.warning("mission.wp_timeout", wp=(wp.x, wp.y, wp.z))

    # nose-tilt 캡처 파라미터 (cinewhoop FPV 한계 보완)
    TILT_CAPTURE_LIMIT_RAD = math.radians(30.0)   # 안전 임계 (TILT_LIMIT_DEG=60° 의 절반 이내)
    TILT_HOLD_SEC = 1.0
    TILT_RECOVER_SEC = 0.6
    YAW_SWEEP_STEPS = 4   # 90° × 4 = 360°

    async def _capture_cell(self, room_idx: int, wp: Waypoint) -> None:
        """
        셀 캡처 트리거.
          - 보조(discrepancy) WP : 추가 통과 캡처만 수행, captured 추적 X (일반 WP 가 추적)
          - FACE_CEILING/FLOOR  : nose-tilt + yaw 360° sweep
          - 그 외(벽/창호)         : RGB/Thermal 동기화 윈도우만
        """
        # 보조 WP — captured set 진입 X (같은 cell_idx 의 일반 WP 가 별도로 진행)
        if wp.purpose == "discrepancy":
            if wp.face_kind in (FACE_CEILING, FACE_FLOOR):
                await self._capture_with_tilt_sweep(wp)
            else:
                await asyncio.sleep(self.CELL_DOUBLE_CAPTURE_SEC)
            return

        cell = wp.cell_idx
        if cell in self.state.captured_cells:
            return

        if wp.face_kind in (FACE_CEILING, FACE_FLOOR):
            await self._capture_with_tilt_sweep(wp)
        else:
            await asyncio.sleep(self.CELL_DOUBLE_CAPTURE_SEC)

        self.state.captured_cells.add(cell)
        await self._persist_cell_captured(room_idx, cell)
        await self._broadcast("coverage.cell", {
            "room_idx": room_idx,
            "cell": list(cell),
            "world": [wp.x, wp.y, wp.z],
            "face_kind": wp.face_kind, "face_idx": wp.face_idx,
        })

    async def _capture_with_tilt_sweep(self, wp: Waypoint) -> None:
        """
        천장/바닥 face WP 에서 nose-tilt + yaw 360° 4-step 회전.
        절차(WP 1점당):
          1) hover 안정 0.4s
          2) yaw 0/90/180/270° 4-step. 각 step에서:
             - SET_ATTITUDE(pitch=cam_pitch_rad clipped to ±30°, yaw=현재 yaw)
             - TILT_HOLD_SEC 동안 캡처 윈도우 — RGB/Thermal 동기화
             - SET_ATTITUDE(pitch=0) 복귀, TILT_RECOVER_SEC 안정
          3) 표류 감지 시 즉시 hover 종료 (safety_monitor 가 별도 가드)
        cinewhoop 자세 기울임은 추력 벡터 분산 → 위치 표류 위험.
        매 step 사이 위치를 재정렬(SET_RAW_RC 0,0,0,0)하고 obstacle_avoider 안전 확인.
        """
        # 안전 클리핑
        pitch = max(-self.TILT_CAPTURE_LIMIT_RAD,
                    min(self.TILT_CAPTURE_LIMIT_RAD, wp.cam_pitch_rad))
        if pitch == 0.0:
            await asyncio.sleep(self.CELL_DOUBLE_CAPTURE_SEC)
            return

        # 1) hover 안정
        await fc_bridge.send(MspCommand.SET_RAW_RC, {
            "vx": 0.0, "vy": 0.0, "vz": 0.0, "yaw_rate": 0.0,
        })
        await asyncio.sleep(0.4)

        base_yaw = wp.yaw_rad
        for step in range(self.YAW_SWEEP_STEPS):
            yaw = base_yaw + step * (math.tau / self.YAW_SWEEP_STEPS)
            # tilt 명령
            await fc_bridge.send(MspCommand.SET_ATTITUDE, {
                "roll_rad": 0.0, "pitch_rad": pitch, "yaw_rad": yaw,
                "thrust_norm": 0.5,    # 정지 추력 — INAV 가 alt-hold 보상 가정
            })
            # 캡처 윈도우
            await asyncio.sleep(self.TILT_HOLD_SEC)
            # 표류 감지 — safety_monitor 가 매 100ms 체크. 추가로 즉시 평가
            pose = self.state.last_pose
            if pose is not None and pose.pos_var_m > 0.4:
                logger.warning("mission.tilt_drift_abort",
                               pos_var_m=pose.pos_var_m, step=step)
                break
            # tilt 복귀 (안전상 매 step 후 복귀)
            await fc_bridge.send(MspCommand.SET_ATTITUDE, {
                "roll_rad": 0.0, "pitch_rad": 0.0, "yaw_rad": yaw,
                "thrust_norm": 0.5,
            })
            await asyncio.sleep(self.TILT_RECOVER_SEC)

        # 명시적 hover 복귀
        await fc_bridge.send(MspCommand.SET_RAW_RC, {
            "vx": 0.0, "vy": 0.0, "vz": 0.0, "yaw_rate": 0.0,
        })

    # ── 안전 가드 루프 ─────────────────────
    async def _safety_guard_loop(self) -> None:
        while True:
            await asyncio.sleep(self.GUARD_INTERVAL_SEC)
            try:
                snap = self._build_safety_snapshot()
                await safety_monitor.check(snap)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("mission.guard_error", error=str(e))

    def _build_safety_snapshot(self) -> TelemetrySnapshot:
        t = self.state.last_telemetry
        p = self.state.last_pose
        # 자세 기반 tilt
        tilt_deg = None
        if t and t.roll is not None and t.pitch is not None:
            tilt_deg = math.degrees(math.sqrt(t.roll * t.roll + t.pitch * t.pitch))
        # 위치 분산: 최근 N 포즈 표준편차
        pos_var_m = None
        if p is not None:
            self._pose_history.append((p.x, p.y, p.z))
            if len(self._pose_history) > self.POSE_VARIANCE_WINDOW:
                self._pose_history = self._pose_history[-self.POSE_VARIANCE_WINDOW:]
            if len(self._pose_history) >= 3:
                arr = list(self._pose_history)
                xs = [a[0] for a in arr]; ys = [a[1] for a in arr]; zs = [a[2] for a in arr]
                def std(v: List[float]) -> float:
                    m = sum(v) / len(v)
                    return math.sqrt(sum((x - m) ** 2 for x in v) / len(v))
                pos_var_m = std(xs) + std(ys) + std(zs)
        # 지오펜스
        geofence_ok = True
        if self.state.geofence_polygon and p is not None:
            geofence_ok = _point_in_polygon((p.x, p.y), self.state.geofence_polygon)
        # 장애물 최소거리(현 위치 안전회랑)
        obs_min = None
        if p is not None:
            obs_min = self.obstacle_avoider._min_dist_in_corridor(
                (p.x, p.y, p.z),
                self.obstacle_avoider.params.safety_radius_m,
                time.time(),
            )
        return TelemetrySnapshot(
            battery_pct=(t.battery_pct if t else None),
            voltage_per_cell=((t.voltage / 4.0) if (t and t.voltage) else None),
            tilt_deg=tilt_deg,
            pos_var_m=pos_var_m,
            last_telemetry_ts=(t.ts if t else 0.0),
            last_pi_heartbeat_ts=fc_bridge.last_heartbeat_ts,
            inside_geofence=geofence_ok,
            obstacle_min_dist_m=obs_min,
            slam_confidence=(p.confidence if p else None),
        )

    # ── 콜백 ──────────────────────────────
    async def _on_telemetry(self, frame: TelemetryFrame) -> None:
        self.state.last_telemetry = frame

    async def _on_pose(self, pose: SlamPose) -> None:
        self.state.last_pose = pose

    async def _on_pointcloud(self, delta) -> None:
        # obstacle_avoider voxel 갱신
        if delta and delta.points_xyz:
            self.obstacle_avoider.update_pointcloud(delta.points_xyz)
        # WS 브로드캐스트 (frontend PointCloudLayer)
        await self._broadcast("pointcloud.delta", {
            "frame_idx": delta.frame_idx,
            "points": delta.points_xyz,
            "colors": delta.points_rgb,
            "pose": {"x": delta.pose.x, "y": delta.pose.y, "z": delta.pose.z},
        })

    async def _on_safety_decision(self, decision: SafetyDecision) -> None:
        if decision.action in (SafetyAction.LAND, SafetyAction.ESTOP):
            await self._fail(decision.reason)
        elif decision.action is SafetyAction.RTL:
            await self.rtl()
        elif decision.action is SafetyAction.HOVER:
            await self.pause()

    # ── 전이 ──────────────────────────────
    async def _transition(self, new_phase: MissionPhase) -> None:
        prev = self.state.phase
        self.state.phase = new_phase
        logger.info("mission.transition", prev=prev.value, next=new_phase.value)
        await self._persist_phase(new_phase)
        await self._broadcast("mission.phase", {
            "prev": prev.value, "next": new_phase.value,
            "mission_id": self.state.mission_id,
            "current_room_idx": self.state.current_room_idx,
        })

    async def _fail(self, reason: str) -> None:
        if self.state.phase is MissionPhase.FAILSAFE:
            return  # 이미 실패 상태
        self.state.failure_reason = reason
        await self._transition(MissionPhase.FAILSAFE)
        await fc_bridge.send(MspCommand.LAND)
        if self._task and not self._task.done():
            self._task.cancel()

    # ── DB 영속화 ─────────────────────────
    async def _persist_phase(self, phase: MissionPhase) -> None:
        if self.state.mission_id is None:
            return
        # phase → mission status 매핑 (DB ENUM mission_status_enum)
        if phase is MissionPhase.FAILSAFE:
            new_status = "failsafe"
        elif phase is MissionPhase.IDLE:
            # IDLE = 미션 종료. 실패 사유가 있으면 aborted, 없으면 completed
            new_status = "aborted" if self.state.failure_reason else "completed"
        else:
            new_status = "running"
        # finished_at: IDLE/COMPLETE/FAILSAFE 중 첫 진입 시 NOW, 그 외는 이미 set 된 값 유지
        finished_at_set = phase in (MissionPhase.IDLE, MissionPhase.COMPLETE, MissionPhase.FAILSAFE)
        try:
            async with async_session_factory() as db:
                values = dict(
                    current_phase=phase.value,
                    status=new_status,
                    current_room_idx=self.state.current_room_idx,
                    failure_reason=self.state.failure_reason,
                )
                if finished_at_set:
                    values["finished_at"] = datetime.now(tz=timezone.utc)
                stmt = (
                    update(MissionPlan)
                    .where(MissionPlan.id == self.state.mission_id)
                    .values(**values)
                )
                await db.execute(stmt)
                await db.commit()
        except Exception as e:
            logger.error("mission.persist_phase_failed", error=str(e))

    async def _persist_topology(self, topo: RoomTopologyGraph) -> None:
        if self.state.mission_id is None:
            return
        try:
            async with async_session_factory() as db:
                row = RoomTopology(
                    mission_id=self.state.mission_id,
                    nodes_json=topo.to_jsonable()["nodes"],
                    edges_json=topo.to_jsonable()["edges"],
                )
                db.add(row)
                await db.commit()
        except Exception as e:
            logger.error("mission.persist_topology_failed", error=str(e))

    async def _seed_coverage_grids(self, plan: MissionGridPlan) -> None:
        if self.state.mission_id is None:
            return
        try:
            async with async_session_factory() as db:
                for r in plan.rooms.values():
                    for w in r.waypoints:
                        if w.purpose == "discrepancy":
                            continue   # 보조 WP — 시드 X
                        cell = CoverageGrid(
                            mission_id=self.state.mission_id,
                            room_idx=r.room_idx,
                            cell_x=w.cell_idx[0], cell_y=w.cell_idx[1], cell_z=w.cell_idx[2],
                            world_x=w.x, world_y=w.y, world_z=w.z,
                            captured=False,
                            face_kind=w.face_kind,
                            face_idx=w.face_idx,
                            cam_pitch_rad=w.cam_pitch_rad,
                        )
                        db.add(cell)
                await db.commit()
        except Exception as e:
            logger.error("mission.seed_coverage_failed", error=str(e))

    async def _persist_area_summary(self, summary: MissionAreaSummary) -> None:
        """plan_json["area_summary"] 에 면적/커버리지 요약 저장."""
        if self.state.mission_id is None:
            return
        try:
            async with async_session_factory() as db:
                stmt = (
                    update(MissionPlan)
                    .where(MissionPlan.id == self.state.mission_id)
                    .values(plan_json={"area_summary": summary.to_jsonable()})
                )
                await db.execute(stmt)
                await db.commit()
        except Exception as e:
            logger.error("mission.persist_area_failed", error=str(e))

    async def _persist_cell_captured(
        self, room_idx: int, cell: Tuple[int, int, int],
    ) -> None:
        if self.state.mission_id is None:
            return
        try:
            async with async_session_factory() as db:
                stmt = (
                    update(CoverageGrid)
                    .where(
                        CoverageGrid.mission_id == self.state.mission_id,
                        CoverageGrid.room_idx == room_idx,
                        CoverageGrid.cell_x == cell[0],
                        CoverageGrid.cell_y == cell[1],
                        CoverageGrid.cell_z == cell[2],
                    )
                    .values(
                        captured=True,
                        captured_at=datetime.now(tz=timezone.utc),
                    )
                )
                await db.execute(stmt)
                await db.commit()
        except Exception as e:
            logger.error("mission.persist_capture_failed", error=str(e))

    # ── WS 브로드캐스트 ────────────────────
    async def _broadcast(self, channel: str, payload: dict) -> None:
        """
        프론트 dispatch 와 정합:
          {"type": <channel>, "data": <payload>} 형태로 모든 활성 WS 에 broadcast.
          기존 useWebSocket messageHandlers 가 type 기준 디스패치하므로 단일 WS 연결로 흡수.
        """
        if _ws_manager is None:
            return
        try:
            await _ws_manager.broadcast_all({"type": channel, "data": payload})
        except Exception as e:
            logger.error("mission.broadcast_failed", channel=channel, error=str(e))


# ── 보조: ray casting point-in-polygon ───
def _point_in_polygon(point: Tuple[float, float], polygon: List[Tuple[float, float]]) -> bool:
    x, y = point
    n = len(polygon)
    inside = False
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            xinters = (y - y1) * (x2 - x1) / (y2 - y1 + 1e-12) + x1
            if x < xinters:
                inside = not inside
    return inside


# ── 싱글톤 ──────────────────────────────
mission_orchestrator = MissionOrchestrator()
