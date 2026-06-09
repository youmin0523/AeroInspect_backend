"""
services/autonomous_flight_simulator.py
역할: 자율비행 + LiDAR 스캔 시뮬레이터 — Gazebo 미가용 환경에서 백엔드 단독 실행 가능
      - 평면도 추출 결과(walls + outline) 를 2D 폴리라인 환경으로 사용
      - 드론이 boustrophedon (Z형) 격자 비행 경로 따라 이동 시뮬레이션
      - 각 위치에서 360° LiDAR raycast → 벽까지 거리 → 3D 점 (z=0~ceiling) 생성
      - WebSocket 'defects' 채널에 다음 이벤트 점진 publish:
          - { type: 'telemetry.update', data: {x,y,z,yaw,...} }    : 1Hz 위치 업데이트
          - { type: 'lidar.points', data: { points: [[x,y,z],...], total } } : 새 점 batch

이 시뮬레이터는 실제 Gazebo / ROS2 / MAVLink 가 없을 때 동등한 데이터 흐름을 제공한다.
실제 Gazebo 환경에서는 이 모듈 대신 ros2 → telemetry / lidar 토픽 → ws_manager.broadcast 를 연결한다.
"""

from __future__ import annotations

import asyncio
import math
import random
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

from app.core.ws_manager import ws_manager


# ── 비행/LiDAR 파라미터 ──────────────────────
DEFAULT_ALTITUDE_M = 1.5            # 단일층 기본 고도 (호환)
DEFAULT_SPEED_MPS = 0.8             # 비행 속도 (m/s)
DEFAULT_LIDAR_HZ = 10               # 초당 스캔 횟수
DEFAULT_LIDAR_BEAMS = 36            # 수평 스캔 당 빔 수 (각 10°)
DEFAULT_LIDAR_RANGE = 12.0          # LiDAR 최대 사거리 (m)
DEFAULT_TELEMETRY_HZ = 5
DEFAULT_BATCH_SIZE = 60
DEFAULT_DRONE_RADIUS_M = 0.25       # 드론 외곽 반경 (DJI Mini 급)
MAX_DETOUR_DEPTH = 3

# //* [Modified Code 2026-05-13 v3] 하자 검출 sweep 모드 강화
# - 기본 layers: 저(걸레받이) / 중(일반) / 고(천장·가구 위)
# - lane_spacing 0.5m 로 격자 빈틈 최소화 (LiDAR 사거리 6m + 36 빔 = 인접 라인 충분 겹침)
# - 벽 마진 0.35m (드론 0.25 + 0.1)
# - 가구 회피 반경 분기: builtin(벽 인접) = 보수적 / freestanding = 짧게 → 둘레 통과
DEFAULT_LANE_SPACING_M = 0.5
DEFAULT_WALL_MARGIN_M = 0.35
DEFAULT_ALTITUDE_LAYERS = (0.4, 1.5, 2.5)    # 걸레받이 / 일반 / 천장·가구위
DEFAULT_FURNITURE_HEIGHT_M = 1.0             # 가구 평균 높이 추정 (over-fly 분기 기준)
BUILTIN_AVOIDANCE_MARGIN_M = 0.4
FREESTANDING_AVOIDANCE_MARGIN_M = 0.15       # freestanding 은 짧게 → 둘레 sweep
VERTICAL_BEAM_RANGE_M = 4.0                  # 천장/바닥 raycast 사거리


@dataclass
class MissionState:
    mission_id: str
    floorplan_id: Optional[str]
    walls_world: list[tuple[float, float, float, float]]
    world_w: float
    world_d: float
    altitude: float = DEFAULT_ALTITUDE_M
    speed: float = DEFAULT_SPEED_MPS
    started_at: float = field(default_factory=time.time)
    ended_at: Optional[float] = None
    status: str = "running"
    cancel_event: asyncio.Event = field(default_factory=asyncio.Event)
    points_emitted: int = 0
    progress: float = 0.0
    furniture_obstacles: list[tuple[float, float, float]] = field(default_factory=list)
    furniture_avoidance_count: int = 0
    # //* [Modified Code 2026-05-13 v3] 다층 sweep 모드
    altitude_layers: tuple = DEFAULT_ALTITUDE_LAYERS
    current_layer_idx: int = 0
    ceiling_height: float = 2.7
    furniture: list[dict] = field(default_factory=list)  # 원본 — 레이어별 회피 재계산용
    outline: list[dict] = field(default_factory=list)


# 모듈 레벨 미션 레지스트리 (단일 프로세스 가정)
_active_missions: dict[str, MissionState] = {}


def get_mission(mission_id: str) -> Optional[MissionState]:
    return _active_missions.get(mission_id)


def list_active_missions() -> list[dict]:
    return [
        {
            "mission_id": m.mission_id,
            "floorplan_id": m.floorplan_id,
            "status": m.status,
            "progress": round(m.progress, 3),
            "points_emitted": m.points_emitted,
            "started_at": m.started_at,
            "ended_at": m.ended_at,
        }
        for m in _active_missions.values()
    ]


def cancel_mission(mission_id: str) -> bool:
    m = _active_missions.get(mission_id)
    if not m or m.status != "running":
        return False
    m.cancel_event.set()
    return True


def _norm_walls_to_world(
    walls: list[dict],
    outline: list[dict],
    world_w: float,
    world_d: float,
    furniture: list[dict] | None = None,
) -> list[tuple[float, float, float, float]]:
    """
    정규화 0-1 벽 + outline (+ 가구 회전 사각형 4변) → 월드 m 단위 선분 리스트.
    가구도 LiDAR raycast 대상에 포함시켜 드론이 인식할 수 있게 함.
    """
    segments: list[tuple[float, float, float, float]] = []

    def to_world_seg(x1n, y1n, x2n, y2n):
        return (
            (x1n - 0.5) * world_w,
            (y1n - 0.5) * world_d,
            (x2n - 0.5) * world_w,
            (y2n - 0.5) * world_d,
        )

    for w in walls or []:
        segments.append(to_world_seg(w["x1"], w["y1"], w["x2"], w["y2"]))

    if outline and len(outline) >= 3:
        for i in range(len(outline)):
            pt = outline[i]
            nxt = outline[(i + 1) % len(outline)]
            segments.append(to_world_seg(pt["x"], pt["y"], nxt["x"], nxt["y"]))

    # //* [Modified Code 2026-05-13] 가구 회전 사각형 → 4 선분 (LiDAR raycast 대상)
    if furniture:
        for f in furniture:
            cx_n = f.get("cx"); cy_n = f.get("cy")
            fw_n = f.get("w");  fd_n = f.get("h")
            if cx_n is None or cy_n is None or not fw_n or not fd_n:
                continue
            cx = (cx_n - 0.5) * world_w
            cy = (cy_n - 0.5) * world_d
            hw = (fw_n * world_w) / 2
            hd = (fd_n * world_d) / 2
            angle = math.radians(f.get("angle", 0))
            cos_a, sin_a = math.cos(angle), math.sin(angle)

            # 회전된 사각형 4 코너
            corners_local = [(-hw, -hd), (hw, -hd), (hw, hd), (-hw, hd)]
            corners_world = [
                (cx + lx * cos_a - ly * sin_a, cy + lx * sin_a + ly * cos_a)
                for (lx, ly) in corners_local
            ]
            for i in range(4):
                x1, y1 = corners_world[i]
                x2, y2 = corners_world[(i + 1) % 4]
                segments.append((x1, y1, x2, y2))

    return segments


def _furniture_obstacle_circles(
    furniture: list[dict] | None,
    world_w: float,
    world_d: float,
    drone_radius: float = DEFAULT_DRONE_RADIUS_M,
    builtin_margin: float = BUILTIN_AVOIDANCE_MARGIN_M,
    freestanding_margin: float = FREESTANDING_AVOIDANCE_MARGIN_M,
    skip_at_altitude: float | None = None,
    furniture_height: float = DEFAULT_FURNITURE_HEIGHT_M,
) -> list[tuple[float, float, float]]:
    """
    가구 → 회피용 원형 장애물 (cx, cy, radius_with_margin).

    //* [Modified Code 2026-05-13 v3] 정책 분기:
      - is_builtin=True  : 큰 회피 (margin 0.4m) — 뒤편이 벽이라 우회 무의미
      - is_builtin=False : 짧은 회피 (margin 0.15m) — 가구 둘레 가까이 통과 → 뒤편도 LiDAR 도달
      - skip_at_altitude > furniture_height + 0.5m : 가구 위 over-fly 가능 → 회피 안 함
    """
    obstacles: list[tuple[float, float, float]] = []
    # 고고도 비행 시 가구 위로 통과 (충돌 마진 충분)
    if skip_at_altitude is not None and skip_at_altitude >= furniture_height + 0.5:
        return obstacles
    for f in furniture or []:
        cx_n = f.get("cx"); cy_n = f.get("cy")
        fw_n = f.get("w");  fd_n = f.get("h")
        if cx_n is None or cy_n is None or not fw_n or not fd_n:
            continue
        cx = (cx_n - 0.5) * world_w
        cy = (cy_n - 0.5) * world_d
        hw = (fw_n * world_w) / 2
        hd = (fd_n * world_d) / 2
        # builtin = 큰 회피 (벽 인접, 뒤편 무의미) / freestanding = 짧은 회피 (둘레 sweep)
        margin = builtin_margin if f.get("is_builtin") else freestanding_margin
        radius = math.hypot(hw, hd) + drone_radius + margin
        obstacles.append((cx, cy, radius))
    return obstacles


def _ray_segment_intersection(
    ox: float, oy: float, dx: float, dy: float,
    x1: float, y1: float, x2: float, y2: float,
) -> Optional[float]:
    """
    Ray (origin=(ox,oy), direction=(dx,dy)) 와 선분 ((x1,y1)-(x2,y2)) 의 교차 거리 t.
    교차 없으면 None.
    """
    sx = x2 - x1
    sy = y2 - y1
    denom = dx * sy - dy * sx
    if abs(denom) < 1e-9:
        return None  # 평행
    t = ((x1 - ox) * sy - (y1 - oy) * sx) / denom
    u = ((x1 - ox) * dy - (y1 - oy) * dx) / denom
    if t >= 0 and 0 <= u <= 1:
        return t
    return None


def _scan_lidar_at(
    cx: float, cy: float,
    walls_world: list[tuple[float, float, float, float]],
    lidar_range: float,
    beam_count: int,
    altitude: float,
    ceiling_height: float = 2.7,
    enable_vertical_beams: bool = True,
) -> list[tuple[float, float, float]]:
    """
    위치 (cx, cy, altitude) 에서 LiDAR 스캔 1회 → 3D 점 리스트.

    //* [Modified Code 2026-05-13 v3]
      - 수평 360° (beam_count 빔) → 벽 + 가구 측면 점
      - 수직 위/아래 빔 → 천장/바닥 직접 측정 (걸레받이/몰딩 점검용)
    """
    points: list[tuple[float, float, float]] = []

    # 수평 360° — 벽/가구 측면
    for i in range(beam_count):
        theta = (2 * math.pi * i) / beam_count
        dx = math.cos(theta)
        dy = math.sin(theta)
        nearest = lidar_range
        for (x1, y1, x2, y2) in walls_world:
            t = _ray_segment_intersection(cx, cy, dx, dy, x1, y1, x2, y2)
            if t is not None and 0 < t < nearest:
                nearest = t
        if nearest < lidar_range - 1e-3:
            hx = cx + dx * nearest
            hy = cy + dy * nearest
            # z 는 비행 고도 + 약간의 jitter (수평 빔이 천장↔바닥 사이 산란)
            hz = max(0.02, min(altitude + random.uniform(-0.4, 0.4), ceiling_height - 0.05))
            points.append((hx, hy, hz))

    # 수직 빔 — 천장 + 바닥 직접 점. 걸레받이/몰딩 검출용
    if enable_vertical_beams:
        # 바닥 (z=0)
        points.append((cx + random.uniform(-0.02, 0.02),
                       cy + random.uniform(-0.02, 0.02), 0.01))
        # 천장
        points.append((cx + random.uniform(-0.02, 0.02),
                       cy + random.uniform(-0.02, 0.02), ceiling_height))
        # 사선 빔 4개 (45° 위·아래 × 4 방향) — 벽/천장 모서리·걸레받이
        for theta_deg in (0, 90, 180, 270):
            theta = math.radians(theta_deg)
            for elev in (-0.6, 0.6):  # 아래/위 약 35°
                dx = math.cos(theta) * math.cos(elev)
                dy = math.sin(theta) * math.cos(elev)
                # 천장/바닥 평면 도달 거리
                if elev > 0:
                    t_plane = (ceiling_height - altitude) / math.sin(elev)
                else:
                    t_plane = -altitude / math.sin(elev)
                # 벽 도달 거리
                nearest = min(t_plane, lidar_range)
                for (x1, y1, x2, y2) in walls_world:
                    t = _ray_segment_intersection(cx, cy, dx, dy, x1, y1, x2, y2)
                    if t is not None and 0 < t < nearest:
                        nearest = t
                if nearest > 0.05:
                    hx = cx + dx * nearest
                    hy = cy + dy * nearest
                    hz = altitude + math.sin(elev) * nearest
                    hz = max(0.01, min(hz, ceiling_height))
                    points.append((hx, hy, hz))

    return points


def _generate_boustrophedon_path(
    world_w: float,
    world_d: float,
    lane_spacing: float,
    margin: float = DEFAULT_WALL_MARGIN_M,
    obstacles: list[tuple[float, float, float]] | None = None,
) -> tuple[list[tuple[float, float]], int]:
    """
    Z-형 격자 비행 경로. 가구 장애물(obstacles)이 있으면 그 주변을 우회하는
    중간 waypoint를 삽입해 충돌 회피.

    Args:
        obstacles: [(cx, cy, radius_with_margin), ...] — 가구별 회피 원

    Returns:
        (waypoints, avoidance_insertions)
        avoidance_insertions: 회피로 인해 삽입된 추가 waypoint 수 (텔레메트리 보고용)
    """
    waypoints: list[tuple[float, float]] = []
    half_w = world_w / 2 - margin
    half_d = world_d / 2 - margin
    if half_w <= 0 or half_d <= 0:
        return [(0.0, 0.0)], 0

    raw: list[tuple[float, float]] = []
    y = -half_d
    direction = 1
    while y <= half_d + 1e-6:
        if direction == 1:
            raw.append((-half_w, y))
            raw.append(( half_w, y))
        else:
            raw.append(( half_w, y))
            raw.append((-half_w, y))
        y += lane_spacing
        direction *= -1

    # 가구 회피 — 각 라인 세그먼트와 장애물 원 교차 시 우회 waypoint 삽입
    obstacles = obstacles or []
    insertions = 0

    if not obstacles:
        return raw, 0

    waypoints.append(raw[0])
    for i in range(1, len(raw)):
        prev = waypoints[-1]
        target = raw[i]
        # //* [Modified Code 2026-05-13 v2] 재귀 우회 — 회피 waypoint 가 다른
        # 가구와 충돌하면 다시 우회를 삽입 (MAX_DETOUR_DEPTH 한계).
        detour_chain = _detour_chain(
            prev, target, obstacles, world_w, world_d, margin,
            depth=0, max_depth=MAX_DETOUR_DEPTH,
        )
        if detour_chain:
            waypoints.extend(detour_chain)
            insertions += len(detour_chain)
        waypoints.append(target)

    return waypoints, insertions


def _detour_chain(
    p_from: tuple[float, float],
    p_to: tuple[float, float],
    obstacles: list[tuple[float, float, float]],
    world_w: float,
    world_d: float,
    margin: float,
    depth: int,
    max_depth: int,
) -> list[tuple[float, float]]:
    """
    p_from → p_to 의 모든 장애물을 재귀 우회. 우회 waypoint 가 또 다른 장애물과
    충돌하면 다시 우회 삽입. 최대 max_depth 까지.
    """
    if depth >= max_depth:
        return []

    detour = _detour_around_obstacles(p_from, p_to, obstacles, world_w, world_d, margin)
    if not detour:
        return []

    chain: list[tuple[float, float]] = []
    cur = p_from
    for d in detour:
        # cur → d 구간이 또 다른 가구와 충돌하면 재귀
        sub = _detour_chain(cur, d, obstacles, world_w, world_d, margin,
                            depth + 1, max_depth)
        chain.extend(sub)
        chain.append(d)
        cur = d
    # 마지막 detour 점 → p_to 구간도 재검사
    final_sub = _detour_chain(cur, p_to, obstacles, world_w, world_d, margin,
                              depth + 1, max_depth)
    chain.extend(final_sub)
    return chain


def _detour_around_obstacles(
    p_from: tuple[float, float],
    p_to: tuple[float, float],
    obstacles: list[tuple[float, float, float]],
    world_w: float,
    world_d: float,
    margin: float,
) -> list[tuple[float, float]]:
    """
    선분 (p_from → p_to) 가 장애물 원과 교차하면 가장 가까운 회피 waypoint 1개 반환.
    원 중심에서 선분 수직방향으로 (radius + 0.3m) 떨어진 점.
    """
    half_w = world_w / 2 - margin
    half_d = world_d / 2 - margin

    for (cx, cy, r) in obstacles:
        if not _segment_circle_intersects(p_from, p_to, cx, cy, r):
            continue

        # 선분 방향 단위벡터
        dx = p_to[0] - p_from[0]
        dy = p_to[1] - p_from[1]
        length = math.hypot(dx, dy)
        if length < 1e-6:
            continue
        ux, uy = dx / length, dy / length
        # 수직 방향 (왼쪽/오른쪽 중 월드 안쪽으로)
        nx, ny = -uy, ux
        offset = r + 0.3

        cand_a = (cx + nx * offset, cy + ny * offset)
        cand_b = (cx - nx * offset, cy - ny * offset)

        def in_bounds(p):
            return -half_w <= p[0] <= half_w and -half_d <= p[1] <= half_d

        if in_bounds(cand_a) and not in_bounds(cand_b):
            return [cand_a]
        if in_bounds(cand_b) and not in_bounds(cand_a):
            return [cand_b]
        # 둘 다 가능 → 출발점에 가까운 쪽
        if in_bounds(cand_a) and in_bounds(cand_b):
            d_a = math.hypot(cand_a[0] - p_from[0], cand_a[1] - p_from[1])
            d_b = math.hypot(cand_b[0] - p_from[0], cand_b[1] - p_from[1])
            return [cand_a if d_a <= d_b else cand_b]
        # 둘 다 경계 밖 → 클램프 후 사용
        clamped = (max(-half_w, min(half_w, cand_a[0])),
                   max(-half_d, min(half_d, cand_a[1])))
        return [clamped]

    return []


def _segment_circle_intersects(
    p1: tuple[float, float],
    p2: tuple[float, float],
    cx: float, cy: float, r: float,
) -> bool:
    """선분과 원의 교차 여부 (가장 가까운 점까지의 거리 ≤ r)."""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    length_sq = dx * dx + dy * dy
    if length_sq < 1e-9:
        return math.hypot(p1[0] - cx, p1[1] - cy) <= r
    t = ((cx - p1[0]) * dx + (cy - p1[1]) * dy) / length_sq
    t = max(0.0, min(1.0, t))
    nearest_x = p1[0] + t * dx
    nearest_y = p1[1] + t * dy
    return math.hypot(nearest_x - cx, nearest_y - cy) <= r


async def run_autonomous_scan(
    walls: list[dict],
    outline: list[dict],
    world_w: float,
    world_d: float,
    floorplan_id: Optional[str] = None,
    altitude: float = DEFAULT_ALTITUDE_M,
    speed: float = DEFAULT_SPEED_MPS,
    lane_spacing: float = DEFAULT_LANE_SPACING_M,
    lidar_hz: int = DEFAULT_LIDAR_HZ,
    lidar_beams: int = DEFAULT_LIDAR_BEAMS,
    lidar_range: float = DEFAULT_LIDAR_RANGE,
    telemetry_hz: int = DEFAULT_TELEMETRY_HZ,
    batch_size: int = DEFAULT_BATCH_SIZE,
    channel: str = "defects",
    furniture: Optional[list[dict]] = None,
    altitude_layers: tuple = DEFAULT_ALTITUDE_LAYERS,
    ceiling_height: float = 2.7,
) -> str:
    """
    자율비행 + LiDAR 스캔 시뮬레이션 (다층 sweep).

    //* [Modified Code 2026-05-13 v3]
      - altitude_layers: 저(걸레받이) / 중(일반) / 고(천장·가구위) 차례로 boustrophedon
      - lane_spacing 기본 0.5m → 격자 빈틈 최소화
      - 벽 margin 0.35m → 벽 가까이 접근
      - 가구 회피 분기: builtin = 큰 회피, freestanding = 짧은 회피 (둘레 sweep)
      - 고고도 layer 에서는 가구 회피 안 함 (over-fly)
    """
    mission_id = str(uuid.uuid4())
    walls_world = _norm_walls_to_world(walls, outline, world_w, world_d, furniture=furniture)

    state = MissionState(
        mission_id=mission_id,
        floorplan_id=floorplan_id,
        walls_world=walls_world,
        world_w=world_w,
        world_d=world_d,
        altitude=altitude_layers[0] if altitude_layers else altitude,
        speed=speed,
        altitude_layers=tuple(altitude_layers) if altitude_layers else (altitude,),
        ceiling_height=ceiling_height,
        furniture=furniture or [],
        outline=outline or [],
    )
    _active_missions[mission_id] = state

    asyncio.create_task(_fly_and_scan(
        state,
        lane_spacing=lane_spacing,
        lidar_hz=lidar_hz,
        lidar_beams=lidar_beams,
        lidar_range=lidar_range,
        telemetry_hz=telemetry_hz,
        batch_size=batch_size,
        channel=channel,
    ))

    return mission_id


async def _fly_and_scan(
    state: MissionState,
    lane_spacing: float,
    lidar_hz: int,
    lidar_beams: int,
    lidar_range: float,
    telemetry_hz: int,
    batch_size: int,
    channel: str,
) -> None:
    """
    다층 비행 + LiDAR sweep 루프.
    각 altitude_layer 마다 boustrophedon 한 바퀴, 가구 회피 분기 (builtin/freestanding/over-fly).
    """
    try:
        # 레이어별 경로 + 거리 사전 산출 (전체 진행률 산정용)
        layer_plans = []  # [(altitude, path, distance, insertions)]
        for layer_idx, alt in enumerate(state.altitude_layers):
            obstacles = _furniture_obstacle_circles(
                state.furniture, state.world_w, state.world_d,
                skip_at_altitude=alt,
                furniture_height=DEFAULT_FURNITURE_HEIGHT_M,
            )
            path, insertions = _generate_boustrophedon_path(
                state.world_w, state.world_d, lane_spacing,
                obstacles=obstacles,
            )
            dist = sum(
                math.hypot(path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
                for i in range(len(path) - 1)
            ) if len(path) > 1 else 0.0
            layer_plans.append((alt, path, dist, insertions, obstacles))

        total_distance = sum(p[2] for p in layer_plans) or 1e-3
        state.furniture_avoidance_count = sum(p[3] for p in layer_plans)
        # 첫 레이어 obstacles 만 store (호환)
        state.furniture_obstacles = layer_plans[0][4]

        dt = 1.0 / max(telemetry_hz, 1)
        scan_interval = 1.0 / max(lidar_hz, 1)
        last_scan_t = 0.0
        elapsed_distance = 0.0

        for layer_idx, (alt, path, dist, _ins, _obs) in enumerate(layer_plans):
            state.current_layer_idx = layer_idx
            state.altitude = alt

            if len(path) < 2:
                pts = _scan_lidar_at(0, 0, state.walls_world,
                                     lidar_range, lidar_beams, alt,
                                     ceiling_height=state.ceiling_height)
                if pts:
                    await _publish_points(channel, state.mission_id, pts)
                    state.points_emitted += len(pts)
                continue

            cx, cy = path[0]
            await _publish_telemetry(channel, state, cx, cy, yaw=0.0)

            for seg_idx in range(len(path) - 1):
                if state.cancel_event.is_set():
                    state.status = "cancelled"
                    state.ended_at = time.time()
                    return

                sx, sy = path[seg_idx]
                ex, ey = path[seg_idx + 1]
                seg_dist = math.hypot(ex - sx, ey - sy)
                if seg_dist < 1e-6:
                    continue
                yaw = math.atan2(ey - sy, ex - sx)

                steps = max(int(seg_dist / (state.speed * dt)), 1)
                for s in range(1, steps + 1):
                    if state.cancel_event.is_set():
                        state.status = "cancelled"
                        state.ended_at = time.time()
                        return

                    ratio = s / steps
                    cx = sx + (ex - sx) * ratio
                    cy = sy + (ey - sy) * ratio
                    step_dist = seg_dist / steps
                    elapsed_distance += step_dist

                    await _publish_telemetry(channel, state, cx, cy, yaw=yaw)
                    state.progress = min(elapsed_distance / total_distance, 1.0)

                    last_scan_t += dt
                    if last_scan_t >= scan_interval:
                        last_scan_t = 0.0
                        pts = _scan_lidar_at(
                            cx, cy, state.walls_world,
                            lidar_range, lidar_beams, alt,
                            ceiling_height=state.ceiling_height,
                        )
                        state.points_emitted += len(pts)
                        for i in range(0, len(pts), batch_size):
                            await _publish_points(channel, state.mission_id, pts[i:i + batch_size])

                    await asyncio.sleep(dt)

        await _publish_mission_complete(channel, state)
        state.status = "completed"
        state.ended_at = time.time()

    except Exception as e:
        state.status = "failed"
        state.ended_at = time.time()
        await ws_manager.broadcast(channel, {
            "type": "mission.failed",
            "data": {"mission_id": state.mission_id, "error": str(e)},
        })


async def _publish_points(channel: str, mission_id: str, points: list[tuple[float, float, float]]):
    if not points:
        return
    await ws_manager.broadcast(channel, {
        "type": "lidar.points",
        "data": {
            "mission_id": mission_id,
            "points": [[round(x, 4), round(y, 4), round(z, 4)] for (x, y, z) in points],
            "count": len(points),
        },
    })


async def _publish_telemetry(channel: str, state: MissionState, cx: float, cy: float, yaw: float):
    await ws_manager.broadcast(channel, {
        "type": "telemetry.update",
        "data": {
            "x": round(cx, 4),
            "y": round(cy, 4),
            "z": round(state.altitude, 4),
            "yaw": round(math.degrees(yaw), 2),
            "armed": True,
            "mode": "AUTO_SCAN",
            "speed": round(state.speed, 2),
        },
    })


async def _publish_mission_complete(channel: str, state: MissionState):
    await ws_manager.broadcast(channel, {
        "type": "mission.completed",
        "data": {
            "mission_id": state.mission_id,
            "points_total": state.points_emitted,
            "duration_s": round(time.time() - state.started_at, 2),
        },
    })
