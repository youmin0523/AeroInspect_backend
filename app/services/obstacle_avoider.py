# =============================================
# app/services/obstacle_avoider.py
# 역할: 코드 기반 장애물 회피 (안전 핵심, 추가 회피 센서 없음 — 점군 단일 의존)
#
# 다층 방어:
#   ① SLAM 신뢰도 임계 (저신뢰 → 즉시 hover)
#   ② 점군 voxel(0.1m) 점유 맵 + 시간 감쇠
#   ③ DWA: 후보 속도 격자 × horizon 시뮬 → 안전회랑 침입 점수 → 최저 위험 선택
#   ④ 능동 yaw 스캔 (사각 보강) — mission_orchestrator 가 호출
#   ⑤ 속도 상한 (룸 0.5 / 도어웨이 0.25 m/s)
#
# voxel 인덱싱: world (x,y,z) / voxel_size → int 좌표
# 점유 맵은 dict[(ix,iy,iz)] = (weight, last_update_ts) — 시간 감쇠 적용
# =============================================
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import numpy as np

from app.core.logging import get_logger

logger = get_logger(__name__)

VoxelKey = Tuple[int, int, int]


@dataclass
class VelocityCommand:
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    yaw_rate: float = 0.0


@dataclass
class AvoidParams:
    voxel_size_m: float = 0.10
    safety_radius_m: float = 0.35
    safety_margin_doorway_m: float = 0.40
    speed_max_room_mps: float = 0.5
    speed_max_doorway_mps: float = 0.25
    horizon_sec: float = 1.0
    horizon_steps: int = 10
    slam_confidence_floor: float = 0.4
    no_feature_hover_sec: float = 1.5
    candidate_grid: int = 5             # vx, vy 각각 -v_max..+v_max 격자 수
    voxel_decay_sec: float = 5.0        # 오래된 voxel 가중치 0으로 감쇠
    voxel_weight_inc: float = 1.0
    voxel_weight_min: float = 0.2       # 가중치 < min 이면 점유 무시


@dataclass
class AvoidEvaluation:
    safe: bool
    chosen: VelocityCommand
    min_obstacle_dist_m: Optional[float]
    reason: str = "ok"
    candidate_count: int = 0


class ObstacleAvoider:
    def __init__(self, params: AvoidParams | None = None) -> None:
        self.params = params or AvoidParams()
        self._voxels: Dict[VoxelKey, Tuple[float, float]] = {}  # key → (weight, ts)
        self._last_low_conf_ts: Optional[float] = None

    # ── 점군 갱신 ─────────────────────────
    def update_pointcloud(
        self,
        points_xyz: Iterable[Tuple[float, float, float]],
        origin_xyz: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> None:
        """
        새 점군 한 묶음을 voxel 인덱스로 양자화하여 가중치 +inc.
        시간 감쇠 적용 — 오래된 voxel 가중치를 떨어뜨림(decay_sec 이상 지난 voxel은 제거).
        origin_xyz: SLAM 로컬 → 월드 보정용(미터). 일반적으로 (0,0,0).
        """
        now = time.time()
        vsize = self.params.voxel_size_m
        ox, oy, oz = origin_xyz

        pts = np.asarray(list(points_xyz), dtype=np.float64) if not isinstance(points_xyz, np.ndarray) else points_xyz
        if pts.size == 0:
            return
        if pts.ndim != 2 or pts.shape[1] != 3:
            logger.warning("avoider.bad_pts_shape", shape=tuple(pts.shape))
            return

        ix = np.floor((pts[:, 0] + ox) / vsize).astype(np.int32)
        iy = np.floor((pts[:, 1] + oy) / vsize).astype(np.int32)
        iz = np.floor((pts[:, 2] + oz) / vsize).astype(np.int32)

        for k in zip(ix.tolist(), iy.tolist(), iz.tolist()):
            w_prev, _ = self._voxels.get(k, (0.0, now))
            self._voxels[k] = (w_prev + self.params.voxel_weight_inc, now)

        # 감쇠 — 매 갱신 시 오래된 항목 제거
        cutoff = now - self.params.voxel_decay_sec
        stale = [k for k, (_, t) in self._voxels.items() if t < cutoff]
        for k in stale:
            self._voxels.pop(k, None)

    # ── 점유 검사 ─────────────────────────
    def _is_voxel_occupied(self, k: VoxelKey, now: float) -> bool:
        wt = self._voxels.get(k)
        if wt is None:
            return False
        w, t = wt
        # 가중치 시간 감쇠
        age = now - t
        decayed = w * max(0.0, 1.0 - age / self.params.voxel_decay_sec)
        return decayed >= self.params.voxel_weight_min

    def _min_dist_in_corridor(
        self,
        center_xyz: Tuple[float, float, float],
        radius_m: float,
        now: float,
    ) -> Optional[float]:
        """
        center 기준 안전회랑(반경 radius_m) 내 점유 voxel 까지 최소거리.
        없으면 None.
        """
        vsize = self.params.voxel_size_m
        cx, cy, cz = center_xyz
        r_vox = int(math.ceil(radius_m / vsize))
        ix0 = int(math.floor(cx / vsize))
        iy0 = int(math.floor(cy / vsize))
        iz0 = int(math.floor(cz / vsize))
        min_d2 = None
        for dx in range(-r_vox, r_vox + 1):
            for dy in range(-r_vox, r_vox + 1):
                for dz in range(-r_vox, r_vox + 1):
                    k = (ix0 + dx, iy0 + dy, iz0 + dz)
                    if not self._is_voxel_occupied(k, now):
                        continue
                    # voxel 중심 좌표
                    vx = (k[0] + 0.5) * vsize
                    vy = (k[1] + 0.5) * vsize
                    vz = (k[2] + 0.5) * vsize
                    d2 = (vx - cx) ** 2 + (vy - cy) ** 2 + (vz - cz) ** 2
                    if min_d2 is None or d2 < min_d2:
                        min_d2 = d2
        return None if min_d2 is None else math.sqrt(min_d2)

    # ── DWA 평가 ─────────────────────────
    def evaluate(
        self,
        target: VelocityCommand,
        current_pose_xyz: Tuple[float, float, float],
        slam_confidence: float,
        in_doorway: bool = False,
    ) -> AvoidEvaluation:
        now = time.time()

        # ① 무특징 영역 정책 — 안전 우선
        if slam_confidence < self.params.slam_confidence_floor:
            self._last_low_conf_ts = self._last_low_conf_ts or now
            return AvoidEvaluation(
                safe=False, chosen=VelocityCommand(),
                min_obstacle_dist_m=None,
                reason="slam_low_confidence",
                candidate_count=0,
            )
        self._last_low_conf_ts = None

        v_max = self.params.speed_max_doorway_mps if in_doorway else self.params.speed_max_room_mps
        margin = self.params.safety_margin_doorway_m if in_doorway else self.params.safety_radius_m

        # 후보 격자: vx, vy ∈ {-v_max .. +v_max}, vz ∈ {-v_max/2, 0, +v_max/2}
        n = max(3, self.params.candidate_grid)
        vxs = np.linspace(-v_max, v_max, n)
        vys = np.linspace(-v_max, v_max, n)
        vzs = np.array([-v_max * 0.5, 0.0, v_max * 0.5])

        best_score = float("inf")
        best_cmd: Optional[VelocityCommand] = None
        best_min_d: Optional[float] = None
        candidate_count = 0

        target_v = np.array([target.vx, target.vy, target.vz])
        target_norm = float(np.linalg.norm(target_v)) + 1e-6

        for vx in vxs:
            for vy in vys:
                for vz in vzs:
                    # 속도 상한 클리핑
                    speed = math.sqrt(vx * vx + vy * vy + vz * vz)
                    if speed > v_max + 1e-6:
                        continue
                    candidate_count += 1
                    # horizon 시뮬 — 직선 외삽으로 충분(짧은 호라이즌)
                    safe = True
                    min_d_in_traj: Optional[float] = None
                    for s in range(1, self.params.horizon_steps + 1):
                        t_s = (s / self.params.horizon_steps) * self.params.horizon_sec
                        px = current_pose_xyz[0] + vx * t_s
                        py = current_pose_xyz[1] + vy * t_s
                        pz = current_pose_xyz[2] + vz * t_s
                        d = self._min_dist_in_corridor((px, py, pz), margin, now)
                        if d is None:
                            continue
                        if d < margin:
                            safe = False
                            break
                        if min_d_in_traj is None or d < min_d_in_traj:
                            min_d_in_traj = d
                    if not safe:
                        continue
                    # 점수: 목표속도 정합 + 위험도(작은 거리일수록 높음)
                    cand_v = np.array([vx, vy, vz])
                    sim = float(np.dot(cand_v, target_v) / (np.linalg.norm(cand_v) * target_norm + 1e-6))
                    risk = 0.0 if min_d_in_traj is None else max(0.0, (margin * 2.0 - min_d_in_traj))
                    score = -sim + risk
                    if score < best_score:
                        best_score = score
                        best_cmd = VelocityCommand(vx=float(vx), vy=float(vy), vz=float(vz), yaw_rate=target.yaw_rate)
                        best_min_d = min_d_in_traj

        if best_cmd is None:
            # 모든 후보가 위험 — hover
            return AvoidEvaluation(
                safe=False, chosen=VelocityCommand(yaw_rate=target.yaw_rate),
                min_obstacle_dist_m=None,
                reason="no_safe_candidate",
                candidate_count=candidate_count,
            )

        return AvoidEvaluation(
            safe=True, chosen=best_cmd,
            min_obstacle_dist_m=best_min_d,
            reason="ok",
            candidate_count=candidate_count,
        )

    # ── 디버그/모니터링 ───────────────────
    def voxel_count(self) -> int:
        return len(self._voxels)
