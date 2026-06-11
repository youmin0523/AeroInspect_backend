# =============================================
# app/services/slam_runner.py
# 역할: Visual-Inertial SLAM 실행기 + 영상 캡처 어댑터
#
# 본 구현은 Strategy 패턴으로 3개 백엔드를 지원:
#   - "orbslam3"  : Docker 이미지에서 띄운 ORB-SLAM3 서브프로세스를 stdin/stdout 으로 호출.
#                   docker compose 로 카메라 디바이스를 컨테이너에 위임 (tools/slam/docker-compose.yml).
#   - "rtabmap"   : rtabmap-standalone 바이너리 (apt install rtabmap) 또는 Docker.
#   - "dummy"     : 합성 점군/pose 생성 (개발/테스트용 — 외부 라이브러리 미설치 환경 폴백).
# 환경변수: SLAM_BACKEND, SLAM_CAPTURE_DEVICE, SLAM_POINTCLOUD_DIR
#
# 캡처 어댑터:
#   - Skydroid FUAV 5.8G OTG 동글을 OS별 백엔드(MSMF/V4L2/DSHOW) 자동 분기로 cv2.VideoCapture 오픈.
#   - 첫 프레임 검증 + asyncio Task 로 프레임 펌프.
#   - 미션 동안 SLAM 백엔드에 (frame_idx, ts, image) 형태로 직접 전달.
#
# 안전:
#   - 디바이스 미발견·디코딩 실패 → start() 에서 RuntimeError("skydroid_otg_not_recognized") raise.
#   - SLAM 백엔드 프로세스 죽음 → soft-fail + WS 알림.
# =============================================
from __future__ import annotations

import asyncio
import json
import os
import platform
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Awaitable, Callable, List, Optional, Tuple

from app.core.logging import get_logger

logger = get_logger(__name__)

try:
    import cv2  # type: ignore
    _CV2_OK = True
except ImportError:  # pragma: no cover
    _CV2_OK = False
    cv2 = None  # type: ignore

try:
    import numpy as np  # type: ignore
except ImportError:  # pragma: no cover
    np = None  # type: ignore


# ── 캡처 어댑터 ──────────────────────────────
class CaptureAdapter:
    """
    Skydroid FUAV 5.8G OTG → cv2.VideoCapture.
      - Windows : MSMF (cv2.CAP_MSMF) → 실패 시 DSHOW 폴백
      - Linux   : V4L2 (cv2.CAP_V4L2)
      - 그 외   : ANY
    device_or_uri 가 정수면 cv2.VideoCapture(int) (USB 인덱스), 문자열이면 RTSP/file URI.
    """

    def __init__(self, device_or_uri: int | str = 0, width: int = 1280, height: int = 720, fps: int = 30) -> None:
        self.dev = device_or_uri
        self.width = width
        self.height = height
        self.fps = fps
        self._cap = None
        self._first_frame_ts: Optional[float] = None

    def open(self) -> None:
        if not _CV2_OK:
            raise RuntimeError("opencv_not_installed")
        os_name = platform.system().lower()
        backends = self._select_backends(os_name)
        last_err: Optional[Exception] = None
        for backend_label, backend_id in backends:
            try:
                cap = cv2.VideoCapture(self.dev, backend_id)
                if not cap.isOpened():
                    cap.release()
                    continue
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                cap.set(cv2.CAP_PROP_FPS, self.fps)
                # 첫 프레임 검증 — 일부 백엔드는 isOpened=True 인데 grab 실패
                ret, _frame = cap.read()
                if not ret:
                    cap.release()
                    continue
                self._cap = cap
                self._first_frame_ts = time.time()
                logger.info("slam.capture.opened", os=os_name, backend=backend_label, dev=str(self.dev))
                return
            except Exception as e:
                last_err = e
                continue
        msg = f"skydroid_otg_not_recognized (last_err={last_err})"
        logger.error("slam.capture.open_failed", error=msg, dev=str(self.dev))
        raise RuntimeError(msg)

    def read(self) -> Optional["np.ndarray"]:
        if self._cap is None:
            return None
        ret, frame = self._cap.read()
        return frame if ret else None

    def close(self) -> None:
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None

    @staticmethod
    def _select_backends(os_name: str) -> List[Tuple[str, int]]:
        if not _CV2_OK:
            return []
        if "windows" in os_name:
            return [("MSMF", cv2.CAP_MSMF), ("DSHOW", cv2.CAP_DSHOW), ("ANY", cv2.CAP_ANY)]
        if "linux" in os_name:
            return [("V4L2", cv2.CAP_V4L2), ("ANY", cv2.CAP_ANY)]
        return [("ANY", cv2.CAP_ANY)]


# ── SLAM 데이터 ─────────────────────────────
@dataclass
class SlamPose:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    qw: float = 1.0
    qx: float = 0.0
    qy: float = 0.0
    qz: float = 0.0
    confidence: float = 0.0
    pos_var_m: float = 0.0


@dataclass
class PointcloudDelta:
    frame_idx: int
    pose: SlamPose
    points_xyz: list = field(default_factory=list)   # [(x,y,z), ...]
    points_rgb: list = field(default_factory=list)   # [(r,g,b), ...] 0~255
    point_count: int = 0
    voxel_downsample_m: float = 0.05


PointcloudCallback = Callable[[PointcloudDelta], Awaitable[None]]
PoseCallback = Callable[[SlamPose], Awaitable[None]]


# ── SLAM Strategy 인터페이스 ────────────────
class SlamBackend:
    name = "base"
    # path_planner 가 사용할 occupancy 메타. SlamRunner.get_latest_occupancy() 가 합쳐 반환.
    occupancy_resolution_m: float = 0.05
    occupancy_origin_xy: Tuple[float, float] = (0.0, 0.0)

    async def start(self, mission_id: str) -> None:
        raise NotImplementedError

    async def stop(self) -> None:
        raise NotImplementedError

    async def feed_frame(self, frame, ts: float) -> None:
        raise NotImplementedError

    def get_latest_pose(self) -> Optional[SlamPose]:
        raise NotImplementedError

    def drain_pointcloud_delta(self) -> Optional[PointcloudDelta]:
        """이번 호출 이후 새로 누적된 키프레임 점군 1건 반환 (없으면 None)."""
        raise NotImplementedError

    def get_latest_occupancy(self):
        """
        2D occupancy grid (numpy ndarray, 0=free, 1=occupied, -1=unknown) 반환.
        path planner / room segmenter 의 입력이 된다. 미지원 백엔드는 None 반환.
        """
        return None


# ── 1) Pseudo 백엔드 (개발/테스트) ───────────
class PseudoSlamBackend(SlamBackend):
    name = "dummy"

    def __init__(self) -> None:
        self._frame_idx = 0
        self._latest_pose: Optional[SlamPose] = None
        self._pending_delta: Optional[PointcloudDelta] = None

    async def start(self, mission_id: str) -> None:
        self._frame_idx = 0
        self._latest_pose = SlamPose(confidence=0.7, pos_var_m=0.05)
        logger.info("slam.dummy.start", mission_id=mission_id)

    async def stop(self) -> None:
        self._latest_pose = None
        logger.info("slam.dummy.stop")

    async def feed_frame(self, frame, ts: float) -> None:
        # PseudoSlam 은 frame 픽셀을 사용하지 않고 frame_idx 만 카운트.
        del frame, ts  # 인터페이스 시그니처 유지용
        self._frame_idx += 1
        if self._latest_pose is None:
            return
        self._latest_pose.x += 0.005
        self._latest_pose.confidence = 0.7
        # 5 프레임마다 가짜 점군 emit
        if self._frame_idx % 5 == 0 and np is not None:
            n = 200
            pts = np.random.uniform(low=-0.5, high=0.5, size=(n, 3)) + [self._latest_pose.x, 0, 1]
            cols = np.random.randint(low=80, high=240, size=(n, 3), dtype=np.uint8)
            self._pending_delta = PointcloudDelta(
                frame_idx=self._frame_idx,
                pose=SlamPose(**self._latest_pose.__dict__),
                points_xyz=pts.tolist(),
                points_rgb=cols.tolist(),
                point_count=n,
            )

    def get_latest_pose(self) -> Optional[SlamPose]:
        return self._latest_pose

    def drain_pointcloud_delta(self) -> Optional[PointcloudDelta]:
        d = self._pending_delta
        self._pending_delta = None
        return d

    def get_latest_occupancy(self):
        """합성 occupancy: 8m × 8m 직사각형 자유공간(0) + 외곽 벽(1).
        resolution 0.05 m/px → 160×160 grid. 개발/SITL 검증용.
        """
        if np is None:
            return None
        grid = np.ones((160, 160), dtype=np.int8)
        grid[10:150, 10:150] = 0
        return grid


# ── 2) ORB-SLAM3 (Docker 서브프로세스) ─────
class OrbSlam3Backend(SlamBackend):
    """
    Docker 이미지에서 ORB-SLAM3 컨테이너를 띄우고 stdin/stdout(JSONL) 로 통신.
    컨테이너는 프레임을 별도 채널(named pipe / shared memory / TCP) 로 받고,
    pose+keyframe 점군을 stdout JSONL 로 출력하는 가정.
    실제 컨테이너 이미지 구성은 tools/slam/docker-compose.yml 에 명시.
    """
    name = "orbslam3"
    DOCKER_SERVICE = "aeroinspect-orbslam3"
    COMPOSE_FILE = str(Path(__file__).resolve().parents[3] / "tools" / "slam" / "docker-compose.yml")

    def __init__(self) -> None:
        self._proc: Optional[subprocess.Popen] = None
        self._reader_task: Optional[asyncio.Task] = None
        self._latest_pose: Optional[SlamPose] = None
        self._pointcloud_q: asyncio.Queue[PointcloudDelta] = asyncio.Queue(maxsize=64)
        self._frame_idx = 0

    async def start(self, mission_id: str) -> None:
        if not Path(self.COMPOSE_FILE).is_file():
            raise RuntimeError(f"slam_compose_not_found:{self.COMPOSE_FILE}")
        # docker compose up
        cmd = ["docker", "compose", "-f", self.COMPOSE_FILE, "up", "-d", self.DOCKER_SERVICE]
        proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        out, err = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(f"docker_compose_up_failed:{err.decode(errors='ignore')[:200]}")
        # 컨테이너 stdout 을 따라가기 위해 docker compose logs -f 를 attach
        self._proc = subprocess.Popen(
            ["docker", "compose", "-f", self.COMPOSE_FILE, "logs", "-f", "--no-color", self.DOCKER_SERVICE],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        self._reader_task = asyncio.create_task(self._read_stdout_loop())
        logger.info("slam.orbslam3.start", mission_id=mission_id)

    async def stop(self) -> None:
        if self._reader_task:
            self._reader_task.cancel()
        if self._proc:
            try:
                self._proc.terminate()
            except Exception:
                pass
        # docker compose down
        try:
            cmd = ["docker", "compose", "-f", self.COMPOSE_FILE, "stop", self.DOCKER_SERVICE]
            await asyncio.create_subprocess_exec(*cmd)
        except Exception:
            pass

    async def feed_frame(self, frame, ts: float) -> None:
        # ORB-SLAM3 컨테이너는 docker-compose.yml 의 device mount 로 카메라를 직접 잡음.
        # 본 어댑터는 frame_idx 만 카운트해 텔레메트리에 노출.
        del frame, ts
        self._frame_idx += 1

    def get_latest_pose(self) -> Optional[SlamPose]:
        return self._latest_pose

    def drain_pointcloud_delta(self) -> Optional[PointcloudDelta]:
        try:
            return self._pointcloud_q.get_nowait()
        except asyncio.QueueEmpty:
            return None

    async def _read_stdout_loop(self) -> None:
        """ORB-SLAM3 컨테이너 stdout JSONL 파서.
        기대 메시지(예):
          {"type":"pose","x":0,"y":0,"z":1,"qw":1,"qx":0,"qy":0,"qz":0,"conf":0.8,"var":0.1}
          {"type":"keyframe","frame":42,"points":[[x,y,z,r,g,b], ...]}
        """
        proc = self._proc
        if proc is None or proc.stdout is None:
            return
        loop = asyncio.get_running_loop()
        while True:
            line = await loop.run_in_executor(None, proc.stdout.readline)
            if not line:
                await asyncio.sleep(0.05)
                continue
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                msg = json.loads(line)
            except Exception:
                continue
            t = msg.get("type")
            if t == "pose":
                self._latest_pose = SlamPose(
                    x=msg.get("x", 0.0), y=msg.get("y", 0.0), z=msg.get("z", 0.0),
                    qw=msg.get("qw", 1.0), qx=msg.get("qx", 0.0),
                    qy=msg.get("qy", 0.0), qz=msg.get("qz", 0.0),
                    confidence=msg.get("conf", 0.0),
                    pos_var_m=msg.get("var", 0.0),
                )
            elif t == "keyframe":
                pts = msg.get("points") or []
                xyz = [(p[0], p[1], p[2]) for p in pts if len(p) >= 3]
                rgb = [(int(p[3]), int(p[4]), int(p[5])) for p in pts if len(p) >= 6]
                delta = PointcloudDelta(
                    frame_idx=int(msg.get("frame", self._frame_idx)),
                    pose=self._latest_pose or SlamPose(),
                    points_xyz=xyz, points_rgb=rgb, point_count=len(xyz),
                )
                try:
                    self._pointcloud_q.put_nowait(delta)
                except asyncio.QueueFull:
                    pass


# ── 3) RTAB-Map (외부 standalone CLI) ───────
class RtabMapBackend(SlamBackend):
    """
    rtabmap CLI 또는 별도 RTAB-Map 데몬과 통신하는 어댑터.
    실시간 모드는 ROS 의존성이 커서 본 어댑터는 standalone 데이터베이스 출력 모드를 가정 —
    현실적으로는 ORB-SLAM3 docker 가 1차 권장. 본 구현은 인터페이스 자리잡기 + 확장 포인트.
    """
    name = "rtabmap"

    def __init__(self) -> None:
        self._latest_pose: Optional[SlamPose] = None

    async def start(self, mission_id: str) -> None:
        logger.info("slam.rtabmap.start", mission_id=mission_id, note="hooks pending")
        # TODO: rtabmap 데몬 기동 (소켓/파이프 연결)

    async def stop(self) -> None:
        logger.info("slam.rtabmap.stop")

    async def feed_frame(self, frame, ts: float) -> None:
        del frame, ts
        return None

    def get_latest_pose(self) -> Optional[SlamPose]:
        return self._latest_pose

    def drain_pointcloud_delta(self) -> Optional[PointcloudDelta]:
        return None


def _make_backend(name: str) -> SlamBackend:
    n = (name or "dummy").lower()
    if n in ("orb", "orbslam3", "orb-slam3"):
        return OrbSlam3Backend()
    if n in ("rtab", "rtabmap"):
        return RtabMapBackend()
    return PseudoSlamBackend()


# ── SlamRunner — 캡처 + 백엔드 + 콜백 결선 ──
class SlamRunner:
    """
    Visual-Inertial SLAM standalone 어댑터.
    백엔드 GPU 서버에서 단일 인스턴스로 동작 (한 번에 1 미션).
    """
    FRAME_PUMP_INTERVAL_SEC = 0.033   # ~30 fps 상한
    POSE_BROADCAST_HZ = 5
    POINTCLOUD_DRAIN_HZ = 5

    def __init__(self) -> None:
        self.capture: Optional[CaptureAdapter] = None
        self.backend: Optional[SlamBackend] = None
        self._pose_callbacks: List[PoseCallback] = []
        self._pc_callbacks: List[PointcloudCallback] = []
        self._frame_idx = 0
        self._running = False
        self._pump_task: Optional[asyncio.Task] = None
        self._pose_emit_task: Optional[asyncio.Task] = None
        self._pc_emit_task: Optional[asyncio.Task] = None

    # ── 라이프사이클 ──────────────────────
    async def start(self, mission_id: str, source: int | str | None = None) -> None:
        if self._running:
            return
        if source is None:
            src_env = os.environ.get("SLAM_CAPTURE_DEVICE", "0")
            try:
                source = int(src_env)
            except ValueError:
                source = src_env
        backend_name = os.environ.get("SLAM_BACKEND", "dummy")
        self.backend = _make_backend(backend_name)
        self.capture = CaptureAdapter(source)
        try:
            self.capture.open()
        except RuntimeError as e:
            logger.error("slam.capture.fatal", error=str(e))
            raise
        await self.backend.start(mission_id)
        self._running = True
        self._pump_task = asyncio.create_task(self._frame_pump_loop())
        self._pose_emit_task = asyncio.create_task(self._pose_emit_loop())
        self._pc_emit_task = asyncio.create_task(self._pointcloud_emit_loop())
        logger.info("slam.runner.start", mission_id=mission_id, backend=self.backend.name)

    async def stop(self) -> None:
        self._running = False
        for t in (self._pump_task, self._pose_emit_task, self._pc_emit_task):
            if t:
                t.cancel()
        if self.backend:
            try:
                await self.backend.stop()
            except Exception as e:
                logger.warning("slam.backend.stop_failed", error=str(e))
            self.backend = None
        if self.capture:
            self.capture.close()
            self.capture = None
        logger.info("slam.runner.stop")

    # ── 콜백 등록 ─────────────────────────
    def subscribe_pose(self, cb: PoseCallback) -> None:
        self._pose_callbacks.append(cb)

    def subscribe_pointcloud(self, cb: PointcloudCallback) -> None:
        self._pc_callbacks.append(cb)

    def get_latest_pose(self) -> Optional[SlamPose]:
        return self.backend.get_latest_pose() if self.backend else None

    def get_latest_occupancy(self):
        """현재 SLAM 백엔드의 occupancy grid 반환. (grid, resolution_m_per_px, origin_xy)."""
        if self.backend is None:
            return None, 0.05, (0.0, 0.0)
        grid = self.backend.get_latest_occupancy()
        return grid, self.backend.occupancy_resolution_m, self.backend.occupancy_origin_xy

    # ── 내부 루프 ─────────────────────────
    async def _frame_pump_loop(self) -> None:
        while self._running:
            try:
                if self.capture is None or self.backend is None:
                    await asyncio.sleep(0.1)
                    continue
                frame = await asyncio.get_running_loop().run_in_executor(None, self.capture.read)
                if frame is not None:
                    self._frame_idx += 1
                    await self.backend.feed_frame(frame, time.time())
                await asyncio.sleep(self.FRAME_PUMP_INTERVAL_SEC)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("slam.frame_pump_error", error=str(e))
                await asyncio.sleep(0.2)

    async def _pose_emit_loop(self) -> None:
        period = 1.0 / max(1, self.POSE_BROADCAST_HZ)
        while self._running:
            try:
                pose = self.backend.get_latest_pose() if self.backend else None
                if pose is not None:
                    for cb in list(self._pose_callbacks):
                        try:
                            await cb(pose)
                        except Exception as e:
                            logger.error("slam.pose_cb_failed", error=str(e))
                await asyncio.sleep(period)
            except asyncio.CancelledError:
                break

    async def _pointcloud_emit_loop(self) -> None:
        period = 1.0 / max(1, self.POINTCLOUD_DRAIN_HZ)
        while self._running:
            try:
                if self.backend is None:
                    await asyncio.sleep(period)
                    continue
                delta = self.backend.drain_pointcloud_delta()
                if delta is not None:
                    for cb in list(self._pc_callbacks):
                        try:
                            await cb(delta)
                        except Exception as e:
                            logger.error("slam.pc_cb_failed", error=str(e))
                await asyncio.sleep(period)
            except asyncio.CancelledError:
                break
