# =============================================
# app/services/recording.py
# 역할: 영상 녹화 서비스 (RGB + Thermal 동시 별도 파일 저장)
#       - 녹화 시작 시 RGB와 Thermal 카메라를 각각 구독하여 별도 mp4 파일로 기록
#       - 사용자 선택에 의한 녹화 (항상 녹화 아님, 스토리지 절약)
#       - 시작/중지/상태조회 API로 제어
#       - 파일 저장 경로: ./recordings/YYYYMMDD_HHMMSS_rgb.mp4 / _thermal.mp4
#
# 사용: from app.services.recording import recording_service
# =============================================

import asyncio
import os
from datetime import datetime
from typing import Optional

import cv2

from app.config import settings
from app.services.camera import CameraService


class _CameraRecorder:
    """
    단일 카메라 녹화 핸들.
    CameraService 구독 → cv2.VideoWriter 기록.
    """

    def __init__(self, name: str, filepath: str):
        self.name = name
        self.filepath = filepath
        self.writer: Optional[cv2.VideoWriter] = None
        self.task: Optional[asyncio.Task] = None
        self.frame_count: int = 0
        self._running = False

    async def start(self, camera: CameraService) -> None:
        self._running = True
        self.frame_count = 0
        self.task = asyncio.create_task(self._loop(camera))

    async def stop(self) -> None:
        self._running = False
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
        if self.writer:
            await asyncio.to_thread(self.writer.release)
            self.writer = None

    async def _loop(self, camera: CameraService) -> None:
        queue = camera.subscribe()
        try:
            while self._running:
                frame = await queue.get()
                if frame is None:
                    continue

                # 첫 프레임에서 VideoWriter 초기화
                if self.writer is None:
                    h, w = frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*settings.RECORDING_CODEC)
                    self.writer = await asyncio.to_thread(
                        cv2.VideoWriter,
                        self.filepath,
                        fourcc,
                        settings.RECORDING_FPS,
                        (w, h),
                    )

                await asyncio.to_thread(self.writer.write, frame)
                self.frame_count += 1
                await asyncio.sleep(0)
        finally:
            camera.unsubscribe(queue)


class RecordingService:
    """
    영상 녹화 서비스.
    녹화 시작 시 RGB와 Thermal 두 카메라를 동시에 각각 별도 파일로 기록.
    """

    def __init__(self):
        self._rgb_recorder: Optional[_CameraRecorder] = None
        self._thermal_recorder: Optional[_CameraRecorder] = None
        self._is_recording = False
        self._start_time: Optional[datetime] = None
        self._timestamp: str = ""

    @property
    def is_recording(self) -> bool:
        return self._is_recording

    @property
    def status(self) -> dict:
        """현재 녹화 상태 반환"""
        if not self._is_recording:
            return {"recording": False}

        elapsed = (
            (datetime.now() - self._start_time).total_seconds()
            if self._start_time
            else 0
        )
        rgb_frames = self._rgb_recorder.frame_count if self._rgb_recorder else 0
        thermal_frames = self._thermal_recorder.frame_count if self._thermal_recorder else 0

        return {
            "recording": True,
            "elapsed_seconds": round(elapsed, 1),
            "files": {
                "rgb": {
                    "filename": f"{self._timestamp}_rgb.mp4",
                    "frame_count": rgb_frames,
                },
                "thermal": {
                    "filename": f"{self._timestamp}_thermal.mp4",
                    "frame_count": thermal_frames,
                },
            },
        }

    async def start(
        self,
        rgb_camera: CameraService,
        thermal_camera: CameraService,
    ) -> dict:
        """
        RGB + Thermal 동시 녹화 시작.

        Args:
            rgb_camera: RGB 카메라 서비스
            thermal_camera: 열화상 카메라 서비스

        Returns:
            생성될 파일명 정보

        Raises:
            RuntimeError: 이미 녹화 중일 때
        """
        if self._is_recording:
            raise RuntimeError("이미 녹화 중입니다. 먼저 중지해주세요.")

        # 출력 디렉토리 생성
        os.makedirs(settings.RECORDING_OUTPUT_DIR, exist_ok=True)

        # 타임스탬프 기반 파일명
        self._timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        rgb_file = os.path.join(
            settings.RECORDING_OUTPUT_DIR, f"{self._timestamp}_rgb.mp4"
        )
        thermal_file = os.path.join(
            settings.RECORDING_OUTPUT_DIR, f"{self._timestamp}_thermal.mp4"
        )

        # 두 카메라 레코더 생성 및 시작
        self._rgb_recorder = _CameraRecorder("RGB", rgb_file)
        self._thermal_recorder = _CameraRecorder("Thermal", thermal_file)

        self._start_time = datetime.now()
        self._is_recording = True

        await self._rgb_recorder.start(rgb_camera)
        await self._thermal_recorder.start(thermal_camera)

        print(f"[Recording] 동시 녹화 시작: {self._timestamp}_rgb.mp4 / {self._timestamp}_thermal.mp4")

        return {
            "rgb_filename": f"{self._timestamp}_rgb.mp4",
            "thermal_filename": f"{self._timestamp}_thermal.mp4",
        }

    async def stop(self) -> dict:
        """
        녹화 중지 및 파일 저장 완료.

        Returns:
            녹화 결과 정보

        Raises:
            RuntimeError: 녹화 중이 아닐 때
        """
        if not self._is_recording:
            raise RuntimeError("녹화 중이 아닙니다.")

        self._is_recording = False

        # 두 레코더 동시 중지
        if self._rgb_recorder:
            await self._rgb_recorder.stop()
        if self._thermal_recorder:
            await self._thermal_recorder.stop()

        duration = (
            round((datetime.now() - self._start_time).total_seconds(), 1)
            if self._start_time
            else 0
        )

        result = {
            "duration_seconds": duration,
            "files": {
                "rgb": {
                    "filename": f"{self._timestamp}_rgb.mp4",
                    "frame_count": (
                        self._rgb_recorder.frame_count if self._rgb_recorder else 0
                    ),
                },
                "thermal": {
                    "filename": f"{self._timestamp}_thermal.mp4",
                    "frame_count": (
                        self._thermal_recorder.frame_count
                        if self._thermal_recorder
                        else 0
                    ),
                },
            },
        }

        print(
            f"[Recording] 녹화 중지: {duration}초 "
            f"(RGB {result['files']['rgb']['frame_count']}프레임, "
            f"Thermal {result['files']['thermal']['frame_count']}프레임)"
        )

        # 상태 초기화
        self._rgb_recorder = None
        self._thermal_recorder = None
        self._start_time = None
        self._timestamp = ""

        return result

    def list_recordings(self) -> list[dict]:
        """
        recordings 디렉토리의 파일 목록 반환.
        동일 타임스탬프의 rgb/thermal 파일을 하나의 세션으로 그룹핑.
        """
        output_dir = settings.RECORDING_OUTPUT_DIR
        if not os.path.exists(output_dir):
            return []

        # 타임스탬프별로 그룹핑
        sessions: dict[str, dict] = {}

        for f in sorted(os.listdir(output_dir), reverse=True):
            if not f.endswith(".mp4"):
                continue
            filepath = os.path.join(output_dir, f)
            stat = os.stat(filepath)
            size_mb = round(stat.st_size / (1024 * 1024), 1)

            # 파일명 파싱: YYYYMMDD_HHMMSS_mode.mp4
            parts = f.replace(".mp4", "").split("_")
            if len(parts) >= 3:
                timestamp_key = f"{parts[0]}_{parts[1]}"
                mode = parts[2]
            else:
                timestamp_key = f.replace(".mp4", "")
                mode = "unknown"

            if timestamp_key not in sessions:
                sessions[timestamp_key] = {
                    "timestamp": timestamp_key,
                    "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                    "files": {},
                }

            sessions[timestamp_key]["files"][mode] = {
                "filename": f,
                "size_bytes": stat.st_size,
                "size_mb": size_mb,
            }

        return list(sessions.values())

    def delete_recording(self, filename: str) -> bool:
        """
        녹화 파일 삭제.
        경로 탐색 공격 방지를 위해 파일명만 허용.
        """
        safe_name = os.path.basename(filename)
        filepath = os.path.join(settings.RECORDING_OUTPUT_DIR, safe_name)

        if os.path.exists(filepath) and filepath.endswith(".mp4"):
            os.remove(filepath)
            print(f"[Recording] 파일 삭제: {safe_name}")
            return True
        return False

    def delete_session(self, timestamp: str) -> int:
        """
        동일 타임스탬프의 녹화 파일(rgb+thermal) 세션 단위 삭제.

        Returns:
            삭제된 파일 수
        """
        safe_ts = os.path.basename(timestamp)
        output_dir = settings.RECORDING_OUTPUT_DIR
        deleted = 0

        for suffix in ["rgb", "thermal"]:
            filepath = os.path.join(output_dir, f"{safe_ts}_{suffix}.mp4")
            if os.path.exists(filepath):
                os.remove(filepath)
                deleted += 1

        if deleted:
            print(f"[Recording] 세션 삭제: {safe_ts} ({deleted}개 파일)")
        return deleted


# ── 모듈 레벨 싱글톤 ─────────────────────────
recording_service = RecordingService()
