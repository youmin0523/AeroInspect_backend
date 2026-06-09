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
import numpy as np

from app.config import settings
from app.services.camera import CameraService

# 진행 중인 레코더 레지스트리. start() 시 등록, stop() 시 제거.
# 앱이 stop() 없이 종료될 때 release_all()로 남은 VideoWriter를 안전하게 release
# (mp4 truncate 방지). main.py 연동은 범위 밖 — 함수/등록만 제공.
_active_recorders: set = set()


async def release_all() -> None:
    """
    아직 진행 중인 모든 레코더의 stop()/writer release를 호출.
    앱 셧다운 시 stop()이 호출되지 않아 VideoWriter가 release되지 않는(파일 손상)
    상황을 방지. 등록 집합 사본을 순회하여 stop() 중 discard로 인한 변경에 안전.
    """
    for recorder in list(_active_recorders):
        try:
            await recorder.stop()
        except Exception as e:
            print(f"[Recording] release_all stop 오류: {e}")
        finally:
            _active_recorders.discard(recorder)


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


class _TestStreamRecorder:
    """
    TEST MODE 전용 녹화 핸들 — test_stream_service 의 _current_rgb_jpeg /
    _current_thermal_jpeg 를 폴링해서 cv2.VideoWriter 로 mp4 저장.

    Why: 운영 recording_service 는 실제 카메라(`CameraService`) 만 구독함.
    test mode 는 카메라가 아닌 자체 MJPEG generator 라서 그대로는 녹화 불가.
    프레임 버전 카운터(`test_stream_service._frame_version`) 변화 감지로
    최신 JPEG 만 디코드해 mp4 로 기록.
    """

    def __init__(self, name: str, filepath: str, channel: str):
        self.name = name
        self.filepath = filepath
        self.channel = channel  # 'rgb' | 'thermal'
        self.writer: Optional[cv2.VideoWriter] = None
        self.task: Optional[asyncio.Task] = None
        self.frame_count: int = 0
        self._running = False

    async def start(self) -> None:
        self._running = True
        self.frame_count = 0
        self.task = asyncio.create_task(self._loop())

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

    async def _loop(self) -> None:
        # 모듈 레벨 import 시 순환 의존 발생 가능 — lazy import.
        from app.services.test_stream import test_stream_service

        last_version = -1
        try:
            while self._running:
                cur_version = test_stream_service._frame_version
                if cur_version == last_version:
                    await asyncio.sleep(0.05)
                    continue
                last_version = cur_version

                jpeg = (
                    test_stream_service._current_rgb_jpeg
                    if self.channel == "rgb"
                    else test_stream_service._current_thermal_jpeg
                )
                if not jpeg:
                    await asyncio.sleep(0.05)
                    continue

                arr = np.frombuffer(jpeg, dtype=np.uint8)
                frame = await asyncio.to_thread(cv2.imdecode, arr, cv2.IMREAD_COLOR)
                if frame is None:
                    continue

                # 첫 프레임에서 VideoWriter 초기화 — JPEG 해상도가 frame마다 일정하지 않을 수
                # 있어 첫 frame 기준으로 고정.
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

                # 해상도가 첫 frame과 다르면 리사이즈 (test mode는 다양한 소스 mix).
                if (frame.shape[1], frame.shape[0]) != (
                    int(self.writer.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    int(self.writer.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                ):
                    frame = await asyncio.to_thread(
                        cv2.resize, frame,
                        (int(self.writer.get(cv2.CAP_PROP_FRAME_WIDTH)),
                         int(self.writer.get(cv2.CAP_PROP_FRAME_HEIGHT))),
                    )

                await asyncio.to_thread(self.writer.write, frame)
                self.frame_count += 1
        except asyncio.CancelledError:
            raise
        except Exception as e:
            print(f"[Recording/TestStream/{self.name}] 루프 오류: {e}")


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

        TEST MODE 자동 분기 — `test_stream_service._playing == True` 면 실제 카메라
        대신 test_stream 의 _current_*_jpeg 를 캡처해서 mp4 저장. R2/CF 보류 상태에서
        로컬 ./recordings 디스크에 우선 저장하여 사용자가 `GET /stream/record/{filename}`
        으로 다운로드 가능.

        Args:
            rgb_camera: RGB 카메라 서비스 (real mode용)
            thermal_camera: 열화상 카메라 서비스 (real mode용)

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

        # ── TEST MODE 자동 분기 ──────────────────
        # test_stream 이 활성 재생 상태면 그쪽을 녹화. 그렇지 않으면 real camera.
        # circular import 회피를 위해 함수 내부 import.
        from app.services.test_stream import test_stream_service

        use_test_source = test_stream_service._playing

        if use_test_source:
            self._rgb_recorder = _TestStreamRecorder("RGB", rgb_file, "rgb")
            self._thermal_recorder = _TestStreamRecorder("Thermal", thermal_file, "thermal")
            self._start_time = datetime.now()
            self._is_recording = True
            _active_recorders.add(self._rgb_recorder)
            _active_recorders.add(self._thermal_recorder)
            await self._rgb_recorder.start()
            await self._thermal_recorder.start()
            print(
                f"[Recording] TEST MODE 동시 녹화 시작: "
                f"{self._timestamp}_rgb.mp4 / {self._timestamp}_thermal.mp4"
            )
        else:
            self._rgb_recorder = _CameraRecorder("RGB", rgb_file)
            self._thermal_recorder = _CameraRecorder("Thermal", thermal_file)
            self._start_time = datetime.now()
            self._is_recording = True
            _active_recorders.add(self._rgb_recorder)
            _active_recorders.add(self._thermal_recorder)
            await self._rgb_recorder.start(rgb_camera)
            await self._thermal_recorder.start(thermal_camera)
            print(
                f"[Recording] REAL CAMERA 동시 녹화 시작: "
                f"{self._timestamp}_rgb.mp4 / {self._timestamp}_thermal.mp4"
            )

        return {
            "rgb_filename": f"{self._timestamp}_rgb.mp4",
            "thermal_filename": f"{self._timestamp}_thermal.mp4",
            "source": "test_stream" if use_test_source else "real_camera",
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

        # 두 레코더 동시 중지 (레지스트리에서도 제거)
        if self._rgb_recorder:
            await self._rgb_recorder.stop()
            _active_recorders.discard(self._rgb_recorder)
        if self._thermal_recorder:
            await self._thermal_recorder.stop()
            _active_recorders.discard(self._thermal_recorder)

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
