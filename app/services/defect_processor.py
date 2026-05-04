# =============================================
# app/services/defect_processor.py
# 역할: 하자 탐지 AI 파이프라인 오케스트레이터
#       - RGB 카메라 + 열화상 카메라 + LiDAR 데이터를 통합 처리
#       - YOLOv8 추론 → 열화상 온도 추출 → LiDAR 3D 좌표 변환
#       - 탐지 결과를 DB 저장 API 호출 (내부 HTTP 요청)
#       - WebSocket 브로드캐스트 트리거
#
# 데이터 흐름:
#   카메라 프레임 → YOLO 추론 → 결과 후처리 → DB 저장 → WS 브로드캐스트
# =============================================

import asyncio
import base64
from typing import Optional

import cv2
import numpy as np

from app.services.camera import rgb_camera_service, thermal_camera_service
from app.services.yolo_inference import yolo_service, DetectionResult
from app.services.thermal import thermal_processor
from app.services.lidar import lidar_service
from app.utils.image_utils import crop_roi, encode_frame_to_base64
from app.core.ws_manager import ws_manager

# 드론 위치 (MAVLink 텔레메트리에서 실시간 업데이트)
_drone_pos = {"x": 0.0, "y": 0.0, "z": 1.5}


class DefectProcessor:
    """
    하자 탐지 파이프라인 오케스트레이터.
    주기적으로 프레임을 가져와 AI 추론 후 결과를 처리.
    """

    # 최소 처리 간격 (초) — 추론 FPS 제한
    PROCESS_INTERVAL = 0.2  # 5fps (너무 자주 하면 CPU 과부하)

    def __init__(self):
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._frame_counter = 0

    async def start(self) -> None:
        """처리 루프 시작"""
        self._running = True
        self._task = asyncio.create_task(self._processing_loop())
        print("[Processor] 하자 탐지 파이프라인 시작")

    async def stop(self) -> None:
        """처리 루프 중지"""
        self._running = False
        if self._task:
            self._task.cancel()

    def update_drone_position(self, x: float, y: float, z: float) -> None:
        """MAVLink 텔레메트리로 드론 위치 업데이트"""
        _drone_pos["x"] = x
        _drone_pos["y"] = y
        _drone_pos["z"] = z

    async def _processing_loop(self) -> None:
        """
        메인 처리 루프.
        RGB 프레임과 열화상 프레임을 동기화하여 AI 추론 실행.
        """
        while self._running:
            try:
                await self._process_single_frame()
            except Exception as e:
                print(f"[Processor] 처리 오류: {e}")
            await asyncio.sleep(self.PROCESS_INTERVAL)

    async def _process_single_frame(self) -> None:
        """단일 프레임 처리 파이프라인"""
        self._frame_counter += 1

        # RGB 프레임 획득
        rgb_frame = await rgb_camera_service.get_single_frame()
        if rgb_frame is None:
            return

        # YOLOv8 추론 (블로킹 → 스레드)
        detections = await yolo_service.infer(rgb_frame)
        if not detections:
            return

        # 열화상 프레임 획득 (있는 경우)
        thermal_colored, temp_map = await thermal_processor.get_processed_frame()

        for detection in detections:
            await self._process_detection(
                detection=detection,
                rgb_frame=rgb_frame,
                temp_map=temp_map,
            )

    async def _process_detection(
        self,
        detection: DetectionResult,
        rgb_frame: np.ndarray,
        temp_map: Optional[np.ndarray],
    ) -> None:
        """탐지 결과 후처리: 온도 추출 + 3D 좌표 + 이미지 크롭 + WS 전송"""

        # 1. 이미지 크롭 (바운딩 박스 영역)
        bbox = (detection.bbox_x, detection.bbox_y, detection.bbox_w, detection.bbox_h)
        cropped = crop_roi(rgb_frame, bbox)
        image_crop_b64 = encode_frame_to_base64(cropped) if cropped is not None else None

        # 2. 열화상 온도 추출
        thermal_max = thermal_min = thermal_avg = None
        if temp_map is not None:
            thermal_max, thermal_min, thermal_avg = thermal_processor.get_roi_temperature(
                temp_map, bbox
            )

        # 3. LiDAR 3D 좌표
        lidar_x = lidar_y = lidar_z = None
        if lidar_service.latest_distance_m is not None:
            lidar_x, lidar_y, lidar_z = lidar_service.compute_3d_position(
                _drone_pos["x"], _drone_pos["y"], _drone_pos["z"]
            )

        # 4. WebSocket "defects" 채널 브로드캐스트
        await ws_manager.broadcast("defects", {
            "type": "defect.new",
            "data": {
                "area": detection.area,
                "category_code": detection.category_code,
                "defect_type": detection.defect_type,
                "severity": detection.severity,
                "confidence": round(detection.confidence, 3),
                "bbox": {"x": detection.bbox_x, "y": detection.bbox_y,
                         "w": detection.bbox_w, "h": detection.bbox_h},
                "lidar_position": {"x": lidar_x, "y": lidar_y, "z": lidar_z},
                "thermal": {"max": thermal_max, "min": thermal_min, "avg": thermal_avg},
                "image_crop": image_crop_b64,
                "frame_id": self._frame_counter,
            },
        })

        print(
            f"[Processor] 하자 탐지: {detection.category_code} "
            f"({detection.severity}) conf={detection.confidence:.2f}"
        )


# 모듈 레벨 싱글톤
defect_processor = DefectProcessor()
