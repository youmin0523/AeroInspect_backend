# =============================================
# app/services/lidar.py
# 역할: Benewake TF-Luna LiDAR 거리 측정 서비스
#       - UART 시리얼 통신으로 TF-Luna 9-byte 프레임 파싱
#       - asyncio.to_thread로 블로킹 serial.read를 비동기 래핑
#       - 10샘플 중앙값 필터로 노이즈 제거
#       - 드론 자세(roll/pitch/yaw) + 거리 → 3D 월드 좌표(ENU) 변환
#
# TF-Luna 프레임 형식 (9 bytes):
#   [0x59, 0x59, Dist_L, Dist_H, Strength_L, Strength_H, Temp_L, Temp_H, Checksum]
#   거리 단위: cm (최대 800cm)
#
# 시리얼 설정: 115200 baud, 8N1
# =============================================

import asyncio
import math
from collections import deque
from typing import Optional, Tuple

import numpy as np

from app.config import settings


class TFLunaService:
    """
    TF-Luna LiDAR 거리 측정 및 3D 좌표 변환 서비스.
    """

    FRAME_HEADER = b'\x59\x59'
    FRAME_LENGTH = 9
    MAX_DISTANCE_CM = 800
    MIN_DISTANCE_CM = 10

    def __init__(self):
        self._serial = None
        self._running = False
        self._distance_buffer: deque = deque(maxlen=10)  # 중앙값 필터
        self._latest_distance: Optional[float] = None     # 미터 단위
        self._read_task: Optional[asyncio.Task] = None

        # 드론 자세 (MAVLink 텔레메트리에서 업데이트)
        self._roll = 0.0    # rad
        self._pitch = 0.0   # rad
        self._yaw = 0.0     # rad

    @property
    def latest_distance_m(self) -> Optional[float]:
        """최신 필터링된 거리 (미터)"""
        return self._latest_distance

    def update_attitude(self, roll: float, pitch: float, yaw: float) -> None:
        """드론 자세 업데이트 (MAVLink 수신 시 호출)"""
        self._roll = roll
        self._pitch = pitch
        self._yaw = yaw

    async def start(self) -> None:
        """시리얼 포트 열기 및 읽기 태스크 시작"""
        try:
            import serial
            self._serial = await asyncio.to_thread(
                serial.Serial,
                port=settings.LIDAR_SERIAL_PORT,
                baudrate=settings.LIDAR_BAUD_RATE,
                timeout=0.1,
            )
            self._running = True
            self._read_task = asyncio.create_task(self._read_loop())
            print(f"[LiDAR] TF-Luna 연결됨: {settings.LIDAR_SERIAL_PORT}")
        except Exception as e:
            print(f"[LiDAR] 연결 실패: {e}. LiDAR 없이 계속.")

    async def stop(self) -> None:
        """시리얼 연결 종료"""
        self._running = False
        if self._read_task:
            self._read_task.cancel()
        if self._serial:
            await asyncio.to_thread(self._serial.close)

    async def _read_loop(self) -> None:
        """TF-Luna 프레임 지속 수신 및 파싱"""
        while self._running and self._serial:
            try:
                raw = await asyncio.to_thread(
                    self._serial.read, self.FRAME_LENGTH
                )
                distance_cm = self._parse_frame(raw)
                if distance_cm is not None:
                    self._distance_buffer.append(distance_cm)
                    # 중앙값 필터 적용
                    median_cm = float(np.median(list(self._distance_buffer)))
                    self._latest_distance = median_cm / 100.0  # cm → m
            except Exception:
                await asyncio.sleep(0.01)

    def _parse_frame(self, data: bytes) -> Optional[int]:
        """
        TF-Luna 9-byte 프레임 파싱.

        Returns:
            거리 (cm) 또는 None (유효하지 않은 프레임)
        """
        if len(data) != self.FRAME_LENGTH:
            return None
        if data[:2] != self.FRAME_HEADER:
            return None

        # 체크섬 검증
        checksum = sum(data[:8]) & 0xFF
        if checksum != data[8]:
            return None

        # 거리 계산 (little-endian 16bit)
        distance_cm = data[2] + (data[3] << 8)

        if distance_cm < self.MIN_DISTANCE_CM or distance_cm > self.MAX_DISTANCE_CM:
            return None

        return distance_cm

    def compute_3d_position(
        self,
        drone_x: float,
        drone_y: float,
        drone_z: float,
    ) -> Tuple[float, float, float]:
        """
        드론 위치 + LiDAR 거리 + 자세 → 탐지 대상 3D 월드 좌표 계산.

        LiDAR가 드론 전방을 향한다고 가정.
        roll/pitch/yaw를 이용한 방향 벡터 계산.

        Args:
            drone_x, drone_y, drone_z: 드론 ENU 좌표 (m)

        Returns:
            (target_x, target_y, target_z) 탐지 대상의 월드 좌표
        """
        d = self._latest_distance or 1.0

        # 전방 벡터 (LiDAR 방향)
        dx = d * math.cos(self._pitch) * math.cos(self._yaw)
        dy = d * math.cos(self._pitch) * math.sin(self._yaw)
        dz = d * math.sin(self._pitch)

        return (drone_x + dx, drone_y + dy, drone_z + dz)


# 모듈 레벨 싱글톤
lidar_service = TFLunaService()
