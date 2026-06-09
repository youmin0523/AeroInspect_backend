# =============================================
# app/core/streaming.py
# 역할: MJPEG 비동기 프레임 스트리밍 제너레이터
#       - RGB, 열화상, 블렌드(합성) 3가지 모드 지원
#       - multipart/x-mixed-replace 형식으로 브라우저 <img> 태그에 직접 스트리밍
#       - 다중 클라이언트: 카메라 서비스의 구독자 큐를 통해 팬아웃
#       - blend 모드: cv2.addWeighted로 RGB + 열화상 알파 합성
# 사용: StreamingResponse(mjpeg_generator(...), media_type="multipart/x-mixed-replace...")
# =============================================

import asyncio
from typing import AsyncGenerator

import cv2
import numpy as np

from app.config import settings
from app.services.camera import CameraService


async def mjpeg_generator(
    camera_service: CameraService,
    quality: int = None,
) -> AsyncGenerator[bytes, None]:
    """
    단일 카메라 MJPEG 스트리밍 제너레이터.
    카메라 서비스에서 프레임을 받아 JPEG 인코딩 후 multipart 형식으로 yield.

    Args:
        camera_service: RGB 또는 열화상 카메라 서비스 인스턴스
        quality: JPEG 압축 품질 (None이면 설정값 사용)
    """
    q = quality or settings.MJPEG_JPEG_QUALITY
    subscriber_queue = camera_service.subscribe()

    try:
        while True:
            # 블로킹 get() 대신 타임아웃 — 카메라가 프레임을 안 주면 영원히 블로킹되어
            # 클라이언트 연결 종료를 감지 못 하고 구독자 큐가 누수됨. 타임아웃 시 yield를
            # 거치며 제어권을 양보하므로, 끊긴 클라이언트는 다음 yield에서 예외가 발생해
            # finally 정리(unsubscribe)가 트리거된다.
            try:
                frame = await asyncio.wait_for(subscriber_queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                # keepalive: 끊긴 클라이언트 감지를 위해 빈 경계 프레임 yield.
                yield b"--frame\r\n\r\n"
                continue
            if frame is None:
                await asyncio.sleep(0.033)  # 30fps 대기
                continue

            # JPEG 인코딩
            success, buf = cv2.imencode(
                ".jpg", frame,
                [cv2.IMWRITE_JPEG_QUALITY, q]
            )
            if not success:
                continue

            # multipart/x-mixed-replace 경계 프레임 형식
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + buf.tobytes()
                + b"\r\n"
            )

            # 이벤트 루프에 제어권 양보 (블로킹 방지)
            await asyncio.sleep(0)

    finally:
        camera_service.unsubscribe(subscriber_queue)


async def mjpeg_blend_generator(
    rgb_service: CameraService,
    thermal_service: CameraService,
    alpha: float = None,
    quality: int = None,
) -> AsyncGenerator[bytes, None]:
    """
    RGB + 열화상 블렌드 MJPEG 스트리밍 제너레이터.
    두 카메라에서 동시에 프레임을 받아 알파 합성 후 스트리밍.

    Args:
        rgb_service: RGB 카메라 서비스
        thermal_service: 열화상 카메라 서비스
        alpha: 열화상 합성 비율 (0.0=RGB만, 1.0=열화상만)
        quality: JPEG 압축 품질
    """
    a = alpha if alpha is not None else settings.THERMAL_BLEND_ALPHA
    q = quality or settings.MJPEG_JPEG_QUALITY

    rgb_queue = rgb_service.subscribe()
    thermal_queue = thermal_service.subscribe()

    try:
        while True:
            # 두 카메라에서 동시에 최신 프레임 획득.
            # 블로킹 get() 대신 타임아웃 — 카메라가 프레임을 안 주면 영원히 블로킹되어
            # 클라이언트 종료 감지 실패 + 구독자 큐 누수. 타임아웃 시 keepalive를 yield하여
            # 끊긴 클라이언트가 finally 정리(unsubscribe)를 트리거하도록 한다.
            try:
                rgb_frame = await asyncio.wait_for(rgb_queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                yield b"--frame\r\n\r\n"
                continue
            # 열화상은 최신 프레임만 사용 (비어있으면 스킵)
            thermal_frame = None
            try:
                thermal_frame = thermal_queue.get_nowait()
            except asyncio.QueueEmpty:
                pass

            if rgb_frame is None:
                await asyncio.sleep(0.033)
                continue

            # 블렌드: 열화상이 있으면 합성, 없으면 RGB만 사용
            if thermal_frame is not None:
                blended = _blend_frames(rgb_frame, thermal_frame, alpha=a)
            else:
                blended = rgb_frame

            success, buf = cv2.imencode(
                ".jpg", blended,
                [cv2.IMWRITE_JPEG_QUALITY, q]
            )
            if not success:
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + buf.tobytes()
                + b"\r\n"
            )

            await asyncio.sleep(0)

    finally:
        rgb_service.unsubscribe(rgb_queue)
        thermal_service.unsubscribe(thermal_queue)


def _blend_frames(
    rgb_frame: np.ndarray,
    thermal_frame: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """
    RGB 프레임과 열화상 프레임을 알파 블렌딩.

    Args:
        rgb_frame: BGR 형식 RGB 프레임
        thermal_frame: BGR 형식 의사색상 열화상 프레임
        alpha: 열화상 비율 (0.0~1.0)

    Returns:
        합성된 BGR 프레임
    """
    # 크기 맞추기 (열화상은 256x192, RGB는 더 클 수 있음)
    h, w = rgb_frame.shape[:2]
    if thermal_frame.shape[:2] != (h, w):
        thermal_frame = cv2.resize(thermal_frame, (w, h))

    # BGR 채널 보장
    if len(thermal_frame.shape) == 2:
        thermal_frame = cv2.cvtColor(thermal_frame, cv2.COLOR_GRAY2BGR)

    return cv2.addWeighted(rgb_frame, 1.0 - alpha, thermal_frame, alpha, 0)
