# =============================================
# app/api/stream.py
# 역할: 카메라 영상 스트리밍, 모드 전환, 녹화 제어 API 엔드포인트
#       - GET  /stream/rgb      → RGB 카메라 MJPEG 스트리밍
#       - GET  /stream/thermal  → 열화상 카메라 MJPEG 스트리밍
#       - GET  /stream/blend    → RGB+열화상 합성 MJPEG 스트리밍
#       - POST /stream/mode     → 카메라 모드 전환 + WS 브로드캐스트
#       - GET  /stream/mode     → 현재 활성 카메라 모드 조회
#       - POST /stream/record/start  → 녹화 시작
#       - POST /stream/record/stop   → 녹화 중지
#       - GET  /stream/record/status → 녹화 상태 조회
#       - GET  /stream/record/list   → 녹화 파일 목록
#       - GET  /stream/record/{filename} → 녹화 파일 다운로드
#       - DELETE /stream/record/{filename} → 녹화 파일 삭제
# =============================================

import os
from typing import Literal, List

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse, FileResponse, Response
from pydantic import BaseModel

from app.config import settings
from app.dependencies import get_rgb_camera, get_thermal_camera, get_ws_manager, get_recording_service
from app.services.camera import CameraService
from app.services.recording import RecordingService
from app.core.streaming import mjpeg_generator, mjpeg_blend_generator
from app.core.stream_inference import stream_inference_worker
from app.core.ws_manager import ConnectionManager
from app.schemas.monitoring import StreamStatsResponse
from app.services.lidar import lidar_service
from app.services.telemetry_cache import telemetry_cache

router = APIRouter()

# 현재 활성 카메라 모드 상태 (모듈 레벨 상태 — 단일 워커 보장)
_active_mode: str = "rgb"

# MJPEG multipart 미디어 타입
MJPEG_CONTENT_TYPE = "multipart/x-mixed-replace; boundary=frame"


class StreamModeRequest(BaseModel):
    mode: Literal["rgb", "thermal", "blend"]


@router.get("/rgb")
async def stream_rgb(
    rgb_camera: CameraService = Depends(get_rgb_camera),
):
    """
    RGB 카메라 MJPEG 스트리밍.
    브라우저 <img src="/api/v1/stream/rgb"> 태그로 직접 소비 가능.
    """
    return StreamingResponse(
        mjpeg_generator(rgb_camera),
        media_type=MJPEG_CONTENT_TYPE,
    )


@router.get("/thermal")
async def stream_thermal(
    thermal_camera: CameraService = Depends(get_thermal_camera),
):
    """
    열화상 카메라 MJPEG 스트리밍.
    IRC-256CA 의사색상(INFERNO 컬러맵) 적용 후 스트리밍.
    """
    return StreamingResponse(
        mjpeg_generator(thermal_camera),
        media_type=MJPEG_CONTENT_TYPE,
    )


@router.get("/blend")
async def stream_blend(
    rgb_camera: CameraService = Depends(get_rgb_camera),
    thermal_camera: CameraService = Depends(get_thermal_camera),
):
    """
    RGB + 열화상 알파 합성 MJPEG 스트리밍.
    config.THERMAL_BLEND_ALPHA 값으로 투명도 조절.
    """
    return StreamingResponse(
        mjpeg_blend_generator(rgb_camera, thermal_camera),
        media_type=MJPEG_CONTENT_TYPE,
    )


@router.post("/mode")
async def set_stream_mode(
    request: StreamModeRequest,
    manager: ConnectionManager = Depends(get_ws_manager),
):
    """
    카메라 모드 전환.
    변경 후 WebSocket "camera" 채널로 mode_changed 이벤트 브로드캐스트.
    프론트엔드에서 이 이벤트를 수신하여 다중 클라이언트 동기화.
    """
    global _active_mode
    _active_mode = request.mode

    # 모든 연결된 클라이언트에게 모드 변경 알림
    await manager.broadcast("camera", {
        "type": "camera.mode_changed",
        "data": {"mode": _active_mode},
    })

    return {"mode": _active_mode, "message": f"카메라 모드가 '{_active_mode}'로 변경되었습니다."}


@router.get("/mode")
async def get_stream_mode():
    """현재 활성 카메라 모드 조회"""
    return {"mode": _active_mode}


@router.get("/stats", response_model=StreamStatsResponse)
async def get_stream_stats() -> StreamStatsResponse:
    """
    WebSocket 실시간 추론 워커 상태 + LiDAR/telemetry 헬스 조회.
    대시보드 좌측 상단 배지 및 운영 모니터링 용도.
    """
    return StreamStatsResponse(
        worker={
            "running": stream_inference_worker.is_running,
            **stream_inference_worker.stats,
        },
        telemetry_cache={
            "ready": telemetry_cache.is_ready,
            "age_sec": telemetry_cache.age_sec,
        },
        lidar={
            "connected": lidar_service.latest_distance_m is not None,
            "distance_m": lidar_service.latest_distance_m,
        },
    )


# ── 녹화 제어 엔드포인트 ─────────────────────────────
# 녹화 시 RGB + Thermal 동시에 별도 파일로 저장


@router.post("/record/start")
async def start_recording(
    rgb_camera: CameraService = Depends(get_rgb_camera),
    thermal_camera: CameraService = Depends(get_thermal_camera),
    recorder: RecordingService = Depends(get_recording_service),
):
    """
    영상 녹화 시작.
    RGB와 Thermal 카메라를 동시에 각각 별도 mp4 파일로 녹화.
    """
    try:
        files = await recorder.start(
            rgb_camera=rgb_camera,
            thermal_camera=thermal_camera,
        )
        return {"message": "녹화가 시작되었습니다.", **files}
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))


@router.post("/record/stop")
async def stop_recording(
    recorder: RecordingService = Depends(get_recording_service),
):
    """영상 녹화 중지 및 파일 저장."""
    try:
        result = await recorder.stop()
        return {"message": "녹화가 중지되었습니다.", **result}
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))


@router.get("/record/status")
async def recording_status(
    recorder: RecordingService = Depends(get_recording_service),
):
    """현재 녹화 상태 조회"""
    return recorder.status


@router.get("/record/list")
async def list_recordings(
    recorder: RecordingService = Depends(get_recording_service),
):
    """저장된 녹화 파일 목록 조회 (타임스탬프 기준 세션 그룹핑)"""
    return {"recordings": recorder.list_recordings()}


@router.get("/record/{filename}")
async def download_recording(filename: str):
    """녹화 파일 다운로드"""
    safe_name = os.path.basename(filename)
    filepath = os.path.join(settings.RECORDING_OUTPUT_DIR, safe_name)

    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="파일을 찾을 수 없습니다.")

    return FileResponse(
        filepath,
        media_type="video/mp4",
        filename=safe_name,
    )


@router.delete("/record/{filename}")
async def delete_recording(
    filename: str,
    recorder: RecordingService = Depends(get_recording_service),
):
    """녹화 파일 삭제 (개별 파일)"""
    if recorder.delete_recording(filename):
        return {"message": f"'{filename}' 파일이 삭제되었습니다."}
    raise HTTPException(status_code=404, detail="파일을 찾을 수 없습니다.")


@router.delete("/record/session/{timestamp}")
async def delete_recording_session(
    timestamp: str,
    recorder: RecordingService = Depends(get_recording_service),
):
    """녹화 세션 삭제 (동일 타임스탬프의 rgb+thermal 파일 일괄 삭제)"""
    deleted = recorder.delete_session(timestamp)
    if deleted:
        return {"message": f"세션 '{timestamp}' 삭제 완료 ({deleted}개 파일)"}
    raise HTTPException(status_code=404, detail="해당 세션을 찾을 수 없습니다.")


# ── 테스트 모드 스트리밍 엔드포인트 ─────────────────────────
# DRONE_CONNECTED=False 환경에서 로컬 이미지/영상으로 하자 검출 프로토타입 테스트.
# 프로젝트 로컬 데이터 또는 사용자 업로드 파일을 MJPEG 스트림으로 서빙하며 AI 추론 실행.

class TestSourceRequest(BaseModel):
    source: Literal["project", "upload"]

class TestDetectionModeRequest(BaseModel):
    mode: Literal["bbox", "detection"]


@router.post("/test/detection-mode")
async def set_test_detection_mode(request: TestDetectionModeRequest):
    """테스트 모드 감지 시각화 전환: 'bbox' (네모박스) ↔ 'detection' (객체감지)."""
    if not settings.TEST_MODE_ENABLED:
        raise HTTPException(status_code=404, detail="Test mode is disabled")
    from app.services.test_stream import test_stream_service
    test_stream_service.set_detection_mode(request.mode)
    return {"mode": test_stream_service.detection_mode}


@router.post("/test/init")
async def init_test_mode():
    """테스트 모드 초기화: 이미지 디렉토리 스캔 + AI 모델 로드. 재생은 시작하지 않음."""
    if not settings.TEST_MODE_ENABLED:
        raise HTTPException(status_code=404, detail="Test mode is disabled")

    from app.services.test_stream import test_stream_service
    scan_result = test_stream_service.scan_images()
    model_result = await test_stream_service.load_models()
    return {
        "status": "ready",
        "images": scan_result,
        "models": model_result,
    }


@router.post("/test/start")
async def start_test_mode():
    """테스트 모드 초기화 + 재생 시작."""
    if not settings.TEST_MODE_ENABLED:
        raise HTTPException(status_code=404, detail="Test mode is disabled")

    from app.services.test_stream import test_stream_service
    if not test_stream_service._scanned:
        test_stream_service.scan_images()
        await test_stream_service.load_models()
    test_stream_service.start_playback()
    return {"status": "playing", "play_state": test_stream_service.play_state}


@router.post("/test/pause")
async def pause_test_mode():
    """테스트 모드 일시중지."""
    if not settings.TEST_MODE_ENABLED:
        raise HTTPException(status_code=404, detail="Test mode is disabled")

    from app.services.test_stream import test_stream_service
    test_stream_service.pause_playback()
    return {"play_state": test_stream_service.play_state}


@router.post("/test/resume")
async def resume_test_mode():
    """테스트 모드 재생 재개."""
    if not settings.TEST_MODE_ENABLED:
        raise HTTPException(status_code=404, detail="Test mode is disabled")

    from app.services.test_stream import test_stream_service
    test_stream_service.resume_playback()
    return {"play_state": test_stream_service.play_state}


@router.post("/test/stop")
async def stop_test_mode():
    """테스트 모드 정지."""
    if not settings.TEST_MODE_ENABLED:
        raise HTTPException(status_code=404, detail="Test mode is disabled")

    from app.services.test_stream import test_stream_service
    test_stream_service.stop_playback()
    return {"play_state": test_stream_service.play_state}


@router.get("/test/state")
async def get_test_state():
    """테스트 모드 현재 상태 조회."""
    if not settings.TEST_MODE_ENABLED:
        raise HTTPException(status_code=404, detail="Test mode is disabled")

    from app.services.test_stream import test_stream_service
    return {"play_state": test_stream_service.play_state, "source": test_stream_service.source}


@router.get("/test/rgb")
async def stream_test_rgb():
    """테스트 모드: 무작위 이미지/영상 MJPEG 스트림 (AI 추론 포함)."""
    if not settings.TEST_MODE_ENABLED:
        raise HTTPException(status_code=404, detail="Test mode is disabled")

    from app.services.test_stream import test_stream_service
    return StreamingResponse(
        test_stream_service.rgb_mjpeg_generator(),
        media_type=MJPEG_CONTENT_TYPE,
    )


@router.get("/test/thermal")
async def stream_test_thermal():
    """테스트 모드: 열화상(IR) 이미지 MJPEG 스트림."""
    if not settings.TEST_MODE_ENABLED:
        raise HTTPException(status_code=404, detail="Test mode is disabled")

    from app.services.test_stream import test_stream_service
    return StreamingResponse(
        test_stream_service.thermal_mjpeg_generator(),
        media_type=MJPEG_CONTENT_TYPE,
    )


@router.post("/test/source")
async def set_test_source(request: TestSourceRequest):
    """테스트 이미지 소스 전환: 'project' (프로젝트 로컬) ↔ 'upload' (직접 업로드)."""
    if not settings.TEST_MODE_ENABLED:
        raise HTTPException(status_code=404, detail="Test mode is disabled")

    from app.services.test_stream import test_stream_service
    test_stream_service.set_source(request.source)
    return {"source": test_stream_service.source}


@router.post("/test/upload")
async def upload_test_files(files: List[UploadFile] = File(...)):
    """테스트용 이미지/영상 파일 대량 업로드."""
    if not settings.TEST_MODE_ENABLED:
        raise HTTPException(status_code=404, detail="Test mode is disabled")

    from app.services.test_stream import test_stream_service
    result = await test_stream_service.add_uploaded_files(files)
    return result


@router.delete("/test/upload")
async def clear_test_uploads():
    """업로드된 테스트 파일 전체 삭제."""
    if not settings.TEST_MODE_ENABLED:
        raise HTTPException(status_code=404, detail="Test mode is disabled")

    from app.services.test_stream import test_stream_service
    result = test_stream_service.clear_uploaded_files()
    return result


@router.get("/test/upload/list")
async def list_test_uploads():
    """업로드된 테스트 파일 목록 조회."""
    if not settings.TEST_MODE_ENABLED:
        raise HTTPException(status_code=404, detail="Test mode is disabled")

    from app.services.test_stream import test_stream_service
    return {"files": test_stream_service.list_uploaded_files()}


@router.get("/test/defect/{defect_id}/{channel}")
async def get_test_defect_frame(
    defect_id: str, channel: str, mode: str = "bbox"
):
    """테스트 모드: 하자 탐지 시점의 프레임을 JPEG로 반환.
    channel: 'rgb' | 'thermal', mode: 'bbox' | 'detection'"""
    if not settings.TEST_MODE_ENABLED:
        raise HTTPException(status_code=404, detail="Test mode is disabled")
    if channel not in ("rgb", "thermal"):
        raise HTTPException(status_code=400, detail="channel must be 'rgb' or 'thermal'")
    if mode not in ("bbox", "detection"):
        mode = "bbox"

    from app.services.test_stream import test_stream_service
    frame_jpeg = test_stream_service.get_defect_frame(defect_id, channel, mode)
    if frame_jpeg is None:
        raise HTTPException(status_code=404, detail="Frame not found or expired")
    return Response(content=frame_jpeg, media_type="image/jpeg")
