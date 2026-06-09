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

import asyncio
import os
from typing import Literal, List

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse, FileResponse, Response
from pydantic import BaseModel

from app.config import settings
from app.dependencies import (
    get_current_user,
    get_rgb_camera,
    get_thermal_camera,
    get_ws_manager,
    get_recording_service,
)
from app.services.camera import CameraService
from app.services.recording import RecordingService
from app.core.streaming import mjpeg_generator, mjpeg_blend_generator
from app.core.stream_inference import stream_inference_worker
from app.core.ws_manager import ConnectionManager
from app.schemas.monitoring import StreamStatsResponse
from app.services.lidar import lidar_service
from app.services.telemetry_cache import telemetry_cache

router = APIRouter()

# 현재 활성 카메라 모드. Redis 가용 시 공유(멀티워커 정합), 아니면 이 메모리 폴백 사용.
_active_mode: str = "rgb"
_MODE_REDIS_KEY = "stream:active_mode"


async def _get_active_mode() -> str:
    """활성 모드 조회 — Redis 우선(멀티워커 공유), 미가용 시 메모리 폴백."""
    from app.core.redis_client import get_redis
    r = await get_redis()
    if r is not None:
        try:
            val = await r.get(_MODE_REDIS_KEY)
            if val:
                return val
        except Exception:
            pass
    return _active_mode


async def _set_active_mode(mode: str) -> None:
    """활성 모드 저장 — 메모리 + (가용 시) Redis 양쪽 갱신."""
    global _active_mode
    _active_mode = mode
    from app.core.redis_client import get_redis
    r = await get_redis()
    if r is not None:
        try:
            await r.set(_MODE_REDIS_KEY, mode)
        except Exception:
            pass

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
    _user=Depends(get_current_user),
):
    """
    카메라 모드 전환.
    변경 후 WebSocket "camera" 채널로 mode_changed 이벤트 브로드캐스트.
    프론트엔드에서 이 이벤트를 수신하여 다중 클라이언트 동기화.
    """
    await _set_active_mode(request.mode)

    # 모든 연결된 클라이언트에게 모드 변경 알림 (Redis WS 백엔드면 전 워커에 전파)
    await manager.broadcast("camera", {
        "type": "camera.mode_changed",
        "data": {"mode": request.mode},
    })

    return {"mode": request.mode, "message": f"카메라 모드가 '{request.mode}'로 변경되었습니다."}


@router.get("/mode")
async def get_stream_mode():
    """현재 활성 카메라 모드 조회 (Redis 공유 상태 우선)"""
    return {"mode": await _get_active_mode()}


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
    _user=Depends(get_current_user),
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
    _user=Depends(get_current_user),
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
    _user=Depends(get_current_user),
):
    """녹화 파일 삭제 (개별 파일)"""
    if recorder.delete_recording(filename):
        return {"message": f"'{filename}' 파일이 삭제되었습니다."}
    raise HTTPException(status_code=404, detail="파일을 찾을 수 없습니다.")


@router.delete("/record/session/{timestamp}")
async def delete_recording_session(
    timestamp: str,
    recorder: RecordingService = Depends(get_recording_service),
    _user=Depends(get_current_user),
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
async def set_test_detection_mode(
    request: TestDetectionModeRequest,
    _user=Depends(get_current_user),
):
    """테스트 모드 감지 시각화 전환: 'bbox' (네모박스) ↔ 'detection' (객체감지)."""
    if not settings.TEST_MODE_ENABLED:
        raise HTTPException(status_code=404, detail="Test mode is disabled")
    from app.services.test_stream import test_stream_service
    test_stream_service.set_detection_mode(request.mode)
    return {"mode": test_stream_service.detection_mode}


@router.post("/test/init")
async def init_test_mode(_user=Depends(get_current_user)):
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


@router.post("/test/warmup")
async def warmup_test_mode(_user=Depends(get_current_user)):
    """모델 사전 로드(비차단). 테스트 모드 진입/업로드 모드 전환 시 프론트가 호출해
    11개 ONNX 콜드 스타트(10~20초)를 사용자가 파일 고르고 업로드하는 시간과 겹쳐 숨긴다.
    이미 로드됐거나 로딩 중이면 멱등 — 중복 호출해도 안전."""
    if not settings.TEST_MODE_ENABLED:
        raise HTTPException(status_code=404, detail="Test mode is disabled")

    from app.services.test_stream import test_stream_service
    if not test_stream_service._scanned:
        test_stream_service.scan_images()
    if not test_stream_service._models_loaded and not test_stream_service._models_loading:
        asyncio.create_task(test_stream_service.load_models())
    return test_stream_service.models_status


@router.post("/test/start")
async def start_test_mode(_user=Depends(get_current_user)):
    """테스트 모드 초기화 + 재생 시작.
    모델 로드는 백그라운드 태스크로 분리 — Fly.io 콜드 스타트 시 11개 ONNX 로드가
    10~20초 걸려 `await` 동기 대기하면 frontend `<img>` 가 first-boundary 오기 전
    edge timeout으로 onError 발화 → '스트림 대기 중' 영구 표시. 즉시 응답해서
    재생 상태(_playing=True) 만 켜놓고, 모델 로드는 generator 백그라운드에서 완료될 때까지
    detection 없이 raw frame 만 흘림(`_detect`는 모델 미로드 시 None 반환).
    사용자 체감: START 직후 영상은 흐름 + 모델 준비 완료 시점부터 검출 카드 등장."""
    if not settings.TEST_MODE_ENABLED:
        raise HTTPException(status_code=404, detail="Test mode is disabled")

    from app.services.test_stream import test_stream_service
    if not test_stream_service._scanned:
        test_stream_service.scan_images()
    # 모델 로드를 백그라운드로 비동기 실행 — 응답 블로킹 회피.
    # 이미 로드 완료/진행 중이면 load_models 가 멱등(`already_loaded`/`loading` 반환)하므로 중복 안전.
    # (사전 warmup 으로 이미 로딩 중이면 _models_loading 가드로 to_thread 중복 로드도 방지.)
    if not test_stream_service._models_loaded and not test_stream_service._models_loading:
        asyncio.create_task(test_stream_service.load_models())
    test_stream_service.start_playback()
    return {
        "status": "playing",
        "play_state": test_stream_service.play_state,
        "models_loaded": test_stream_service._models_loaded,
    }


@router.post("/test/pause")
async def pause_test_mode(_user=Depends(get_current_user)):
    """테스트 모드 일시중지."""
    if not settings.TEST_MODE_ENABLED:
        raise HTTPException(status_code=404, detail="Test mode is disabled")

    from app.services.test_stream import test_stream_service
    test_stream_service.pause_playback()
    return {"play_state": test_stream_service.play_state}


@router.post("/test/resume")
async def resume_test_mode(_user=Depends(get_current_user)):
    """테스트 모드 재생 재개."""
    if not settings.TEST_MODE_ENABLED:
        raise HTTPException(status_code=404, detail="Test mode is disabled")

    from app.services.test_stream import test_stream_service
    test_stream_service.resume_playback()
    return {"play_state": test_stream_service.play_state}


@router.post("/test/stop")
async def stop_test_mode(_user=Depends(get_current_user)):
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
    return {
        "play_state": test_stream_service.play_state,
        "source": test_stream_service.source,
        **test_stream_service.models_status,
    }


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
async def set_test_source(
    request: TestSourceRequest,
    _user=Depends(get_current_user),
):
    """테스트 이미지 소스 전환: 'project' (프로젝트 로컬) ↔ 'upload' (직접 업로드)."""
    if not settings.TEST_MODE_ENABLED:
        raise HTTPException(status_code=404, detail="Test mode is disabled")

    from app.services.test_stream import test_stream_service
    test_stream_service.set_source(request.source)
    return {"source": test_stream_service.source}


@router.post("/test/upload")
async def upload_test_files(
    files: List[UploadFile] = File(...),
    _user=Depends(get_current_user),
):
    """테스트용 이미지/영상 파일 대량 업로드.
    저장 1건 이상이면 source='upload'로 자동 전환 — 머신 재시작/새 세션에서
    백엔드 in-memory _source가 'project'로 초기화돼 화면이 AWAITING SIGNAL로
    멎는 사고 재발 방지."""
    if not settings.TEST_MODE_ENABLED:
        raise HTTPException(status_code=404, detail="Test mode is disabled")

    from app.services.test_stream import test_stream_service
    result = await test_stream_service.add_uploaded_files(files)
    if result.get("saved", 0) > 0 and test_stream_service.source != "upload":
        test_stream_service.set_source("upload")
        result["source"] = "upload"
    return result


@router.delete("/test/upload")
async def clear_test_uploads(_user=Depends(get_current_user)):
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


@router.get("/test/active")
async def get_test_active_media():
    """현재 재생 대상이 영상인지 이미지인지 메타 반환.
    프론트 useTestActiveMedia 가 폴링 → 영상이면 <video src=/test/upload/file/{name}>
    직접 재생, 이미지면 기존 MJPEG <img src=/test/rgb>. 인증 불요(GET 스트림 계열과 동일).
    """
    if not settings.TEST_MODE_ENABLED:
        raise HTTPException(status_code=404, detail="Test mode is disabled")

    from app.services.test_stream import test_stream_service
    # active_media(kind/filename/…) 에 모델 로딩 상태를 합쳐 한 번의 폴링으로 프론트가
    # '영상 vs 이미지' + 'AI 모델 로딩 중 여부' 를 동시에 알게 한다(별도 poller 불필요).
    return {**test_stream_service.active_media, **test_stream_service.models_status}


@router.get("/test/upload/file/{filename}")
async def serve_test_upload_file(filename: str):
    """업로드된 원본 파일(주로 영상)을 <video>/<img> src 로 직접 서빙.
    인증 불요 — <video> 태그는 Authorization 헤더를 못 붙임(스트림 계열과 동일 정책).
    경로 traversal 방어: basename 만 취하고 실제 경로가 업로드 디렉터리 내부인지 재확인.
    """
    if not settings.TEST_MODE_ENABLED:
        raise HTTPException(status_code=404, detail="Test mode is disabled")

    upload_dir = os.path.abspath(settings.TEST_UPLOAD_DIR)
    safe_name = os.path.basename(filename)  # ../ 류 제거
    full_path = os.path.abspath(os.path.join(upload_dir, safe_name))
    # 정규화 후에도 업로드 디렉터리 밖이면 거부
    if os.path.commonpath([upload_dir, full_path]) != upload_dir:
        raise HTTPException(status_code=400, detail="잘못된 파일 경로입니다.")
    if not os.path.isfile(full_path):
        raise HTTPException(status_code=404, detail="파일을 찾을 수 없습니다.")
    # FileResponse 는 Range 요청(영상 탐색)을 자동 처리.
    return FileResponse(full_path)


@router.get("/test/defect/{defect_id}/{channel}")
async def get_test_defect_frame(
    defect_id: str, channel: str, mode: str = "bbox"
):
    """테스트 모드: 하자 탐지 시점의 프레임을 JPEG로 반환.
    channel: 'rgb' | 'thermal', mode: 'bbox' | 'detection' | 'raw'.
    'raw' 는 프론트가 SVG로 자체 오버레이를 그릴 때 사용 (스캔 sweep + bbox 페이드인 UX).

    [v1.1 / R31 노트]
    영상 수신기 도착 후 진짜 현장점검 활성 시 RealStreamService 가 동일 시그니처의
    `GET /api/v1/stream/defect/{defect_id}/{channel}?mode=...` 를 노출해야 함.
    프론트 LiveVideoFeed.jsx 는 isTestMode 분기로 testMode/real URL 을 자동 전환하도록
    이미 source-agnostic 설계 — backend가 real 경로를 추가하면 프론트 수정 없이 동작.
    구현 시 반드시 적용할 패턴 (test_stream R30 에서 확립):
      1) detection 발생 시 raw 프레임 JPEG 를 detection 딕셔너리에 _rgb_snapshot/_thermal_snapshot
         으로 굳혀둘 것 (broadcast 지연 사이 _current_*_jpeg 가 다음 프레임으로 갱신되어
         bbox/jpeg 짝이 어긋나는 프레임 드리프트 방지).
      2) store_defect_frame 호출 시 위 스냅샷을 명시 전달.
      3) get_defect_frame 에서 mode='raw' 분기 지원."""
    if not settings.TEST_MODE_ENABLED:
        raise HTTPException(status_code=404, detail="Test mode is disabled")
    if channel not in ("rgb", "thermal"):
        raise HTTPException(status_code=400, detail="channel must be 'rgb' or 'thermal'")
    if mode not in ("bbox", "detection", "raw"):
        mode = "bbox"

    from app.services.test_stream import test_stream_service
    frame_jpeg = test_stream_service.get_defect_frame(defect_id, channel, mode)
    if frame_jpeg is None:
        raise HTTPException(status_code=404, detail="Frame not found or expired")
    return Response(content=frame_jpeg, media_type="image/jpeg")
