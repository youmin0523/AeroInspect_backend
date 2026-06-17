# =============================================
# app/api/ai_webhook.py
# 역할: Python AI 서버 → FastAPI 백엔드 연동 웹훅
#       - POST /ai/detection  → AI 탐지 이벤트 수신 → DB 저장 + WS Push
#       - POST /ai/thermal    → 열화상 분석 결과 수신 → WS Push
#       - POST /ai/batch      → 다건 탐지 결과 일괄 저장
#       Python AI 서버(YOLO/PatchCore/RANSAC 등)에서 탐지 완료 시
#       이 엔드포인트를 호출하여 결과를 백엔드로 전달.
# =============================================

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import get_db, get_ws_manager, verify_ai_webhook
from app.models.defect import DefectLog
from app.schemas.defect import (
    DefectLogCreate,
    DefectLogResponse,
    BoundingBox,
    LidarPosition,
    ThermalData,
)
from app.core.ws_manager import ConnectionManager
from app.services.image_storage import image_storage


def _build_response(defect: DefectLog) -> DefectLogResponse:
    """ORM → 응답 스키마. image_crop_path가 있으면 URL까지 채움."""
    resp = DefectLogResponse.model_validate(defect)
    if defect.image_crop_path:
        resp.image_crop_url = image_storage.get_url(defect.image_crop_path)
    return resp

router = APIRouter()


class ThermalAnalysisResult(BaseModel):
    """열화상 분석 결과 (AI 서버에서 전송)"""
    zone: str = Field(..., description="분석 영역 (예: wall_north, floor)")
    max_temp: float
    min_temp: float
    avg_temp: float
    cold_spots: Optional[list[dict]] = Field(None, description="냉점 좌표 리스트")
    hot_spots: Optional[list[dict]] = Field(None, description="열점 좌표 리스트")
    frame_id: Optional[int] = None


class BatchDetectionRequest(BaseModel):
    """다건 탐지 결과 일괄 저장 요청"""
    detections: list[DefectLogCreate]


class BatchDetectionResponse(BaseModel):
    """다건 저장 응답"""
    saved_count: int
    items: list[DefectLogResponse]


@router.post(
    "/detection",
    response_model=DefectLogResponse,
    status_code=201,
    dependencies=[Depends(verify_ai_webhook)],
)
async def receive_detection(
    payload: DefectLogCreate,
    db: AsyncSession = Depends(get_db),
    manager: ConnectionManager = Depends(get_ws_manager),
):
    """
    AI 서버에서 단건 탐지 이벤트 수신.
    DB 저장 + WebSocket 'defects' 채널로 실시간 Push.

    AI 서버 호출 예시:
        import httpx
        httpx.post("http://localhost:8000/api/v1/ai/detection", json={
            "area": "A", "category_code": "A-02",
            "defect_type": "구조 균열", "severity": "HIGH",
            "confidence": 0.87,
            "bbox": {"x": 0.3, "y": 0.4, "w": 0.1, "h": 0.05},
            "lidar_position": {"x": 2.5, "y": 1.0, "z": 1.8},
        })
    """
    # Base64 이미지 → 파일 저장 (실패하면 None 저장, 하자 기록은 계속)
    image_crop_path = await image_storage.save_base64_jpeg(payload.image_crop)

    defect = DefectLog(
        area=payload.area.upper() if payload.area else None,
        defect_source=payload.defect_source,
        defect_class=payload.defect_class,
        defect_class_display_en=payload.defect_class_display_en,
        defect_class_display_ko=payload.defect_class_display_ko,
        category_code=payload.category_code,
        defect_type=payload.defect_type,
        severity=payload.severity.upper(),
        confidence=payload.confidence,
        bbox_x=payload.bbox.x if payload.bbox else None,
        bbox_y=payload.bbox.y if payload.bbox else None,
        bbox_w=payload.bbox.w if payload.bbox else None,
        bbox_h=payload.bbox.h if payload.bbox else None,
        lidar_x=payload.lidar_position.x if payload.lidar_position else None,
        lidar_y=payload.lidar_position.y if payload.lidar_position else None,
        lidar_z=payload.lidar_position.z if payload.lidar_position else None,
        image_crop=None,  # Base64는 더이상 DB에 저장 안 함
        image_crop_path=image_crop_path,
        thermal_max=payload.thermal_data.max if payload.thermal_data else None,
        thermal_min=payload.thermal_data.min if payload.thermal_data else None,
        thermal_avg=payload.thermal_data.avg if payload.thermal_data else None,
        frame_id=payload.frame_id,
        raw_payload=payload.raw_payload,
    )

    db.add(defect)
    await db.flush()

    response = _build_response(defect)

    await manager.broadcast("defects", {
        "type": "defect.new",
        "data": response.model_dump(mode="json"),
    })

    return response


@router.post("/thermal", dependencies=[Depends(verify_ai_webhook)])
async def receive_thermal_analysis(
    payload: ThermalAnalysisResult,
    manager: ConnectionManager = Depends(get_ws_manager),
):
    """
    열화상 분석 결과 수신 → WebSocket 'thermal' 채널로 Push.
    DB 저장 없이 실시간 스트리밍만 수행 (대시보드 Recharts 용).
    """
    await manager.broadcast("thermal", {
        "type": "thermal.analysis",
        "data": payload.model_dump(mode="json"),
    })

    return {"status": "ok", "zone": payload.zone}


@router.post(
    "/batch",
    response_model=BatchDetectionResponse,
    status_code=201,
    dependencies=[Depends(verify_ai_webhook)],
)
async def receive_batch_detections(
    payload: BatchDetectionRequest,
    db: AsyncSession = Depends(get_db),
    manager: ConnectionManager = Depends(get_ws_manager),
):
    """
    다건 탐지 결과 일괄 저장.
    한 프레임에서 여러 하자가 동시에 탐지된 경우 사용.
    """
    saved = []
    for det in payload.detections:
        image_crop_path = await image_storage.save_base64_jpeg(det.image_crop)
        defect = DefectLog(
            area=det.area.upper() if det.area else None,
            category_code=det.category_code,
            defect_type=det.defect_type,
            severity=det.severity.upper(),
            confidence=det.confidence,
            bbox_x=det.bbox.x if det.bbox else None,
            bbox_y=det.bbox.y if det.bbox else None,
            bbox_w=det.bbox.w if det.bbox else None,
            bbox_h=det.bbox.h if det.bbox else None,
            lidar_x=det.lidar_position.x if det.lidar_position else None,
            lidar_y=det.lidar_position.y if det.lidar_position else None,
            lidar_z=det.lidar_position.z if det.lidar_position else None,
            image_crop=None,  # Base64는 더이상 DB에 저장 안 함
            image_crop_path=image_crop_path,
            thermal_max=det.thermal_data.max if det.thermal_data else None,
            thermal_min=det.thermal_data.min if det.thermal_data else None,
            thermal_avg=det.thermal_data.avg if det.thermal_data else None,
            frame_id=det.frame_id,
            raw_payload=det.raw_payload,
        )
        db.add(defect)
        await db.flush()
        saved.append(_build_response(defect))

    # 일괄 WS Push
    await manager.broadcast("defects", {
        "type": "defect.batch",
        "data": {
            "count": len(saved),
            "items": [item.model_dump(mode="json") for item in saved],
        },
    })

    return BatchDetectionResponse(saved_count=len(saved), items=saved)
