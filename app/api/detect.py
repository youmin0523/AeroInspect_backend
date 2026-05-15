# =============================================
# app/api/detect.py
# 역할: 3-모델 추론 REST 엔드포인트 (이미지 업로드 → 즉시 추론)
#       - POST /detect        → multipart 단건 업로드 → DetectionResult
#       - POST /detect/batch  → 최대 10장 일괄 업로드 → List[DetectionResult]
#
# 에러 코드:
#   400: 이미지 디코딩 실패 / 지원 안 되는 파일
#   413: 파일 크기 초과 (batch 10장 초과)
#   503: 모델 미로드 (가중치 없음)
# =============================================

from __future__ import annotations

from typing import List

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from app.dependencies import verify_ai_webhook_or_user
from app.schemas.detection import DetectionResult
from app.services.inference_pipeline import detect_defects_async, pipeline

router = APIRouter()

MAX_BATCH_SIZE = 10
ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/bmp",
    "image/tiff",
    # 일부 클라이언트는 content-type을 안 보냄 → application/octet-stream 허용
    "application/octet-stream",
}


def _ensure_loaded() -> None:
    if not pipeline.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="추론 모델이 로드되지 않았습니다. weights/ 폴더에 3개 가중치 파일을 배치하고 서버를 재시작하세요.",
        )


def _validate_content_type(upload: UploadFile) -> None:
    ct = (upload.content_type or "").lower()
    if ct and ct not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"지원하지 않는 이미지 타입입니다: {ct}",
        )


@router.post(
    "",
    response_model=DetectionResult,
    summary="단일 이미지 하자 탐지",
    description="multipart로 이미지 1장을 업로드하면 3-모델(YOLO thermal + delam + ResNet 벽지) 추론 결과를 반환합니다.",
)
async def detect_single(
    image: UploadFile = File(..., description="업로드 이미지 파일 (JPEG/PNG/WEBP/BMP/TIFF)"),
    _auth=Depends(verify_ai_webhook_or_user),
) -> DetectionResult:
    _ensure_loaded()
    _validate_content_type(image)

    raw = await image.read()
    if not raw:
        raise HTTPException(status_code=400, detail="빈 파일입니다.")

    try:
        return await detect_defects_async(raw)
    except ValueError as e:
        # 이미지 디코딩 실패
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))


@router.post(
    "/batch",
    response_model=List[DetectionResult],
    summary="다중 이미지 하자 탐지 (배치)",
    description=f"최대 {MAX_BATCH_SIZE}장을 한 번에 업로드. 각 이미지마다 독립 추론 후 리스트로 반환.",
)
async def detect_batch(
    images: List[UploadFile] = File(..., description=f"이미지 파일 리스트 (최대 {MAX_BATCH_SIZE}장)"),
    _auth=Depends(verify_ai_webhook_or_user),
) -> List[DetectionResult]:
    _ensure_loaded()

    if len(images) == 0:
        raise HTTPException(status_code=400, detail="이미지가 하나 이상 필요합니다.")
    if len(images) > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"배치 크기 초과: {len(images)} > {MAX_BATCH_SIZE}",
        )

    results: List[DetectionResult] = []
    for upload in images:
        _validate_content_type(upload)
        raw = await upload.read()
        if not raw:
            raise HTTPException(status_code=400, detail=f"빈 파일: {upload.filename}")
        try:
            result = await detect_defects_async(raw)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"{upload.filename}: {e}")
        results.append(result)
    return results
