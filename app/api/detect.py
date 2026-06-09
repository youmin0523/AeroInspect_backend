# =============================================
# app/api/detect.py
# 역할: 3-모델 추론 REST 엔드포인트 (이미지 업로드 → 즉시 추론)
#       - POST /detect        → multipart 단건 업로드 → DetectionResult
#       - POST /detect/batch  → 최대 10장 일괄 업로드 → BatchDetectionResponse
#                                (per-item 격리: 한 장 실패가 배치 전체를 죽이지 않음)
#
# 에러 코드:
#   400: 이미지 디코딩 실패 / 지원 안 되는 파일
#   413: 파일 크기 초과 (batch 10장 초과)
#   503: 모델 미로드 (가중치 없음)
# =============================================

from __future__ import annotations

import asyncio
from typing import List, Optional

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from fastapi.responses import Response

from app.config import settings
from app.dependencies import verify_ai_webhook_or_user
from app.schemas.detection import (
    BatchDetectionItem,
    BatchDetectionResponse,
    CompareResult,
    DetectionResult,
    HybridDetectionResult,
    VLMDetectionResult,
)
from app.services.hybrid_detector import detect_hybrid_async
from app.services.inference_pipeline import detect_defects_async, pipeline
from app.services.vlm_detector import VLMQuotaExceeded, detect_vlm_async

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
        # 이미지 디코딩 실패 — 내부 detail 로깅, 클라이언트엔 일반 메시지 (R-v1.1.17 보안)
        import logging
        logging.getLogger(__name__).warning("detect ValueError: %s", e)
        raise HTTPException(status_code=400, detail="이미지 디코딩 실패 — 지원되지 않는 형식이거나 손상되었습니다.")
    except RuntimeError as e:
        import logging
        logging.getLogger(__name__).error("detect RuntimeError: %s", e)
        raise HTTPException(status_code=503, detail="추론 서비스 일시 중단 — 잠시 후 다시 시도해주세요.")


@router.post(
    "/batch",
    response_model=BatchDetectionResponse,
    summary="다중 이미지 하자 탐지 (배치, per-item 격리)",
    description=(
        f"최대 {MAX_BATCH_SIZE}장을 한 번에 업로드. 각 이미지마다 독립 추론 후 "
        "per-item 결과(success/result/error)로 반환. 한 장이 실패해도 나머지는 정상 처리."
    ),
)
async def detect_batch(
    images: List[UploadFile] = File(..., description=f"이미지 파일 리스트 (최대 {MAX_BATCH_SIZE}장)"),
    _auth=Depends(verify_ai_webhook_or_user),
) -> BatchDetectionResponse:
    # ⚠️ 응답 스키마 변경(승인됨): 과거 List[DetectionResult] → BatchDetectionResponse.
    #    이전 구현은 asyncio.gather(return_exceptions 없이) + HTTPException 재발생이라
    #    한 장 디코딩 실패가 배치 전체를 4xx 로 죽였다. 이제 per-item 으로 격리한다.
    #    프론트엔드는 items[].success 로 분기 처리해야 함.
    _ensure_loaded()

    if len(images) == 0:
        raise HTTPException(status_code=400, detail="이미지가 하나 이상 필요합니다.")
    if len(images) > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"배치 크기 초과: {len(images)} > {MAX_BATCH_SIZE}",
        )

    # 1) 이미지별 추론을 per-item 격리로 실행 (입력 순서 보존).
    #    읽기/검증/디코딩/추론 전 과정을 태스크 단위 try/except 로 감싸 한 장의
    #    실패(content-type 오류, 빈 파일, 디코딩 실패, 추론 오류)가 배치 전체를
    #    중단시키지 않게 한다. 반환값은 (성공 result | 오류 메시지) 튜플.
    async def _detect_one(upload: UploadFile) -> tuple[Optional[DetectionResult], Optional[str]]:
        try:
            _validate_content_type(upload)
            raw = await upload.read()
            if not raw:
                return None, "빈 파일입니다."
            result = await detect_defects_async(raw)
            return result, None
        except HTTPException as e:
            # _validate_content_type 등이 던지는 4xx 도 per-item 오류로 흡수
            return None, str(e.detail)
        except ValueError:
            return None, "이미지 디코딩 실패 — 지원되지 않는 형식이거나 손상되었습니다."
        except Exception as e:  # noqa: BLE001 — 한 장 실패가 배치를 죽이지 않도록 흡수
            import logging
            logging.getLogger(__name__).error("batch detect 실패(%s): %s", upload.filename, e)
            return None, "추론 실패 — 잠시 후 다시 시도해주세요."

    # 2) 동시 실행. return_exceptions=True 로 예기치 못한 예외도 결과 슬롯에 담아
    #    순서를 보존(gather 는 입력 순서대로 결과를 정렬).
    outcomes = await asyncio.gather(
        *(_detect_one(upload) for upload in images),
        return_exceptions=True,
    )

    items: List[BatchDetectionItem] = []
    for idx, (upload, outcome) in enumerate(zip(images, outcomes)):
        if isinstance(outcome, BaseException):
            # _detect_one 이 모든 예외를 흡수하므로 여기 도달은 이론상 없으나
            # 방어적으로 처리 (gather return_exceptions 안전망).
            result, error = None, "내부 처리 오류"
        else:
            result, error = outcome
        items.append(BatchDetectionItem(
            index=idx,
            filename=upload.filename,
            success=result is not None,
            result=result,
            error=error,
        ))

    success_count = sum(1 for it in items if it.success)
    return BatchDetectionResponse(
        items=items,
        total=len(items),
        success_count=success_count,
        failure_count=len(items) - success_count,
    )


# =============================================
# VLM (비전 LLM) 검출 — 기존 ONNX와 병행 비교 PoC
# =============================================

def _ensure_vlm_enabled() -> None:
    if not settings.VLM_DETECTION_ENABLED:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="VLM 검출이 비활성화됨. VLM_DETECTION_ENABLED=true 로 설정하세요.",
        )


async def _read_image(image: UploadFile) -> bytes:
    _validate_content_type(image)
    raw = await image.read()
    if not raw:
        raise HTTPException(status_code=400, detail="빈 파일입니다.")
    return raw


async def _run_vlm(raw: bytes, mode: Optional[str], provider: Optional[str]) -> VLMDetectionResult:
    try:
        return await detect_vlm_async(raw, mode=mode, provider=provider)
    except VLMQuotaExceeded as e:
        raise HTTPException(status_code=429, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail="이미지 디코딩 실패 — 지원되지 않는 형식이거나 손상되었습니다.")
    except Exception as e:  # noqa: BLE001 — 외부 API 실패는 내부 로깅 후 503
        import logging
        logging.getLogger(__name__).error("VLM 검출 실패: %s", e)
        raise HTTPException(status_code=503, detail="VLM 검출 서비스 일시 중단 — API 키/네트워크를 확인하세요.")


@router.post(
    "/vlm",
    response_model=VLMDetectionResult,
    summary="비전 LLM 단일 이미지 하자 검출",
    description="Gemini/Claude/GPT-4o 비전 모델로 이미지 1장을 판정. mode=classify(기본)|grounding.",
)
async def detect_vlm(
    image: UploadFile = File(..., description="업로드 이미지 파일 (JPEG/PNG/...)"),
    mode: Optional[str] = Query(None, description="classify | grounding (미지정 시 서버 기본값)"),
    provider: Optional[str] = Query(None, description="gemini | claude | openai (미지정 시 서버 기본값)"),
    _auth=Depends(verify_ai_webhook_or_user),
) -> VLMDetectionResult:
    _ensure_vlm_enabled()
    raw = await _read_image(image)
    return await _run_vlm(raw, mode, provider)


@router.post(
    "/compare",
    response_model=CompareResult,
    summary="ONNX vs VLM 병행 비교",
    description="동일 이미지에 기존 3-모델 ONNX와 VLM을 동시 실행해 결과를 나란히 반환 (PoC 평가용).",
)
async def detect_compare(
    image: UploadFile = File(..., description="업로드 이미지 파일 (JPEG/PNG/...)"),
    mode: Optional[str] = Query(None, description="VLM mode: classify | grounding"),
    provider: Optional[str] = Query(None, description="VLM provider: gemini | claude | openai"),
    _auth=Depends(verify_ai_webhook_or_user),
) -> CompareResult:
    _ensure_loaded()
    _ensure_vlm_enabled()
    raw = await _read_image(image)

    # ONNX와 VLM 동시 실행 (VLM은 네트워크 지연, ONNX는 로컬 — gather로 병렬)
    onnx_task = detect_defects_async(raw)
    vlm_task = _run_vlm(raw, mode, provider)
    try:
        onnx_res, vlm_res = await asyncio.gather(onnx_task, vlm_task)
    except HTTPException:
        raise
    except ValueError:
        raise HTTPException(status_code=400, detail="이미지 디코딩 실패 — 지원되지 않는 형식이거나 손상되었습니다.")

    return CompareResult(
        onnx=onnx_res,
        vlm=vlm_res,
        onnx_defect_count=onnx_res.defect_count,
        vlm_defect_count=vlm_res.defect_count,
        image_shape=vlm_res.image_shape,
    )


@router.post(
    "/hybrid",
    response_model=HybridDetectionResult,
    summary="ONNX+VLM 하이브리드 검출 (상업용 캐스케이드 판정)",
    description=(
        "ONNX가 후보를 제안하고 VLM이 검증/종류교정/기각 + 누락 보완. "
        "단일 엔진 단독 CONFIRMED 불가 — 합의/교정만 보고서 등재. provenance 포함."
    ),
)
async def detect_hybrid(
    image: UploadFile = File(..., description="업로드 이미지 파일 (JPEG/PNG/...)"),
    provider: Optional[str] = Query(None, description="VLM provider: gemini | claude | openai"),
    _auth=Depends(verify_ai_webhook_or_user),
) -> HybridDetectionResult:
    _ensure_vlm_enabled()
    raw = await _read_image(image)
    try:
        return await detect_hybrid_async(raw, provider=provider)
    except VLMQuotaExceeded as e:
        raise HTTPException(status_code=429, detail=str(e))
    except ValueError:
        raise HTTPException(status_code=400, detail="이미지 디코딩 실패 — 지원되지 않는 형식이거나 손상되었습니다.")
    except Exception as e:  # noqa: BLE001
        import logging
        logging.getLogger(__name__).error("하이브리드 검출 실패: %s", e)
        raise HTTPException(status_code=503, detail="하이브리드 검출 서비스 일시 중단 — API 키/모델 상태를 확인하세요.")


@router.post(
    "/hybrid/visualize",
    summary="하이브리드 검출 결과 시각화 (bbox+종류+등급 주석 이미지)",
    description="ONNX+VLM 하이브리드 검출 후 원본 이미지에 박스·하자종류·등급을 그려 JPEG로 반환.",
    responses={200: {"content": {"image/jpeg": {}}}},
)
async def detect_hybrid_visualize(
    image: UploadFile = File(..., description="업로드 이미지 파일 (JPEG/PNG/...)"),
    provider: Optional[str] = Query(None, description="VLM provider: gemini | claude | openai"),
    _auth=Depends(verify_ai_webhook_or_user),
) -> Response:
    import cv2
    import numpy as np
    from app.services.detection_overlay import annotate_hybrid, encode_jpeg

    _ensure_vlm_enabled()
    raw = await _read_image(image)
    try:
        result = await detect_hybrid_async(raw, provider=provider)
    except VLMQuotaExceeded as e:
        raise HTTPException(status_code=429, detail=str(e))
    except ValueError:
        raise HTTPException(status_code=400, detail="이미지 디코딩 실패 — 지원되지 않는 형식이거나 손상되었습니다.")
    except Exception as e:  # noqa: BLE001
        import logging
        logging.getLogger(__name__).error("하이브리드 시각화 실패: %s", e)
        raise HTTPException(status_code=503, detail="하이브리드 검출 서비스 일시 중단.")

    frame = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    annotated = annotate_hybrid(frame, result)
    headers = {
        "X-Defect-Count": str(result.defect_count),
        "X-Confirmed-Count": str(result.confirmed_count),
        "X-Review-Count": str(result.review_count),
    }
    return Response(content=encode_jpeg(annotated), media_type="image/jpeg", headers=headers)
