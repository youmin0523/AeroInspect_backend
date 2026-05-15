# =============================================
# app/api/floorplan.py
# 역할: 평면도 이미지 업로드 & 처리 API
#       - POST /floorplan/upload      → JPG/PDF/DXF 업로드
#       - POST /floorplan/{id}/process → OpenCV 벽체 추출 트리거
#       - GET  /floorplan             → 업로드 목록 조회
#       - GET  /floorplan/{id}        → 상세 조회 (처리 결과 포함)
#       - DELETE /floorplan/{id}      → 삭제
# =============================================

import math
import os
import uuid as uuid_mod
from uuid import UUID

import aiofiles
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from sqlalchemy import select, func, desc
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import get_current_org_member, get_db
from app.models.floorplan import Floorplan
from app.schemas.floorplan import (
    FloorplanCalibrateRequest,
    FloorplanCalibrateResponse,
    FloorplanUploadResponse,
    FloorplanProcessResponse,
    FloorplanAnalyzeResponse,
    FloorplanListResponse,
    FloorplanValidateResponse,
)
from app.services.floorplan_processor import extract_walls_from_bytes, validate_floorplan_quality
from app.services.gazebo_world_generator import write_world_file

router = APIRouter()

# 업로드 저장 디렉토리
UPLOAD_DIR = "./uploads/floorplans"
WORLD_DIR = "./uploads/gazebo_worlds"

ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "application/pdf",
    "application/dxf",
    "application/octet-stream",  # .dxf fallback
}


async def _get_org_floorplan(db: AsyncSession, org_id, floorplan_id: UUID) -> Floorplan:
    """현재 조직 소유의 평면도만 조회. 없거나 타 조직 것이면 404."""
    result = await db.execute(
        select(Floorplan)
        .where(Floorplan.id == floorplan_id)
        .where(Floorplan.organization_id == org_id)
    )
    fp = result.scalar_one_or_none()
    if not fp:
        raise HTTPException(status_code=404, detail="평면도를 찾을 수 없습니다.")
    return fp


@router.get("", response_model=FloorplanListResponse)
async def list_floorplans(
    db: AsyncSession = Depends(get_db),
    org_tuple=Depends(get_current_org_member),
):
    """업로드된 평면도 목록 조회 (현재 조직 소유분만)."""
    _user, _member, org = org_tuple
    base = select(Floorplan).where(Floorplan.organization_id == org.id)
    total = await db.scalar(
        select(func.count()).select_from(base.subquery())
    )
    result = await db.execute(base.order_by(desc(Floorplan.created_at)))
    items = result.scalars().all()

    return FloorplanListResponse(
        items=[FloorplanUploadResponse.model_validate(item) for item in items],
        total=total or 0,
    )


@router.get("/{floorplan_id}", response_model=FloorplanUploadResponse)
async def get_floorplan(
    floorplan_id: UUID,
    db: AsyncSession = Depends(get_db),
    org_tuple=Depends(get_current_org_member),
):
    """평면도 상세 조회 (현재 조직 소유분만)."""
    _user, _member, org = org_tuple
    fp = await _get_org_floorplan(db, org.id, floorplan_id)
    return FloorplanUploadResponse.model_validate(fp)


@router.post("/upload", response_model=FloorplanUploadResponse, status_code=201)
async def upload_floorplan(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    org_tuple=Depends(get_current_org_member),
):
    """
    평면도 이미지(JPG/PNG/PDF/DXF) 업로드.
    파일을 서버에 저장하고 DB에 메타데이터 기록 (소유 조직 자동 기록).
    이후 /process 엔드포인트로 OpenCV 처리 트리거.
    """
    _user, _member, org = org_tuple

    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"지원하지 않는 파일 형식입니다: {file.content_type}"
        )

    # 업로드 디렉토리 생성
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    # 고유 파일명 생성
    ext = os.path.splitext(file.filename or "unknown")[1]
    saved_filename = f"{uuid_mod.uuid4()}{ext}"
    file_path = os.path.join(UPLOAD_DIR, saved_filename)

    # 파일 저장
    async with aiofiles.open(file_path, "wb") as f:
        content = await file.read()
        await f.write(content)

    # DB 기록 — 소유 조직 자동 기록
    floorplan = Floorplan(
        organization_id=org.id,
        filename=file.filename or "unknown",
        content_type=file.content_type,
        file_path=file_path,
        status="uploaded",
    )
    db.add(floorplan)
    await db.flush()

    return FloorplanUploadResponse.model_validate(floorplan)


@router.post("/{floorplan_id}/process", response_model=FloorplanProcessResponse)
async def process_floorplan(
    floorplan_id: UUID,
    db: AsyncSession = Depends(get_db),
    org_tuple=Depends(get_current_org_member),
):
    """
    업로드된 평면도에서 벽체 라인 추출 트리거.
    OpenCV로 처리 후 벽체 좌표 데이터 반환.

    지원 파일 형식:
      - JPG/PNG/WEBP: OpenCV 벽체 추출 (extract_walls_from_bytes)
      - PDF/DXF: 추후 지원 (pdf2image / ezdxf 의존성 추가 필요)
    """
    _user, _member, org = org_tuple
    fp = await _get_org_floorplan(db, org.id, floorplan_id)

    if fp.status == "processing":
        raise HTTPException(status_code=409, detail="이미 처리 중입니다.")

    # 파일 존재 확인
    if not fp.file_path or not os.path.exists(fp.file_path):
        raise HTTPException(status_code=404, detail="업로드된 파일을 찾을 수 없습니다.")

    # 상태 업데이트
    fp.status = "processing"
    await db.flush()

    try:
        # 파일 읽기
        async with aiofiles.open(fp.file_path, "rb") as f:
            file_bytes = await f.read()

        # 파일 형식별 처리
        content_type = (fp.content_type or "").lower()

        if content_type in {"image/jpeg", "image/png", "image/webp", "application/octet-stream"}:
            # JPG/PNG/WEBP ��� OpenCV 벽체 추출
            extraction = extract_walls_from_bytes(file_bytes)
        elif content_type == "application/pdf":
            # PDF → 이미지 변환 후 처리 (pdf2image 필요)
            try:
                from pdf2image import convert_from_bytes
                import cv2
                import numpy as np

                images = convert_from_bytes(file_bytes, dpi=200, first_page=1, last_page=1)
                img_array = np.array(images[0])
                img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                _, img_bytes = cv2.imencode(".png", img_bgr)
                extraction = extract_walls_from_bytes(img_bytes.tobytes())
            except ImportError:
                fp.status = "failed"
                await db.flush()
                raise HTTPException(
                    status_code=422,
                    detail="PDF 처리를 위한 pdf2image 패키지가 설치되어 있지 않습니다.",
                )
        elif content_type in {"application/dxf"}:
            # //* [Modified Code 2026-05-13] DXF 처리를 dxf_parser 모듈로 이관 —
            # LINE 외에 INSERT(블록)/CIRCLE/ARC/LWPOLYLINE/POLYLINE 까지 처리해
            # CAD 가구 심볼이 자율비행 충돌 회피 입력으로 들어가도록 함.
            try:
                from app.services.dxf_parser import parse_dxf
                extraction = parse_dxf(fp.file_path)
            except ImportError:
                fp.status = "failed"
                await db.flush()
                raise HTTPException(
                    status_code=422,
                    detail="DXF 처리를 위한 ezdxf 패키지가 설치되어 있지 않습니다.",
                )
        else:
            fp.status = "failed"
            await db.flush()
            raise HTTPException(
                status_code=422,
                detail=f"지원하지 않는 파일 형식입니다: {fp.content_type}",
            )

        # DB 업데이트 — 처리 완료
        fp.status = "completed"
        fp.wall_count = extraction["wall_count"]
        fp.walls_data = extraction["walls"]
        # //* [Modified Code 2026-05-13] DXF 분기는 가구 키 없음 → 안전 폴백
        fp.furniture_data = extraction.get("furniture", [])
        fp.furniture_count = extraction.get("furniture_count", 0)
        await db.flush()

        return FloorplanProcessResponse(
            id=fp.id,
            filename=fp.filename,
            status=fp.status,
            wall_count=fp.wall_count,
            walls=fp.walls_data,
            furniture=fp.furniture_data or [],
            furniture_count=fp.furniture_count or 0,
            gazebo_world=fp.gazebo_world_path,
        )

    except HTTPException:
        raise
    except Exception as e:
        fp.status = "failed"
        await db.flush()
        raise HTTPException(status_code=500, detail=f"벽체 추출 처리 중 오류: {str(e)}")


@router.post("/analyze", response_model=FloorplanAnalyzeResponse)
async def analyze_floorplan(
    file: UploadFile = File(...),
    _org=Depends(get_current_org_member),
):
    """
    평면도 이미지에서 벽체 라인 추출 (Stateless — DB 불필요).
    JPG/PNG/WEBP 이미지를 받아 OpenCV로 처리 후 정규화 벽체 좌표 JSON 반환.
    프론트엔드 /employee/pre-work 에서 호출.
    """
    if file.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(
            status_code=400,
            detail=f"이미지 파일만 지원합니다 (JPG/PNG/WEBP): {file.content_type}",
        )

    content = await file.read()

    try:
        result = extract_walls_from_bytes(content)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"이미지 처리 실패: {str(e)}")

    return FloorplanAnalyzeResponse(**result)


@router.post("/validate", response_model=FloorplanValidateResponse)
async def validate_floorplan(
    file: UploadFile = File(...),
    _org=Depends(get_current_org_member),
):
    """
    평면도 이미지 품질 검증 (Stateless — DB 불필요).
    업로드 전 이미지가 벽체 추출에 적합한지 사전 판별.

    검증 항목:
      - 해상도 (최소 1000×1000px 권장)
      - 선명도 (Laplacian variance)
      - 대비 (흑백 표준편차)
      - 직선 비율 (평면도 특성 확인)
      - 직각 교차점 수
      - 기울기 (수평/수직 정렬)
      - 벽체 감지 수

    응답 status:
      - "ok": 양호 — 진행 허용
      - "warning": 주의사항 있으나 진행 가능
      - "rejected": 품질 부족 — 재업로드 권장
    """
    if file.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(
            status_code=400,
            detail=f"이미지 파일만 지원합니다 (JPG/PNG/WEBP): {file.content_type}",
        )

    content = await file.read()

    # 파일 크기 체크 (50KB 미만 거부)
    if len(content) < 50 * 1024:
        return FloorplanValidateResponse(
            status="rejected",
            score=0,
            checks={},
            warnings=[],
            errors=[f"파일 크기가 너무 작습니다 ({len(content) // 1024}KB). 최소 50KB 이상이 필요합니다."],
        )

    try:
        result = validate_floorplan_quality(content)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"이미지 검증 실패: {str(e)}")

    return FloorplanValidateResponse(**result)


@router.post("/{floorplan_id}/calibrate", response_model=FloorplanCalibrateResponse)
async def calibrate_floorplan_scale(
    floorplan_id: UUID,
    payload: FloorplanCalibrateRequest,
    db: AsyncSession = Depends(get_db),
    org_tuple=Depends(get_current_org_member),
):
    """
    평면도 스케일 보정 (FR-015).
    사용자가 찍은 두 점(p1, p2) + 실측 거리(real_length_m) →
      scale_px_per_meter = pixel_length / real_length_m
    이후 벽체 길이·면적을 미터 단위로 표기 가능.
    """
    _user, _member, org = org_tuple
    fp = await _get_org_floorplan(db, org.id, floorplan_id)

    dx = payload.p2[0] - payload.p1[0]
    dy = payload.p2[1] - payload.p1[1]
    pixel_length = math.hypot(dx, dy)
    if pixel_length < 1e-6:
        raise HTTPException(
            status_code=400,
            detail="두 점이 동일합니다. 서로 다른 지점을 지정하세요.",
        )

    px_per_m = pixel_length / payload.real_length_m
    fp.scale_px_per_meter = px_per_m
    fp.scale_reference = {
        "p1": list(payload.p1),
        "p2": list(payload.p2),
        "real_length_m": payload.real_length_m,
    }
    await db.flush()

    return FloorplanCalibrateResponse(
        id=fp.id,
        scale_px_per_meter=round(px_per_m, 4),
        pixel_length=round(pixel_length, 2),
        real_length_m=payload.real_length_m,
    )


@router.post("/{floorplan_id}/generate-world")
async def generate_gazebo_world(
    floorplan_id: UUID,
    db: AsyncSession = Depends(get_db),
    org_tuple=Depends(get_current_org_member),
):
    """
    추출된 walls + outline + scale_px_per_meter → Gazebo SDF .world 파일 생성.
    파일은 ./uploads/gazebo_worlds/{floorplan_id}.world 에 저장되고
    DB 의 gazebo_world_path 에 경로 기록.

    실제 Gazebo 컨테이너에 입력하면 평면도와 동일한 구조의 시뮬레이션 환경이 띄워진다.
    Gazebo 가 없는 환경에서는 backend 자율비행 시뮬레이터(autonomous_flight_simulator)
    가 동일한 walls 데이터로 LiDAR raycast 시뮬레이션을 대신 수행.
    """
    _user, _member, org = org_tuple
    fp = await _get_org_floorplan(db, org.id, floorplan_id)

    if not fp.walls_data:
        raise HTTPException(
            status_code=409,
            detail="아직 벽체 추출이 완료되지 않았습니다. 먼저 /process 를 실행하세요.",
        )

    output_path = os.path.join(WORLD_DIR, f"{floorplan_id}.world")

    # outline 은 Floorplan 모델에 별도 저장 필드가 없어 빈 배열 (현재 스키마 한계)
    # 향후 outline_data JSONB 필드 추가 시 fp.outline_data 사용
    meta = write_world_file(
        output_path=output_path,
        world_name=f"floorplan_{floorplan_id}",
        walls=fp.walls_data,
        outline=[],
        image_width=None,
        image_height=None,
        scale_px_per_meter=fp.scale_px_per_meter,
        # //* [Modified Code 2026-05-13] 가구도 .world 에 collision geometry 로 추가
        # → 실 Gazebo 시뮬에서 드론이 가구를 인식하고 회피 가능
        furniture=fp.furniture_data or [],
    )

    fp.gazebo_world_path = output_path
    await db.flush()

    return {
        "id": str(fp.id),
        "gazebo_world_path": output_path,
        **meta,
    }


@router.get("/{floorplan_id}/world")
async def download_gazebo_world(
    floorplan_id: UUID,
    db: AsyncSession = Depends(get_db),
    org_tuple=Depends(get_current_org_member),
):
    """생성된 Gazebo .world 파일 다운로드."""
    _user, _member, org = org_tuple
    fp = await _get_org_floorplan(db, org.id, floorplan_id)
    if not fp.gazebo_world_path or not os.path.exists(fp.gazebo_world_path):
        raise HTTPException(
            status_code=404,
            detail=".world 파일이 아직 생성되지 않았습니다. /generate-world 를 먼저 호출하세요.",
        )

    return FileResponse(
        path=fp.gazebo_world_path,
        media_type="application/xml",
        filename=f"floorplan_{floorplan_id}.world",
    )


@router.delete("/{floorplan_id}", status_code=204)
async def delete_floorplan(
    floorplan_id: UUID,
    db: AsyncSession = Depends(get_db),
    org_tuple=Depends(get_current_org_member),
):
    """평면도 및 관련 파일 삭제 (현재 조직 소유분만)."""
    _user, _member, org = org_tuple
    fp = await _get_org_floorplan(db, org.id, floorplan_id)

    # 파일 삭제
    if fp.file_path and os.path.exists(fp.file_path):
        os.remove(fp.file_path)

    await db.delete(fp)
