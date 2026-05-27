# =============================================
# app/schemas/defect.py
# 역할: 하자 탐지 로그 Pydantic 입출력 스키마 정의
#       - DefectLogCreate: POST 요청 시 클라이언트 입력 검증
#       - DefectLogResponse: GET 응답 시 직렬화 형식
#       - DefectLogFilter: 목록 조회 시 필터 파라미터
# 사용: API 라우터에서 request body 및 response_model로 사용
# =============================================

from datetime import datetime
from typing import Optional, List
from uuid import UUID

from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    """바운딩 박스 좌표 (0.0 ~ 1.0 정규화)"""
    x: float = Field(ge=0.0, le=1.0)
    y: float = Field(ge=0.0, le=1.0)
    w: float = Field(ge=0.0, le=1.0)
    h: float = Field(ge=0.0, le=1.0)


class LidarPosition(BaseModel):
    """LiDAR 기반 3D 월드 좌표 (미터 단위)"""
    x: Optional[float] = None
    y: Optional[float] = None
    z: Optional[float] = None


class ThermalData(BaseModel):
    """열화상 온도 데이터 (섭씨)"""
    max: Optional[float] = None
    min: Optional[float] = None
    avg: Optional[float] = None


# ── 생성 요청 스키마 ─────────────────────────
class DefectLogCreate(BaseModel):
    """
    하자 탐지 결과 저장 요청.
    AI 파이프라인에서 탐지 후 백엔드 DB에 저장할 때 사용.
    레거시 A-E taxonomy (area/category_code/defect_type)는
    신규 3-모델 중 매핑 불가 클래스의 경우 None 허용.
    """
    # 레거시 taxonomy (신규 모델 결과 중 매핑 없으면 None)
    area: Optional[str] = Field(None, pattern="^[A-E]$", description="하자 영역 코드 (A-E)")
    category_code: Optional[str] = Field(None, description="하자 카테고리 코드 (예: A-01)")
    defect_type: Optional[str] = Field(None, description="하자 유형명 (한글)")

    severity: str = Field(..., pattern="^(HIGH|MED|LOW)$", description="심각도 등급")
    confidence: float = Field(..., ge=0.0, le=1.0, description="AI 탐지 신뢰도")

    # 신규 3-모델 파이프라인 분류 (⚠️ 'good'은 벽지 '터짐(Burst)')
    defect_source: Optional[str] = Field(
        None, pattern="^(yolo_thermal|yolo_delam|wallpaper)$",
        description="탐지 모델 종류"
    )
    defect_class: Optional[str] = Field(None, description="모델 내부 클래스명 (예: 'Crack', 'good')")
    defect_class_display_en: Optional[str] = Field(None, description="영문 표시명")
    defect_class_display_ko: Optional[str] = Field(None, description="한글 표시명")

    bbox: Optional[BoundingBox] = None
    lidar_position: Optional[LidarPosition] = None
    thermal_data: Optional[ThermalData] = None
    image_crop: Optional[str] = None  # Base64 JPEG
    frame_id: Optional[int] = None
    raw_payload: Optional[dict] = None

    model_config = {
        "json_schema_extra": {
            "example": {
                "area": "A",
                "category_code": "A-01",
                "defect_type": "벽체 균열",
                "severity": "HIGH",
                "confidence": 0.87,
                "defect_source": "yolo_thermal",
                "defect_class": "Crack",
                "defect_class_display_en": "Crack",
                "defect_class_display_ko": "균열",
                "bbox": {"x": 0.32, "y": 0.45, "w": 0.18, "h": 0.12},
                "lidar_position": {"x": 2.4, "y": 1.1, "z": 1.5},
                "thermal_data": {"max": 28.4, "min": 22.1, "avg": 25.0},
                "frame_id": 1024,
            },
        },
    }


# ── 응답 스키마 ──────────────────────────────
class DefectLogResponse(BaseModel):
    """
    하자 탐지 로그 응답.
    목록 조회 및 단건 조회 시 반환 형식.
    """
    id: UUID
    # 레거시 taxonomy (신규 모델 중 매핑 없으면 None)
    area: Optional[str]
    category_code: Optional[str]
    defect_type: Optional[str]
    severity: str
    confidence: float

    # 신규 3-모델 파이프라인 분류
    defect_source: Optional[str] = None
    defect_class: Optional[str] = None
    defect_class_display_en: Optional[str] = None
    defect_class_display_ko: Optional[str] = None

    bbox_x: Optional[float]
    bbox_y: Optional[float]
    bbox_w: Optional[float]
    bbox_h: Optional[float]
    lidar_x: Optional[float]
    lidar_y: Optional[float]
    lidar_z: Optional[float]
    image_crop: Optional[str] = Field(None, description="[DEPRECATED] Base64 JPEG. 신규 레코드는 image_crop_url 사용.")
    image_crop_path: Optional[str] = Field(None, description="크롭 이미지 상대 경로 (uploads/ 기준)")
    image_crop_url: Optional[str] = Field(None, description="클라이언트 접근 URL (/uploads/...). StaticFiles 서빙.")
    thermal_max: Optional[float]
    thermal_min: Optional[float]
    thermal_avg: Optional[float]
    timestamp: datetime
    frame_id: Optional[int]

    # ── 검수 메타 (감사·신뢰성) ────────────────
    review_status: str = Field("pending", description="검수 상태 (pending/approved/rejected/flagged_false_positive)")
    reviewed_by_user_id: Optional[UUID] = Field(None, description="검수자 ID")
    reviewed_at: Optional[datetime] = Field(None, description="검수 시각")
    review_note: Optional[str] = Field(None, description="검수 사유/메모")

    # ── 탐지 모델 출처 ───────────────────────
    detection_model_id: Optional[str] = Field(None, description="탐지 모델 (M1_YOLO/M2_YOLO/M3_YOLO/M4_CONTEXT/M5_SEG/furniture_aware 등)")

    # ── GPS WGS84 (현장 정확 위치) ────────────
    gps_lat: Optional[float] = Field(None, description="GPS 위도 (WGS84)")
    gps_lon: Optional[float] = Field(None, description="GPS 경도 (WGS84)")
    gps_alt: Optional[float] = Field(None, description="GPS 고도 (m, MSL)")

    class Config:
        from_attributes = True  # SQLAlchemy ORM 객체 직접 변환


# ── 검수 요청 스키마 (Track C 의존성) ─────────
class DefectReviewRequest(BaseModel):
    """
    하자 검수 요청.
    현장 작업자/사무실 검수자가 AI 탐지 결과를 승인/반려/오탐 플래그할 때 사용.
    rejected/flagged_false_positive 는 review_note 필수 (감사 추적용).
    """
    review_status: str = Field(
        ...,
        pattern="^(approved|rejected|flagged_false_positive|pending)$",
        description="검수 상태",
    )
    review_note: Optional[str] = Field(
        None,
        max_length=2000,
        description="검수 사유. rejected/flagged_false_positive 는 필수.",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "review_status": "rejected",
                "review_note": "벽지 무늬를 균열로 오탐. 실제 균열 아님 확인.",
            },
        },
    }


# ── 목록 조회 필터 ────────────────────────────
class DefectLogFilter(BaseModel):
    """하자 목록 조회 시 필터 파라미터"""
    area: Optional[str] = None          # A-E
    severity: Optional[str] = None      # HIGH / MED / LOW
    category_code: Optional[str] = None  # A-01 등
    limit: int = Field(default=50, ge=1, le=500)
    offset: int = Field(default=0, ge=0)


# ── 요약 통계 응답 ────────────────────────────
class DefectSummary(BaseModel):
    """대시보드용 하자 요약 통계"""
    total: int
    by_severity: dict  # {"HIGH": 3, "MED": 7, "LOW": 12}
    by_area: dict      # {"A": 5, "B": 3, ...}
    latest: Optional[DefectLogResponse] = None


class DefectLogListResponse(BaseModel):
    """하자 목록 페이지네이션 응답"""
    items: List[DefectLogResponse]
    total: int
    limit: int
    offset: int
