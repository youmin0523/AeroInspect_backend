# =============================================
# app/schemas/detection.py
# 역할: 3-모델 추론 파이프라인 응답 Pydantic 스키마
#       - BBox: xyxy 픽셀 좌표 (API/WS 공통, 프론트 Canvas 렌더용)
#       - YoloDetection: YOLOv8 단건 (thermal / delam 공통 포맷)
#       - WallpaperPrediction: ResNet50 top1 + top3 분류 결과
#       - DetectionResult: detect_defects() 통합 응답
#       - WSStreamMessage: WebSocket 스트림 송수신 메시지
#       - HealthResponse: /health 엔드포인트 응답
# 주의: API 응답은 bbox_xyxy 유지, DB 저장 시에만 xywhn 변환
# =============================================

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


Severity = Literal["HIGH", "MED", "LOW"]
# 신뢰도 등급 (confidence_grader.py 와 1:1 매칭)
#   CONFIRMED : 하자목록 등재 — 분쟁 시 책임질 수준
#   REVIEW    : 점검자 추가 확인 권장 — 목록 X, 별도 섹션
#   REFERENCE : 참고용 — 점검자 모드 토글 시만
Grade = Literal["CONFIRMED", "REVIEW", "REFERENCE"]
DefectSource = Literal["yolo_thermal", "yolo_delam", "wallpaper"]
# 20종 파이프라인 defect source
DefectSource20 = Literal[
    "yolo_structural", "yolo_surface", "yolo_floor_window",
    "thermal_unet", "geometric", "patchcore",
]


class BBox(BaseModel):
    """바운딩 박스 — xyxy 픽셀 좌표."""
    x1: float
    y1: float
    x2: float
    y2: float

    def to_list(self) -> List[float]:
        return [self.x1, self.y1, self.x2, self.y2]


class YoloDetection(BaseModel):
    """YOLOv8 탐지 단건 (thermal / delam 동일 포맷)."""
    class_: str = Field(..., alias="class", description="모델 내부 클래스명 (예: 'Crack')")
    class_display_en: str
    class_display_ko: str
    conf: float = Field(..., ge=0.0, le=1.0)
    bbox_xyxy: List[float] = Field(..., min_length=4, max_length=4, description="[x1, y1, x2, y2] 픽셀 좌표")

    model_config = {"populate_by_name": True}


class Top3Prediction(BaseModel):
    """ResNet50 top3 단건."""
    class_: str = Field(..., alias="class")
    class_display_en: str
    class_display_ko: str
    conf: float = Field(..., ge=0.0, le=1.0)

    model_config = {"populate_by_name": True}


class WallpaperPrediction(BaseModel):
    """ResNet50 벽지 분류 결과."""
    top1_class: str = Field(..., description="모델 내부명. ⚠️ 'good'은 실제로 '터짐' 하자임")
    top1_class_display_en: str
    top1_class_display_ko: str
    top1_conf: float = Field(..., ge=0.0, le=1.0)
    is_confident: bool = Field(
        ...,
        description=(
            "(top1_conf >= WALLPAPER_CONF_THRESHOLD) AND "
            "(top1_conf - top2_conf >= WALLPAPER_MARGIN_THRESHOLD). "
            "false면 모호/노이즈로 취급하여 severity 판정 보류."
        )
    )
    top3: List[Top3Prediction] = Field(default_factory=list, max_length=3)


class ImageShape(BaseModel):
    """프레임 크기 (xyxy → xywhn 변환에 필요)."""
    width: int
    height: int


class DetectionResult(BaseModel):
    """
    3-모델 통합 추론 응답.
    inference_pipeline.detect_defects()의 반환 타입과 1:1 매칭.
    """
    yolo_thermal: List[YoloDetection] = Field(default_factory=list)
    yolo_delam: List[YoloDetection] = Field(default_factory=list)
    wallpaper_cls: Optional[WallpaperPrediction] = None
    severity: Optional[Severity] = Field(
        None,
        description="HIGH/MED/LOW. 신뢰도 부족 시 null (판단 보류)"
    )
    has_defect: bool = False
    defect_count: int = 0
    image_shape: ImageShape = Field(
        ...,
        description="프레임 W/H. DB 저장 시 xyxy → xywhn 변환에 사용"
    )


class WSStreamMessage(BaseModel):
    """WebSocket /ws/stream 송신 메시지."""
    type: Literal["detection", "pong", "error"]
    timestamp: float
    frame_id: Optional[int] = None
    result: Optional[DetectionResult] = None
    message: Optional[str] = None


class ModelsLoadedStatus(BaseModel):
    yolo_thermal: bool
    yolo_delam: bool
    wallpaper: bool


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    device: Literal["cuda", "cpu"]
    models_loaded: ModelsLoadedStatus
    wallpaper_classes_count: int
    stream_worker_running: bool
    frame_skip: int


# =============================================
# 20종 하자 검출 파이프라인 스키마 (신규)
# =============================================

class DefectDetection(BaseModel):
    """20종 파이프라인 통합 검출 단건."""
    class_: str = Field(..., alias="class", description="severity_mapper class_name")
    class_display_en: str = ""
    class_display_ko: str = ""
    code: str = Field("", description="A-01 ~ E-02 카테고리 코드")
    conf: float = Field(..., ge=0.0, le=1.0)
    bbox_xyxy: List[float] = Field(default_factory=list, description="[x1,y1,x2,y2] 픽셀")
    severity: Optional[Severity] = None
    grade: Grade = Field(
        "REVIEW",
        description="신뢰도 등급. CONFIRMED만 보고서 하자목록 등재.",
    )
    grade_display_ko: str = ""
    defect_source: str = ""
    ensemble_boosted: bool = False
    cross_model_boosted: bool = False

    model_config = {"populate_by_name": True}


class InsulationDetection(BaseModel):
    """M4 열화상 단열/기밀/난방 하자."""
    class_: str = Field(..., alias="class")
    code: str
    display_ko: str = ""
    conf: float = Field(..., ge=0.0, le=1.0)
    bbox_xyxy: List[float] = Field(default_factory=list)
    delta_temperature: float = Field(..., description="주변 대비 온도차 (°C)")
    max_temperature: float = 0.0
    min_temperature: float = 0.0
    severity: Optional[Severity] = None
    grade: Grade = "REVIEW"
    grade_display_ko: str = ""
    defect_source: str = "thermal_unet"

    model_config = {"populate_by_name": True}


class AlignmentDetection(BaseModel):
    """M5+G1 기하학 수직수평/직각도 하자."""
    class_: str = Field(..., alias="class")
    code: str
    display_ko: str = ""
    conf: float = Field(..., ge=0.0, le=1.0)
    bbox_xyxy: List[float] = Field(default_factory=list)
    deviation_degrees: float = Field(..., description="편차 각도 (도)")
    deviation_mm_per_m: float = Field(0.0, description="편차 mm/m")
    direction: str = Field("", description="vertical | horizontal | both")
    severity: Optional[Severity] = None
    grade: Grade = "REVIEW"
    grade_display_ko: str = ""
    defect_source: str = "geometric"

    model_config = {"populate_by_name": True}


class DetectionResult20(BaseModel):
    """
    20종 하자 통합 추론 응답.
    기존 DetectionResult와 병존 — USE_20DEFECT_PIPELINE 플래그로 전환.

    detections/insulation/alignment 모두 grade 필드 포함.
    보고서 등재용은 confirmed_count, 점검자 표시용은 review_count.
    """
    detections: List[DefectDetection] = Field(default_factory=list)
    insulation: List[InsulationDetection] = Field(default_factory=list)
    alignment: List[AlignmentDetection] = Field(default_factory=list)
    anomaly_score: Optional[float] = None
    has_defect: bool = False
    defect_count: int = 0
    confirmed_count: int = Field(
        0, description="CONFIRMED 등급 합계 — 보고서 하자목록 등재 대상"
    )
    review_count: int = Field(
        0, description="REVIEW 등급 합계 — 점검자 추가 확인 권장"
    )
    image_shape: ImageShape = Field(
        ..., description="프레임 W/H"
    )
    tier_executed: int = Field(1, description="실행된 최고 Tier (1/2/3)")


# =============================================
# VLM (비전 LLM) 하자 검출 스키마 — 기존 ONNX와 병행 비교 PoC
#   - 기존 YoloDetection/DefectDetection과 필드 정렬해 나란히 비교 용이
#   - classify 모드: bbox 없음(localization="image_level", 전체 프레임 사용)
#   - grounding 모드: Gemini normalized bbox(0~1000) → 픽셀 xyxy 변환
# =============================================

VLMLocalization = Literal["image_level", "bbox"]


class VLMDetection(BaseModel):
    """비전 LLM 검출 단건."""
    class_: str = Field(..., alias="class", description="severity_mapper class_name")
    class_display_ko: str = ""
    code: str = Field("", description="A-01 ~ E-02 카테고리 코드")
    area: str = Field("", description="영역 A-E")
    conf: float = Field(..., ge=0.0, le=1.0, description="VLM 자가 보고 신뢰도")
    severity: Optional[Severity] = None
    bbox_xyxy: List[float] = Field(
        default_factory=list,
        description="[x1,y1,x2,y2] 픽셀. classify 모드면 전체 프레임",
    )
    localization: VLMLocalization = "image_level"
    reasoning: str = Field("", description="VLM 판정 근거 (짧은 설명)")

    model_config = {"populate_by_name": True}


class VLMDetectionResult(BaseModel):
    """비전 LLM 통합 검출 응답."""
    detections: List[VLMDetection] = Field(default_factory=list)
    has_defect: bool = False
    defect_count: int = 0
    provider: str = ""
    model: str = ""
    mode: str = ""
    latency_ms: float = 0.0
    cached: bool = False
    image_shape: ImageShape


class CompareResult(BaseModel):
    """
    동일 이미지에 ONNX(기존 3-모델)와 VLM을 동시 실행한 병행 비교 결과.
    PoC 평가용 — 두 방식의 검출을 나란히 확인.
    """
    onnx: DetectionResult
    vlm: VLMDetectionResult
    onnx_defect_count: int = 0
    vlm_defect_count: int = 0
    image_shape: ImageShape


# =============================================
# 하이브리드 검출 스키마 — ONNX 제안 + VLM 판정 (캐스케이드 판정)
#   상업용 원칙: 단일 엔진 단독 CONFIRMED 금지.
#     - ONNX+VLM 합의/종류교정 → CONFIRMED 가능 (위치=ONNX 정밀, 종류=VLM 권위)
#     - ONNX 단독 / VLM 단독 추가 → REVIEW 상한 (점검자 확인)
#     - VLM 기각 → REFERENCE (감사 로그)
#   모든 검출에 provenance(onnx_conf/vlm_conf/agreement/reasoning) 기록.
# =============================================

HybridStatus = Literal[
    "confirmed_by_both",  # ONNX 검출 + VLM 동의
    "reclassified",       # ONNX 위치 + VLM 종류 교정
    "onnx_only",          # ONNX 검출, VLM 미언급 (미검증)
    "vlm_only",           # ONNX 놓침, VLM 추가 (위치 미검증)
    "rejected",           # ONNX 검출, VLM 기각 (오탐)
]


class HybridDetection(BaseModel):
    """ONNX+VLM 캐스케이드 판정 단건 (provenance 포함)."""
    class_: str = Field(..., alias="class", description="최종 확정 class_name")
    class_display_ko: str = ""
    code: str = ""
    area: str = ""
    conf: float = Field(..., ge=0.0, le=1.0, description="등급 산정에 사용된 최종 신뢰도")
    severity: Optional[Severity] = None
    bbox_xyxy: List[float] = Field(default_factory=list, description="[x1,y1,x2,y2] 픽셀")
    localization: VLMLocalization = "image_level"
    status: HybridStatus
    grade: Grade = "REVIEW"
    grade_display_ko: str = ""
    listable: bool = Field(False, description="보고서 하자목록 등재 여부 (CONFIRMED만 True)")
    # ── provenance (감사 추적) ──
    onnx_conf: Optional[float] = Field(None, description="ONNX 원본 신뢰도 (없으면 VLM 단독)")
    vlm_conf: Optional[float] = Field(None, description="VLM 판정 신뢰도")
    agreement: bool = Field(False, description="ONNX·VLM 합의 여부 (위치+종류)")
    source: str = Field("", description="onnx+vlm | onnx | vlm")
    reasoning: str = Field("", description="VLM 판정 근거")

    model_config = {"populate_by_name": True}


class HybridDetectionResult(BaseModel):
    """ONNX 제안 → VLM 판정 → 결정론적 병합 통합 응답."""
    detections: List[HybridDetection] = Field(default_factory=list)
    has_defect: bool = False
    defect_count: int = 0
    confirmed_count: int = Field(0, description="CONFIRMED — 보고서 등재 대상")
    review_count: int = Field(0, description="REVIEW — 점검자 추가 확인")
    rejected_count: int = Field(0, description="VLM이 기각한 ONNX 오탐 수")
    onnx_engine: str = Field("", description="pipeline20 | pipeline3 | none")
    vlm_provider: str = ""
    vlm_model: str = ""
    vlm_calls: int = Field(0, description="VLM 호출 수 (충돌 재판정 시 2)")
    latency_ms: float = 0.0
    image_shape: ImageShape


class ModelsLoadedStatus20(BaseModel):
    """20종 파이프라인 모델 상태."""
    m1_yolo: bool = False
    m1_resnet: bool = False
    m2_yolo: bool = False
    m2_resnet: bool = False
    m3_yolo: bool = False
    m3_resnet: bool = False
    m4_unet: bool = False
    m4_context: bool = False
    m5_seg: bool = False
    m6_patchcore: bool = False
    furniture_aware: bool = False
