# =============================================
# app/schemas/thermal_screening.py
# 역할: 의사색 단열 스크리닝(보조) 검수 피드백 요청/응답 스키마
#       - 스크리닝 항목은 DB 영속화되지 않음(WS 방출 전용) → 검수는 audit_logs 로만 회수.
#       - 본 검출(DefectReviewRequest) 패턴을 미러링하되, 스크리닝 정체성(파일/시각/bbox/점수)을
#         함께 받아 피드백 레코드에 그대로 적재(재학습 export 시 식별 가능하도록).
# =============================================

from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, Field


class ScreeningBBox(BaseModel):
    """스크리닝 anomaly 의 원본 프레임 픽셀 좌표 bbox."""
    x1: float
    y1: float
    x2: float
    y2: float


class ThermalScreeningReviewRequest(BaseModel):
    """
    단열 스크리닝 항목 검수 피드백.
      - confirmed: 점검자가 '실제 단열 의심부로 채택'(참고용 확인)
      - dismissed: '무시'(추적 불필요)
      - flagged_false_positive: '오탐' — 재학습 데이터로 회수(사유 필수)

    스크리닝은 영속 id 가 없으므로 프런트가 식별 메타(파일/시각/bbox/kind/score)를 함께 보낸다.
    client_item_id 는 프런트 세션 내 합성 id(`${ts}_${i}`) — 같은 세션 UI 매칭/중복 방지에만 사용.
    """
    # ── 스크리닝 항목 식별/맥락 (영속화 안 됨 → 그대로 피드백에 적재) ──
    video_timestamp_sec: float = Field(..., description="키프레임 영상 시각(초)")
    filename: Optional[str] = Field(None, max_length=255, description="대상 영상 파일명")
    frame_w: Optional[int] = Field(None, ge=0)
    frame_h: Optional[int] = Field(None, ge=0)
    bbox: Optional[ScreeningBBox] = None
    kind: Optional[str] = Field(None, pattern="^(spot|patch|area)$")
    severity: Optional[str] = Field(None, pattern="^(HIGH|MED|LOW)$")
    score: Optional[float] = Field(None, description="상대 냉점 강도(0-100 등)")
    client_item_id: Optional[str] = Field(None, max_length=64, description="프런트 합성 id")

    # ── 검수 액션 ──
    review_status: str = Field(
        ...,
        pattern="^(confirmed|dismissed|flagged_false_positive)$",
        description="검수 상태",
    )
    review_note: Optional[str] = Field(
        None,
        max_length=2000,
        description="검수 사유. flagged_false_positive 는 필수.",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "video_timestamp_sec": 45.3,
                "filename": "thermal_20260616.mp4",
                "frame_w": 1920,
                "frame_h": 1080,
                "bbox": {"x1": 820, "y1": 410, "x2": 980, "y2": 560},
                "kind": "patch",
                "severity": "MED",
                "score": 7.3,
                "client_item_id": "45.300_1",
                "review_status": "flagged_false_positive",
                "review_note": "창틀 반사(핫소스 경계)를 단열 의심으로 오인.",
            },
        },
    }


class ThermalScreeningReviewResponse(BaseModel):
    """검수 피드백 접수 결과(에코). 영속 레코드가 아니므로 접수 확인 용도."""
    ok: bool = True
    client_item_id: Optional[str] = None
    review_status: str
    review_note: Optional[str] = None
    reviewed_by_user_id: UUID
    reviewed_at: datetime
