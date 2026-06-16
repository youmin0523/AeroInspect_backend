# =============================================
# app/api/thermal_screening.py
# 역할: 의사색 단열 스크리닝(보조) 검수 피드백 엔드포인트
#       - POST /thermal-screening/review
#       - 스크리닝 항목은 DB 영속화되지 않음(thermal.screening WS 방출 전용) →
#         검수 피드백을 audit_logs 에 적재해 회수(본 검출의 flagged_false_positive 와 동일 export 경로).
#       - 다른 클라이언트도 즉시 반영하도록 thermal.screening.reviewed WS 브로드캐스트.
#
#   설계 메모(경량 선택):
#     스크리닝은 '확정 진단 아님' 보조 신호라 본 검출처럼 테이블/UUID 를 두지 않는다.
#     오탐(특히 단열 클래스)의 피드백 회수가 핵심 가치이므로, 마이그레이션 없이 audit_logs 만으로
#     회수한다. 영구 검수상태가 필요해지면 추후 thermal_screening_items 테이블로 승격 가능.
# =============================================

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import get_db, get_ws_manager, get_current_org_member
from app.core.ws_manager import ConnectionManager
from app.services.audit_logger import write_audit
from app.schemas.thermal_screening import (
    ThermalScreeningReviewRequest,
    ThermalScreeningReviewResponse,
)

router = APIRouter()

# 검수 상태 → 감사 로그 action (review_status 별 세분화 → 통계/재학습 export 검색 용이)
_ACTION_MAP = {
    "confirmed": "thermal_screening.review.confirm",
    "dismissed": "thermal_screening.review.dismiss",
    "flagged_false_positive": "thermal_screening.review.flag_false_positive",
}


def _validate_review_note(payload: ThermalScreeningReviewRequest) -> None:
    """오탐 신고는 review_note 필수 (재학습 데이터 품질 + 감사 추적)."""
    if payload.review_status == "flagged_false_positive":
        if not payload.review_note or not payload.review_note.strip():
            raise HTTPException(
                status_code=400,
                detail="flagged_false_positive 검수는 review_note(사유)가 필수입니다.",
            )


@router.post("/review", response_model=ThermalScreeningReviewResponse)
async def review_thermal_screening(
    payload: ThermalScreeningReviewRequest,
    request: Request,
    org_tuple=Depends(get_current_org_member),
    db: AsyncSession = Depends(get_db),
    manager: ConnectionManager = Depends(get_ws_manager),
):
    """
    단열 스크리닝 항목 검수 (확인/무시/오탐 플래그).
    스크리닝은 영속 레코드가 없으므로 검수 결과를 audit_logs 에 적재(조직 격리는 인증으로 보장).
    """
    _validate_review_note(payload)
    user, member, org = org_tuple

    reviewed_at = datetime.now(timezone.utc)

    # 스크리닝 정체성 + 검수 내용을 after 스냅샷으로 — 재학습 export 시 bbox/score 까지 식별 가능.
    after = {
        "review_status": payload.review_status,
        "reviewed_by_user_id": str(user.id),
        "reviewed_at": reviewed_at.isoformat(),
        "review_note": payload.review_note,
        "screening": {
            "filename": payload.filename,
            "video_timestamp_sec": payload.video_timestamp_sec,
            "frame_w": payload.frame_w,
            "frame_h": payload.frame_h,
            "bbox": payload.bbox.model_dump() if payload.bbox else None,
            "kind": payload.kind,
            "severity": payload.severity,
            "score": payload.score,
            "client_item_id": payload.client_item_id,
        },
    }

    await write_audit(
        db,
        action=_ACTION_MAP.get(payload.review_status, "thermal_screening.review"),
        resource_type="thermal_screening",
        resource_id=None,  # 영속 UUID 없음 — 정체성은 after.screening 에 보존
        user_id=user.id,
        organization_id=org.id,
        before=None,
        after=after,
        note=payload.review_note,
        request=request,
    )

    # WS broadcast — 같은 세션의 다른 화면도 검수 결과 즉시 반영(오버레이 마킹).
    await manager.broadcast("defects", {
        "type": "thermal.screening.reviewed",
        "data": {
            "client_item_id": payload.client_item_id,
            "video_timestamp_sec": payload.video_timestamp_sec,
            "review_status": payload.review_status,
            "reviewed_by_user_id": str(user.id),
            "reviewed_at": reviewed_at.isoformat(),
        },
    })

    return ThermalScreeningReviewResponse(
        ok=True,
        client_item_id=payload.client_item_id,
        review_status=payload.review_status,
        review_note=payload.review_note,
        reviewed_by_user_id=user.id,
        reviewed_at=reviewed_at,
    )
