# =============================================
# app/services/confidence_grader.py
# 신뢰도 3단계 등급 분류 — Precision↔Recall 균형
#
# 등급:
#   CONFIRMED  : 하자목록 등재(보고서). 분쟁 시 책임질 수준.
#   REVIEW     : 점검자 추가 확인 권장. 하자목록 X, 별도 섹션.
#   REFERENCE  : 참고용. 점검자 모드 토글 시만 노출.
#   DROP       : 표시 X (REFERENCE 미달).
#
# 규칙 (20종 클래스 통일 — 단열 특례 X):
#   1. CONFIRMED:
#      - conf >= 0.85 (단일 모델 매우 강함)
#      - OR (conf >= 0.7 AND voting 통과)
#      - 단, M6 PatchCore 단독은 CONFIRMED 불가 (anomaly만으로 단정 X)
#   2. REVIEW: 0.40 <= conf < 0.70 (단일 모델)
#   3. REFERENCE: 0.20 <= conf < 0.40
#   4. DROP: conf < 0.20
#
# voting 통과 마커:
#   - cross_model_boosted = True (다른 defect_source 동의)
#   - ensemble_boosted = True (PatchCore anomaly 동의)
# =============================================

from __future__ import annotations

from typing import Literal


Grade = Literal["CONFIRMED", "REVIEW", "REFERENCE", "DROP"]


# 2026-05-30 GT 검증 결과 반영: Precision 0.535 (목표 0.90 미달) — 임계 상향
# M2/M3가 cross-domain (crack 이미지)에서 자기 클래스로 잘못 탐지 → CONFIRMED 임계 0.85→0.90
# voting 동반 임계도 0.70→0.75로 +0.05 상향
CONFIRMED_STRONG = 0.90        # 기존 0.85 → 0.90 (단일 모델 강검출 임계)
CONFIRMED_WITH_VOTING = 0.75   # 기존 0.70 → 0.75 (voting 동반 시 임계)
REVIEW_THRESHOLD = 0.40
REFERENCE_THRESHOLD = 0.20

# PatchCore anomaly 단독은 CONFIRMED 불가 — 라벨 없는 비지도 신호라
# 단일 신호로 분쟁 책임 불가. 다른 모델과 동의해야 등급 상승 허용.
#
# 2026-05-30 시도: M2/M3 voting 필수 적용 → Recall 0.748 → 0.195 폭락. 즉시 revert.
# voting은 cross-domain 검증 도구가 아님. M2/M3 도메인 mismatch는 conf threshold
# 또는 학습 단계에서 해결해야 함.
PATCHCORE_ONLY_SOURCES = {"patchcore", "thermal_anomaly"}


GRADE_DISPLAY_KO = {
    "CONFIRMED": "확정",
    "REVIEW": "권장점검",
    "REFERENCE": "참고용",
    "DROP": "표시안함",
}


def grade_detection(det: dict) -> Grade:
    """검출 단건의 등급을 산정."""
    conf = float(det.get("conf", 0.0))
    source = det.get("defect_source", "")
    voted = bool(
        det.get("cross_model_boosted", False)
        or det.get("ensemble_boosted", False)
    )

    if conf < REFERENCE_THRESHOLD:
        return "DROP"

    if conf < REVIEW_THRESHOLD:
        return "REFERENCE"

    if conf < CONFIRMED_WITH_VOTING:
        return "REVIEW"

    # conf >= 0.75 — CONFIRMED 후보
    # PatchCore/anomaly 단독은 voting 없으면 REVIEW로 강등
    if source in PATCHCORE_ONLY_SOURCES and not voted:
        return "REVIEW"

    if conf >= CONFIRMED_STRONG:
        return "CONFIRMED"

    if voted:
        return "CONFIRMED"

    # 0.75 <= conf < 0.90, voting X — 단일 모델 강검출이지만 voting 없음 → REVIEW
    return "REVIEW"


def grade_display_ko(grade: Grade) -> str:
    return GRADE_DISPLAY_KO.get(grade, "")


def is_listable(grade: Grade) -> bool:
    """보고서 하자목록 등재 여부 — CONFIRMED만 True."""
    return grade == "CONFIRMED"


def is_inspector_visible(grade: Grade) -> bool:
    """점검자 모드 노출 여부 — CONFIRMED + REVIEW."""
    return grade in ("CONFIRMED", "REVIEW")


__all__ = [
    "Grade",
    "grade_detection",
    "grade_display_ko",
    "is_listable",
    "is_inspector_visible",
    "GRADE_DISPLAY_KO",
    "CONFIRMED_STRONG",
    "CONFIRMED_WITH_VOTING",
    "REVIEW_THRESHOLD",
    "REFERENCE_THRESHOLD",
]
