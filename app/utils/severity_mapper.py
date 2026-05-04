# =============================================
# app/utils/severity_mapper.py
# 역할: 하자 카테고리 코드 ↔ 심각도/상세정보 매핑 테이블
#       - 20종 하자 항목을 5개 영역(A-E)으로 분류
#       - YOLOv8 클래스 ID → 카테고리 코드 매핑
#       - severity_mapper: 코드 → {name, area, severity} 딕셔너리
#
# 계획서 섹션 9 기준 20종 하자 항목:
#   A. 구조·기하학 (4종): A-01~A-04
#   B. 단열·방수·기밀 (5종): B-01~B-05
#   C. 마감재·표면 (5종): C-01~C-05
#   D. 바닥 (4종): D-01~D-04
#   E. 창호·문 외관 (2종): E-01~E-02
# =============================================

from typing import Dict

# ── 하자 카테고리 전체 정의 ───────────────────
DEFECT_CATALOG: Dict[str, Dict] = {
    # A. 구조·기하학 영역
    "A-01": {"name": "벽·천장 수직·수평도 불량", "area": "A", "severity": "HIGH", "class_name": "vertical_horizontal_defect"},
    "A-02": {"name": "균열 (구조 균열)",          "area": "A", "severity": "HIGH", "class_name": "crack_structural"},
    "A-03": {"name": "균열 (마감 균열)",           "area": "A", "severity": "MED",  "class_name": "crack_finishing"},
    "A-04": {"name": "문·창호 틀 직각도 불량",    "area": "A", "severity": "MED",  "class_name": "frame_squareness_defect"},

    # B. 단열·방수·기밀 영역
    "B-01": {"name": "창호 단열 불량 (결로·냉교)", "area": "B", "severity": "HIGH", "class_name": "window_insulation_defect"},
    "B-02": {"name": "벽체 단열 공백·탈락",        "area": "B", "severity": "HIGH", "class_name": "wall_insulation_gap"},
    "B-03": {"name": "코킹 누락·불량",             "area": "B", "severity": "HIGH", "class_name": "caulking_defect"},
    "B-04": {"name": "방수층 들뜸 / 누수 흔적",   "area": "B", "severity": "HIGH", "class_name": "waterproof_defect"},
    "B-05": {"name": "창호 기밀 불량 (틈새)",      "area": "B", "severity": "MED",  "class_name": "window_airtight_defect"},

    # C. 마감재·표면 영역
    "C-01": {"name": "도배 이음매 불량",           "area": "C", "severity": "MED",  "class_name": "wallpaper_seam_defect"},
    "C-02": {"name": "도배지 기포·들뜸",           "area": "C", "severity": "MED",  "class_name": "wallpaper_bubble"},
    "C-03": {"name": "도색 얼룩·붓자국",           "area": "C", "severity": "LOW",  "class_name": "paint_stain"},
    "C-04": {"name": "찍힘·스크래치 (벽·천장)",   "area": "C", "severity": "LOW",  "class_name": "scratch_wall"},
    "C-05": {"name": "걸레받이 오염·파손",         "area": "C", "severity": "LOW",  "class_name": "baseboard_damage"},

    # D. 바닥 영역
    "D-01": {"name": "바닥 난방 불량 (온도 편차)", "area": "D", "severity": "HIGH", "class_name": "floor_heating_defect"},
    "D-02": {"name": "바닥재 들뜸 (공명 감지)",   "area": "D", "severity": "MED",  "class_name": "floor_lifting"},
    "D-03": {"name": "바닥 오염·스크래치",         "area": "D", "severity": "LOW",  "class_name": "floor_stain"},
    "D-04": {"name": "줄눈 불량 (타일·마루)",     "area": "D", "severity": "LOW",  "class_name": "grout_defect"},

    # E. 창호·문 외관 영역
    "E-01": {"name": "창호 유리 스크래치·파손",   "area": "E", "severity": "MED",  "class_name": "glass_scratch"},
    "E-02": {"name": "창틀·문틀 도장 불량",       "area": "E", "severity": "LOW",  "class_name": "frame_paint_defect"},
}

# ── class_name → category_code 역방향 매핑 ──
_CLASS_NAME_TO_CODE: Dict[str, str] = {
    v["class_name"]: k for k, v in DEFECT_CATALOG.items()
}

# ── YOLOv8 클래스 ID (0-based) → class_name 매핑 ──
# 학습 시 클래스 순서와 일치해야 함
DEFECT_CLASS_NAMES: Dict[int, str] = {
    idx: info["class_name"]
    for idx, (code, info) in enumerate(DEFECT_CATALOG.items())
}

# ── class_id → code 직접 매핑 ────────────────
DEFECT_CLASS_ID_TO_CODE: Dict[int, str] = {
    idx: code
    for idx, code in enumerate(DEFECT_CATALOG.keys())
}


def get_severity_by_code(class_name_or_code: str) -> Dict:
    """
    class_name 또는 category_code로 하자 정보 조회.

    Args:
        class_name_or_code: "crack_structural" 또는 "A-02"

    Returns:
        {"code": "A-02", "name": "...", "area": "A", "severity": "HIGH"}
    """
    # category_code 직접 조회
    if class_name_or_code in DEFECT_CATALOG:
        info = DEFECT_CATALOG[class_name_or_code]
        return {"code": class_name_or_code, **info}

    # class_name → code 역방향 조회
    code = _CLASS_NAME_TO_CODE.get(class_name_or_code)
    if code:
        info = DEFECT_CATALOG[code]
        return {"code": code, **info}

    # 알 수 없는 클래스
    return {
        "code": "X-00",
        "name": class_name_or_code,
        "area": "A",
        "severity": "MED",
        "class_name": class_name_or_code,
    }


def get_all_by_area(area: str) -> Dict[str, Dict]:
    """특정 영역(A-E)의 모든 하자 항목 반환"""
    return {
        code: info
        for code, info in DEFECT_CATALOG.items()
        if info["area"] == area.upper()
    }


def get_all_by_severity(severity: str) -> Dict[str, Dict]:
    """특정 심각도(HIGH/MED/LOW)의 모든 하자 항목 반환"""
    return {
        code: info
        for code, info in DEFECT_CATALOG.items()
        if info["severity"] == severity.upper()
    }
