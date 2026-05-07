# =============================================
# app/services/defect_taxonomy.py
# 역할: 신규 3-모델 클래스 ↔ 표시명 ↔ 기존 A-E taxonomy 매핑 테이블
#       - WALLPAPER_CLASSES: ResNet50 체크포인트의 정확한 19 클래스 순서
#       - CLASS_DISPLAY_MAP: 내부명 → (영문 표시명, 한글 표시명)
#       - YOLO_DISPLAY_MAP: YOLO 3 클래스 표시명
#       - LEGACY_MAP_THERMAL / LEGACY_MAP_WALLPAPER: 신규 → 기존 A-E taxonomy
#       - xyxy_to_xywhn: API용 xyxy 좌표를 DB 저장용 xywhn로 변환
#
# ⚠️ 매우 중요:
#   벽지 클래스 "good"은 실제로 "터짐(Burst)" 하자 클래스임.
#   "정상"으로 취급하면 안 되며, severity는 MED로 격상 처리한다.
# =============================================

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

# ── ResNet50 체크포인트에 baked-in된 19 클래스 (ImageFolder 알파벳 순) ──
# 이 순서 절대 바꾸지 말 것 — 체크포인트 class_names와 assert 검증된다.
WALLPAPER_CLASSES: List[str] = [
    "Baseboard", "Crying", "Damage", "Defective_Joint", "Exploded",
    "Furniture", "Gypsum", "Kink", "Many_niches", "Mold",
    "Molding", "Piece", "Plane", "Pollution", "Rust",
    "Spot", "W.F_D.F", "Wrong_punch", "good",
]


# ── 내부명(weights) → (영문 표시명, 한글 표시명) ──
# 프론트/DB에는 반드시 이 매핑을 거친 표시명을 전달한다.
CLASS_DISPLAY_MAP: Dict[str, Tuple[str, str]] = {
    "Baseboard":       ("Baseboard",          "걸레받이"),
    "Crying":          ("Crying",             "울음"),
    "Damage":          ("Damage",             "훼손"),
    "Defective_Joint": ("Defective Joint",    "이음부 불량"),
    "Exploded":        ("Exploded",           "들뜸"),
    "Furniture":       ("Furniture",          "가구 수정"),
    "Gypsum":          ("Gypsum",             "석고"),
    "Kink":            ("Kink",               "꼬임"),
    "Many_niches":     ("Many niches",        "틈새 과다"),
    "Mold":            ("Mold",               "곰팡이"),
    "Molding":         ("Molding",            "몰딩"),
    "Piece":           ("Piece",              "조각"),
    "Plane":           ("Plane",              "면 불량"),
    "Pollution":       ("Pollution",          "오염"),
    "Rust":            ("Rust",               "녹 오염"),
    "Spot":            ("Spot",               "반점"),
    "W.F_D.F":         ("Window/Door Frame",  "창틀/문틀"),
    "Wrong_punch":     ("Wrong punch",        "오타공"),
    "good":            ("Burst",              "터짐"),  # ⚠️ 실제 의미는 '터짐'
}

YOLO_DISPLAY_MAP: Dict[str, Tuple[str, str]] = {
    "Crack":        ("Crack",        "균열"),
    "Moisture":     ("Moisture",     "습기"),
    "delamination": ("Delamination", "박리"),
}


def get_display_names(class_name: str) -> Tuple[str, str]:
    """내부명 → (영문, 한글) 표시명. 매핑 없으면 내부명을 그대로 반환."""
    if class_name in CLASS_DISPLAY_MAP:
        return CLASS_DISPLAY_MAP[class_name]
    if class_name in YOLO_DISPLAY_MAP:
        return YOLO_DISPLAY_MAP[class_name]
    return (class_name, class_name)


# ── severity 격상 대상 벽지 클래스 ──
# top1이 이 집합에 속하고 is_confident=True이면 severity=MED
# (구조적/방수적 영향이 큰 유형 — good(=터짐) 포함)
WALLPAPER_SEVERE_CLASSES = frozenset({
    "Mold", "Damage", "Exploded", "Defective_Joint", "good",
})


# =============================================
# 레거시 A-E taxonomy 매핑
# =============================================
# 기존 [app/utils/severity_mapper.py](severity_mapper.py)의 20-코드 체계와
# 병존시키기 위한 매핑. DefectLog의 area/category_code/defect_type 컬럼은
# 프론트 보고서/통계에서 계속 사용되므로 신규 모델 결과도 가능하면 채운다.
#
# 매핑 없는 클래스는 (None, None, class_display_ko)로 폴백하고,
# DB 컬럼 area/category_code/defect_type은 nullable=True로 완화한다.

# YOLO thermal (crack_moisture) + YOLO delam → 기존 taxonomy
LEGACY_MAP_THERMAL: Dict[str, Tuple[str, str, str]] = {
    # 내부명: (area, category_code, defect_type)
    "Crack":        ("A", "A-02", "균열 (구조 균열)"),
    "Moisture":     ("B", "B-04", "방수층 들뜸 / 누수 흔적"),
    "delamination": ("B", "B-02", "벽체 단열 공백·탈락"),
}

# Wallpaper ResNet → 기존 taxonomy (대부분 C 영역: 마감재/표면)
# 매핑 불가 / 데이터셋 특유 레이블은 None으로 두고 프론트에선 display_ko로 표기
LEGACY_MAP_WALLPAPER: Dict[str, Optional[Tuple[str, str, str]]] = {
    # 명확히 기존 코드로 매핑되는 클래스
    "Defective_Joint": ("C", "C-01", "도배 이음매 불량"),
    "Exploded":        ("C", "C-02", "도배지 기포·들뜸"),
    "Pollution":       ("C", "C-03", "도색 얼룩·붓자국"),
    "Damage":          ("C", "C-04", "찍힘·스크래치 (벽·천장)"),
    "Baseboard":       ("C", "C-05", "걸레받이 오염·파손"),
    "W.F_D.F":         ("E", "E-02", "창틀·문틀 도장 불량"),
    # good(=터짐)은 도배지 파손으로 C-04 (찍힘·스크래치)에 편입
    "good":            ("C", "C-04", "도배지 터짐"),
    # Mold는 방수/단열 문제 — B-04 (방수층/누수)에 편입
    "Mold":            ("B", "B-04", "곰팡이 (누수 의심)"),

    # 매핑 없음 — area=None 폴백 (defect_type은 한글 표시명 사용)
    "Crying":          None,
    "Furniture":       None,
    "Gypsum":          None,
    "Kink":            None,
    "Many_niches":     None,
    "Molding":         None,
    "Piece":           None,
    "Plane":           None,
    "Rust":            None,
    "Spot":            None,
    "Wrong_punch":     None,
}


def map_to_legacy(
    source: str,
    class_name: str,
) -> Tuple[Optional[str], Optional[str], str]:
    """
    신규 모델 클래스를 기존 A-E taxonomy로 매핑.

    Args:
        source: 'yolo_thermal' | 'yolo_delam' | 'wallpaper'
        class_name: 모델 내부명 (예: 'Crack', 'good')

    Returns:
        (area, category_code, defect_type):
            매핑되면 (A-E, A-01~E-02, 한글명) 3-튜플.
            매핑 없으면 (None, None, class_display_ko) — defect_type은 한글 표시명.
    """
    _, display_ko = get_display_names(class_name)

    if source in ("yolo_thermal", "yolo_delam"):
        mapping = LEGACY_MAP_THERMAL.get(class_name)
        if mapping:
            return mapping
    elif source == "wallpaper":
        mapping = LEGACY_MAP_WALLPAPER.get(class_name)
        if mapping:
            return mapping

    return (None, None, display_ko)


# =============================================
# 좌표 변환 유틸
# =============================================

# =============================================
# 20종 하자 파이프라인 매핑 (신규)
# =============================================
# severity_mapper.py의 DEFECT_CATALOG와 1:1 매핑.
# 각 모델 내부 class_name → (code, display_ko, severity, area)

DEFECT_20_MAP: Dict[str, Tuple[str, str, str, str]] = {
    # (class_name): (code, display_ko, severity, area)
    # M1: 구조·방수
    "crack_structural":       ("A-02", "균열 (구조 균열)",       "HIGH", "A"),
    "crack_finishing":        ("A-03", "균열 (마감 균열)",        "MED",  "A"),
    "caulking_defect":        ("B-03", "코킹 누락·불량",         "HIGH", "B"),
    "waterproof_defect":      ("B-04", "방수층 들뜸 / 누수 흔적", "HIGH", "B"),
    # M2: 마감·표면
    "wallpaper_seam":         ("C-01", "도배 이음매 불량",        "MED",  "C"),
    "wallpaper_bubble":       ("C-02", "도배지 기포·들뜸",        "MED",  "C"),
    "paint_stain":            ("C-03", "도색 얼룩·붓자국",        "LOW",  "C"),
    "scratch":                ("C-04", "찍힘·스크래치 (벽·천장)",  "LOW",  "C"),
    "baseboard_damage":       ("C-05", "걸레받이 오염·파손",      "LOW",  "C"),
    # M3: 바닥·창호
    "floor_stain":            ("D-03", "바닥 오염·스크래치",      "LOW",  "D"),
    "grout_defect":           ("D-04", "줄눈 불량 (타일·마루)",   "LOW",  "D"),
    "glass_scratch":          ("E-01", "창호 유리 스크래치·파손",  "MED",  "E"),
    "frame_paint_defect":     ("E-02", "창틀·문틀 도장 불량",     "LOW",  "E"),
    # M4: 열화상
    "window_insulation_defect": ("B-01", "창호 단열 불량 (결로·냉교)", "HIGH", "B"),
    "wall_insulation_gap":      ("B-02", "벽체 단열 공백·탈락",      "HIGH", "B"),
    "window_airtight_defect":   ("B-05", "창호 기밀 불량 (틈새)",     "MED",  "B"),
    "floor_heating_defect":     ("D-01", "바닥 난방 불량 (온도 편차)", "HIGH", "D"),
    # M5+G1: 기하학
    "vertical_horizontal_defect": ("A-01", "벽·천장 수직·수평도 불량", "HIGH", "A"),
    "frame_squareness_defect":    ("A-04", "문·창호 틀 직각도 불량",   "MED",  "A"),
    # D-02: 바닥재 들뜸 (열화상 간접 / 시각적)
    "floor_lifting":          ("D-02", "바닥재 들뜸 (공명 감지)",  "MED",  "D"),

    # ── ResNet/YOLO 원본 출력 raw 클래스 매핑 ────────────────
    # 학습 모델이 위 정식 코드를 직접 출력하지 않고 sub-class 라벨을 내므로
    # 등록 누락 시 get_20defect_info 폴백("X-00", raw_name, ...) → 화면에
    # "X-00 caulking_indicator" 같은 영문 raw 라벨이 노출되는 사고 방지.

    # M1-ResNet 출력 (5-class sub-분류, ImageFolder 알파벳 순)
    "caulking_indicator":   ("B-03", "코킹 누락·불량",         "HIGH", "B"),
    "crack_indicator":      ("A-03", "균열 (마감 균열)",        "MED",  "A"),
    "moisture_indicator":   ("B-04", "방수층 들뜸 / 누수 흔적", "HIGH", "B"),
    "structural_damage":    ("A-02", "균열 (구조 균열)",       "HIGH", "A"),
    # M2-ResNet/YOLO 통합 출력 — sub 분류 정보 없음 → 일반 표면 결함
    "surface_defect":       ("C-04", "벽 표면 결함",            "MED",  "C"),
    "surface_defect_wall":  ("C-04", "벽 표면 결함",            "MED",  "C"),
    # M3-YOLO/ResNet 출력 (sub 분류 정보 없음 → 카테고리 default)
    "floor_defect":         ("D-03", "바닥 오염·스크래치",      "LOW",  "D"),
    "glass_defect":         ("E-01", "창호 유리 스크래치·파손",  "MED",  "E"),
    "frame_defect":         ("E-02", "창틀·문틀 도장 불량",     "LOW",  "E"),
}


def get_20defect_info(class_name: str) -> Tuple[str, str, str, str]:
    """
    20종 class_name → (code, display_ko, severity, area).
    매핑 없으면 ("X-00", class_name, "MED", "A") 폴백.
    """
    if class_name in DEFECT_20_MAP:
        return DEFECT_20_MAP[class_name]
    return ("X-00", class_name, "MED", "A")


def xyxy_to_xywhn(
    xyxy: List[float],
    img_w: int,
    img_h: int,
) -> Tuple[float, float, float, float]:
    """
    픽셀 xyxy → 정규화 xywhn (YOLO 포맷).
    DB의 기존 bbox_x/y/w/h 컬럼(xywhn) 저장용.

    Args:
        xyxy: [x1, y1, x2, y2] 픽셀 좌표
        img_w: 원본 이미지 너비 (픽셀)
        img_h: 원본 이미지 높이 (픽셀)

    Returns:
        (cx, cy, w, h) 정규화 (0.0~1.0). 이미지 크기 0이면 (0,0,0,0).
    """
    if img_w <= 0 or img_h <= 0:
        return (0.0, 0.0, 0.0, 0.0)

    x1, y1, x2, y2 = xyxy
    cx = (x1 + x2) / 2.0 / img_w
    cy = (y1 + y2) / 2.0 / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h

    def clip(v: float) -> float:
        return max(0.0, min(1.0, v))

    return (clip(cx), clip(cy), clip(w), clip(h))
