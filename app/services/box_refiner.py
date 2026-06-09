# =============================================
# app/services/box_refiner.py
# 역할: VLM이 준 거친 bbox(또는 image_level)를 실제 하자 픽셀에 스냅(snap).
#       - VLM = "무엇이 / 대략 어디" (semantic prior)
#       - 고전 CV = ROI 안에서 실제 결함 픽셀을 찾아 박스를 조임 (pixel precision)
#       하자 종류별 단서:
#         · 균열/줄눈 : 어두운 선형 구조 (black-hat morphology)
#         · 녹물/철근 : 주황·갈색 색상 (HSV mask)
#         · 얼룩/오염 : 국부 배경 대비 색·밝기 이상
#         · 박리/spalling : 텍스처(거칠기) 이상 영역
#         · 기타 : 엣지 밀도
#
# 설계 원칙: 강한 단서가 없으면 원본 박스 유지(절대 더 나쁘게 만들지 않음).
# API 키 불필요 — 로컬 CV만 사용.
# =============================================

from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np

# 하자 class_name → 보정 전략
_CRACK = {
    "crack_structural", "crack_finishing", "vertical_horizontal_defect",
    "grout_defect", "frame_squareness_defect", "crack_indicator", "structural_damage",
}
_RUST = {"rebar_corrosion"}
_STAIN = {
    "paint_stain", "floor_stain", "pollution", "baseboard_damage",
    "waterproof_defect", "moisture_indicator", "thermal_anomaly_area",
}
_SPALL = {"finish_delamination", "wallpaper_bubble", "floor_lifting"}


def refine_box(
    image_bgr: np.ndarray,
    bbox_xyxy: List[float],
    class_name: str,
    *,
    expand: Optional[float] = None,
    min_area_frac: float = 0.0004,
) -> Tuple[Optional[List[float]], str]:
    """거친 박스를 실제 하자 픽셀에 스냅.

    원리: VLM 박스의 '중심'을 신뢰(대략 어디)하고, CV는 그 중심 근처에서
    실제 결함 픽셀의 정확한 '범위'를 찾는다.

    Returns:
        (refined_xyxy, method). 보정 실패 시 (None, "none") — 호출측이 원본 유지.
    """
    H, W = image_bgr.shape[:2]
    if not bbox_xyxy or len(bbox_xyxy) != 4:
        return None, "none"

    x1, y1, x2, y2 = bbox_xyxy
    # 색상(녹물)은 확장 작게 — 주변 엉뚱한 색 유입 방지. 형태 기반은 넉넉히.
    if expand is None:
        expand = 0.15 if class_name in _RUST else 0.18
    bw, bh = x2 - x1, y2 - y1
    ex, ey = bw * expand, bh * expand
    rx1 = int(max(0, x1 - ex)); ry1 = int(max(0, y1 - ey))
    rx2 = int(min(W, x2 + ex)); ry2 = int(min(H, y2 + ey))
    if rx2 - rx1 < 8 or ry2 - ry1 < 8:
        return None, "none"

    roi = image_bgr[ry1:ry2, rx1:rx2]
    # VLM 박스 중심을 ROI 로컬 좌표로 (CV 후보 선택의 prior)
    cx = (x1 + x2) / 2 - rx1
    cy = (y1 + y2) / 2 - ry1
    center = (cx, cy)
    # 후보 게이트 반경 — VLM 박스 크기 기준. 멀리 떨어진 blob 배제.
    max_dist = 0.6 * max(bw, bh)
    horiz_hint = bw >= bh  # VLM 박스 가로>세로 → 가로 균열

    if class_name in _CRACK:
        local, method = _refine_crack(roi, center, max_dist, horiz_hint)
    elif class_name in _RUST:
        local, method = _refine_color(roi, _rust_mask, center, max_dist)
        method = "rust:" + method
    elif class_name in _STAIN:
        local, method = _refine_stain(roi, center, max_dist)
    elif class_name in _SPALL:
        local, method = _refine_texture(roi, center, max_dist)
    else:
        local, method = _refine_edges(roi, center, max_dist)

    if local is None:
        return None, "none"

    lx1, ly1, lx2, ly2 = local
    # 최소 면적 가드 (너무 작은 노이즈면 무시)
    if (lx2 - lx1) * (ly2 - ly1) < min_area_frac * W * H:
        return None, "none"

    refined = [
        round(rx1 + lx1, 1), round(ry1 + ly1, 1),
        round(rx1 + lx2, 1), round(ry1 + ly2, 1),
    ]
    return refined, method


def _pick_near_center(cnts, center, roi_shape, max_dist, *, prefer="area", min_area=20):
    """후보 컨투어 중 VLM 중심에 가깝고 단서가 강한 것 선택.

    - max_dist 밖(중심에서 너무 먼) 후보는 배제 (VLM prior 신뢰).
    - ROI 테두리에 닿는 후보는 배경 유입 가능성 → 페널티.
    prefer: "area"(큰 것) | "long"(긴 것).
    """
    rh, rw = roi_shape[:2]
    cx, cy = center
    best, best_score = None, 0.0
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        bcx, bcy = x + w / 2, y + h / 2
        dist = ((bcx - cx) ** 2 + (bcy - cy) ** 2) ** 0.5
        if max_dist and dist > max_dist:
            continue  # VLM 중심에서 너무 멀면 배제
        norm = dist / (max(w, h) + 40)
        strength = (max(w, h) if prefer == "long" else area)
        score = strength / (1.0 + norm)
        # 테두리 접촉 = 배경 유입 의심 → 강한 페널티
        if x <= 1 or y <= 1 or x + w >= rw - 1 or y + h >= rh - 1:
            score *= 0.25
        if score > best_score:
            best_score, best = score, (x, y, x + w, y + h)
    return best


# ── 균열: 어두운 선형 구조 (방향성 연결 + 밴드 추출) ──
def _refine_crack(roi: np.ndarray, center, max_dist, horiz: bool) -> Tuple[Optional[List[float]], str]:
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    k = max(9, (min(h, w) // 12) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    blackhat = cv2.normalize(blackhat, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    thr = max(22, int(np.mean(blackhat) + 1.8 * np.std(blackhat)))
    _, mask = cv2.threshold(blackhat, thr, 255, cv2.THRESH_BINARY)

    # 방향은 VLM 박스 종횡비로 결정 (horiz 인자) — 투영보다 신뢰도 높음.
    # 균열 방향으로 길게 연결 (끊긴 세그먼트 이어붙임), 직교 방향은 얇게 유지.
    if horiz:
        ck = cv2.getStructuringElement(cv2.MORPH_RECT, (max(15, w // 6), 3))
    else:
        ck = cv2.getStructuringElement(cv2.MORPH_RECT, (3, max(15, h // 6)))
    connected = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, ck)

    cnts, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, "crack"
    box = _pick_near_center(cnts, center, roi.shape, max_dist, prefer="long",
                            min_area=max(15, 0.0008 * h * w))
    if box is None:
        return None, "crack"

    # 선택된 균열의 '밴드'를 따라 같은 선상의 세그먼트까지 확장 (전체 균열 회수)
    x1, y1, x2, y2 = box
    if horiz:
        band = mask[y1:y2, :]
        cols = np.where(band.sum(axis=0) > 0)[0]
        if cols.size:
            x1, x2 = int(cols.min()), int(cols.max())
    else:
        band = mask[:, x1:x2]
        rows = np.where(band.sum(axis=1) > 0)[0]
        if rows.size:
            y1, y2 = int(rows.min()), int(rows.max())

    pad = max(2, int(min(h, w) * 0.01))
    return [max(0, x1 - pad), max(0, y1 - pad),
            min(w, x2 + pad), min(h, y2 + pad)], f"crack:{'H' if horiz else 'V'}"


# ── 색상 기반 (녹물) ───────────────────────
def _rust_mask(roi: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    # 주황~갈색 (녹물): 채도 충분(엷은 회갈색 배경 배제), 명도 중간 이상
    lower = np.array([6, 70, 50]); upper = np.array([26, 255, 235])
    return cv2.inRange(hsv, lower, upper)


def _refine_color(roi: np.ndarray, mask_fn, center, max_dist) -> Tuple[Optional[List[float]], str]:
    mask = mask_fn(roi)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 색 특징(녹물)은 조각나기 쉬움 → 중심 근처·비테두리 blob들을 통합
    box = _union_near_center(cnts, center, roi.shape, max_dist, min_area=10)
    if box is None:
        return None, "color"
    return list(box), "color"


def _union_near_center(cnts, center, roi_shape, max_dist, *, min_area=10):
    """중심(VLM prior) 근처 컨투어들의 bbox 를 통합. 거리 게이트로 배경 배제."""
    cx, cy = center
    boxes = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if cv2.contourArea(c) < min_area:
            continue
        bcx, bcy = x + w / 2, y + h / 2
        if max_dist and ((bcx - cx) ** 2 + (bcy - cy) ** 2) ** 0.5 > max_dist:
            continue  # VLM 중심에서 너무 멀면 배경 → 제외
        boxes.append((x, y, x + w, y + h))
    if not boxes:
        return None
    return [min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes)]


# ── 얼룩/오염: 국부 배경 대비 이상 ─────────
def _refine_stain(roi: np.ndarray, center, max_dist) -> Tuple[Optional[List[float]], str]:
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=max(3, min(gray.shape) / 20))
    diff = cv2.absdiff(gray, blur)
    diff = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    thr = max(20, int(np.mean(diff) + 1.8 * np.std(diff)))
    _, mask = cv2.threshold(diff, thr, 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    box = _pick_near_center(cnts, center, roi.shape, max_dist, prefer="area", min_area=20)
    return (list(box), "stain") if box else (None, "stain")


# ── 박리/들뜸: 텍스처(거칠기) 이상 ─────────
def _refine_texture(roi: np.ndarray, center, max_dist) -> Tuple[Optional[List[float]], str]:
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY).astype(np.float32)
    mean = cv2.boxFilter(gray, -1, (9, 9))
    sq = cv2.boxFilter(gray * gray, -1, (9, 9))
    var = np.clip(sq - mean * mean, 0, None)
    std = cv2.normalize(np.sqrt(var), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    thr = max(30, int(np.mean(std) + 1.5 * np.std(std)))
    _, mask = cv2.threshold(std, thr, 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    box = _pick_near_center(cnts, center, roi.shape, max_dist, prefer="area", min_area=30)
    return (list(box), "texture") if box else (None, "texture")


# ── 기타: 엣지 밀도 ────────────────────────
def _refine_edges(roi: np.ndarray, center, max_dist) -> Tuple[Optional[List[float]], str]:
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    box = _pick_near_center(cnts, center, roi.shape, max_dist, prefer="area", min_area=20)
    return (list(box), "edges") if box else (None, "edges")


__all__ = ["refine_box"]
