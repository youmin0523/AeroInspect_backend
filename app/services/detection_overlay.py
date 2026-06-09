# =============================================
# app/services/detection_overlay.py
# 역할: 검출 결과를 원본 이미지에 bbox + 하자종류 + 등급으로 시각화
#       - 하이브리드/VLM 결과를 사람이 눈으로 검증할 수 있게 주석 이미지 생성
#       - 등급별 색상: CONFIRMED=빨강, REVIEW=주황, REFERENCE=회색
#       - 한글 라벨: Windows 맑은고딕(malgun.ttf), 없으면 ASCII 폴백
#       - localization=bbox → 실제 박스, image_level → 상단 라벨 목록
#
# 사용: /detect/hybrid/visualize 엔드포인트, training/eval/annotate_test_folder.py
# =============================================

from __future__ import annotations

import os
from typing import Any, List, Optional, Tuple

import cv2
import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
    _PIL_OK = True
except ImportError:
    _PIL_OK = False

# 등급별 색상 (R, G, B)
_GRADE_COLOR = {
    "CONFIRMED": (220, 40, 40),    # 빨강 — 보고서 등재
    "REVIEW": (240, 150, 20),      # 주황 — 점검자 확인
    "REFERENCE": (150, 150, 150),  # 회색 — 참고
    "DROP": (110, 110, 110),
}
_STATUS_TAG = {
    "confirmed_by_both": "합의",
    "reclassified": "교정",
    "onnx_only": "ONNX",
    "vlm_only": "VLM",
    "rejected": "기각",
}

_FONT_CANDIDATES = [
    r"C:\Windows\Fonts\malgun.ttf",      # 맑은 고딕 (한글)
    r"C:\Windows\Fonts\malgunbd.ttf",
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
]


def _load_font(size: int):
    if not _PIL_OK:
        return None
    for path in _FONT_CANDIDATES:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    try:
        return ImageFont.load_default()
    except Exception:
        return None


class _OverlayItem:
    __slots__ = ("bbox", "located", "label", "color")

    def __init__(self, bbox, located, label, color):
        self.bbox = bbox
        self.located = located
        self.label = label
        self.color = color


def _draw(image_bgr: np.ndarray, items: List[_OverlayItem]) -> np.ndarray:
    """공통 드로잉 — bbox 있는 건 박스, image_level 은 상단 목록."""
    if not _PIL_OK:
        return _draw_cv2_fallback(image_bgr, items)

    img = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    H, W = image_bgr.shape[:2]
    fsize = max(14, int(H * 0.018))
    font = _load_font(fsize)
    lw = max(2, int(H * 0.003))

    # image_level(전체프레임) 라벨은 상단에 쌓아서 표시
    top_y = 6
    for it in items:
        if it.located and it.bbox and len(it.bbox) == 4:
            x1, y1, x2, y2 = [int(v) for v in it.bbox]
            draw.rectangle([x1, y1, x2, y2], outline=it.color, width=lw)
            _label(draw, it.label, x1, y1, it.color, font, W)
        else:
            # 위치 미상 — 상단에 누적
            _label(draw, "▣ " + it.label, 6, top_y, it.color, font, W, anchor_top=True)
            top_y += fsize + 8

    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def _label(draw, text, x, y, color, font, W, anchor_top: bool = False):
    """라벨 배경 박스 + 텍스트."""
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    except Exception:
        tw, th = len(text) * 8, 16
    pad = 3
    ty = y - th - 2 * pad
    if ty < 0 or anchor_top:
        ty = y
    tx = min(x, W - tw - 2 * pad)
    tx = max(0, tx)
    draw.rectangle([tx, ty, tx + tw + 2 * pad, ty + th + 2 * pad], fill=color)
    draw.text((tx + pad, ty + pad), text, fill=(255, 255, 255), font=font)


def _draw_cv2_fallback(image_bgr: np.ndarray, items: List[_OverlayItem]) -> np.ndarray:
    """PIL 없을 때 — ASCII 라벨만(cv2.putText는 한글 불가)."""
    img = image_bgr.copy()
    for it in items:
        col_bgr = (it.color[2], it.color[1], it.color[0])
        if it.located and it.bbox and len(it.bbox) == 4:
            x1, y1, x2, y2 = [int(v) for v in it.bbox]
            cv2.rectangle(img, (x1, y1), (x2, y2), col_bgr, 2)
    return img


# ── 공개 API ──────────────────────────────
def annotate_hybrid(image_bgr: np.ndarray, result: Any, *, show_rejected: bool = True) -> np.ndarray:
    """HybridDetectionResult → 주석 이미지 (bbox + 종류 + 등급)."""
    items: List[_OverlayItem] = []
    for d in result.detections:
        if d.status == "rejected" and not show_rejected:
            continue
        color = _GRADE_COLOR.get(d.grade, (110, 110, 110))
        tag = _STATUS_TAG.get(d.status, "")
        conf_pct = f"{d.conf:.0%}" if d.conf is not None else ""
        label = f"{d.code} {d.class_display_ko or d.class_} [{tag}|{d.grade} {conf_pct}]"
        located = d.localization == "bbox" and bool(d.bbox_xyxy)
        items.append(_OverlayItem(d.bbox_xyxy, located, label, color))
    return _draw(image_bgr, items)


def annotate_vlm(image_bgr: np.ndarray, result: Any) -> np.ndarray:
    """VLMDetectionResult → 주석 이미지."""
    items: List[_OverlayItem] = []
    for d in result.detections:
        sev = d.severity or ""
        color = (220, 40, 40) if sev == "HIGH" else (240, 150, 20) if sev == "MED" else (150, 150, 150)
        label = f"{d.code} {d.class_display_ko or d.class_} ({d.conf:.0%})"
        located = d.localization == "bbox" and bool(d.bbox_xyxy)
        items.append(_OverlayItem(d.bbox_xyxy, located, label, color))
    return _draw(image_bgr, items)


def encode_jpeg(image_bgr: np.ndarray, quality: int = 90) -> bytes:
    ok, buf = cv2.imencode(".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("JPEG 인코딩 실패")
    return buf.tobytes()
