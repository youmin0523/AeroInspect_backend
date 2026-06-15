# =============================================
# app/services/thermal_pseudo.py
# 역할: 의사색(FLIR iron) 열화상에서 '단열 의심부'를 라디오메트릭 데이터 없이 스크리닝.
#   - 절대온도(°C) 없이 프레임 내 상대 온도지도(=의사색)만으로 국소 cold 이상을 검출.
#   - 점형 열교/결로(dot)·프레임 틈(patch)을 잡고, 정상 발열체(창·조명)/구조선은 억제.
#   ⚠️ '스크리닝 보조'용 — 절대 ΔT 정량/확정 진단이 아니며, 컬러바 부재·압축 화질 한계가 있다.
#      검출은 Drone2 오버레이로만 노출하고 보고서 DB 에는 자동 적재하지 않는다(점검자 수동 채택).
#
# 알고리즘(프로토타입 v4 검증본):
#   (1) iron 팔레트 LUT 역매핑 → 0-1 상대 온도 스칼라 T
#   (2) 멀티스케일 DoG 점검출 (거리에 따라 점 크기 가변 대응)
#   (3) 영역 국소대비 냉각패치 (코너 열교/넓은 결로)
#   (4) 고온원 마진 억제 + 테두리/선형 blob 필터
# =============================================
from __future__ import annotations

from typing import List, Dict

import cv2
import numpy as np

# ── iron 팔레트 제어색 (temp fraction → BGR) — 보라→자홍→주황→노랑→백 근사 ──
_IRON_CTRL = [
    (0.00, (40, 0, 20)),    (0.18, (130, 0, 70)),
    (0.38, (150, 0, 160)),  (0.55, (40, 30, 220)),
    (0.72, (0, 120, 255)),  (0.88, (60, 220, 255)),
    (1.00, (255, 255, 255)),
]


def _build_iron_lut() -> np.ndarray:
    fr = np.array([c[0] for c in _IRON_CTRL])
    cols = np.array([c[1] for c in _IRON_CTRL], dtype=np.float32)  # BGR
    xs = np.linspace(0, 1, 256)
    lut = np.stack([np.interp(xs, fr, cols[:, ch]) for ch in range(3)], axis=1)
    return lut.astype(np.float32)   # (256,3) BGR


_IRON_LUT = _build_iron_lut()


def _palette_to_temp(bgr: np.ndarray) -> np.ndarray:
    """각 픽셀을 iron LUT 256색 중 최근접 인덱스로 역매핑 → 0-1 상대온도."""
    h, w = bgr.shape[:2]
    flat = bgr.reshape(-1, 3).astype(np.float32)
    d = np.linalg.norm(flat[:, None, :] - _IRON_LUT[None, :, :], axis=2)
    idx = np.argmin(d, axis=1).astype(np.float32) / 255.0
    return idx.reshape(h, w)


def _hot_margin_mask(T: np.ndarray, short: int, hot_pct: float = 98.0,
                     grow_frac: float = 0.03) -> np.ndarray:
    """정상 발열체(창·조명=초고온) + 주변 마진 — 고온원 경계 오탐 억제."""
    hot = (T >= np.percentile(T, hot_pct)).astype(np.uint8)
    g = int(max(3, short * grow_frac)) | 1
    return cv2.dilate(hot, np.ones((g, g), np.uint8)) > 0


def _severity(sev: float, kind: str) -> str:
    """상대 cold 강도(sev)를 표시용 등급으로. (보고서 적재 아님 — 색/우선순위용)"""
    hi = 8 if kind == "spot" else 14
    mid = 5 if kind == "spot" else 9
    if sev >= hi:
        return "HIGH"
    if sev >= mid:
        return "MED"
    return "LOW"


def _cold_map(T: np.ndarray, sh: int, k_frac: float, neutralize_pct: float | None = None) -> np.ndarray:
    """국소 배경(blur) 대비 'cold(주변보다 차가움)' 맵. neutralize_pct 지정 시 그 백분위 이상
    밝은 픽셀을 중앙값으로 치환 후 배경을 구해 hot 구조의 halo 거짓-cold 를 억제."""
    T_bg = T
    if neutralize_pct is not None:
        bright = T >= np.percentile(T, neutralize_pct)
        if bright.any() and (~bright).any():
            T_bg = T.copy()
            T_bg[bright] = float(np.median(T[~bright]))
    k = int(sh * k_frac) | 1
    return np.clip(cv2.GaussianBlur(T_bg, (k, k), 0) - T, 0, 1)


def _detect_cold_bands(T: np.ndarray, hot_m: np.ndarray, sh: int, h: int, w: int) -> List[Dict]:
    """방향성 모폴로지로 '두꺼운 cold 띠'(수직/수평 = 코너 열교)를 검출.
    얇은 그리드선은 두께 open 으로, 끊긴 조각은 방향 close 로 이어 붙인 뒤 길이로 확인."""
    cold = _cold_map(T, sh, k_frac=0.30, neutralize_pct=94.0)
    base = ((cold >= 0.04) & (~hot_m)).astype(np.uint8)
    wmin = max(3, int(sh * 0.022))                      # 최소 띠 두께 — 1~2px 그리드선 배제
    base = cv2.morphologyEx(base, cv2.MORPH_OPEN, np.ones((wmin, wmin), np.uint8))
    gap = max(8, int(sh * 0.20))                        # 조각 사이 brige 길이
    vbridge = cv2.morphologyEx(base, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (1, gap)))
    hbridge = cv2.morphologyEx(base, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (gap, 1)))
    bands = cv2.bitwise_or(vbridge, hbridge)
    out: List[Dict] = []
    n, lbl, st, _ = cv2.connectedComponentsWithStats(bands, 8)
    for i in range(1, n):
        x, y, ww, hh, area = st[i]
        length = max(ww, hh)
        thick = min(ww, hh)
        if length < sh * 0.24:                          # 충분히 긴 띠만
            continue
        if length / (thick + 1e-6) < 2.6:               # 확실히 가늘고 긴 띠
            continue
        if area < 0.0025 * h * w:
            continue
        sev = float(cold[lbl == i].mean()) * 100
        if sev < 5.0:
            continue
        out.append({"kind": "patch",
                    "bbox": {"x1": int(x), "y1": int(y), "x2": int(x + ww), "y2": int(y + hh)},
                    "severity": _severity(sev, "patch"), "score": round(sev, 1)})
    return out


def detect_pseudocolor_anomalies(bgr: np.ndarray, max_results: int = 20) -> List[Dict]:
    """의사색 열화상 1프레임에서 단열 의심 영역을 반환.
    각 원소: {kind:'spot'|'patch', bbox:{x1,y1,x2,y2}, severity, score}
    bbox 좌표는 입력 프레임 픽셀 기준(원본 해상도)."""
    if bgr is None or bgr.ndim != 3:
        return []
    h, w = bgr.shape[:2]
    # 매핑 비용 절감: 256px 폭으로 다운샘플해 T 계산 후 업스케일
    small = cv2.resize(bgr, (256, max(1, int(h * 256.0 / w))))
    Ts = cv2.GaussianBlur(_palette_to_temp(small), (0, 0), 1.0)
    T = cv2.resize(Ts, (w, h))
    sh = min(T.shape[:2])

    hot_m = _hot_margin_mask(T, sh)
    results: List[Dict] = []

    # (2) 멀티스케일 점검출 — 작은 스케일만(큰 냉각영역은 patch 담당)
    for sf in (0.020, 0.032, 0.045):
        s = max(2.0, sh * sf)
        dog = cv2.GaussianBlur(T, (0, 0), s * 1.6) - cv2.GaussianBlur(T, (0, 0), s * 0.5)
        m = ((dog >= 0.06) & (~hot_m)).astype(np.uint8) * 255
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n, lbl, st, _ = cv2.connectedComponentsWithStats(m, 8)
        amin, amax = (0.3 * s) ** 2 * 3.14, (2.5 * s) ** 2 * 3.14
        for i in range(1, n):
            x, y, ww, hh, area = st[i]
            if area < amin or area > amax:
                continue
            if area / (ww * hh + 1e-6) < 0.45:       # 선형 구조 제거
                continue
            if max(ww, hh) / (min(ww, hh) + 1e-6) > 1.8:  # 길쭉하면 점 아님
                continue
            sev = float(dog[lbl == i].mean()) * 100
            if sev < 4.5:
                continue
            results.append({"kind": "spot",
                            "bbox": {"x1": int(x), "y1": int(y), "x2": int(x + ww), "y2": int(y + hh)},
                            "severity": _severity(sev, "spot"), "score": round(sev, 1)})

    # (3) 영역 냉각패치 — 넓은 결로/문틈.  배경(국소평균) 산출 시 '밝은(=뜨거운) 구조'가
    #     배경을 부풀려 인접 정상부를 거짓 cold 로 만드는 halo 오탐을 막는다 →
    #     밝은 상위 픽셀을 중앙값으로 중성화한 뒤 blur 해서 배경을 구한다.
    cold = _cold_map(T, sh, k_frac=0.18, neutralize_pct=94.0)
    m = ((cold >= 0.065) & (~hot_m)).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    ck = int(sh * 0.02) | 1
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((ck, ck), np.uint8))
    n, lbl, st, _ = cv2.connectedComponentsWithStats(m, 8)
    amin, amax = 0.0035 * h * w, 0.32 * h * w
    bm = int(sh * 0.015)
    line_thick = 0.018 * sh        # 이보다 얇고 채움 낮으면 구조 그리드선 → 배제
    patches: List[Dict] = []
    for i in range(1, n):
        x, y, ww, hh, area = st[i]
        if area < amin or area > amax:
            continue
        thick = min(ww, hh)
        length = max(ww, hh)
        ext = area / (ww * hh + 1e-6)
        if thick < line_thick and ext < 0.55:        # 얇은 선형 구조(그리드/창틀) 배제
            continue
        elongated = (length / (thick + 1e-6)) >= 2.3  # 코너 열교 같은 cold 띠
        # L자형 코너(수직+수평 띠)는 bbox 채움이 낮음 — ext 하한을 완화해 살린다.
        if not elongated and ext < 0.18:
            continue
        sev = float(cold[lbl == i].mean()) * 100
        if x <= bm or y <= bm or x + ww >= w - bm or y + hh >= h - bm:
            sev *= 0.75                               # 테두리 접촉 패널티(완화)
        if sev < 6.5:
            continue
        patches.append({"kind": "patch",
                        "bbox": {"x1": int(x), "y1": int(y), "x2": int(x + ww), "y2": int(y + hh)},
                        "severity": _severity(sev, "patch"), "score": round(sev, 1)})

    # (4) 방향성 cold 띠(코너 열교) — 위 패치가 놓치는 얇고 긴 띠를 보강. 기존 패치와 겹치면 제외.
    def _iou_overlap(a, b):
        ix1, iy1 = max(a["x1"], b["x1"]), max(a["y1"], b["y1"])
        ix2, iy2 = min(a["x2"], b["x2"]), min(a["y2"], b["y2"])
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = iw * ih
        aarea = (a["x2"] - a["x1"]) * (a["y2"] - a["y1"])
        return inter / (aarea + 1e-6)
    for band in _detect_cold_bands(T, hot_m, sh, h, w):
        if not any(_iou_overlap(band["bbox"], p["bbox"]) > 0.4 for p in patches):
            patches.append(band)

    # 간단 NMS: 패치 안에 들어간 점은 흡수
    def _center_in(b, p):
        cx = (b["x1"] + b["x2"]) / 2
        cy = (b["y1"] + b["y2"]) / 2
        return p["x1"] <= cx <= p["x2"] and p["y1"] <= cy <= p["y2"]

    keep_spots = [s for s in results
                  if not any(_center_in(s["bbox"], p["bbox"]) for p in patches)]
    out = patches + keep_spots

    # FLIR 오버레이 UI 영역 오탐 제거 — 상단 스케일바 / 하단 로고(워터마크)는 '차가운' 글자라
    # cold-spot 으로 오인된다. 프레임 상·하단 띠에 중심이 있는 검출은 제외.
    def _in_ui_band(b):
        cy = (b["y1"] + b["y2"]) / 2.0
        return cy >= 0.93 * h or cy <= 0.04 * h
    out = [r for r in out if not _in_ui_band(r["bbox"])]

    out.sort(key=lambda r: r["score"], reverse=True)
    return out[:max_results]
