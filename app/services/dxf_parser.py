"""
services/dxf_parser.py
역할: DXF (CAD) 도면 → walls + furniture 추출
      - LINE         → 벽 후보 (긴 선분만)
      - LWPOLYLINE   → 닫힌 다각형이면 가구, 열린 거면 벽 선분으로 분해
      - CIRCLE/ARC   → 가구 (식탁·세면대·욕조 등)
      - INSERT       → 블록 참조 펼치기 (블록 내부 엔티티 재귀 처리)

ezdxf 1.x API. 평면도→3D 라인의 가구 누락 보완 (자율비행 충돌 회피용).
LINE-only 처리는 DXF 가구 심볼을 모두 놓치므로 이 파서로 대체.
"""

from __future__ import annotations

import math
from typing import Iterable

# ezdxf 는 try/except 가드 로 import — 미설치 환경에서도 모듈 로드 자체는 통과
try:
    import ezdxf
    from ezdxf.math import Vec3
    EZDXF_AVAILABLE = True
except ImportError:
    EZDXF_AVAILABLE = False


# ── 분류 임계값 ──────────────────────────
MIN_WALL_LENGTH_RATIO = 0.04   # 도면 긴 변의 4% 이상이어야 벽 후보
MAX_FURNITURE_AREA_RATIO = 0.18  # 너무 큰 폴리라인은 방 외곽 — 가구 아님


def parse_dxf(file_path: str) -> dict:
    """
    DXF 파일 → {walls, furniture, outline, image_width, image_height} 딕셔너리.

    Returns:
        {
            "walls": [{"x1","y1","x2","y2"}, ...],
            "furniture": [{"cx","cy","w","h","angle","label"}, ...],
            "outline": [{"x","y"}, ...],   # 가장 큰 닫힌 폴리라인 (있으면)
            "image_width": int,             # mm 단위 → 1px = 1mm 가정
            "image_height": int,
            "wall_count": int,
            "furniture_count": int,
        }
    """
    if not EZDXF_AVAILABLE:
        raise ImportError("ezdxf 패키지가 설치되어 있지 않습니다.")

    doc = ezdxf.readfile(file_path)
    msp = doc.modelspace()

    # ── 1. 모든 엔티티 펼치기 (INSERT 블록 포함) ──
    flat_entities: list = []
    for e in msp:
        flat_entities.extend(_flatten_entity(e, doc, depth=0))

    # ── 2. 좌표 범위 산출 (정규화용) ──
    all_pts = list(_collect_points(flat_entities))
    if not all_pts:
        return _empty_result()

    xs = [p[0] for p in all_pts]
    ys = [p[1] for p in all_pts]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    w = max_x - min_x if max_x - min_x > 1e-6 else 1.0
    h = max_y - min_y if max_y - min_y > 1e-6 else 1.0
    long_side = max(w, h)
    min_wall_len = long_side * MIN_WALL_LENGTH_RATIO

    walls: list[dict] = []
    furniture: list[dict] = []
    outline: list[dict] = []
    largest_closed_area = 0.0

    def to_norm(x: float, y: float) -> tuple[float, float]:
        return round((x - min_x) / w, 4), round((y - min_y) / h, 4)

    def add_wall(x1, y1, x2, y2):
        length = math.hypot(x2 - x1, y2 - y1)
        if length < min_wall_len:
            return  # 짧은 선분은 가구 디테일일 가능성 — 벽 아님
        nx1, ny1 = to_norm(x1, y1)
        nx2, ny2 = to_norm(x2, y2)
        walls.append({"x1": nx1, "y1": ny1, "x2": nx2, "y2": ny2})

    def add_furniture_bbox(min_fx, min_fy, max_fx, max_fy, label="rectangular"):
        cx = (min_fx + max_fx) / 2
        cy = (min_fy + max_fy) / 2
        fw = max_fx - min_fx
        fh = max_fy - min_fy
        area_ratio = (fw * fh) / (w * h)
        if area_ratio > MAX_FURNITURE_AREA_RATIO:
            return False  # 너무 크면 outline 후보 (방 외곽)
        if fw < long_side * 0.005 or fh < long_side * 0.005:
            return False
        ncx, ncy = to_norm(cx, cy)
        furniture.append({
            "cx": ncx, "cy": ncy,
            "w": round(fw / w, 4),
            "h": round(fh / h, 4),
            "angle": 0.0,
            "label": label,
        })
        return True

    # ── 3. 엔티티별 분류 ──
    for e in flat_entities:
        etype = e.dxftype()

        if etype == "LINE":
            sx, sy = e.dxf.start.x, e.dxf.start.y
            ex, ey = e.dxf.end.x, e.dxf.end.y
            add_wall(sx, sy, ex, ey)

        elif etype == "CIRCLE":
            cx, cy = e.dxf.center.x, e.dxf.center.y
            r = float(e.dxf.radius)
            ncx, ncy = to_norm(cx, cy)
            furniture.append({
                "cx": ncx, "cy": ncy,
                "w": round((2 * r) / w, 4),
                "h": round((2 * r) / h, 4),
                "angle": 0.0,
                "label": "circular",
            })

        elif etype == "ARC":
            # 호는 곡선 가구 일부 (욕조·세면대 등) — bbox 로 근사 가구 등록
            cx, cy = e.dxf.center.x, e.dxf.center.y
            r = float(e.dxf.radius)
            # 호의 bbox 는 부정확하지만 회피 입장에서 근사 OK
            ncx, ncy = to_norm(cx, cy)
            furniture.append({
                "cx": ncx, "cy": ncy,
                "w": round((2 * r) / w, 4),
                "h": round((2 * r) / h, 4),
                "angle": 0.0,
                "label": "circular",
            })

        elif etype == "LWPOLYLINE":
            pts = [(p[0], p[1]) for p in e.get_points("xy")]
            if len(pts) < 2:
                continue
            is_closed = bool(e.closed)
            if is_closed and len(pts) >= 3:
                # 닫힌 폴리라인 — 가구 후보 (단, 면적 큰 건 outline)
                xs_p = [p[0] for p in pts]
                ys_p = [p[1] for p in pts]
                area = _polygon_area(pts)
                ratio = area / (w * h)
                if ratio > MAX_FURNITURE_AREA_RATIO:
                    # 가장 큰 닫힌 폴리라인 = 외곽 후보
                    if area > largest_closed_area:
                        largest_closed_area = area
                        outline = [{"x": to_norm(p[0], p[1])[0],
                                    "y": to_norm(p[0], p[1])[1]} for p in pts]
                else:
                    add_furniture_bbox(min(xs_p), min(ys_p), max(xs_p), max(ys_p))
            else:
                # 열린 폴리라인 — 각 세그먼트를 벽으로
                for i in range(len(pts) - 1):
                    add_wall(pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1])

        elif etype == "POLYLINE":
            try:
                vertices = list(e.vertices)
                pts = [(v.dxf.location.x, v.dxf.location.y) for v in vertices]
            except Exception:
                continue
            if len(pts) < 2:
                continue
            is_closed = bool(getattr(e, "is_closed", False))
            if is_closed and len(pts) >= 3:
                xs_p = [p[0] for p in pts]
                ys_p = [p[1] for p in pts]
                area = _polygon_area(pts)
                if area / (w * h) <= MAX_FURNITURE_AREA_RATIO:
                    add_furniture_bbox(min(xs_p), min(ys_p), max(xs_p), max(ys_p))
            else:
                for i in range(len(pts) - 1):
                    add_wall(pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1])

    # 너무 많으면 컷
    if len(walls) > 80:
        walls.sort(key=lambda wd: math.hypot(wd["x2"] - wd["x1"], wd["y2"] - wd["y1"]), reverse=True)
        walls = walls[:80]
    if len(furniture) > 50:
        furniture.sort(key=lambda f: f["w"] * f["h"], reverse=True)
        furniture = furniture[:50]

    return {
        "walls": walls,
        "furniture": furniture,
        "outline": outline,
        "image_width": int(round(w)),
        "image_height": int(round(h)),
        "wall_count": len(walls),
        "furniture_count": len(furniture),
    }


def _flatten_entity(entity, doc, depth: int) -> Iterable:
    """INSERT 블록 참조를 재귀 펼치기. 너무 깊은 중첩은 안전 한계."""
    if depth > 5:
        return
    etype = entity.dxftype()
    if etype != "INSERT":
        yield entity
        return

    # INSERT — 블록 정의 참조 + 위치/스케일/회전 변환 적용
    try:
        block_name = entity.dxf.name
        block = doc.blocks.get(block_name)
        if block is None:
            return
        # 가상 변환 — virtual_entities() 가 변환된 엔티티 yield
        for sub in entity.virtual_entities():
            for inner in _flatten_entity(sub, doc, depth + 1):
                yield inner
    except Exception:
        return


def _collect_points(entities: Iterable) -> Iterable[tuple[float, float]]:
    """좌표 범위 산출용 — 모든 엔티티 좌표 yield."""
    for e in entities:
        et = e.dxftype()
        if et == "LINE":
            yield (e.dxf.start.x, e.dxf.start.y)
            yield (e.dxf.end.x, e.dxf.end.y)
        elif et == "CIRCLE":
            cx, cy = e.dxf.center.x, e.dxf.center.y
            r = float(e.dxf.radius)
            yield (cx - r, cy - r)
            yield (cx + r, cy + r)
        elif et == "ARC":
            cx, cy = e.dxf.center.x, e.dxf.center.y
            r = float(e.dxf.radius)
            yield (cx - r, cy - r)
            yield (cx + r, cy + r)
        elif et == "LWPOLYLINE":
            for p in e.get_points("xy"):
                yield (p[0], p[1])
        elif et == "POLYLINE":
            try:
                for v in e.vertices:
                    yield (v.dxf.location.x, v.dxf.location.y)
            except Exception:
                continue


def _polygon_area(pts: list[tuple[float, float]]) -> float:
    """Shoelace 공식. 음수면 절대값."""
    n = len(pts)
    if n < 3:
        return 0.0
    s = 0.0
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return abs(s) / 2.0


def _empty_result() -> dict:
    return {
        "walls": [], "furniture": [], "outline": [],
        "image_width": 1, "image_height": 1,
        "wall_count": 0, "furniture_count": 0,
    }
