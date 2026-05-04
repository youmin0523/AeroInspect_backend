# =============================================
# tests/test_coverage_geometry.py
# 역할: coverage 라우터 내 순수 기하 유틸 회귀 테스트
#       - convex hull 알고리즘 (Andrew's monotone chain)
#       - shoelace 면적
# 엔드포인트 자체는 DB 의존 → 여기선 순수 함수만 검증
# =============================================

import math

import pytest

from app.api.coverage import _convex_hull, _polygon_area


def test_convex_hull_square():
    """완전 정사각형 → hull 4꼭짓점, 면적 = 변²."""
    pts = [(0, 0), (10, 0), (10, 10), (0, 10), (5, 5)]  # 내부점 포함
    hull = _convex_hull(pts)
    assert len(hull) == 4
    assert _polygon_area(hull) == pytest.approx(100.0)


def test_convex_hull_triangle():
    """세 점 → 면적 = 0.5 × base × height."""
    pts = [(0, 0), (4, 0), (0, 3)]
    hull = _convex_hull(pts)
    assert len(hull) == 3
    assert _polygon_area(hull) == pytest.approx(6.0)


def test_convex_hull_duplicates_removed():
    """중복 점이 있어도 hull은 고유 꼭짓점만 반환."""
    pts = [(0, 0), (0, 0), (2, 0), (2, 2), (0, 2), (2, 0)]
    hull = _convex_hull(pts)
    assert len(hull) == 4
    assert _polygon_area(hull) == pytest.approx(4.0)


def test_polygon_area_degenerate():
    """점이 3개 미만이면 면적 0."""
    assert _polygon_area([]) == 0.0
    assert _polygon_area([(0, 0)]) == 0.0
    assert _polygon_area([(0, 0), (1, 1)]) == 0.0


def test_convex_hull_collinear_points():
    """일직선 위 점만 있으면 hull은 2점(또는 그 미만) → 면적 0."""
    pts = [(0, 0), (1, 1), (2, 2), (3, 3)]
    hull = _convex_hull(pts)
    assert _polygon_area(hull) == pytest.approx(0.0)


def test_convex_hull_rotated_square():
    """45도 회전 정사각형(대각선 2)도 shoelace로 정확히 2.0."""
    pts = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    hull = _convex_hull(pts)
    assert _polygon_area(hull) == pytest.approx(2.0)
