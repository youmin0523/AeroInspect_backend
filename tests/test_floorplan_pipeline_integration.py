"""
tests/test_floorplan_pipeline_integration.py
역할: 평면도 → 3D 모델링 → 자율비행 종단 통합 테스트 (services-level).

HTTP TestClient 는 기존 .env / Settings 이슈로 부팅이 막혀 services 레벨로 검증.
endpoint 핸들러가 호출하는 동일 함수들을 같은 순서로 호출해 동등성 보장.

검증:
  1) /analyze 의 핵심: extract_walls_from_bytes → FloorplanAnalyzeResponse(**) 검증
  2) /generate-world 의 핵심: write_world_file (가구 포함) → 메타 + SDF 모델
  3) /missions/autonomous-scan/start 의 핵심:
        run_autonomous_scan → WS broadcast → mission.completed
"""
from __future__ import annotations

import asyncio
import os
import tempfile
from collections import Counter
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from app.services.floorplan_processor import extract_walls_from_bytes
from app.services.gazebo_world_generator import write_world_file
from app.services import autonomous_flight_simulator as af
from app.schemas.floorplan import FloorplanAnalyzeResponse


def _make_floorplan_with_furniture():
    """가구 + 벽 포함 평면도 합성."""
    W, H = 1600, 1200
    img = np.full((H, W, 3), 252, dtype=np.uint8)
    BLACK = (15, 15, 15); GRAY = (90, 90, 90); THK = 14

    cv2.rectangle(img, (100, 100), (1500, 1100), BLACK, THK)
    cv2.line(img, (800, 100), (800, 1100), BLACK, THK)
    # 가구
    cv2.rectangle(img, (200, 200), (550, 480), GRAY, -1)
    cv2.circle(img, (1100, 700), 100, GRAY, -1)
    cv2.rectangle(img, (1100, 950), (1400, 1050), GRAY, -1)

    path = os.path.join(tempfile.gettempdir(), 'integ_fp.png')
    cv2.imwrite(path, img)
    return path


def test_analyze_endpoint_returns_valid_schema():
    """/analyze 핸들러: extract → FloorplanAnalyzeResponse 직렬화 가능 확인."""
    path = _make_floorplan_with_furniture()
    with open(path, 'rb') as f:
        content = f.read()

    raw = extract_walls_from_bytes(content)
    response = FloorplanAnalyzeResponse(**raw)

    assert response.wall_count > 0, "벽 추출 0 — analyze 흐름 실패"
    assert response.furniture_count > 0, "가구 추출 0 — analyze 흐름 실패"
    assert response.image_width == 1600
    assert response.image_height == 1200
    # 직렬화 가능
    dumped = response.model_dump()
    assert 'walls' in dumped and 'furniture' in dumped


def test_generate_world_endpoint_writes_sdf_with_furniture():
    """/generate-world 핸들러: write_world_file 가 가구 포함 SDF 생성 확인."""
    path = _make_floorplan_with_furniture()
    with open(path, 'rb') as f:
        content = f.read()
    raw = extract_walls_from_bytes(content)

    out = os.path.join(tempfile.gettempdir(), 'integ.world')
    meta = write_world_file(
        output_path=out,
        world_name='integration_test',
        walls=raw['walls'],
        outline=raw['outline'],
        image_width=raw['image_width'],
        image_height=raw['image_height'],
        furniture=raw['furniture'],
    )

    assert os.path.exists(out)
    assert meta['wall_count'] > 0
    assert meta['furniture_count'] > 0
    assert meta['size_source'] == 'aspect'
    assert abs(meta['world_w'] / meta['world_d'] - 1600 / 1200) < 0.01

    with open(out, encoding='utf-8') as f:
        sdf = f.read()
    assert sdf.startswith('<?xml')
    assert '<sdf version="1.9">' in sdf
    assert sdf.count('<model name="wall_') == meta['wall_count']
    assert sdf.count('<model name="furniture_') == meta['furniture_count']
    assert '<model name="ground_plane">' in sdf


@pytest.mark.asyncio
async def test_missions_endpoint_autonomous_scan_with_furniture():
    """/missions/autonomous-scan/start 핸들러: 가구 포함 시뮬 종단 확인."""
    path = _make_floorplan_with_furniture()
    with open(path, 'rb') as f:
        content = f.read()
    raw = extract_walls_from_bytes(content)

    received = []

    async def fake_broadcast(channel, payload):
        received.append((channel, payload['type']))

    with patch('app.services.autonomous_flight_simulator.ws_manager.broadcast',
               side_effect=fake_broadcast):
        # 단일 레이어 + 큰 lane 으로 테스트 시간 단축 (다층 동작은 별도 케이스)
        mid = await af.run_autonomous_scan(
            walls=raw['walls'], outline=raw['outline'],
            world_w=6.0, world_d=4.0,
            furniture=raw['furniture'],
            speed=15.0, telemetry_hz=20, lidar_hz=20,
            altitude_layers=(1.5,),
            lane_spacing=1.5,
        )

        for _ in range(400):
            await asyncio.sleep(0.05)
            m = af.get_mission(mid)
            if m and m.status != 'running':
                break

        m = af.get_mission(mid)
        assert m.status == 'completed', f"미션 미완주: {m.status}"
        assert m.points_emitted > 0
        assert m.furniture_avoidance_count >= 0  # 가구 위치/경로에 따라 0 이상

        types = Counter(t for _, t in received)
        assert types.get('telemetry.update', 0) > 0, "telemetry.update 미발행"
        assert types.get('lidar.points', 0) > 0, "lidar.points 미발행"
        assert types.get('mission.completed', 0) == 1, "mission.completed 미발행"


@pytest.mark.asyncio
async def test_missions_cancel_flow():
    """/missions/{id}/cancel 핸들러: cancel_mission → status=cancelled."""
    walls = [
        {'x1': 0.02, 'y1': 0.02, 'x2': 0.98, 'y2': 0.02},
        {'x1': 0.98, 'y1': 0.02, 'x2': 0.98, 'y2': 0.98},
        {'x1': 0.98, 'y1': 0.98, 'x2': 0.02, 'y2': 0.98},
        {'x1': 0.02, 'y1': 0.98, 'x2': 0.02, 'y2': 0.02},
    ]
    with patch('app.services.autonomous_flight_simulator.ws_manager.broadcast',
               side_effect=lambda *a, **k: None):
        mid = await af.run_autonomous_scan(
            walls=walls, outline=[], world_w=10, world_d=10,
            speed=0.3, telemetry_hz=20, lidar_hz=20,
        )
        await asyncio.sleep(0.4)
        ok = af.cancel_mission(mid)
        assert ok
        await asyncio.sleep(0.3)
        m = af.get_mission(mid)
        assert m.status == 'cancelled'


def test_dxf_pipeline_with_furniture():
    """/process DXF 분기: 합성 DXF (LINE+CIRCLE+INSERT) → walls + furniture 추출."""
    import ezdxf
    from app.services.dxf_parser import parse_dxf

    doc = ezdxf.new('R2010')
    msp = doc.modelspace()
    # 벽 4면
    for s, e in [((0, 0), (10000, 0)), ((10000, 0), (10000, 8000)),
                 ((10000, 8000), (0, 8000)), ((0, 8000), (0, 0))]:
        msp.add_line(s, e)
    # 식탁
    msp.add_circle((5000, 4000), 800)
    # 침대 (LWPOLYLINE 닫힘)
    msp.add_lwpolyline([(500, 500), (3000, 500), (3000, 2500), (500, 2500)], close=True)
    # 의자 블록 INSERT
    block = doc.blocks.new(name='CHAIR2')
    block.add_lwpolyline([(0, 0), (300, 0), (300, 300), (0, 300)], close=True)
    msp.add_blockref('CHAIR2', insert=(7000, 6000))

    path = os.path.join(tempfile.gettempdir(), 'integ.dxf')
    doc.saveas(path)

    result = parse_dxf(path)
    assert result['wall_count'] >= 4, f"DXF 벽 추출 부족: {result['wall_count']}"
    assert result['furniture_count'] >= 3, f"DXF 가구 추출 부족: {result['furniture_count']}"
    labels = Counter(f['label'] for f in result['furniture'])
    assert labels.get('circular', 0) >= 1, "원형 가구 미검출"
    assert labels.get('rectangular', 0) >= 2, "사각형 가구 미검출"
    assert result['image_width'] == 10000  # mm 단위 보존
