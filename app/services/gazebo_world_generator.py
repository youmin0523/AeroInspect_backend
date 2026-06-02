# =============================================
# app/services/gazebo_world_generator.py
# 역할: 도면 벽체(walls) → Gazebo SDF .world 파일 생성
#       드론 자율비행 시뮬레이션(SLAM/경로계획)용 3D 월드.
#   - walls: [{x1,y1,x2,y2}] 정규화(0-1) 선분 (floorplan_processor 출력)
#   - 각 벽 선분을 일정 높이의 box 모델로 변환
#   - scale_px_per_meter 로 실척(meter) 변환, 없으면 기본 가정치
# =============================================

from __future__ import annotations

import math
import os
from typing import List, Dict, Optional, Any

# 기본 가정치 (도면 스케일 미보정 시)
DEFAULT_WALL_HEIGHT_M = 2.4      # 표준 천장고
DEFAULT_WALL_THICKNESS_M = 0.15  # 표준 벽 두께
# 정규화 좌표를 실척으로 풀 때, 스케일 없으면 도면 1.0 = 이 길이(m)로 가정
DEFAULT_PLAN_SPAN_M = 10.0
# 정규화 좌표 기준 px 환산용 가정 해상도 (scale_px_per_meter 가 px 기준이므로)
ASSUMED_IMAGE_PX = 1000.0


def derive_world_size(
    image_width: Optional[int],
    image_height: Optional[int],
    scale_px_per_meter: Optional[float],
) -> tuple:
    """도면 px + 스케일 → 실척 월드 크기 (world_w, world_d, source).

    scale 있으면 px/scale = meter (정확). 없으면 기본 가정치(정사각 span).
    return: (world_w_m, world_d_m, source_str)
    """
    if scale_px_per_meter and scale_px_per_meter > 0 and image_width and image_height:
        return (
            round(image_width / scale_px_per_meter, 4),
            round(image_height / scale_px_per_meter, 4),
            "scale",
        )
    # 스케일 없음 — 가로세로비 유지하되 기본 span 가정
    if image_width and image_height and image_height > 0:
        aspect = image_width / image_height
        return (round(DEFAULT_PLAN_SPAN_M * aspect, 4), DEFAULT_PLAN_SPAN_M, "assumed_aspect")
    return (DEFAULT_PLAN_SPAN_M, DEFAULT_PLAN_SPAN_M, "assumed_default")


def _wall_to_box(
    wall: Dict[str, float],
    idx: int,
    span_m: float,
    height_m: float,
    thickness_m: float,
) -> Optional[str]:
    """단일 벽 선분(정규화) → Gazebo box model SDF 조각."""
    try:
        x1, y1 = float(wall["x1"]), float(wall["y1"])
        x2, y2 = float(wall["x2"]), float(wall["y2"])
    except (KeyError, TypeError, ValueError):
        return None

    # 정규화 → 실척(m). y축은 이미지(아래로+)→월드(위로+) 뒤집기.
    wx1, wy1 = x1 * span_m, (1.0 - y1) * span_m
    wx2, wy2 = x2 * span_m, (1.0 - y2) * span_m

    dx, dy = wx2 - wx1, wy2 - wy1
    length = math.hypot(dx, dy)
    if length < 1e-3:
        return None  # 점에 가까운 선분 스킵

    cx, cy = (wx1 + wx2) / 2.0, (wy1 + wy2) / 2.0
    yaw = math.atan2(dy, dx)
    cz = height_m / 2.0

    return f"""    <model name='wall_{idx}'>
      <static>true</static>
      <pose>{cx:.4f} {cy:.4f} {cz:.4f} 0 0 {yaw:.5f}</pose>
      <link name='link'>
        <collision name='collision'>
          <geometry><box><size>{length:.4f} {thickness_m:.4f} {height_m:.4f}</size></box></geometry>
        </collision>
        <visual name='visual'>
          <geometry><box><size>{length:.4f} {thickness_m:.4f} {height_m:.4f}</size></box></geometry>
          <material><ambient>0.8 0.8 0.8 1</ambient><diffuse>0.8 0.8 0.8 1</diffuse></material>
        </visual>
      </link>
    </model>"""


def write_world_file(
    output_path: str,
    world_name: str,
    walls: List[Dict[str, float]],
    outline: Optional[List[Dict[str, float]]] = None,
    image_width: Optional[int] = None,
    image_height: Optional[int] = None,
    scale_px_per_meter: Optional[float] = None,
    wall_height_m: float = DEFAULT_WALL_HEIGHT_M,
    wall_thickness_m: float = DEFAULT_WALL_THICKNESS_M,
) -> Dict[str, Any]:
    """
    벽체 리스트를 Gazebo SDF .world 파일로 저장.

    Args:
        output_path: 저장할 .world 경로
        world_name: <world name=...>
        walls: [{x1,y1,x2,y2}] 정규화(0-1) 선분
        outline: 외곽선(선택, 현재 walls와 동일 처리)
        image_width/height: 원본 도면 px (스케일 보정용, 선택)
        scale_px_per_meter: px→meter 변환 계수 (있으면 실척 정확)
    Returns:
        {n_walls, n_models, span_m, world_path, wall_height_m, ...}
    """
    walls = walls or []

    # 실척 span 계산: scale_px_per_meter 있으면 도면 전체 px폭/scale = m, 없으면 기본 가정.
    if scale_px_per_meter and scale_px_per_meter > 0:
        px = float(image_width) if image_width else ASSUMED_IMAGE_PX
        span_m = px / scale_px_per_meter
    else:
        span_m = DEFAULT_PLAN_SPAN_M

    models = []
    for i, wall in enumerate(walls):
        sdf = _wall_to_box(wall, i, span_m, wall_height_m, wall_thickness_m)
        if sdf:
            models.append(sdf)

    models_xml = "\n".join(models)
    world_xml = f"""<?xml version='1.0' ?>
<sdf version='1.7'>
  <world name='{world_name}'>
    <include><uri>model://ground_plane</uri></include>
    <include><uri>model://sun</uri></include>
{models_xml}
  </world>
</sdf>
"""

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(world_xml)

    return {
        "n_walls": len(walls),
        "n_models": len(models),
        "span_m": round(span_m, 3),
        "wall_height_m": wall_height_m,
        "wall_thickness_m": wall_thickness_m,
        "world_path": output_path,
        "scale_applied": bool(scale_px_per_meter and scale_px_per_meter > 0),
    }
