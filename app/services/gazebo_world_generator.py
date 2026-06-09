"""
services/gazebo_world_generator.py
역할: 평면도/CAD 추출 결과(walls + outline) → Gazebo SDF .world 파일 생성
      - walls (정규화 0-1 선분) → 박스 모델 (벽 두께/높이 부여)
      - outline (정규화 0-1 다각형) → 외벽 (선택 사항)
      - 결과 .world 는 향후 Gazebo Garden / Ignition 컨테이너에 입력 가능
      - ROS2 / Gazebo 의존성 없이 순수 텍스트(SDF/XML) 생성

레퍼런스:
  - SDF 1.9 spec: http://sdformat.org/spec
  - Gazebo Garden: https://gazebosim.org/docs/garden

생성 대상:
  - <world name="..."> 루트
    - 표준 sun + ground_plane
    - <model> per wall  (box 1개)
    - <model> per outline segment (반투명 유리, 외부 경계)
"""

from __future__ import annotations

import math
import os
import xml.etree.ElementTree as ET
from typing import Optional


# ── 기본값 (m) ──────────────────────────────
DEFAULT_WORLD_LONG_SIDE = 12.0   # calibrate 안 됐을 때 긴 변 = 12m (BuildingMesh 와 일치)
DEFAULT_HEIGHT = 3.0             # 천장 높이
DEFAULT_WALL_THICKNESS = 0.12    # 벽 두께
DEFAULT_OUTLINE_THICKNESS = 0.04
MIN_SEGMENT_LEN = 0.05           # 너무 짧은 선분은 노이즈 — 무시


def derive_world_size(
    image_width: Optional[int],
    image_height: Optional[int],
    scale_px_per_meter: Optional[float],
    long_side_fallback: float = DEFAULT_WORLD_LONG_SIDE,
) -> tuple[float, float, str]:
    """
    실세계 m 단위 (world_w, world_d) 산출. BuildingMesh.deriveSceneSize 와 동일 정책.

    Returns:
        (world_w, world_d, source) — source ∈ {'calibrated', 'aspect', 'fallback'}
    """
    if scale_px_per_meter and scale_px_per_meter > 0 and image_width and image_height:
        return (
            image_width / scale_px_per_meter,
            image_height / scale_px_per_meter,
            'calibrated',
        )

    if image_width and image_height:
        aspect = image_width / image_height
        if aspect >= 1:
            return (long_side_fallback, long_side_fallback / aspect, 'aspect')
        return (long_side_fallback * aspect, long_side_fallback, 'aspect')

    return (long_side_fallback, long_side_fallback * 0.75, 'fallback')


def _make_wall_model(
    name: str,
    midx: float,
    midz: float,
    length: float,
    angle: float,
    height: float,
    thickness: float,
    color_rgba: str = "0.4 0.45 0.5 1",
) -> ET.Element:
    """
    벽 1개를 정적 SDF <model> 엘리먼트로 생성.
    Gazebo 좌표계: X=동쪽, Y=북쪽, Z=위. 평면도의 (X,Z) 를 (X,Y) 로 매핑.
    """
    model = ET.Element("model", attrib={"name": name})
    ET.SubElement(model, "static").text = "true"

    # 위치 + yaw (벽이 X축 방향으로 길다고 가정 → angle 만큼 yaw 회전)
    pose = ET.SubElement(model, "pose")
    pose.text = f"{midx:.4f} {midz:.4f} {height/2:.4f} 0 0 {angle:.4f}"

    link = ET.SubElement(model, "link", attrib={"name": "link"})

    # collision (시뮬레이션 충돌체)
    collision = ET.SubElement(link, "collision", attrib={"name": "collision"})
    geom_c = ET.SubElement(collision, "geometry")
    box_c = ET.SubElement(geom_c, "box")
    ET.SubElement(box_c, "size").text = f"{length:.4f} {thickness:.4f} {height:.4f}"

    # visual (시각화)
    visual = ET.SubElement(link, "visual", attrib={"name": "visual"})
    geom_v = ET.SubElement(visual, "geometry")
    box_v = ET.SubElement(geom_v, "box")
    ET.SubElement(box_v, "size").text = f"{length:.4f} {thickness:.4f} {height:.4f}"

    material = ET.SubElement(visual, "material")
    ET.SubElement(material, "ambient").text = color_rgba
    ET.SubElement(material, "diffuse").text = color_rgba

    return model


def _norm_segment_to_world(
    x1n: float, y1n: float, x2n: float, y2n: float,
    world_w: float, world_d: float,
) -> tuple[float, float, float, float] | None:
    """
    정규화 0-1 선분 (x1n, y1n)-(x2n, y2n) → 월드 좌표계의 (midx, midy, length, angle).
    이미지 원점(좌상단) → 월드 원점(중앙) 변환. y축 부호는 그대로 (위→아래 → 북→남).
    """
    x1 = (x1n - 0.5) * world_w
    z1 = (y1n - 0.5) * world_d
    x2 = (x2n - 0.5) * world_w
    z2 = (y2n - 0.5) * world_d

    midx = (x1 + x2) / 2
    midz = (z1 + z2) / 2
    length = math.hypot(x2 - x1, z2 - z1)
    if length < MIN_SEGMENT_LEN:
        return None
    angle = math.atan2(z2 - z1, x2 - x1)
    return midx, midz, length, angle


def _make_furniture_model(
    name: str,
    cx: float, cy: float,
    fw: float, fd: float, fh: float,
    angle_rad: float,
    color_rgba: str = "0.55 0.40 0.25 1",  # 갈색 톤 (가구)
) -> ET.Element:
    """
    가구 1개를 정적 SDF box collision + visual 로 생성.
    드론 시뮬레이터의 충돌 검사·LiDAR raycast 대상이 됨.
    """
    model = ET.Element("model", attrib={"name": name})
    ET.SubElement(model, "static").text = "true"
    pose = ET.SubElement(model, "pose")
    pose.text = f"{cx:.4f} {cy:.4f} {fh/2:.4f} 0 0 {angle_rad:.4f}"

    link = ET.SubElement(model, "link", attrib={"name": "link"})
    collision = ET.SubElement(link, "collision", attrib={"name": "collision"})
    geom_c = ET.SubElement(collision, "geometry")
    box_c = ET.SubElement(geom_c, "box")
    ET.SubElement(box_c, "size").text = f"{fw:.4f} {fd:.4f} {fh:.4f}"

    visual = ET.SubElement(link, "visual", attrib={"name": "visual"})
    geom_v = ET.SubElement(visual, "geometry")
    box_v = ET.SubElement(geom_v, "box")
    ET.SubElement(box_v, "size").text = f"{fw:.4f} {fd:.4f} {fh:.4f}"
    material = ET.SubElement(visual, "material")
    ET.SubElement(material, "ambient").text = color_rgba
    ET.SubElement(material, "diffuse").text = color_rgba

    return model


def _furniture_height_for_label(label: str) -> float:
    """가구 라벨별 추정 높이 (m). 충돌 회피 입장에선 보수적으로 큰 값."""
    return {
        'rectangular': 1.0,   # 침대·소파·식탁 등
        'small': 0.85,        # 의자·세면대·변기
        'unknown': 1.2,       # 안전 마진
    }.get(label, 1.0)


def build_world_xml(
    world_name: str,
    walls: list[dict],
    outline: list[dict],
    world_w: float,
    world_d: float,
    height: float = DEFAULT_HEIGHT,
    wall_thickness: float = DEFAULT_WALL_THICKNESS,
    outline_thickness: float = DEFAULT_OUTLINE_THICKNESS,
    furniture: list[dict] | None = None,
) -> str:
    """
    SDF 1.9 .world XML 문자열 생성.

    Args:
        world_name: <world name="...">
        walls: [{x1,y1,x2,y2}] 정규화 0-1 좌표 리스트
        outline: [{x,y}] 정규화 0-1 다각형 꼭짓점 (닫힘)
        world_w, world_d: 월드 가로/세로 (m)
        height: 천장 높이 (m)
    """
    root = ET.Element("sdf", attrib={"version": "1.9"})
    world = ET.SubElement(root, "world", attrib={"name": world_name})

    # 표준 물리 + 조명 + 바닥
    physics = ET.SubElement(world, "physics", attrib={"name": "default_physics", "type": "ode"})
    ET.SubElement(physics, "max_step_size").text = "0.001"
    ET.SubElement(physics, "real_time_factor").text = "1.0"
    ET.SubElement(physics, "real_time_update_rate").text = "1000"

    # sun
    light = ET.SubElement(world, "light", attrib={"name": "sun", "type": "directional"})
    ET.SubElement(light, "cast_shadows").text = "true"
    ET.SubElement(light, "pose").text = "0 0 10 0 0 0"
    ET.SubElement(light, "diffuse").text = "0.8 0.8 0.8 1"
    ET.SubElement(light, "specular").text = "0.2 0.2 0.2 1"
    ET.SubElement(light, "direction").text = "-0.5 0.1 -0.9"

    # //* [Modified Code 2026-05-13 v3] ceiling_plane — 천장 평면 모델
    # 자율비행 시뮬에서 천장 raycast / 충돌 검사 가능
    ceil = ET.SubElement(world, "model", attrib={"name": "ceiling_plane"})
    ET.SubElement(ceil, "static").text = "true"
    ET.SubElement(ceil, "pose").text = f"0 0 {height:.4f} 0 0 0"
    c_link = ET.SubElement(ceil, "link", attrib={"name": "link"})
    c_col = ET.SubElement(c_link, "collision", attrib={"name": "collision"})
    c_geom_c = ET.SubElement(c_col, "geometry")
    c_plane_c = ET.SubElement(c_geom_c, "plane")
    ET.SubElement(c_plane_c, "normal").text = "0 0 -1"
    ET.SubElement(c_plane_c, "size").text = f"{world_w * 2:.4f} {world_d * 2:.4f}"
    c_vis = ET.SubElement(c_link, "visual", attrib={"name": "visual"})
    c_geom_v = ET.SubElement(c_vis, "geometry")
    c_plane_v = ET.SubElement(c_geom_v, "plane")
    ET.SubElement(c_plane_v, "normal").text = "0 0 -1"
    ET.SubElement(c_plane_v, "size").text = f"{world_w * 2:.4f} {world_d * 2:.4f}"
    c_mat = ET.SubElement(c_vis, "material")
    ET.SubElement(c_mat, "ambient").text = "0.92 0.92 0.92 1"
    ET.SubElement(c_mat, "diffuse").text = "0.92 0.92 0.92 1"

    # ground_plane
    ground = ET.SubElement(world, "model", attrib={"name": "ground_plane"})
    ET.SubElement(ground, "static").text = "true"
    g_link = ET.SubElement(ground, "link", attrib={"name": "link"})
    g_col = ET.SubElement(g_link, "collision", attrib={"name": "collision"})
    g_geom_c = ET.SubElement(g_col, "geometry")
    g_plane_c = ET.SubElement(g_geom_c, "plane")
    ET.SubElement(g_plane_c, "normal").text = "0 0 1"
    ET.SubElement(g_plane_c, "size").text = f"{world_w * 2:.4f} {world_d * 2:.4f}"
    g_vis = ET.SubElement(g_link, "visual", attrib={"name": "visual"})
    g_geom_v = ET.SubElement(g_vis, "geometry")
    g_plane_v = ET.SubElement(g_geom_v, "plane")
    ET.SubElement(g_plane_v, "normal").text = "0 0 1"
    ET.SubElement(g_plane_v, "size").text = f"{world_w * 2:.4f} {world_d * 2:.4f}"
    g_mat = ET.SubElement(g_vis, "material")
    ET.SubElement(g_mat, "ambient").text = "0.18 0.22 0.28 1"
    ET.SubElement(g_mat, "diffuse").text = "0.18 0.22 0.28 1"

    # walls
    for i, wall in enumerate(walls or []):
        seg = _norm_segment_to_world(
            wall["x1"], wall["y1"], wall["x2"], wall["y2"],
            world_w, world_d,
        )
        if seg is None:
            continue
        midx, midz, length, angle = seg
        world.append(_make_wall_model(
            name=f"wall_{i}",
            midx=midx, midz=midz, length=length, angle=angle,
            height=height, thickness=wall_thickness,
        ))

    # outline (창호 갭 포함된 외벽 — 반투명 유리)
    if outline and len(outline) >= 3:
        for i in range(len(outline)):
            pt = outline[i]
            nxt = outline[(i + 1) % len(outline)]
            seg = _norm_segment_to_world(
                pt["x"], pt["y"], nxt["x"], nxt["y"],
                world_w, world_d,
            )
            if seg is None:
                continue
            midx, midz, length, angle = seg
            world.append(_make_wall_model(
                name=f"outline_{i}",
                midx=midx, midz=midz, length=length, angle=angle,
                height=height, thickness=outline_thickness,
                color_rgba="0.22 0.74 0.97 0.25",  # cyan, 반투명
            ))

    # //* [Modified Code 2026-05-13] 가구/빌트인 — 자율비행 충돌 회피용
    # 정규화 (cx,cy,w,h,angle°) → 월드 m 단위 박스. 천장 닿지 않도록 가구별 추정 높이.
    if furniture:
        for i, f in enumerate(furniture):
            cx_n = f.get("cx"); cy_n = f.get("cy")
            fw_n = f.get("w");  fd_n = f.get("h")
            if cx_n is None or cy_n is None or fw_n is None or fd_n is None:
                continue
            cx_m = (cx_n - 0.5) * world_w
            cy_m = (cy_n - 0.5) * world_d
            fw_m = max(fw_n * world_w, 0.05)
            fd_m = max(fd_n * world_d, 0.05)
            angle_rad = math.radians(f.get("angle", 0))
            fh_m = min(_furniture_height_for_label(f.get("label", "rectangular")),
                       height - 0.1)  # 천장 미만
            world.append(_make_furniture_model(
                name=f"furniture_{i}_{f.get('label','obj')}",
                cx=cx_m, cy=cy_m,
                fw=fw_m, fd=fd_m, fh=fh_m,
                angle_rad=angle_rad,
            ))

    return ET.tostring(root, encoding="unicode")


def write_world_file(
    output_path: str,
    world_name: str,
    walls: list[dict],
    outline: list[dict],
    image_width: Optional[int] = None,
    image_height: Optional[int] = None,
    scale_px_per_meter: Optional[float] = None,
    furniture: Optional[list[dict]] = None,
) -> dict:
    """
    .world 파일을 디스크에 작성하고 메타정보 반환.

    Args:
        furniture: 가구/빌트인 회전 사각형 [{cx,cy,w,h,angle,label}, ...] (정규화 0-1)
                   자율비행 충돌 회피용 SDF static box 로 변환됨.

    Returns:
        {
            "path": str, "world_name": str,
            "world_w": float, "world_d": float,
            "size_source": str,    # 'calibrated' | 'aspect' | 'fallback'
            "wall_count": int, "outline_count": int, "furniture_count": int,
        }
    """
    world_w, world_d, size_source = derive_world_size(
        image_width, image_height, scale_px_per_meter,
    )

    xml = build_world_xml(
        world_name=world_name,
        walls=walls or [],
        outline=outline or [],
        world_w=world_w,
        world_d=world_d,
        furniture=furniture or [],
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0" ?>\n')
        f.write(xml)

    return {
        "path": output_path,
        "world_name": world_name,
        "world_w": round(world_w, 4),
        "world_d": round(world_d, 4),
        "size_source": size_source,
        "wall_count": len(walls or []),
        "furniture_count": len(furniture or []),
        "outline_count": len(outline or []),
    }
