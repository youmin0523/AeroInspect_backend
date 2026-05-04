# =============================================
# app/services/furniture_gate.py
# 역할: furniture_aware (10-class) 기반 가구 위 false positive 차단
#       - 빌트인 가구(냉장고/싱크대/아일랜드/캐비닛 등) 위에서 발생한
#         하자 검출은 거의 100% 오탐 → 차단
#       - 단, 일부 클래스(열교/단열 등)는 가구 뒤·옆에서도 유효 → 예외
#
# 정책:
#   - postprocess_config.yaml의 furniture_gate 섹션
#   - 검출 bbox와 가구 bbox의 IoU/containment >= threshold면 가구 위 판정
#   - exempt_classes에 포함된 클래스는 게이트 무시 (열교 등)
#   - furniture_aware 미사용 시 graceful pass-through
# =============================================

from __future__ import annotations

from typing import List, Optional, Set


def _containment(detection_box: List[float], furniture_box: List[float]) -> float:
    """검출 박스가 가구 박스 안에 얼마나 포함되는가. inter / detection_area."""
    x1 = max(detection_box[0], furniture_box[0])
    y1 = max(detection_box[1], furniture_box[1])
    x2 = min(detection_box[2], furniture_box[2])
    y2 = min(detection_box[3], furniture_box[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    det_area = (detection_box[2] - detection_box[0]) * (detection_box[3] - detection_box[1])
    if det_area <= 0:
        return 0.0
    return inter / det_area


class FurnitureGate:
    """
    빌트인 가구 영역 위 검출 차단 게이트.

    Args:
        furniture_classes: 가구로 간주할 클래스 (예: ['cabinet_builtin', 'kitchen_appliance', ...])
        containment_threshold: 검출의 N% 이상이 가구 위에 있으면 차단 (기본 0.5)
        exempt_classes: 가구 위에 있어도 통과시킬 하자 클래스 (예: 열교 결함)
        weak_mode: True면 차단 안 하고 conf 감소만
        weak_conf_penalty: weak_mode 시 곱할 페널티 (기본 0.4)
    """

    def __init__(
        self,
        furniture_classes: Optional[List[str]] = None,
        containment_threshold: float = 0.5,
        exempt_classes: Optional[List[str]] = None,
        weak_mode: bool = False,
        weak_conf_penalty: float = 0.4,
    ):
        self.furniture_classes: Set[str] = set(furniture_classes or [
            "cabinet_builtin", "kitchen_appliance",
            "countertop_sink", "kitchen_island", "shelf",
        ])
        self.containment_threshold = containment_threshold
        self.exempt_classes: Set[str] = set(exempt_classes or [])
        self.weak_mode = weak_mode
        self.weak_conf_penalty = weak_conf_penalty

    def filter(
        self,
        detections: List[dict],
        furniture_detections: Optional[List[dict]] = None,
    ) -> List[dict]:
        """
        가구 위 검출 차단.

        Args:
            detections: 하자 검출 [{class, conf, bbox_xyxy, ...}, ...]
            furniture_detections: furniture_aware 결과
                                  [{class: 'cabinet_builtin'|..., bbox_xyxy: [...]}, ...]
                                  None이면 graceful pass.

        Returns:
            필터된 검출. 통과/차단 결정에 'furniture_gate_decision' 필드 추가.
        """
        if not detections:
            return []
        if not furniture_detections:
            return [{**d, "furniture_gate_decision": "no_furniture_data_pass"} for d in detections]

        # 가구 박스만 추출
        furniture_boxes: List[List[float]] = []
        for f in furniture_detections:
            cls = f.get("class")
            bbox = f.get("bbox_xyxy")
            if cls in self.furniture_classes and bbox:
                furniture_boxes.append(bbox)

        if not furniture_boxes:
            # 가구 검출 자체 없음 (정상 환경) → 모두 통과
            return [{**d, "furniture_gate_decision": "no_furniture_in_scene"} for d in detections]

        kept: List[dict] = []
        for det in detections:
            det_class = det.get("class")
            det_bbox = det.get("bbox_xyxy")

            # bbox 없는 검출(분류만)은 게이트 적용 어려움 → 통과
            if not det_bbox:
                kept.append({**det, "furniture_gate_decision": "no_bbox_skip"})
                continue

            # 면제 클래스 (예: 창호 단열 — 빌트인 가전 옆에서 결로 발생 가능)
            if det_class in self.exempt_classes:
                kept.append({**det, "furniture_gate_decision": "exempt"})
                continue

            # 가구 위 판정
            max_containment = 0.0
            for fb in furniture_boxes:
                c = _containment(det_bbox, fb)
                if c > max_containment:
                    max_containment = c

            if max_containment >= self.containment_threshold:
                # 가구 위 검출
                if self.weak_mode:
                    kept.append({
                        **det,
                        "conf": det["conf"] * self.weak_conf_penalty,
                        "furniture_gate_decision": "weak_block",
                        "furniture_containment": round(max_containment, 4),
                    })
                # 강한 모드: 완전 차단 (kept에 추가 안 함)
            else:
                kept.append({
                    **det,
                    "furniture_gate_decision": "pass",
                    "furniture_containment": round(max_containment, 4),
                })

        return kept


def load_furniture_gate_from_config(config: dict) -> FurnitureGate:
    """postprocess_config.yaml의 furniture_gate 섹션에서 인스턴스 생성.

    NOTE: furniture_aware mAP 0.38로 부정확 → weak_mode=True로 전환.
    잘못된 가구 인식으로 정상 검출 차단 위험 줄임.
    """
    enabled = config.get("enabled", False)
    if not enabled:
        # 비활성화 — 빈 furniture_classes로 모두 통과
        return FurnitureGate(furniture_classes=[], containment_threshold=1.1)

    return FurnitureGate(
        furniture_classes=config.get("furniture_classes", []),
        containment_threshold=config.get("iou_with_furniture_threshold", 0.5),
        exempt_classes=config.get("exempt_classes", []),
        weak_mode=True,                # 부정확한 가구 인식 → conf 감소만 (차단 X)
        weak_conf_penalty=0.6,         # 가구 위 검출은 conf 60%로 (recall 보존)
    )


__all__ = ["FurnitureGate", "load_furniture_gate_from_config"]
