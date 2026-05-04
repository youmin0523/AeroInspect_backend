# =============================================
# app/services/geometric_gate.py
# 역할: M4 Context (wall/ceiling/floor/window/door) 기반 검출 게이트
#       - 하자 검출이 적합한 표면 위에 있는지 검증
#       - 부적합 표면(예: 창호 결함이 floor에 있다면 부적합) → 차단 또는 신뢰도 감소
#       - M4 Context 미로드/미실행 시 graceful degradation (pass 또는 block)
#
# 정책:
#   - postprocess_config.yaml의 geometric_gate 섹션에서 valid_context 매핑 로드
#   - 검출 bbox와 context bbox/mask의 IoU >= threshold면 통과
#   - context 정보 없으면 fallback 정책 적용
# =============================================

from __future__ import annotations

from typing import Dict, List, Optional


def _iou(box_a: List[float], box_b: List[float]) -> float:
    """xyxy bbox IoU."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    aa = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    bb = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    return inter / (aa + bb - inter + 1e-9)


def _containment(detection_box: List[float], context_box: List[float]) -> float:
    """
    검출 박스가 컨텍스트 박스 안에 얼마나 포함되는가.
    inter / detection_area — 1.0이면 검출이 100% 컨텍스트 안에 있음.

    IoU 대신 사용하는 이유: 컨텍스트(예: wall) 영역이 검출(예: 작은 균열)보다
    훨씬 클 때 IoU는 작게 나옴. containment가 더 적합.
    """
    x1 = max(detection_box[0], context_box[0])
    y1 = max(detection_box[1], context_box[1])
    x2 = min(detection_box[2], context_box[2])
    y2 = min(detection_box[3], context_box[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    det_area = (detection_box[2] - detection_box[0]) * (detection_box[3] - detection_box[1])
    if det_area <= 0:
        return 0.0
    return inter / det_area


class GeometricGate:
    """
    M4 Context 기반 기하학적 게이트.

    Args:
        valid_context_map: {defect_class: [valid_context_class, ...]}
        containment_threshold: 검출의 N% 이상이 컨텍스트 위에 있어야 통과 (기본 0.4)
        fallback: M4 Context 미사용 시 동작 — "pass" | "block"
        weak_mode: True면 부적합 시 conf 감소만, False면 차단
    """

    def __init__(
        self,
        valid_context_map: Optional[Dict[str, List[str]]] = None,
        containment_threshold: float = 0.4,
        fallback: str = "pass",
        weak_mode: bool = False,
        weak_conf_penalty: float = 0.5,
    ):
        self.valid_context_map = valid_context_map or {}
        self.containment_threshold = containment_threshold
        self.fallback = fallback
        self.weak_mode = weak_mode
        self.weak_conf_penalty = weak_conf_penalty

    def filter(
        self,
        detections: List[dict],
        context_detections: Optional[List[dict]] = None,
    ) -> List[dict]:
        """
        하자 검출을 컨텍스트 게이트로 필터링.

        Args:
            detections: 하자 검출 리스트 [{class, conf, bbox_xyxy, ...}, ...]
            context_detections: M4 Context 결과
                                [{class: 'wall'|'ceiling'|'floor'|'window'|'door',
                                  bbox_xyxy: [x1,y1,x2,y2]}, ...]
                                None이면 fallback 정책 적용.

        Returns:
            필터링된 검출 리스트. weak_mode 시 차단 안 하고 conf 감소만.
            각 통과 검출에는 'gate_decision' = 'pass'|'fallback'|'weak_pass' 필드 추가.
        """
        if not detections:
            return []

        # M4 Context 결과 없음 → fallback
        if not context_detections:
            return self._apply_fallback(detections)

        # 컨텍스트별 그룹화 (효율 위해)
        contexts_by_class: Dict[str, List[List[float]]] = {}
        for c in context_detections:
            cls = c.get("class")
            bbox = c.get("bbox_xyxy")
            if cls and bbox:
                contexts_by_class.setdefault(cls, []).append(bbox)

        kept: List[dict] = []
        for det in detections:
            det_class = det.get("class")
            det_bbox = det.get("bbox_xyxy")
            if not det_class or not det_bbox:
                # bbox 없는 검출(분류만)은 게이트 통과 (분류기는 컨텍스트 게이트 적용 어려움)
                kept.append({**det, "gate_decision": "no_bbox_skip"})
                continue

            valid_ctx_classes = self.valid_context_map.get(det_class)
            if not valid_ctx_classes:
                # 매핑 없는 클래스는 게이트 통과 (보수적)
                kept.append({**det, "gate_decision": "no_mapping_pass"})
                continue

            # 유효 컨텍스트 클래스 중 하나라도 충분히 겹치면 통과
            best_containment = 0.0
            best_ctx_class = None
            for ctx_class in valid_ctx_classes:
                for ctx_bbox in contexts_by_class.get(ctx_class, []):
                    c = _containment(det_bbox, ctx_bbox)
                    if c > best_containment:
                        best_containment = c
                        best_ctx_class = ctx_class

            if best_containment >= self.containment_threshold:
                kept.append({
                    **det,
                    "gate_decision": "pass",
                    "context_class": best_ctx_class,
                    "context_containment": round(best_containment, 4),
                })
            elif self.weak_mode:
                # 약한 모드: conf 감소시켜 통과
                kept.append({
                    **det,
                    "conf": det["conf"] * self.weak_conf_penalty,
                    "gate_decision": "weak_pass",
                    "context_class": best_ctx_class,
                    "context_containment": round(best_containment, 4),
                })
            # else: 차단 (kept에 추가 안 함)

        return kept

    def _apply_fallback(self, detections: List[dict]) -> List[dict]:
        """M4 Context 결과 없을 때 fallback 정책 적용."""
        if self.fallback == "block":
            return []
        # "pass"
        return [{**d, "gate_decision": "fallback_pass"} for d in detections]


def load_geometric_gate_from_config(config: dict) -> GeometricGate:
    """
    postprocess_config.yaml의 geometric_gate 섹션에서 GeometricGate 인스턴스 생성.

    Args:
        config: yaml에서 로드한 dict 중 geometric_gate 섹션
    """
    enabled = config.get("enabled", False)
    valid_context = config.get("valid_context", {})
    threshold = config.get("context_iou_threshold", 0.4)  # yaml 키명은 IoU지만 containment로 사용
    fallback = config.get("fallback_when_unavailable", "pass")

    if not enabled:
        # 비활성화 — 모든 검출 그대로 통과
        return GeometricGate(
            valid_context_map={},  # 빈 매핑 → no_mapping_pass로 모두 통과
            containment_threshold=threshold,
            fallback="pass",
            weak_mode=False,
        )

    return GeometricGate(
        valid_context_map=valid_context,
        containment_threshold=threshold,
        fallback=fallback,
        weak_mode=True,                # M4 Context mAP 0.55 — strict 차단 위험, weak로 conf 감소만
        weak_conf_penalty=0.6,         # 기본 0.5보다 약하게 (recall 보존)
    )


__all__ = ["GeometricGate", "load_geometric_gate_from_config"]
