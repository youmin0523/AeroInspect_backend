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
        strict_classes: weak_mode 여부와 무관하게 맥락 불일치 시 hard block 하는
                        클래스 집합. 물리적으로 불가능한 조합(유리=바닥 등)용.
                        매핑은 있으나 context 결과가 비어 판단 불가일 때는 차단하지 않음
                        (M4 미검출을 불일치로 오인하면 정상 하자를 놓침).
        relabel_group: 표면이 유일한 구분자라 상호 혼동되는 클래스 집합
                       (예: {floor_defect, glass_defect, frame_defect}). 이 집합의 검출은
                       gate filter 전에 surface_to_class 로 표면 기반 재라벨 후보가 된다.
        surface_to_class: {context_surface: 정답 class}. 모호하지 않은 표면만 등록한다.
                          (floor 는 바닥 결함만 → 안전. window 는 유리·창틀 공존 → 제외.)
                          relabel_group 검출이 이 표면 위(containment>=threshold)에 있고
                          그 표면의 정답 class 가 현재 class 와 다르면 재라벨 → 차단(미탐)이
                          아니라 올바른 코드로 교정한다. (2026-06-08 바닥균열→E-01 사고 대응)
    """

    def __init__(
        self,
        valid_context_map: Optional[Dict[str, List[str]]] = None,
        containment_threshold: float = 0.4,
        fallback: str = "pass",
        weak_mode: bool = False,
        weak_conf_penalty: float = 0.5,
        strict_classes: Optional[List[str]] = None,
        relabel_group: Optional[List[str]] = None,
        surface_to_class: Optional[Dict[str, str]] = None,
    ):
        self.valid_context_map = valid_context_map or {}
        self.containment_threshold = containment_threshold
        self.fallback = fallback
        self.weak_mode = weak_mode
        self.weak_conf_penalty = weak_conf_penalty
        self.strict_classes = frozenset(strict_classes or ())
        self.relabel_group = frozenset(relabel_group or ())
        self.surface_to_class = surface_to_class or {}

    def _maybe_relabel(
        self,
        det: dict,
        contexts_by_class: Dict[str, List[List[float]]],
    ) -> dict:
        """표면이 유일한 구분자인 혼동 클래스를 M4 표면 기반으로 교정.

        det['class'] 가 relabel_group 에 속하고, 검출이 surface_to_class 에 등록된
        표면 위(containment>=threshold)에 충분히 올라가 있으며, 그 표면의 정답 class 가
        현재와 다르면 class 를 교정한다. 그 외에는 원본 그대로 반환.
        """
        det_class = det.get("class")
        det_bbox = det.get("bbox_xyxy")
        if det_class not in self.relabel_group or not det_bbox or not self.surface_to_class:
            return det
        best_c = 0.0
        best_target = None
        for surface, target_class in self.surface_to_class.items():
            for ctx_bbox in contexts_by_class.get(surface, []):
                c = _containment(det_bbox, ctx_bbox)
                if c > best_c:
                    best_c = c
                    best_target = target_class
        if best_target and best_target != det_class and best_c >= self.containment_threshold:
            print(
                f"[GeometricGate] context relabel: {det_class} -> {best_target} "
                f"(containment={best_c:.2f})"
            )
            return {**det, "class": best_target, "context_relabeled_from": det_class}
        return det

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
            # 표면 기반 재라벨(혼동 클래스) — 차단 전에 먼저 교정 시도.
            # 교정되면 아래 valid_context 검사를 새 class 로 통과한다.
            det = self._maybe_relabel(det, contexts_by_class)
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
                continue

            # ── strict hard block 판정 ──
            # 유효 표면 위에 없을 때, "다른(부적합) 표면 위에 확실히 올라가 있는가"를
            # 따로 본다. 부적합 표면에 충분히 포함되면 물리적 불일치(유리=바닥)로 확정 →
            # weak_mode 라도 차단. M4가 유효 표면을 단순히 못 잡은 경우(부적합 표면도 없음)는
            # 차단하지 않아 정상 하자를 놓치지 않는다.
            if det_class in self.strict_classes:
                wrong_containment = 0.0
                wrong_ctx_class = None
                for ctx_class, boxes in contexts_by_class.items():
                    if ctx_class in valid_ctx_classes:
                        continue
                    for ctx_bbox in boxes:
                        c = _containment(det_bbox, ctx_bbox)
                        if c > wrong_containment:
                            wrong_containment = c
                            wrong_ctx_class = ctx_class
                if wrong_containment >= self.containment_threshold:
                    # 차단 — kept에 추가 안 함. (진단용으로 흔적만 남기지 않음: 노출 차단)
                    print(
                        f"[GeometricGate] strict block: {det_class} on "
                        f"{wrong_ctx_class}(containment={wrong_containment:.2f}) "
                        f"valid={valid_ctx_classes}"
                    )
                    continue

            if self.weak_mode:
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
    strict_classes = config.get("strict_classes", [])
    relabel_cfg = config.get("context_relabel", {}) or {}
    relabel_group = relabel_cfg.get("group", [])
    surface_to_class = relabel_cfg.get("surface_to_class", {})

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
        strict_classes=strict_classes, # 물리적 불일치(유리=바닥 등)는 weak 대신 hard block
        relabel_group=relabel_group,   # 표면이 유일한 구분자인 혼동 클래스(floor/glass/frame)
        surface_to_class=surface_to_class,  # 모호하지 않은 표면→정답 class (floor→floor_defect)
    )


__all__ = ["GeometricGate", "load_geometric_gate_from_config"]
