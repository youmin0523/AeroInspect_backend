# =============================================
# tests/test_geometric_gate.py
# 역할: GeometricGate 단위 테스트 (M4 Context 기반 검출 게이트)
# =============================================

import pytest

from app.services.geometric_gate import (
    GeometricGate,
    load_geometric_gate_from_config,
    _containment,
    _iou,
)


@pytest.fixture
def basic_gate():
    return GeometricGate(
        valid_context_map={
            "crack_structural": ["wall", "ceiling"],
            "wall_insulation_gap": ["wall"],
            "floor_stain": ["floor"],
        },
        containment_threshold=0.4,
    )


@pytest.fixture
def wall_context():
    return [{"class": "wall", "bbox_xyxy": [0, 0, 1000, 1000]}]


@pytest.fixture
def floor_context():
    return [{"class": "floor", "bbox_xyxy": [0, 800, 1000, 1000]}]


# ── 유틸 함수 테스트 ──

class TestUtils:
    def test_iou_complete_overlap(self):
        assert _iou([0, 0, 100, 100], [0, 0, 100, 100]) == pytest.approx(1.0)

    def test_iou_no_overlap(self):
        assert _iou([0, 0, 50, 50], [100, 100, 150, 150]) == 0.0

    def test_containment_full(self):
        # 작은 박스가 큰 박스 안에 100% 포함
        assert _containment([10, 10, 20, 20], [0, 0, 100, 100]) == pytest.approx(1.0)

    def test_containment_zero(self):
        assert _containment([0, 0, 50, 50], [100, 100, 150, 150]) == 0.0

    def test_containment_partial(self):
        # 검출의 절반이 컨텍스트 안
        c = _containment([0, 0, 100, 100], [50, 50, 150, 150])
        assert c == pytest.approx(0.25)  # 50x50 / 100x100 = 0.25


# ── 핵심 동작 ──

class TestGeometricGate:
    def test_pass_on_valid_context(self, basic_gate, wall_context):
        """wall 위에 있는 crack은 통과."""
        det = [{"class": "crack_structural", "conf": 0.7, "bbox_xyxy": [100, 100, 200, 200]}]
        out = basic_gate.filter(det, wall_context)
        assert len(out) == 1
        assert out[0]["gate_decision"] == "pass"
        assert out[0]["context_class"] == "wall"
        assert out[0]["context_containment"] == pytest.approx(1.0)

    def test_block_on_invalid_context(self, basic_gate, floor_context):
        """floor 위 crack은 차단 (crack은 wall/ceiling만 valid)."""
        det = [{"class": "crack_structural", "conf": 0.7, "bbox_xyxy": [100, 850, 200, 950]}]
        out = basic_gate.filter(det, floor_context)
        assert len(out) == 0

    def test_no_mapping_pass(self, basic_gate, wall_context):
        """매핑되지 않은 클래스는 보수적으로 통과."""
        det = [{"class": "unknown_defect_xyz", "conf": 0.5, "bbox_xyxy": [10, 10, 50, 50]}]
        out = basic_gate.filter(det, wall_context)
        assert len(out) == 1
        assert out[0]["gate_decision"] == "no_mapping_pass"

    def test_no_bbox_skip(self, basic_gate, wall_context):
        """bbox 없는 검출(분류기)는 게이트 적용 안 됨."""
        det = [{"class": "crack_structural", "conf": 0.6, "bbox_xyxy": None}]
        out = basic_gate.filter(det, wall_context)
        assert len(out) == 1
        assert out[0]["gate_decision"] == "no_bbox_skip"

    def test_fallback_pass_when_no_context(self, wall_context):
        """M4 Context 미사용 시 fallback=pass면 모두 통과."""
        gate = GeometricGate(
            valid_context_map={"crack_structural": ["wall"]},
            fallback="pass",
        )
        det = [{"class": "crack_structural", "conf": 0.5, "bbox_xyxy": [10, 10, 50, 50]}]
        out = gate.filter(det, None)
        assert len(out) == 1
        assert out[0]["gate_decision"] == "fallback_pass"

    def test_fallback_block_when_no_context(self):
        """fallback=block이면 모두 차단."""
        gate = GeometricGate(
            valid_context_map={"crack_structural": ["wall"]},
            fallback="block",
        )
        det = [{"class": "crack_structural", "conf": 0.5, "bbox_xyxy": [10, 10, 50, 50]}]
        out = gate.filter(det, None)
        assert len(out) == 0

    def test_weak_mode_reduces_conf(self, floor_context):
        """weak_mode=True면 차단 대신 conf 감소."""
        gate = GeometricGate(
            valid_context_map={"crack_structural": ["wall"]},
            weak_mode=True,
            weak_conf_penalty=0.5,
        )
        det = [{"class": "crack_structural", "conf": 0.8, "bbox_xyxy": [100, 850, 200, 950]}]
        out = gate.filter(det, floor_context)
        assert len(out) == 1
        assert out[0]["conf"] == pytest.approx(0.4)
        assert out[0]["gate_decision"] == "weak_pass"

    def test_empty_detections(self, basic_gate, wall_context):
        out = basic_gate.filter([], wall_context)
        assert out == []

    def test_multiple_contexts_pick_best(self, basic_gate):
        """여러 컨텍스트 중 가장 잘 맞는 거 선택."""
        contexts = [
            {"class": "wall", "bbox_xyxy": [0, 0, 100, 100]},     # det과 0% 겹침
            {"class": "ceiling", "bbox_xyxy": [200, 200, 400, 400]},  # det과 100% 겹침
        ]
        det = [{"class": "crack_structural", "conf": 0.7, "bbox_xyxy": [250, 250, 350, 350]}]
        out = basic_gate.filter(det, contexts)
        assert len(out) == 1
        assert out[0]["context_class"] == "ceiling"
        assert out[0]["context_containment"] == pytest.approx(1.0)


# ── config 로딩 ──

class TestConfigLoading:
    def test_disabled_config_passes_all(self):
        cfg = {"enabled": False}
        gate = load_geometric_gate_from_config(cfg)
        # 빈 매핑 → 모든 검출이 no_mapping_pass
        det = [{"class": "any_class", "conf": 0.5, "bbox_xyxy": [10, 10, 50, 50]}]
        out = gate.filter(det, [{"class": "wall", "bbox_xyxy": [0, 0, 100, 100]}])
        assert len(out) == 1

    def test_enabled_config(self):
        cfg = {
            "enabled": True,
            "valid_context": {"crack_structural": ["wall"]},
            "context_iou_threshold": 0.4,
            "fallback_when_unavailable": "pass",
        }
        gate = load_geometric_gate_from_config(cfg)
        assert gate.containment_threshold == 0.4
        assert "crack_structural" in gate.valid_context_map

    def test_strict_classes_loaded_from_config(self):
        cfg = {
            "enabled": True,
            "valid_context": {"glass_defect": ["window"]},
            "strict_classes": ["glass_defect"],
        }
        gate = load_geometric_gate_from_config(cfg)
        assert "glass_defect" in gate.strict_classes


# ── strict hard-block (물리적 불일치 — 유리=바닥 등) ──
# 회귀 가드: 2026-06-08 바닥 균열이 M3 raw 'glass_defect'로 E-01 노출된 사고.
# 게이트가 raw 클래스명을 매핑하지 못해 no_mapping_pass로 새던 문제 + weak penalty로는
# 고신뢰 오분류를 못 막던 문제를 strict hard block으로 차단한다.

class TestStrictHardBlock:
    @pytest.fixture
    def strict_gate(self):
        return GeometricGate(
            valid_context_map={"glass_defect": ["window"], "floor_defect": ["floor"]},
            containment_threshold=0.4,
            weak_mode=True,            # 일반 클래스는 weak, strict만 hard block
            weak_conf_penalty=0.6,
            strict_classes=["glass_defect", "floor_defect"],
        )

    def test_strict_block_glass_on_floor(self, strict_gate):
        """바닥 위 유리 결함 → 물리적 불가능 → 차단 (weak_mode라도)."""
        det = [{"class": "glass_defect", "conf": 1.0, "bbox_xyxy": [300, 140, 1060, 240]}]
        ctx = [{"class": "floor", "bbox_xyxy": [0, 0, 1485, 690]}]
        out = strict_gate.filter(det, ctx)
        assert len(out) == 0

    def test_strict_pass_glass_on_window(self, strict_gate):
        """창문 위 유리 결함 → 정상 통과 (진짜 하자 보존)."""
        det = [{"class": "glass_defect", "conf": 0.9, "bbox_xyxy": [300, 140, 400, 240]}]
        ctx = [{"class": "window", "bbox_xyxy": [200, 100, 500, 300]}]
        out = strict_gate.filter(det, ctx)
        assert len(out) == 1
        assert out[0]["gate_decision"] == "pass"

    def test_strict_no_block_when_context_missing(self, strict_gate):
        """M4가 유효 표면도 부적합 표면도 못 잡으면 차단 안 함 (놓침 방지)."""
        det = [{"class": "glass_defect", "conf": 0.9, "bbox_xyxy": [300, 140, 400, 240]}]
        # 검출과 안 겹치는 wall만 존재 → 부적합 표면 containment < threshold
        ctx = [{"class": "wall", "bbox_xyxy": [900, 900, 1000, 1000]}]
        out = strict_gate.filter(det, ctx)
        assert len(out) == 1  # weak_pass (차단 아님)

    def test_relabel_glass_on_floor_to_floor(self):
        """바닥 위 glass_defect → floor_defect 로 교정(차단 아님). 하자 보존 + 올바른 코드."""
        gate = GeometricGate(
            valid_context_map={"glass_defect": ["window"], "floor_defect": ["floor"]},
            containment_threshold=0.4,
            weak_mode=True,
            strict_classes=["glass_defect", "floor_defect"],
            relabel_group=["floor_defect", "glass_defect", "frame_defect"],
            surface_to_class={"floor": "floor_defect"},
        )
        det = [{"class": "glass_defect", "conf": 1.0, "bbox_xyxy": [300, 140, 1060, 240]}]
        ctx = [{"class": "floor", "bbox_xyxy": [0, 0, 1485, 690]}]
        out = gate.filter(det, ctx)
        assert len(out) == 1                              # 차단 아님 — 미탐 방지
        assert out[0]["class"] == "floor_defect"          # 표면 기반 교정
        assert out[0]["context_relabeled_from"] == "glass_defect"
        assert out[0]["gate_decision"] == "pass"          # 교정 후 floor 위라 valid

    def test_relabel_skips_window_ambiguous(self):
        """window 표면은 surface_to_class 에 없으므로 재라벨 안 함(유리·창틀 모호)."""
        gate = GeometricGate(
            valid_context_map={"floor_defect": ["floor"], "glass_defect": ["window"]},
            containment_threshold=0.4,
            weak_mode=True,
            relabel_group=["floor_defect", "glass_defect", "frame_defect"],
            surface_to_class={"floor": "floor_defect"},
        )
        det = [{"class": "floor_defect", "conf": 0.9, "bbox_xyxy": [250, 150, 450, 250]}]
        ctx = [{"class": "window", "bbox_xyxy": [200, 100, 500, 300]}]
        out = gate.filter(det, ctx)
        # window는 매핑 없음 → 재라벨 안 됨, floor_defect는 window valid 아님 → weak_pass
        assert all(d.get("context_relabeled_from") is None for d in out)

    def test_non_strict_class_still_weak(self, strict_gate):
        """strict 목록에 없는 클래스는 기존 weak_mode(conf 감소) 유지."""
        gate = GeometricGate(
            valid_context_map={"scratch": ["wall"], "glass_defect": ["window"]},
            weak_mode=True, weak_conf_penalty=0.6,
            strict_classes=["glass_defect"],
        )
        det = [{"class": "scratch", "conf": 1.0, "bbox_xyxy": [100, 850, 200, 950]}]
        ctx = [{"class": "floor", "bbox_xyxy": [0, 800, 1000, 1000]}]
        out = gate.filter(det, ctx)
        assert len(out) == 1
        assert out[0]["gate_decision"] == "weak_pass"
        assert out[0]["conf"] == pytest.approx(0.6)
