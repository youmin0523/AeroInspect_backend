# =============================================
# tests/test_furniture_gate.py
# 역할: FurnitureGate 단위 테스트 (가구 위 검출 차단)
# =============================================

import pytest

from app.services.furniture_gate import FurnitureGate, load_furniture_gate_from_config


@pytest.fixture
def basic_gate():
    return FurnitureGate(
        furniture_classes=["cabinet_builtin", "kitchen_appliance"],
        containment_threshold=0.5,
    )


@pytest.fixture
def gate_with_exempt():
    return FurnitureGate(
        furniture_classes=["cabinet_builtin"],
        exempt_classes=["window_insulation_defect", "wall_insulation_gap"],
        containment_threshold=0.5,
    )


@pytest.fixture
def cabinet_furniture():
    return [{"class": "cabinet_builtin", "bbox_xyxy": [50, 50, 250, 250]}]


@pytest.fixture
def appliance_furniture():
    return [{"class": "kitchen_appliance", "bbox_xyxy": [300, 300, 500, 500]}]


# ── 기본 동작 ──

class TestFurnitureGate:
    def test_block_detection_on_furniture(self, basic_gate, cabinet_furniture):
        """가구 위 검출은 차단."""
        det = [{"class": "wallpaper_seam", "conf": 0.7, "bbox_xyxy": [100, 100, 200, 200]}]
        out = basic_gate.filter(det, cabinet_furniture)
        assert len(out) == 0

    def test_pass_detection_off_furniture(self, basic_gate, cabinet_furniture):
        """가구 밖 검출은 통과."""
        det = [{"class": "wallpaper_seam", "conf": 0.7, "bbox_xyxy": [400, 400, 500, 500]}]
        out = basic_gate.filter(det, cabinet_furniture)
        assert len(out) == 1
        assert out[0]["furniture_gate_decision"] == "pass"
        assert out[0]["furniture_containment"] == 0.0

    def test_exempt_class_passes_on_furniture(self, gate_with_exempt, cabinet_furniture):
        """면제 클래스는 가구 위에 있어도 통과 (열교/단열 등)."""
        det = [{"class": "window_insulation_defect", "conf": 0.7, "bbox_xyxy": [100, 100, 200, 200]}]
        out = gate_with_exempt.filter(det, cabinet_furniture)
        assert len(out) == 1
        assert out[0]["furniture_gate_decision"] == "exempt"

    def test_no_furniture_data_passes_all(self, basic_gate):
        """furniture_aware 결과 없을 때 모두 통과 (graceful degradation)."""
        det = [{"class": "crack", "conf": 0.6, "bbox_xyxy": [100, 100, 200, 200]}]
        out = basic_gate.filter(det, None)
        assert len(out) == 1
        assert out[0]["furniture_gate_decision"] == "no_furniture_data_pass"

    def test_empty_furniture_scene(self, basic_gate):
        """furniture_aware 결과 있지만 가구 클래스 0개 (빈 방)."""
        det = [{"class": "crack", "conf": 0.6, "bbox_xyxy": [100, 100, 200, 200]}]
        # wall만 검출됨 (가구 아님)
        non_furniture = [{"class": "wall", "bbox_xyxy": [0, 0, 1000, 1000]}]
        out = basic_gate.filter(det, non_furniture)
        assert len(out) == 1
        assert out[0]["furniture_gate_decision"] == "no_furniture_in_scene"

    def test_no_bbox_skip(self, basic_gate, cabinet_furniture):
        """bbox 없는 검출은 게이트 적용 안 됨."""
        det = [{"class": "wallpaper_seam", "conf": 0.5, "bbox_xyxy": None}]
        out = basic_gate.filter(det, cabinet_furniture)
        assert len(out) == 1
        assert out[0]["furniture_gate_decision"] == "no_bbox_skip"

    def test_partial_overlap_below_threshold(self, basic_gate, cabinet_furniture):
        """가구와 30%만 겹치는 검출은 통과 (threshold 0.5 미만)."""
        # det 100x100, cabinet [50,50,250,250]에 30%만 들어가도록
        det = [{"class": "crack", "conf": 0.6, "bbox_xyxy": [220, 220, 320, 320]}]
        # 검출 100x100, 겹치는 영역 30x30 = 900. 검출 면적 10000. containment = 0.09
        out = basic_gate.filter(det, cabinet_furniture)
        assert len(out) == 1
        assert out[0]["furniture_gate_decision"] == "pass"

    def test_weak_mode_penalizes_conf(self, cabinet_furniture):
        """weak_mode=True면 차단 안 하고 conf 감소."""
        gate = FurnitureGate(
            furniture_classes=["cabinet_builtin"],
            containment_threshold=0.5,
            weak_mode=True,
            weak_conf_penalty=0.4,
        )
        det = [{"class": "crack", "conf": 0.8, "bbox_xyxy": [100, 100, 200, 200]}]
        out = gate.filter(det, cabinet_furniture)
        assert len(out) == 1
        assert out[0]["conf"] == pytest.approx(0.32)
        assert out[0]["furniture_gate_decision"] == "weak_block"

    def test_multiple_furniture_pieces(self, basic_gate):
        """여러 가구 중 하나라도 충분히 겹치면 차단."""
        furniture = [
            {"class": "cabinet_builtin", "bbox_xyxy": [0, 0, 100, 100]},
            {"class": "kitchen_appliance", "bbox_xyxy": [200, 200, 300, 300]},
        ]
        # 두 번째 가구와 100% 겹침
        det = [{"class": "crack", "conf": 0.6, "bbox_xyxy": [220, 220, 280, 280]}]
        out = basic_gate.filter(det, furniture)
        assert len(out) == 0

    def test_empty_detections(self, basic_gate, cabinet_furniture):
        out = basic_gate.filter([], cabinet_furniture)
        assert out == []


# ── config 로딩 ──

class TestConfigLoading:
    def test_disabled_passes_all(self):
        gate = load_furniture_gate_from_config({"enabled": False})
        # furniture_classes 비어있어 매칭 안 됨 → 통과
        det = [{"class": "crack", "conf": 0.5, "bbox_xyxy": [10, 10, 50, 50]}]
        out = gate.filter(det, [{"class": "cabinet_builtin", "bbox_xyxy": [0, 0, 100, 100]}])
        assert len(out) == 1

    def test_enabled_with_full_config(self):
        cfg = {
            "enabled": True,
            "furniture_classes": ["cabinet_builtin", "shelf"],
            "iou_with_furniture_threshold": 0.6,
            "exempt_classes": ["wall_insulation_gap"],
        }
        gate = load_furniture_gate_from_config(cfg)
        assert gate.containment_threshold == 0.6
        assert "cabinet_builtin" in gate.furniture_classes
        assert "wall_insulation_gap" in gate.exempt_classes
