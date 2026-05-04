# =============================================
# dry_run_full_pipeline.py
# 역할: 전 시스템 dry-run 검증
#       - 모든 ONNX 모델 로딩 가능 여부
#       - 후처리 모듈 (geometric_gate, furniture_gate, tta, temporal_filter, tracker, ensemble) import 가능
#       - 샘플 이미지 한 장으로 end-to-end 추론 실행
#       - 결과 dict 구조 검증
#
# 출력: PASS/FAIL 리포트. 하나라도 FAIL이면 exit 1.
#
# 사용:
#   cd backend/training
#   python eval/dry_run_full_pipeline.py
# =============================================

from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path
from typing import Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
BACKEND = ROOT / "backend"
sys.path.insert(0, str(BACKEND))

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


PASS = "[PASS]"
FAIL = "[FAIL]"


def log(level: str, msg: str):
    print(f"{level} {msg}")


# ─────────────────────────────────────────────
# 1) 모델 가중치 존재 여부
# ─────────────────────────────────────────────

REQUIRED_MODELS = [
    ("M1_YOLO", "m1_yolo_structural.onnx"),
    ("M1_RESNET", "m1_resnet_crack_classifier.onnx"),
    ("M2_YOLO", "m2_yolo_surface.onnx"),
    ("M2_RESNET", "m2_resnet_surface_classifier.onnx"),
    ("M3_YOLO", "m3_yolo_floor_window.onnx"),
    ("M3_RESNET", "m3_resnet_floor_window_classifier.onnx"),
    ("M4_UNET", "m4_unet_thermal_insulation.onnx"),
    ("M5_SEG", "m5_yolo_seg_frames.onnx"),
    ("M6_PATCHCORE", "m6_patchcore_feature_extractor.onnx"),
]
OPTIONAL_MODELS = [
    ("M4_CONTEXT", "m4_yolo_context_elements.onnx"),  # M4v2 학습 중
    ("FURNITURE_AWARE", "furniture_aware.onnx"),       # Colab D 학습 중
]


def check_model_files(weights_dir: Path) -> Tuple[int, int]:
    """가중치 파일 존재 여부 검증."""
    print("\n=== [1] 모델 가중치 파일 ===")
    passed = failed = 0
    for key, fname in REQUIRED_MODELS:
        p = weights_dir / fname
        if p.exists():
            size_mb = p.stat().st_size / (1024 * 1024)
            log(PASS, f"{key}: {fname} ({size_mb:.1f}MB)")
            passed += 1
        else:
            log(FAIL, f"{key}: {fname} 없음")
            failed += 1
    for key, fname in OPTIONAL_MODELS:
        p = weights_dir / fname
        if p.exists():
            size_mb = p.stat().st_size / (1024 * 1024)
            log(PASS, f"{key} (optional): {fname} ({size_mb:.1f}MB)")
            passed += 1
        else:
            log("[SKIP]", f"{key} (optional): 학습 중, 미존재")
    return passed, failed


# ─────────────────────────────────────────────
# 2) Python 모듈 import
# ─────────────────────────────────────────────

REQUIRED_MODULES = [
    "app.services.temporal_filter",
    "app.services.object_tracker",
    "app.services.ensemble",
    "app.services.geometric_gate",
    "app.services.furniture_gate",
    "app.services.tta",
    "app.services.tiled_inference",
    "app.services.inference_pipeline_20",
    "app.services.defect_taxonomy",
    "app.services.alignment_detector",
    "app.services.insulation_detector",
    "app.services.onnx_inference",
]


def check_module_imports() -> Tuple[int, int]:
    """필수 모듈 import 가능 여부."""
    print("\n=== [2] Python 모듈 import ===")
    passed = failed = 0
    for mod in REQUIRED_MODULES:
        try:
            __import__(mod)
            log(PASS, f"{mod}")
            passed += 1
        except Exception as e:
            log(FAIL, f"{mod}: {type(e).__name__}: {e}")
            failed += 1
    return passed, failed


# ─────────────────────────────────────────────
# 3) 후처리 모듈 단위 동작
# ─────────────────────────────────────────────

def check_postprocess_units() -> Tuple[int, int]:
    """각 후처리 모듈 단위 동작 검증."""
    print("\n=== [3] 후처리 모듈 단위 동작 ===")
    passed = failed = 0

    # TemporalFilter
    try:
        from app.services.temporal_filter import TemporalFilter
        tf = TemporalFilter(window_size=3, min_detections=2, instant_threshold=0.85)
        det = [{"class": "crack", "conf": 0.95, "bbox_xyxy": [10, 10, 50, 50]}]
        out = tf.update(det, frame_id=1)
        assert len(out) == 1, "Instant report failed"
        log(PASS, "TemporalFilter: instant report 동작")
        passed += 1
    except Exception as e:
        log(FAIL, f"TemporalFilter: {e}")
        failed += 1

    # ObjectTracker
    try:
        from app.services.object_tracker import DefectTracker
        dt = DefectTracker(min_hits=2, max_age=5, iou_threshold=0.3)
        det = [{"class": "crack", "conf": 0.7, "bbox_xyxy": [10, 10, 50, 50]}]
        dt.update(det, frame_id=1)
        out2 = dt.update(det, frame_id=2)
        assert len(out2) == 1 and out2[0]["track_id"] == 1, f"Tracker confirm failed: {out2}"
        log(PASS, "DefectTracker: track_id 부여 + 확정")
        passed += 1
    except Exception as e:
        log(FAIL, f"DefectTracker: {e}")
        failed += 1

    # GeometricGate
    try:
        from app.services.geometric_gate import GeometricGate
        gate = GeometricGate(
            valid_context_map={"crack_structural": ["wall"]},
            containment_threshold=0.4,
        )
        det = [{"class": "crack_structural", "conf": 0.7, "bbox_xyxy": [100, 100, 150, 150]}]
        ctx = [{"class": "wall", "bbox_xyxy": [50, 50, 500, 500]}]
        out = gate.filter(det, ctx)
        assert len(out) == 1 and out[0]["gate_decision"] == "pass"
        log(PASS, "GeometricGate: 적합 컨텍스트 통과")
        passed += 1
    except Exception as e:
        log(FAIL, f"GeometricGate: {e}")
        failed += 1

    # FurnitureGate
    try:
        from app.services.furniture_gate import FurnitureGate
        gate = FurnitureGate(furniture_classes=["cabinet_builtin"], containment_threshold=0.5)
        det = [{"class": "crack", "conf": 0.6, "bbox_xyxy": [100, 100, 150, 150]}]
        furn = [{"class": "cabinet_builtin", "bbox_xyxy": [50, 50, 200, 200]}]
        out = gate.filter(det, furn)
        assert len(out) == 0, "Furniture gate should block"
        log(PASS, "FurnitureGate: 가구 위 검출 차단")
        passed += 1
    except Exception as e:
        log(FAIL, f"FurnitureGate: {e}")
        failed += 1

    # TTA
    try:
        from app.services.tta import TTAEnsemble
        tta = TTAEnsemble(augmentations=["horizontal_flip"], merge_method="wbf")
        def fake(_img):
            return [{"class": "crack", "conf": 0.5, "bbox_xyxy": [40.0, 40.0, 60.0, 60.0]}]
        img = np.ones((100, 100, 3), dtype=np.uint8)
        out = tta.predict(fake, img)
        assert len(out) >= 1
        log(PASS, "TTAEnsemble: hflip + WBF 동작")
        passed += 1
    except Exception as e:
        log(FAIL, f"TTAEnsemble: {e}")
        failed += 1

    # Ensemble
    try:
        from app.services.ensemble import (
            cross_model_nms, compute_combined_confidence,
        )
        det = [
            {"class": "crack", "conf": 0.7, "bbox_xyxy": [10, 10, 50, 50], "defect_source": "yolo_structural"},
            {"class": "crack", "conf": 0.6, "bbox_xyxy": [12, 12, 48, 48], "defect_source": "yolo_structural"},
        ]
        out = cross_model_nms(det)
        assert len(out) == 1 and out[0]["conf"] == 0.7
        c = compute_combined_confidence(0.5, 0.6)
        assert abs(c - 0.8) < 1e-9
        log(PASS, "Ensemble: cross_model_nms + combined_confidence")
        passed += 1
    except Exception as e:
        log(FAIL, f"Ensemble: {e}")
        failed += 1

    return passed, failed


# ─────────────────────────────────────────────
# 4) 추론 파이프라인 로딩
# ─────────────────────────────────────────────

def check_pipeline_load() -> Tuple[int, int]:
    """InferencePipeline20 로딩 가능 여부 (실제 ONNX 로딩 시도).

    NOTE: settings.AEROINSPECT_WEIGHTS_DIR은 './models_weights' 상대경로이므로
    cwd가 backend/이어야 정상 동작. dry-run을 backend/training/에서 실행하면
    cwd 임시 변경이 필요.
    """
    print("\n=== [4] InferencePipeline20 로딩 ===")
    import os
    orig_cwd = os.getcwd()
    backend_dir = ROOT / "backend"
    try:
        os.chdir(backend_dir)
        from app.services.inference_pipeline_20 import InferencePipeline20
        pipe = InferencePipeline20()
        pipe.load_models()  # 가용 모델 로드 시도
        loaded_status = pipe.models_loaded
        n_loaded = sum([
            loaded_status.m1_yolo, loaded_status.m1_resnet,
            loaded_status.m2_yolo, loaded_status.m2_resnet,
            loaded_status.m3_yolo, loaded_status.m3_resnet,
            loaded_status.m4_unet, loaded_status.m5_seg,
            loaded_status.m6_patchcore,
        ])
        if pipe.is_loaded:
            log(PASS, f"InferencePipeline20: 로드 완료 ({n_loaded}/9 모델)")
            return 1, 0
        else:
            log(FAIL, f"InferencePipeline20: critical 모델 미로드 ({n_loaded}/9)")
            return 0, 1
    except Exception as e:
        log(FAIL, f"InferencePipeline20: {type(e).__name__}: {e}")
        traceback.print_exc()
        return 0, 1
    finally:
        os.chdir(orig_cwd)


# ─────────────────────────────────────────────
# 5) 샘플 이미지 end-to-end 추론
# ─────────────────────────────────────────────

def check_sample_inference(orig_cwd: Path) -> Tuple[int, int]:
    """샘플 이미지 1장으로 end-to-end 추론 실행. cwd를 backend/로 임시 변경."""
    print("\n=== [5] 샘플 이미지 end-to-end 추론 ===")
    import os
    saved_cwd = os.getcwd()
    backend_dir = ROOT / "backend"
    try:
        # 1) 데이터셋은 backend/training/datasets/* 에 있음 — orig_cwd 기준으로 찾기
        sample_img_path = None
        for ds in ["structural", "surface", "floor_window"]:
            img_dir = orig_cwd / "datasets" / ds / "images" / "test"
            if img_dir.exists():
                imgs = [f for f in img_dir.iterdir() if f.suffix.lower() in {".jpg", ".png", ".jpeg"}]
                if imgs:
                    sample_img_path = imgs[0]
                    break
        if sample_img_path is None:
            log(FAIL, "샘플 이미지 없음 (datasets/{structural,surface,floor_window}/images/test 비어있음)")
            return 0, 1

        import cv2
        img = cv2.imread(str(sample_img_path))
        if img is None:
            log(FAIL, f"이미지 로드 실패: {sample_img_path}")
            return 0, 1
        log("[INFO]", f"샘플: {sample_img_path.name} {img.shape}")

        # 2) Pipeline 로딩 + 추론은 cwd=backend/ 에서
        os.chdir(backend_dir)
        from app.services.inference_pipeline_20 import InferencePipeline20
        pipe = InferencePipeline20()
        pipe.load_models()

        if not pipe.is_loaded:
            log(FAIL, "Pipeline 미로드 — 추론 불가")
            return 0, 1

        t0 = time.time()
        result = pipe.detect(img, tier=1)
        elapsed = time.time() - t0

        # 결과 구조 검증
        assert hasattr(result, "detections"), "DetectionResult20.detections 없음"
        assert hasattr(result, "image_shape"), "image_shape 없음"
        n_dets = len(result.detections)
        log(PASS, f"Tier 1 추론: {n_dets} detections, {elapsed*1000:.0f}ms")

        # Tier 2 추론
        t0 = time.time()
        result2 = pipe.detect(img, tier=2)
        elapsed2 = time.time() - t0
        n_dets2 = len(result2.detections)
        log(PASS, f"Tier 2 추론: {n_dets2} detections, {elapsed2*1000:.0f}ms")

        return 2, 0
    except Exception as e:
        log(FAIL, f"Sample inference: {type(e).__name__}: {e}")
        traceback.print_exc()
        return 0, 1
    finally:
        os.chdir(saved_cwd)


# ─────────────────────────────────────────────
# 6) Config 로딩
# ─────────────────────────────────────────────

def check_postprocess_config(_cwd: Path) -> Tuple[int, int]:
    """postprocess_config.yaml 로딩 가능 여부."""
    print("\n=== [6] postprocess_config.yaml 로딩 ===")
    try:
        cfg_path = ROOT / "backend" / "app" / "services" / "postprocess_config.yaml"
        if not cfg_path.exists():
            log(FAIL, f"config 없음: {cfg_path}")
            return 0, 1

        import yaml
        with open(cfg_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        required_keys = ["baseline", "model_strength", "strength_modifiers",
                         "class_overrides", "furniture_gate", "geometric_gate", "tta"]
        for k in required_keys:
            if k not in cfg:
                log(FAIL, f"config 키 없음: {k}")
                return 0, 1
        log(PASS, f"postprocess_config.yaml: {len(cfg)}개 섹션 모두 존재")

        # GeometricGate, FurnitureGate, TTA 인스턴스 생성 가능?
        from app.services.geometric_gate import load_geometric_gate_from_config
        from app.services.furniture_gate import load_furniture_gate_from_config
        from app.services.tta import load_tta_from_config

        load_geometric_gate_from_config(cfg["geometric_gate"])
        load_furniture_gate_from_config(cfg["furniture_gate"])
        tta = load_tta_from_config(cfg["tta"], "M4_CONTEXT")
        log(PASS, f"config 기반 인스턴스 생성: GeometricGate, FurnitureGate, TTA(M4_CONTEXT={tta is not None})")
        return 2, 0
    except Exception as e:
        log(FAIL, f"config: {type(e).__name__}: {e}")
        traceback.print_exc()
        return 0, 1


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    cwd = Path.cwd()
    weights_dir = cwd.parent / "models_weights" if cwd.name == "training" else cwd / "backend" / "models_weights"

    print("=" * 70)
    print("전 시스템 dry-run 검증")
    print(f"  cwd: {cwd}")
    print(f"  weights_dir: {weights_dir}")
    print("=" * 70)

    total_passed = total_failed = 0

    p, f = check_model_files(weights_dir)
    total_passed += p
    total_failed += f

    p, f = check_module_imports()
    total_passed += p
    total_failed += f

    p, f = check_postprocess_units()
    total_passed += p
    total_failed += f

    p, f = check_postprocess_config(cwd if cwd.name != "training" else cwd)
    total_passed += p
    total_failed += f

    p, f = check_pipeline_load()
    total_passed += p
    total_failed += f

    p, f = check_sample_inference(cwd)
    total_passed += p
    total_failed += f

    print("\n" + "=" * 70)
    print(f"결과: {total_passed} PASS / {total_failed} FAIL")
    print("=" * 70)

    return 0 if total_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
