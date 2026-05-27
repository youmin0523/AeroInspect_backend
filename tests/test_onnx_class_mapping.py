# =============================================
# tests/test_onnx_class_mapping.py
# 역할: ONNX 출력 dim ↔ data.yaml names ↔ defect_taxonomy.EXPECTED_CLASS_NAMES
#       ↔ inference_pipeline_20.py 의 로드 인자, 4-way 회귀 가드.
#
# 배경:
#   2026-05-07 5건 동시 거짓 라벨 사고는 위 4 출처가 한쪽씩 어긋난 채
#   배포되어 발생함. 신규 ONNX 통합/모델 교체 시 반드시 본 테스트를
#   먼저 통과시켜야 한다.
#
# 의존:
#   - onnxruntime (CPUExecutionProvider 만 사용 → CUDA 없는 환경 OK)
#   - PyYAML (이미 backend requirements 에 포함)
#
# 실행:
#   pytest tests/test_onnx_class_mapping.py -v
# =============================================
from __future__ import annotations

from pathlib import Path

import pytest

from app.services.defect_taxonomy import (
    EXPECTED_CLASS_NAMES,
    validate_class_mapping,
)


# ── 모델 ↔ ONNX 파일명 ↔ 데이터셋 폴더명 매핑 ──
# (model_name, onnx_filename, dataset_subdir or None)
# dataset_subdir=None 이면 yaml 검증을 스킵하고 ONNX↔code 2-way 만 검증
# (ResNet 분류기는 ImageFolder 학습이라 data.yaml 이 없음).
MODEL_MATRIX = [
    # YOLO detection — 4-way (ONNX + yaml + code + pipeline 로드 인자)
    ("M1_YOLO",         "m1_yolo_structural.onnx",        "structural"),
    ("M2_YOLO",         "m2_yolo_surface.onnx",           "surface"),
    ("M3_YOLO",         "m3_yolo_floor_window.onnx",      "floor_window"),
    ("M4_CONTEXT",      "m4_yolo_context_elements.onnx",  "m4_context_refined"),
    ("M5_SEG",          "m5_yolo_seg_frames.onnx",        None),  # frames data.yaml 라벨은 placeholder
    ("FURNITURE_AWARE", "furniture_aware.onnx",           "furniture_aware"),
    # ResNet 분류기 — 3-way (ONNX + code + pipeline 로드 인자)
    ("M1_RESNET", "m1_resnet_crack_classifier.onnx",        None),
    ("M2_RESNET", "m2_resnet_surface_classifier.onnx",      None),
    ("M3_RESNET", "m3_resnet_floor_window_classifier.onnx", None),
]


@pytest.fixture(scope="module")
def _onnxruntime_available():
    """onnxruntime 미설치 시 모듈 전체 skip."""
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        pytest.skip("onnxruntime 미설치 — ONNX 매핑 검증 스킵")


@pytest.mark.parametrize(
    "model_name, onnx_filename, dataset_subdir",
    MODEL_MATRIX,
    ids=[m[0] for m in MODEL_MATRIX],
)
def test_onnx_class_mapping_4way(
    _onnxruntime_available,
    onnx_weights_dir: Path,
    datasets_dir: Path,
    model_name: str,
    onnx_filename: str,
    dataset_subdir: str | None,
):
    """ONNX dim ↔ data.yaml names ↔ EXPECTED_CLASS_NAMES 일치 검증."""
    onnx_path = onnx_weights_dir / onnx_filename
    if not onnx_path.exists():
        pytest.skip(f"{model_name}: ONNX 파일 없음 ({onnx_path}) — CI graceful skip")

    yaml_path: str | None = None
    if dataset_subdir is not None:
        candidate = datasets_dir / dataset_subdir / "data.yaml"
        if candidate.exists():
            yaml_path = str(candidate)
        else:
            pytest.skip(
                f"{model_name}: data.yaml 없음 ({candidate}) — CI graceful skip"
            )

    errors = validate_class_mapping(model_name, str(onnx_path), yaml_path)

    assert not errors, (
        f"\n=== {model_name} 4-way 매핑 검증 실패 ===\n"
        + "\n".join(f"  - {e}" for e in errors)
        + "\n조치: defect_taxonomy.EXPECTED_CLASS_NAMES, data.yaml names, "
        "inference_pipeline_20.py 로드 인자, ONNX export 중 어디가 어긋났는지 확인."
    )


def test_pipeline_loader_args_match_expected_class_names():
    """inference_pipeline_20.py 의 _try_load_yolo / _try_load_resnet 에 전달되는
    class_names 가 EXPECTED_CLASS_NAMES 와 정확히 동일한지 AST 정적 비교.

    파이프라인 코드가 갱신될 때 EXPECTED_CLASS_NAMES 와의 동기화를 보장.
    AST 기반이라 주석/줄바꿈에 영향 받지 않음.
    """
    import ast

    pipeline_path = (
        Path(__file__).resolve().parents[1]
        / "app" / "services" / "inference_pipeline_20.py"
    )
    src = pipeline_path.read_text(encoding="utf-8")
    tree = ast.parse(src)

    # 마지막 인자(label 문자열) → EXPECTED_CLASS_NAMES 키
    label_to_key = {
        "M1-YOLO": "M1_YOLO",
        "M2-YOLO": "M2_YOLO",
        "M3-YOLO": "M3_YOLO",
        "M4-Context": "M4_CONTEXT",
        "FurnitureAware": "FURNITURE_AWARE",
        "M1-ResNet": "M1_RESNET",
        "M2-ResNet": "M2_RESNET",
        "M3-ResNet": "M3_RESNET",
    }
    target_funcs = {"_try_load_yolo", "_try_load_resnet"}

    def _literal_str_list(node: ast.AST) -> list[str] | None:
        """ast.List 노드를 [str, ...] 으로 평가. 문자열 리터럴만 허용."""
        if not isinstance(node, ast.List):
            return None
        out: list[str] = []
        for elt in node.elts:
            if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                out.append(elt.value)
            else:
                return None
        return out

    mismatches: list[str] = []
    seen_keys: set[str] = set()

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        # self._try_load_yolo(...) / cls._try_load_yolo(...) / 직접호출 모두 잡기
        fname = (
            func.attr if isinstance(func, ast.Attribute)
            else func.id if isinstance(func, ast.Name)
            else None
        )
        if fname not in target_funcs:
            continue

        # 시그니처: (weights_dir, filename, class_names, label)
        # 호출 형태: self._try_load_yolo(wd, ..., [list], "label")
        # positional 인자에서 list 와 label 추출
        args = node.args
        list_node = None
        label = None
        for a in args:
            cand = _literal_str_list(a)
            if cand is not None:
                list_node = cand
            elif isinstance(a, ast.Constant) and isinstance(a.value, str):
                # 마지막 str constant 가 label
                label = a.value

        if list_node is None or label is None:
            continue
        if label not in label_to_key:
            continue
        key = label_to_key[label]
        if key in seen_keys:
            continue  # 보조 ckpt 등 동일 label 재등장 시 첫 번째만 사용
        seen_keys.add(key)

        expected = EXPECTED_CLASS_NAMES[key]
        if list_node != expected:
            mismatches.append(
                f"{key} ({label}): pipeline={list_node} vs EXPECTED={expected}"
            )

    required = {
        "M1_YOLO", "M2_YOLO", "M3_YOLO", "M4_CONTEXT", "FURNITURE_AWARE",
        "M1_RESNET", "M2_RESNET", "M3_RESNET",
    }
    missing = required - seen_keys
    assert not missing, (
        f"파이프라인 로드 인자 추출 실패: {sorted(missing)} — "
        f"inference_pipeline_20.py 구조 변경 또는 라벨 이름 변경 의심"
    )

    assert not mismatches, (
        "\n=== 파이프라인 로드 인자 ↔ EXPECTED_CLASS_NAMES 불일치 ===\n"
        + "\n".join(f"  - {m}" for m in mismatches)
    )


def test_expected_class_names_has_no_duplicates():
    """EXPECTED_CLASS_NAMES 각 모델의 클래스 리스트에 중복 라벨이 없어야 함."""
    bad = []
    for model_name, names in EXPECTED_CLASS_NAMES.items():
        if len(set(names)) != len(names):
            bad.append(f"{model_name}: {names} (중복 존재)")
    assert not bad, "중복 라벨 발견:\n" + "\n".join(bad)
