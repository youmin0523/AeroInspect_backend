# =============================================
# export_colab_results.py
# 역할: Colab에서 받은 best.pt / last.pt 중 mAP 높은 쪽으로 ONNX export + 배포
#       - furniture_aware (10-class)
#       - M1 Plan A (구조 균열, structural data.yaml)
#
# 사용:
#   cd backend/training
#   # 기본: Downloads에서 자동 검색
#   python export_colab_results.py
#
#   # 또는 명시적 경로
#   python export_colab_results.py \
#     --m1-best /path/to/m1_best.pt --m1-last /path/to/m1_last.pt \
#     --fur-best /path/to/fur_best.pt --fur-last /path/to/fur_last.pt
# =============================================

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Optional, Tuple

from ultralytics import YOLO

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


WEIGHTS_DIR = Path("../models_weights")
DOWNLOADS = Path("C:/Users/Codelab/Downloads")


def auto_find(name_patterns: list, search_dirs: list) -> Optional[Path]:
    """주어진 패턴 중 매칭되는 첫 .pt 파일 반환."""
    for d in search_dirs:
        if not d.exists():
            continue
        for p in d.iterdir():
            if not p.is_file() or p.suffix != ".pt":
                continue
            name_lower = p.name.lower()
            if any(pat.lower() in name_lower for pat in name_patterns):
                return p
    return None


def evaluate_pt(pt_path: Path, data_yaml: Path, imgsz: int = 640) -> Optional[float]:
    """ultralytics val()로 mAP50 측정. GPU 우선, 실패 시 CPU fallback."""
    if not pt_path.exists() or not data_yaml.exists():
        return None
    for device in [0, "cpu"]:
        try:
            model = YOLO(str(pt_path))
            metrics = model.val(
                data=str(data_yaml), imgsz=imgsz, batch=8,
                device=device, workers=0, plots=False, save_json=False, verbose=False,
            )
            return float(metrics.box.map50)
        except Exception as e:
            if device == 0:
                print(f"  [WARN] GPU 실패, CPU fallback: {e}")
                continue
            print(f"  [WARN] val 실패: {pt_path.name} — {e}")
            return None
    return None


def pick_better_and_export(
    pt_candidates: list,        # [(label, Path), ...]
    data_yaml: Path,
    output_onnx_name: str,
    imgsz: int,
    label: str,
) -> Tuple[Optional[Path], Optional[float]]:
    """여러 .pt 후보 중 mAP 가장 높은 것 ONNX export + 배포."""
    print(f"\n{'='*60}")
    print(f"=== {label} ===")
    print(f"{'='*60}")
    for lab, p in pt_candidates:
        print(f"{lab}: {p if p and p.exists() else 'MISSING'}")
    print(f"data.yaml: {data_yaml if data_yaml.exists() else 'MISSING'}")

    if not data_yaml.exists():
        print(f"[ERROR] data.yaml 없음: {data_yaml}")
        return None, None

    # 각 후보 평가
    results = []
    for lab, p in pt_candidates:
        if p is None or not p.exists():
            continue
        m = evaluate_pt(p, data_yaml, imgsz)
        if m is not None:
            print(f"  {lab}: mAP50={m:.4f}")
            results.append((m, p, lab))

    if not results:
        print(f"[ERROR] 평가 가능한 .pt 없음")
        return None, None

    results.sort(key=lambda x: x[0], reverse=True)
    chosen_map, chosen_pt, chosen_type = results[0]
    print(f"\n채택: {chosen_type} (mAP50={chosen_map:.4f})")
    print(f"경로: {chosen_pt}")

    # ONNX export
    print(f"\nONNX export...")
    model = YOLO(str(chosen_pt))
    model.export(format="onnx", opset=17, dynamic=True, simplify=True)
    onnx_src = chosen_pt.with_suffix(".onnx")
    if not onnx_src.exists():
        print(f"[ERROR] export 결과 없음: {onnx_src}")
        return None, chosen_map

    # 배포
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    dst = WEIGHTS_DIR / output_onnx_name
    shutil.copy2(onnx_src, dst)
    size_mb = dst.stat().st_size / (1024 * 1024)
    print(f"\n배포: {dst} ({size_mb:.1f} MB)")
    return dst, chosen_map


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m1-best", type=str, default=None)
    parser.add_argument("--m1-last", type=str, default=None)
    parser.add_argument("--fur-best", type=str, default=None)
    parser.add_argument("--fur-last", type=str, default=None)
    parser.add_argument("--search-dirs", nargs="+", default=[
        str(DOWNLOADS),
        "C:/Users/Codelab/Desktop",
    ], help="자동 검색 디렉토리")
    args = parser.parse_args()

    cwd = Path.cwd()
    print(f"cwd: {cwd}")

    # 압축 해제된 autosave 디렉토리 자동 탐지
    extracted_root = DOWNLOADS / "colab_pt_extracted"
    m1_dir = extracted_root / "m1_plan_a_autosave"
    fur_dir = extracted_root / "furniture_aware_autosave"

    # M1 Plan A 후보 4개
    m1_candidates = []
    for stage_name in ["stage1_best", "stage1_last", "stage2_best", "stage2_last"]:
        p = m1_dir / f"{stage_name}.pt"
        if p.exists():
            m1_candidates.append((stage_name, p))
        elif args.m1_best and stage_name == "stage1_best":
            m1_candidates.append((stage_name, Path(args.m1_best)))

    # furniture_aware 후보 4개
    fur_candidates = []
    for stage_name in ["stage1_best", "stage1_last", "stage2_best", "stage2_last"]:
        p = fur_dir / f"{stage_name}.pt"
        if p.exists():
            fur_candidates.append((stage_name, p))

    results = {}

    # M1 Plan A → m1_yolo_structural.onnx
    # 학습 시 imgsz=640 (SAHI tile 데이터, m1_plan_a 노트북) → 평가도 640으로 일치
    if m1_candidates:
        # SAHI tile 데이터셋이 있으면 그걸 사용, 아니면 일반 structural
        m1_tiled_yaml = cwd / "datasets" / "structural_tiled" / "data.yaml"
        m1_yaml = m1_tiled_yaml if m1_tiled_yaml.exists() else (cwd / "datasets" / "structural" / "data.yaml")
        dst, m1_map = pick_better_and_export(
            m1_candidates, m1_yaml,
            output_onnx_name="m1_yolo_structural.onnx",
            imgsz=640,  # 학습 imgsz와 일치
            label="M1 Plan A → m1_yolo_structural.onnx",
        )
        results["m1"] = (dst, m1_map)
    else:
        print("\n[SKIP] M1 Plan A — pt 파일 없음")
        results["m1"] = (None, None)

    # furniture_aware → furniture_aware.onnx
    if fur_candidates:
        # furniture_aware data.yaml — Colab 노트북에서 사용한 것과 같은 구조로 생성 필요
        fur_yaml_candidate = cwd / "datasets" / "furniture_aware" / "data.yaml"
        if not fur_yaml_candidate.exists():
            # Colab에서 만들어진 것 — 동일 구조로 임시 생성
            fur_dir = cwd / "datasets" / "furniture_aware"
            fur_yaml_candidate = fur_dir / "data.yaml"
            if fur_dir.exists() and not fur_yaml_candidate.exists():
                yaml_text = f"""path: {fur_dir.resolve()}
train: images/train
val: images/val
test: images/test

nc: 10
names:
  0: wall
  1: ceiling
  2: floor
  3: window
  4: door
  5: cabinet_builtin
  6: kitchen_appliance
  7: countertop_sink
  8: kitchen_island
  9: shelf
"""
                fur_yaml_candidate.write_text(yaml_text, encoding="utf-8")
                print(f"\n[INFO] furniture_aware data.yaml 자동 생성: {fur_yaml_candidate}")

        dst, fur_map = pick_better_and_export(
            fur_candidates, fur_yaml_candidate,
            output_onnx_name="furniture_aware.onnx",
            imgsz=640,
            label="furniture_aware → furniture_aware.onnx",
        )
        results["fur"] = (dst, fur_map)
    else:
        print("\n[SKIP] furniture_aware — best/last 둘 다 없음")
        results["fur"] = (None, None)

    # 종합
    print(f"\n{'='*60}")
    print("종합")
    print(f"{'='*60}")
    for k, (dst, mp) in results.items():
        status = f"{dst} (mAP50={mp:.4f})" if dst else "스킵/실패"
        print(f"  {k}: {status}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
