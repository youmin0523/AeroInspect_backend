# =============================================
# export_m4v2_best.py
# 역할: Stage 1/2 best.pt 중 mAP 높은 쪽을 ONNX로 export
#       train_m4v2_local.py 학습 완료 후 자동 export가 잘못된 best 선택 시
#       수동으로 재export하기 위한 standalone 스크립트
#
# 사용:
#   cd backend/training
#   python export_m4v2_best.py
# =============================================

from __future__ import annotations

import shutil
import sys
from pathlib import Path

from ultralytics import YOLO

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


PROJECT = Path("../../runs/detect/runs/m4v2")
WEIGHTS_DIR = Path("../models_weights")
REFINED = Path("datasets/m4_context_refined")
ONNX_OUT_NAME = "m4_yolo_context_elements.onnx"


def read_best_map_from_csv(csv_path: Path) -> float:
    if not csv_path.exists():
        return -1.0
    best = -1.0
    for i, line in enumerate(csv_path.read_text(encoding="utf-8").splitlines()):
        if i == 0:
            continue
        parts = line.split(",")
        if len(parts) < 8:
            continue
        try:
            v = float(parts[7])
            if v > best:
                best = v
        except ValueError:
            continue
    return best


def main():
    stage1_best = PROJECT / "stage1" / "weights" / "best.pt"
    stage2_best = PROJECT / "stage2" / "weights" / "best.pt"
    stage1_csv = PROJECT / "stage1" / "results.csv"
    stage2_csv = PROJECT / "stage2" / "results.csv"

    s1_map = read_best_map_from_csv(stage1_csv) if stage1_best.exists() else -1.0
    s2_map = read_best_map_from_csv(stage2_csv) if stage2_best.exists() else -1.0

    print("=" * 60)
    print("M4v2 best.pt 비교 + ONNX export")
    print("=" * 60)
    print(f"Stage 1 best.pt: {'exists' if stage1_best.exists() else 'MISSING'} | mAP50={s1_map:.5f}")
    print(f"Stage 2 best.pt: {'exists' if stage2_best.exists() else 'MISSING'} | mAP50={s2_map:.5f}")

    if s1_map < 0 and s2_map < 0:
        print("[ERROR] best.pt 둘 다 없음")
        return 1

    if s2_map >= s1_map and stage2_best.exists():
        best_path = stage2_best
        chosen = f"Stage 2 (mAP {s2_map:.5f} >= {s1_map:.5f})"
    else:
        best_path = stage1_best
        chosen = f"Stage 1 (mAP {s1_map:.5f} > {s2_map:.5f})"

    print(f"\n채택: {chosen}")
    print(f"경로: {best_path.resolve()}")

    if not best_path.exists():
        print(f"[ERROR] 선택된 best.pt 없음: {best_path}")
        return 1

    # ONNX export
    print("\nONNX export...")
    model = YOLO(str(best_path))
    model.export(format="onnx", opset=17, dynamic=True, simplify=True)
    onnx_src = best_path.with_suffix(".onnx")

    # 평가
    print("\n검증...")
    metrics = model.val(data=str(REFINED / "data.yaml"), imgsz=960, batch=4, device="cpu")
    print("\n=== 최종 결과 ===")
    print(f"  mAP50:    {metrics.box.map50:.4f}")
    print(f"  mAP50-95: {metrics.box.map:.4f}")
    print(f"  precision: {metrics.box.mp:.4f}")
    print(f"  recall:    {metrics.box.mr:.4f}")
    print(f"  0.9 도달? {'YES' if metrics.box.map50 >= 0.9 else 'NO'}")

    # 배포 위치 복사
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    dst = WEIGHTS_DIR / ONNX_OUT_NAME
    shutil.copy2(onnx_src, dst)
    size_mb = dst.stat().st_size / (1024 * 1024)
    print(f"\n배포 위치 저장: {dst} ({size_mb:.1f}MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
