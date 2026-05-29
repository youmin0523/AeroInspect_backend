# =============================================
# export_furniture_onnx.py
# Furniture best.pt → ONNX export (cuDNN 사고 복구용)
#
# 배경:
#   - chain v1.2 Furniture가 epoch 18에서 cuDNN_STATUS_EXECUTION_FAILED 사망
#   - best.pt 0.349 / last.pt는 보존됨 (mAP50-95)
#   - 학습 스크립트 내장 export 단계 도달 못함
#
# 동작:
#   - runs/detect/runs/furniture_aware_v2/weights/best.pt 로드
#   - ONNX export → models_weights/furniture_aware.onnx 교체 (백업 포함)
# =============================================

from __future__ import annotations

import shutil
import sys
from pathlib import Path

from ultralytics import YOLO

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent.parent  # repo root
BEST = ROOT / "runs" / "detect" / "runs" / "furniture_aware_v2" / "weights" / "best.pt"
WEIGHTS_DIR = Path(__file__).resolve().parent.parent / "models_weights"
OUTPUT_NAME = "furniture_aware"


def main():
    print("=" * 60)
    print("[export_furniture] best.pt → ONNX (cuDNN 사고 복구)")
    print("=" * 60)

    if not BEST.exists():
        print(f"[ERROR] best.pt 없음: {BEST}")
        return

    print(f"[info] best.pt: {BEST} ({BEST.stat().st_size/1024/1024:.1f}MB)")

    model = YOLO(str(BEST))
    model.export(format="onnx", opset=14, dynamic=False, simplify=True)

    onnx_src = BEST.with_suffix(".onnx")
    if not onnx_src.exists():
        print(f"[ERROR] ONNX 생성 실패: {onnx_src}")
        return

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    dst = WEIGHTS_DIR / f"{OUTPUT_NAME}.onnx"
    if dst.exists():
        prev = WEIGHTS_DIR / f"{OUTPUT_NAME}_prev.onnx"
        shutil.copy2(dst, prev)
        print(f"[info] 기존 백업: {prev}")
    shutil.copy2(onnx_src, dst)
    print(f"[done] ONNX 저장: {dst} ({dst.stat().st_size/1024/1024:.1f}MB)")
    print(f"[note] mAP50-95 ≈ 0.349 (epoch 18, cuDNN 사고로 조기 종료)")


if __name__ == "__main__":
    main()
