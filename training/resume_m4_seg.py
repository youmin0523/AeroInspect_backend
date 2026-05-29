# =============================================
# resume_m4_seg.py
# M4 seg 학습 재개 (노트북 종료 / 사고 복구용)
#
# ultralytics resume=True: last.pt + args.yaml 자동 로드, optimizer state 복원
# 학습 진행률·hyperparameter 동일하게 이어서 진행
#
# 사용: cd backend/training && python resume_m4_seg.py
# =============================================

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from ultralytics import YOLO

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent.parent
LAST = ROOT / "runs" / "segment" / "runs" / "m4_context_seg" / "train" / "weights" / "last.pt"
WEIGHTS_DIR = Path(__file__).resolve().parent.parent / "models_weights"
OUTPUT_NAME = "m4_yolo_context_elements"
DATA_YAML = "datasets/m4_context/data.yaml"
IMGSZ = 768
BATCH = 4


def train():
    print("=" * 60)
    print("[M4-Seg resume] last.pt에서 학습 재개")
    print("=" * 60)

    if not LAST.exists():
        print(f"[ERROR] last.pt 없음 — 새 학습 필요: {LAST}")
        return

    print(f"[info] last.pt: {LAST} ({LAST.stat().st_size/1024/1024:.1f}MB)")

    model = YOLO(str(LAST))
    model.train(resume=True)

    # 학습 완료 후 export + 평가 (train_m4_context_seg.py 동일 로직)
    best = LAST.parent / "best.pt"
    if not best.exists():
        print(f"[ERROR] best.pt 없음 — {best}")
        return

    print(f"\n[M4-Seg resume] best.pt: {best} ({best.stat().st_size/1024/1024:.1f}MB)")
    best_model = YOLO(str(best))
    best_model.export(format="onnx", opset=17, dynamic=True, simplify=True)

    onnx_path = best.with_suffix(".onnx")
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    dst = WEIGHTS_DIR / f"{OUTPUT_NAME}.onnx"
    if dst.exists():
        shutil.copy2(dst, WEIGHTS_DIR / f"{OUTPUT_NAME}_prev.onnx")
    shutil.copy2(onnx_path, dst)
    print(f"[M4-Seg resume] ONNX 저장: {dst} ({dst.stat().st_size/1024/1024:.1f}MB)")

    print("\n[M4-Seg resume] val 평가...")
    metrics = best_model.val(data=DATA_YAML, imgsz=IMGSZ, batch=BATCH)
    print(f"  box mAP50:     {metrics.box.map50:.4f}")
    print(f"  box mAP50-95:  {metrics.box.map:.4f}")
    print(f"  seg mAP50:     {metrics.seg.map50:.4f}")
    print(f"  seg mAP50-95:  {metrics.seg.map:.4f}")
    print(f"  baseline 0.355 돌파? "
          f"{'YES' if metrics.box.map >= 0.355 else 'NO'}")

    print("\n" + "=" * 60)
    print("[M4-Seg resume → verify] test_external 자동 검증 시작")
    print("=" * 60)
    try:
        subprocess.run(
            ["python", "verify_test_mode.py"],
            cwd=str(Path(__file__).resolve().parent),
            check=False,
        )
    except Exception as e:
        print(f"[verify] 실행 실패 (수동): {e}")


if __name__ == "__main__":
    train()
