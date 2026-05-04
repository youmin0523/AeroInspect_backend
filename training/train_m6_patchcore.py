# =============================================
# train_m6_patchcore.py
# M6: PatchCore 비지도 학습 (Anomalib)
# 정상 표면 이미지만으로 학습 → 이상 영역 탐지
# 출력: models_weights/m6_patchcore_surface.onnx
#
# 데이터셋 구조:
#   datasets/normal/
#     good/       정상 표면 이미지 (벽, 바닥, 창호)
#     defective/  하자 이미지 (검증용, 선택)
#
# 사용법:
#   cd backend/training
#   python train_m6_patchcore.py
# =============================================

from __future__ import annotations

import shutil
from pathlib import Path

WEIGHTS_DIR = Path("../models_weights")
OUTPUT_NAME = "m6_patchcore_surface"


def train():
    print("=" * 60)
    print("[M6-PatchCore] 비지도 학습 (정상 표면)")
    print("=" * 60)

    try:
        from anomalib.data import Folder
        from anomalib.engine import Engine
        from anomalib.models import Patchcore
    except ImportError:
        print("[M6-PatchCore] anomalib 미설치. pip install anomalib 후 재시도.")
        return

    datamodule = Folder(
        name="normal_surfaces",
        root="datasets/normal",
        normal_dir="good",
        abnormal_dir="defective",
        image_size=(256, 256),
        train_batch_size=32,
        eval_batch_size=32,
    )

    model = Patchcore(
        backbone="wide_resnet50_2",
        layers=["layer2", "layer3"],
        coreset_sampling_ratio=0.1,
        num_neighbors=9,
    )

    engine = Engine()
    engine.fit(model=model, datamodule=datamodule)

    # ONNX export
    export_root = Path("runs/m6_patchcore")
    export_root.mkdir(parents=True, exist_ok=True)
    engine.export(model=model, export_type="onnx", export_root=str(export_root))

    # models_weights로 복사
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    onnx_files = list(export_root.rglob("*.onnx"))
    if onnx_files:
        shutil.copy2(onnx_files[0], WEIGHTS_DIR / f"{OUTPUT_NAME}.onnx")
        print(f"[M6-PatchCore] ONNX 저장 완료: {WEIGHTS_DIR / OUTPUT_NAME}.onnx")
    else:
        print("[M6-PatchCore] 경고: ONNX 파���을 찾을 수 없습니다.")


if __name__ == "__main__":
    train()
