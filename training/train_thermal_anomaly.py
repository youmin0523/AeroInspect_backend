# =============================================
# train_thermal_anomaly.py
# Thermal anomaly detection (PatchCore) — Moisture/delam 대안
#
# 왜 YOLO를 포기했는가:
#   - 1788 이미지에 평균 8.8 / 최대 170 인스턴스 = 라벨 과밀
#   - 박스 라벨 자체가 노이즈 (열화상은 경계 모호)
#   - 10번 재학습해도 mAP50-95 0.18 한계
#
# 대안 (이 모델):
#   - 정상 천장/벽 패치만으로 unsupervised 학습 (라벨 불필요)
#   - 추론 시 anomaly map heatmap → 영역 단위 결함 표시
#   - 박스 라벨 노이즈 영향 사라짐, Recall 100% 가능
#
# 의도된 통합:
#   - 기존 thermal YOLO(Crack용)는 유지
#   - Moisture/delam은 이 anomaly 모델로 대체
#   - inference_pipeline_20에서 thermal_anomaly 분기 추가 예정
#
# 데이터셋 구조:
#   datasets/thermal_anomaly/good/  ← prepare_thermal_anomaly.py 결과
# =============================================

from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

WEIGHTS_DIR = Path("../models_weights")
OUTPUT_NAME = "thermal_anomaly"


def train():
    print("=" * 60)
    print("[thermal_anomaly] PatchCore 비지도 학습 (정상 패치)")
    print("=" * 60)

    try:
        from anomalib.data import Folder
        from anomalib.engine import Engine
        from anomalib.models import Patchcore
    except ImportError:
        print("[thermal_anomaly] anomalib 미설치. pip install anomalib 후 재시도.")
        return

    datamodule = Folder(
        name="thermal_normal",
        root="datasets/thermal_anomaly",
        normal_dir="good",
        train_batch_size=8,
        eval_batch_size=8,
        num_workers=4,
    )

    model = Patchcore(
        backbone="wide_resnet50_2",
        layers=["layer2", "layer3"],
        coreset_sampling_ratio=0.01,
        num_neighbors=9,
    )

    # 8GB VRAM 보호: limit_train_batches (M6에서 검증된 값)
    engine = Engine(max_epochs=1, limit_train_batches=50, limit_val_batches=25)
    engine.fit(model=model, datamodule=datamodule)

    export_root = Path("runs/thermal_anomaly")
    export_root.mkdir(parents=True, exist_ok=True)
    engine.export(model=model, export_type="onnx", export_root=str(export_root))

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    onnx_files = list(export_root.rglob("*.onnx"))
    if onnx_files:
        dst = WEIGHTS_DIR / f"{OUTPUT_NAME}.onnx"
        if dst.exists():
            shutil.copy2(dst, WEIGHTS_DIR / f"{OUTPUT_NAME}_prev.onnx")
        shutil.copy2(onnx_files[0], dst)
        print(f"[thermal_anomaly] ONNX 저장 완료: {dst}")
    else:
        print("[thermal_anomaly] 경고: ONNX 파일 못 찾음.")


if __name__ == "__main__":
    train()
