"""
남은 학습 + 재학습 파이프라인
M2 YOLO 완료 대기 → 열화상 → M1 재학습(nc=3) → M3 재학습(820장) → M6 재학습(446장) → 리포트
"""
import os, sys, time, shutil, json
from pathlib import Path
from datetime import datetime

TRAIN_DIR = Path(__file__).parent.resolve()
WEIGHTS_DIR = TRAIN_DIR.parent / "models_weights"
WEIGHTS_DIR.mkdir(exist_ok=True)
LOG_FILE = TRAIN_DIR / "auto_train_remaining_log.txt"

def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")

import torch
log(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only'}")

os.chdir(str(TRAIN_DIR))

# ══════════════════════════════════════════════
# 1. 열화상 YOLO (GPU) — auto_train_all.py가 시작할 수도 있으니 확인
# ══════════════════════════════════════════════
def train_thermal_yolo():
    log("=" * 60)
    log("열화상 YOLO: thermal_yolo (Crack+Moisture+delamination) — GPU")
    log("=" * 60)
    from ultralytics import YOLO

    model = YOLO("yolov8m.pt")
    model.train(
        data="datasets/thermal_yolo/data.yaml",
        epochs=100, imgsz=640, batch=16, patience=20,
        project=str(TRAIN_DIR / "runs" / "thermal"),
        name="yolov8m", device=0, optimizer="AdamW",
        lr0=0.001, cos_lr=True,
        hsv_h=0.0, hsv_s=0.3, hsv_v=0.3,
        save=True, exist_ok=True,
    )
    metrics = model.val()
    log(f"열화상 YOLO 완료 — mAP@0.5: {metrics.box.map50:.4f}, mAP@0.5:0.95: {metrics.box.map:.4f}")

    model.export(format="onnx", imgsz=640, dynamic=True, simplify=True)
    best_onnx = TRAIN_DIR / "runs" / "thermal" / "yolov8m" / "weights" / "best.onnx"
    if best_onnx.exists():
        shutil.copy(best_onnx, WEIGHTS_DIR / "thermal_yolo.onnx")
        log("열화상 ONNX 저장 완료")

# ══════════════════════════════════════════════
# 2. M1 YOLO 재학습 — structural nc=3 (crack + waterproof + caulking)
# ══════════════════════════════════════════════
def retrain_m1_yolo():
    log("=" * 60)
    log("M1 YOLO 재학습: structural nc=3 (caulking 추가) — GPU")
    log("=" * 60)
    from ultralytics import YOLO

    model = YOLO("yolov8m.pt")
    model.train(
        data="datasets/structural/data.yaml",
        epochs=100, imgsz=640, batch=16, patience=20,
        project=str(TRAIN_DIR / "runs" / "m1_structural_v2"),
        name="yolov8m", device=0, optimizer="AdamW",
        lr0=0.001, cos_lr=True, save=True, exist_ok=True,
    )
    metrics = model.val()
    log(f"M1 YOLO v2 완료 — mAP@0.5: {metrics.box.map50:.4f}, mAP@0.5:0.95: {metrics.box.map:.4f}")

    model.export(format="onnx", imgsz=640, dynamic=True, simplify=True)
    best_onnx = TRAIN_DIR / "runs" / "m1_structural_v2" / "yolov8m" / "weights" / "best.onnx"
    if best_onnx.exists():
        shutil.copy(best_onnx, WEIGHTS_DIR / "m1_yolo_structural.onnx")
        log("M1 YOLO v2 ONNX 저장 완료 (기존 덮어씀)")

# ══════════════════════════════════════════════
# 3. M3 YOLO 재학습 — floor_window 3클래스 820장
# ══════════════════════════════════════════════
def retrain_m3_yolo():
    log("=" * 60)
    log("M3 YOLO 재학습: floor_window 3클래스 820장 — GPU")
    log("=" * 60)
    from ultralytics import YOLO

    model = YOLO("yolov8m.pt")
    model.train(
        data="datasets/floor_window/data.yaml",
        epochs=100, imgsz=640, batch=16, patience=20,
        project=str(TRAIN_DIR / "runs" / "m3_floor_window_v2"),
        name="yolov8m", device=0, optimizer="AdamW",
        lr0=0.001, cos_lr=True, save=True, exist_ok=True,
    )
    metrics = model.val()
    log(f"M3 YOLO v2 완료 — mAP@0.5: {metrics.box.map50:.4f}, mAP@0.5:0.95: {metrics.box.map:.4f}")

    model.export(format="onnx", imgsz=640, dynamic=True, simplify=True)
    best_onnx = TRAIN_DIR / "runs" / "m3_floor_window_v2" / "yolov8m" / "weights" / "best.onnx"
    if best_onnx.exists():
        shutil.copy(best_onnx, WEIGHTS_DIR / "m3_yolo_floor_window.onnx")
        log("M3 YOLO v2 ONNX 저장 완료 (기존 덮어씀)")

# ══════════════════════════════════════════════
# 4. M6 PatchCore 재학습 — normal 446장
# ══════════════════════════════════════════════
def retrain_m6_patchcore():
    log("=" * 60)
    log("M6 PatchCore 재학습: normal 446장 — CPU")
    log("=" * 60)
    import torch.nn as nn
    from torchvision import models, transforms
    from PIL import Image
    import numpy as np

    backbone = models.wide_resnet50_2(weights=models.Wide_ResNet50_2_Weights.IMAGENET1K_V2)
    backbone.eval()

    features = {}
    def hook_fn(name):
        def hook(module, input, output):
            features[name] = output.detach()
        return hook

    backbone.layer2.register_forward_hook(hook_fn('layer2'))
    backbone.layer3.register_forward_hook(hook_fn('layer3'))

    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    normal_dir = Path('datasets/normal/good')
    all_images = list(normal_dir.glob('*.jpg')) + list(normal_dir.glob('*.png'))
    log(f"정상 이미지: {len(all_images)}장")

    all_features = []
    with torch.no_grad():
        for img_path in all_images:
            img = Image.open(img_path).convert('RGB')
            x = tf(img).unsqueeze(0)
            _ = backbone(x)
            f2 = features['layer2']
            f3 = nn.functional.interpolate(features['layer3'], size=f2.shape[2:], mode='bilinear', align_corners=False)
            combined = torch.cat([f2, f3], dim=1)
            pooled = nn.functional.adaptive_avg_pool2d(combined, (7, 7))
            patches = pooled.squeeze(0).reshape(pooled.shape[1], -1).T
            all_features.append(patches)

    memory_bank = torch.cat(all_features, dim=0)
    n_samples = max(int(memory_bank.shape[0] * 0.1), 200)
    indices = torch.randperm(memory_bank.shape[0])[:n_samples]
    coreset = memory_bank[indices]
    log(f"Coreset: {coreset.shape}")

    os.makedirs('runs/m6_patchcore_v2', exist_ok=True)
    torch.save({
        'coreset': coreset,
        'backbone': 'wide_resnet50_2',
        'n_normal_images': len(all_images),
    }, 'runs/m6_patchcore_v2/patchcore_model.pt')

    np.save(str(WEIGHTS_DIR / 'm6_patchcore_coreset.npy'), coreset.numpy())
    log("M6 PatchCore v2 coreset 저장 완료")

# ══════════════════════════════════════════════
# 5. 최종 리포트
# ══════════════════════════════════════════════
def final_report():
    log("=" * 60)
    log("전체 학습 파이프라인 완료 — 최종 리포트")
    log("=" * 60)
    log("")
    log("생성된 ONNX/모델 파일:")
    for f in sorted(WEIGHTS_DIR.glob("*")):
        size = f.stat().st_size / 1024 / 1024
        log(f"  {f.name}: {size:.1f} MB")
    log("")
    log("내일 확인해주세요.")

# ══════════════════════════════════════════════
# 메인
# ══════════════════════════════════════════════
if __name__ == "__main__":
    log("=" * 60)
    log("남은 학습 + 재학습 파이프라인 시작")
    log("=" * 60)

    # auto_train_all.py가 열화상 YOLO를 이미 시작했는지 확인
    thermal_runs = TRAIN_DIR / "runs" / "thermal" / "yolov8m"
    if thermal_runs.exists() and (thermal_runs / "weights" / "best.pt").exists():
        log("열화상 YOLO — 이미 완료됨 (auto_train_all.py가 처리), 건너뜀")
    else:
        try:
            train_thermal_yolo()
        except Exception as e:
            log(f"열화상 YOLO 에러: {e}")

    try:
        retrain_m1_yolo()
    except Exception as e:
        log(f"M1 YOLO 재학습 에러: {e}")

    try:
        retrain_m3_yolo()
    except Exception as e:
        log(f"M3 YOLO 재학습 에러: {e}")

    try:
        retrain_m6_patchcore()
    except Exception as e:
        log(f"M6 PatchCore 에러: {e}")

    final_report()
