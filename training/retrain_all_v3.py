"""전체 재학습 v3 — 새 데이터 통합 후"""
import os, sys, time, shutil
from pathlib import Path
from datetime import datetime

TRAIN_DIR = Path(__file__).parent.resolve()
WEIGHTS_DIR = TRAIN_DIR.parent / "models_weights"
WEIGHTS_DIR.mkdir(exist_ok=True)
LOG = TRAIN_DIR / "retrain_v3_log.txt"

def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")

os.chdir(str(TRAIN_DIR))

import torch
log(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

# ── GPU 모델 순차 학습 ──

def train_yolo(name, data_yaml, project, nc, epochs=100):
    log(f"{'='*50}")
    log(f"{name} 학습 시작 (nc={nc}, {epochs} epochs)")
    log(f"{'='*50}")
    from ultralytics import YOLO
    model = YOLO("yolov8m.pt")
    model.train(
        data=data_yaml, epochs=epochs, imgsz=640, batch=16,
        patience=20, project=str(TRAIN_DIR / "runs" / project),
        name="yolov8m_v3", device=0, optimizer="AdamW",
        lr0=0.001, cos_lr=True, save=True, exist_ok=True,
    )
    metrics = model.val()
    log(f"{name} 완료 — mAP@0.5: {metrics.box.map50:.4f}")
    model.export(format="onnx", imgsz=640, dynamic=True, simplify=True)
    best_onnx = TRAIN_DIR / "runs" / project / "yolov8m_v3" / "weights" / "best.onnx"
    if best_onnx.exists():
        dst = WEIGHTS_DIR / f"{project.replace('/', '_')}.onnx"
        shutil.copy(best_onnx, dst)
        log(f"ONNX 저장: {dst}")

def train_yolo_seg(name, data_yaml, project, nc, epochs=100):
    log(f"{'='*50}")
    log(f"{name} SEG 학습 시작 (nc={nc}, {epochs} epochs)")
    log(f"{'='*50}")
    from ultralytics import YOLO
    model = YOLO("yolov8m-seg.pt")
    model.train(
        data=data_yaml, epochs=epochs, imgsz=640, batch=8,
        patience=20, project=str(TRAIN_DIR / "runs" / project),
        name="yolov8m_seg_v3", device=0, optimizer="AdamW",
        lr0=0.001, cos_lr=True, save=True, exist_ok=True,
    )
    metrics = model.val()
    log(f"{name} 완료 — mAP@0.5: {metrics.box.map50:.4f}")
    model.export(format="onnx", imgsz=640, dynamic=True, simplify=True)
    best_onnx = TRAIN_DIR / "runs" / project / "yolov8m_seg_v3" / "weights" / "best.onnx"
    if best_onnx.exists():
        dst = WEIGHTS_DIR / f"{project.replace('/', '_')}_seg.onnx"
        shutil.copy(best_onnx, dst)
        log(f"ONNX 저장: {dst}")

if __name__ == "__main__":
    log("전체 재학습 v3 시작")

    # 1. M1 YOLO structural (nc=3, 5731장)
    try:
        train_yolo("M1 YOLO structural", "datasets/structural/data.yaml", "m1_structural_v3", 3)
    except Exception as e:
        log(f"M1 YOLO 에러: {e}")

    # 2. M2 YOLO surface (nc=2, 8957장)
    try:
        train_yolo("M2 YOLO surface", "datasets/surface/data.yaml", "m2_surface_v3", 2)
    except Exception as e:
        log(f"M2 YOLO 에러: {e}")

    # 3. M3 YOLO floor_window (nc=3, 820장)
    try:
        train_yolo("M3 YOLO floor_window", "datasets/floor_window/data.yaml", "m3_floor_window_v3", 3)
    except Exception as e:
        log(f"M3 YOLO 에러: {e}")

    # 4. 열화상 YOLO (nc=3, 504장)
    try:
        train_yolo("열화상 YOLO", "datasets/thermal_yolo/data.yaml", "thermal_v3", 3)
    except Exception as e:
        log(f"열화상 YOLO 에러: {e}")

    # 5. M5 YOLO-seg frames (nc=5, 7244장) — 신규!
    try:
        train_yolo_seg("M5 frames seg", "datasets/frames/data.yaml", "m5_frames", 5)
    except Exception as e:
        log(f"M5 frames 에러: {e}")

    # 6. M6 PatchCore (5346장) — CPU
    try:
        log("M6 PatchCore v3 (5346장)")
        import torch.nn as nn
        from torchvision import models, transforms
        from PIL import Image
        import numpy as np

        backbone = models.wide_resnet50_2(weights=models.Wide_ResNet50_2_Weights.IMAGENET1K_V2)
        backbone.eval()
        features = {}
        def hook_fn(name):
            def hook(m, i, o): features[name] = o.detach()
            return hook
        backbone.layer2.register_forward_hook(hook_fn('layer2'))
        backbone.layer3.register_forward_hook(hook_fn('layer3'))
        tf = transforms.Compose([
            transforms.Resize((224, 224)), transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        normal_dir = Path('datasets/normal/good')
        all_imgs = list(normal_dir.glob('*.jpg')) + list(normal_dir.glob('*.png'))
        log(f"정상 이미지: {len(all_imgs)}장")

        all_feats = []
        with torch.no_grad():
            for i, p in enumerate(all_imgs):
                img = Image.open(p).convert('RGB')
                x = tf(img).unsqueeze(0)
                _ = backbone(x)
                f2 = features['layer2']
                f3 = nn.functional.interpolate(features['layer3'], size=f2.shape[2:], mode='bilinear', align_corners=False)
                combined = torch.cat([f2, f3], dim=1)
                pooled = nn.functional.adaptive_avg_pool2d(combined, (7, 7))
                patches = pooled.squeeze(0).reshape(pooled.shape[1], -1).T
                all_feats.append(patches)
                if (i+1) % 500 == 0:
                    log(f"  {i+1}/{len(all_imgs)}")

        bank = torch.cat(all_feats, dim=0)
        n = max(int(bank.shape[0] * 0.05), 500)
        idx = torch.randperm(bank.shape[0])[:n]
        coreset = bank[idx]
        np.save(str(WEIGHTS_DIR / 'm6_patchcore_coreset.npy'), coreset.numpy())
        log(f"M6 PatchCore v3 완료 — coreset {coreset.shape}")
    except Exception as e:
        log(f"M6 에러: {e}")

    log("=" * 50)
    log("전체 재학습 v3 완료")
    log("ONNX 파일:")
    for f in sorted(WEIGHTS_DIR.glob("*")):
        log(f"  {f.name}: {f.stat().st_size / 1024 / 1024:.1f} MB")
