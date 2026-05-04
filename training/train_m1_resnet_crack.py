# =============================================
# train_m1_resnet_crack.py
# M1-Stage2: ResNet50 균열 유형 분류 (구조균열 vs 마감균열)
# 입력: YOLO가 검출한 crack bbox의 ROI 크롭 (224x224)
# 출력: models_weights/m1_resnet_crack_classifier.onnx
#
# 데이터셋 구조 (ImageFolder):
#   datasets/structural_crops/
#     train/ crack_structural/ *.jpg
#            crack_finishing/  *.jpg
#     val/   ...
#     test/  ...
#
# 사용법:
#   cd backend/training
#   python train_m1_resnet_crack.py
# =============================================

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


# ── 하이퍼파라미터 ──────────────────────────
NUM_CLASSES = 5   # 실제 crop 데이터 클래스 수 (ImageFolder 알파벳순)
CLASS_NAMES = ["caulking_indicator", "crack_indicator", "moisture_indicator", "structural_damage", "waterproof_defect"]
INPUT_SIZE = 224
BATCH_SIZE = 16  # M5 YOLO과 GPU 공유 (8GB VRAM 분할)
EPOCHS = 50
LR = 1e-4
WEIGHT_DECAY = 1e-4
T_MAX = 50
ETA_MIN = 1e-6

DATA_DIR = Path("datasets/structural_crops")
WEIGHTS_DIR = Path("../models_weights")
OUTPUT_NAME = "m1_resnet_crack_classifier"


# ── 데이터 증강 ──────────────────────────────
train_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(INPUT_SIZE),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.RandomGrayscale(p=0.1),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.2),
])

val_transforms = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def build_model() -> nn.Module:
    """ResNet50 + NUM_CLASSES fc head."""
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.fc.in_features, NUM_CLASSES),
    )
    return model


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[M1-ResNet] Device: {device}")

    train_dataset = datasets.ImageFolder(DATA_DIR / "train", train_transforms)
    val_dataset = datasets.ImageFolder(DATA_DIR / "val", val_transforms)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=(device.type == "cuda"),
    )

    model = build_model().to(device)

    # 클래스 가중치 (구조손상에 높은 가중치 — HIGH severity, 소수 클래스 보정)
    class_weights = torch.tensor([1.0, 1.0, 2.0, 2.0, 1.0], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_MAX, eta_min=ETA_MIN)

    best_val_acc = 0.0
    run_dir = Path("runs/m1_resnet_crack")
    run_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(EPOCHS):
        # ── Train ──
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        scheduler.step()

        # ── Validate ──
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        val_acc = correct / total

        avg_loss = train_loss / len(train_loader)
        print(f"  Epoch {epoch+1:3d}/{EPOCHS} | Loss: {avg_loss:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": CLASS_NAMES,
                    "val_acc": val_acc,
                },
                run_dir / "best.pt",
            )

    print(f"\n[M1-ResNet] 최고 검증 정확도: {best_val_acc:.4f}")

    # 최적 가중치 재로드 → ONNX 변환
    ckpt = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    export_to_onnx(model, device)


def export_to_onnx(model: nn.Module, device: torch.device):
    """ResNet50 → ONNX 변환."""
    model.eval()
    dummy = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE).to(device)

    onnx_path = WEIGHTS_DIR / f"{OUTPUT_NAME}.onnx"
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        opset_version=17,
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={
            "image": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
    )

    # ONNX 검증
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    print(f"[M1-ResNet] ONNX 저장 + 검증 완료: {onnx_path}")


if __name__ == "__main__":
    train()
