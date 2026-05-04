# =============================================
# train_m3_resnet_floor_window.py
# M3-Stage2: ResNet50 바닥·창호 하자 유형 분류
# 클래스: floor_stain, grout_defect, glass_scratch, frame_paint_defect
# 출력: models_weights/m3_resnet_floor_window_classifier.onnx
# =============================================

from __future__ import annotations

from pathlib import Path

import onnx
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

NUM_CLASSES = 3   # 실제 crop 데이터 클래스 수
CLASS_NAMES = [
    "floor_defect",     # D-02~D-04
    "glass_defect",     # E-01
    "frame_defect",     # E-02
]
INPUT_SIZE = 224
BATCH_SIZE = 16  # M5 YOLO과 GPU 공유
EPOCHS = 60
LR = 1e-4

DATA_DIR = Path("datasets/floor_window_crops")
WEIGHTS_DIR = Path("../models_weights")
OUTPUT_NAME = "m3_resnet_floor_window_classifier"


train_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(INPUT_SIZE),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.15),
])

val_transforms = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def build_model() -> nn.Module:
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(model.fc.in_features, NUM_CLASSES))
    return model


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[M3-ResNet] Device: {device}")

    train_loader = DataLoader(
        datasets.ImageFolder(DATA_DIR / "train", train_transforms),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(
        datasets.ImageFolder(DATA_DIR / "val", val_transforms),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=(device.type == "cuda"))

    model = build_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    best_val_acc = 0.0
    run_dir = Path("runs/m3_resnet_floor_window")
    run_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(EPOCHS):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                _, predicted = model(images).max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        val_acc = correct / total
        print(f"  Epoch {epoch+1:3d}/{EPOCHS} | Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({"model_state_dict": model.state_dict(),
                         "class_names": CLASS_NAMES, "val_acc": val_acc},
                        run_dir / "best.pt")

    print(f"\n[M3-ResNet] 최고 검증 정확도: {best_val_acc:.4f}")

    ckpt = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    onnx_path = WEIGHTS_DIR / f"{OUTPUT_NAME}.onnx"
    torch.onnx.export(model, torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE).to(device),
                       str(onnx_path), opset_version=17,
                       input_names=["image"], output_names=["logits"],
                       dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}})
    onnx.checker.check_model(onnx.load(str(onnx_path)))
    print(f"[M3-ResNet] ONNX 저장 완료: {onnx_path}")


if __name__ == "__main__":
    train()
