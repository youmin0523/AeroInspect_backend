# =============================================
# train_m2_resnet_surface.py
# M2-Stage2: ResNet50 마감·표면 하자 유형 분류
# 클래스: wallpaper_seam, wallpaper_bubble, paint_stain, scratch, baseboard_damage
# 입력: YOLO Stage 1이 검출한 ROI 크롭 (224x224)
# 출력: models_weights/m2_resnet_surface_classifier.onnx
#
# 데이터셋 구조 (ImageFolder):
#   datasets/surface_crops/
#     train/ wallpaper_seam/ *.jpg
#            wallpaper_bubble/ *.jpg
#            paint_stain/ *.jpg
#            scratch/ *.jpg
#            baseboard_damage/ *.jpg
#     val/   ...
#     test/  ...
# =============================================

from __future__ import annotations

from pathlib import Path

import onnx
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

NUM_CLASSES = 2   # 26K crop: surface_defect, baseboard_damage
CLASS_NAMES = [
    "baseboard_damage",     # C-05 (ImageFolder 알파벳순)
    "surface_defect",       # C-01~C-04 통합
]
INPUT_SIZE = 224
BATCH_SIZE = 8   # M1-YOLO와 GPU 공유 (여유 1.4GB)
EPOCHS = 80
LR = 1e-4

DATA_DIR = Path("datasets/surface_crops_v2")  # 26K crop
WEIGHTS_DIR = Path("../models_weights")
OUTPUT_NAME = "m2_resnet_surface_classifier"


train_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(INPUT_SIZE),
    transforms.RandomHorizontalFlip(p=0.5),
    # 마감·표면 특화: 강한 색상 변형
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1),
    transforms.RandomGrayscale(p=0.05),
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
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(model.fc.in_features, NUM_CLASSES),
    )
    return model


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[M2-ResNet] Device: {device}")

    train_dataset = datasets.ImageFolder(DATA_DIR / "train", train_transforms)
    val_dataset = datasets.ImageFolder(DATA_DIR / "val", val_transforms)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=0, pin_memory=(device.type == "cuda"))

    model = build_model().to(device)

    # baseboard(소수)에 높은 가중치
    class_weights = torch.tensor([2.0, 1.0], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    best_val_acc = 0.0
    run_dir = Path("runs/m2_resnet_surface")
    run_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
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

        print(f"  Epoch {epoch+1:3d}/{EPOCHS} | Loss: {train_loss/len(train_loader):.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({"model_state_dict": model.state_dict(),
                         "class_names": CLASS_NAMES, "val_acc": val_acc},
                        run_dir / "best.pt")

    print(f"\n[M2-ResNet] 최고 검증 정확도: {best_val_acc:.4f}")

    ckpt = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    dummy = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE).to(device)
    onnx_path = WEIGHTS_DIR / f"{OUTPUT_NAME}.onnx"
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(model, dummy, str(onnx_path), opset_version=17,
                       input_names=["image"], output_names=["logits"],
                       dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}})
    onnx.checker.check_model(onnx.load(str(onnx_path)))
    print(f"[M2-ResNet] ONNX 저장 + 검증 완료: {onnx_path}")


if __name__ == "__main__":
    train()
