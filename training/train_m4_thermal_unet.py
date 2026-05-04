# =============================================
# train_m4_thermal_unet.py
# M4: U-Net (EfficientNet-B3 backbone) 열화상 세그멘테이션
# 클래스: background, window_insulation, wall_insulation, window_airtight, floor_heating
# 출력: models_weights/m4_unet_thermal_insulation.onnx
#
# 데이터셋 구조:
#   datasets/thermal/
#     thermal_maps/  *.npy  (float32, H×W 온도맵)
#     masks/         *_mask.npy  (uint8, H×W 클래스 인덱스 0-4)
#     rgb/           *.jpg  (M4 컨텍스트 모델용, 선택)
#
# 사용법:
#   cd backend/training
#   python train_m4_thermal_unet.py
# =============================================

from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import onnx
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader, Dataset, random_split

# segmentation_models_pytorch: pip install segmentation-models-pytorch
import segmentation_models_pytorch as smp


# ── 설정 로드 ──────────────────────────────
with open("configs/thermal_unet.yaml", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

NUM_CLASSES = CFG["model"]["num_classes"]
CLASS_NAMES = [CFG["classes"][i] for i in range(NUM_CLASSES)]
INPUT_H = CFG["input"]["height"]
INPUT_W = CFG["input"]["width"]
BATCH_SIZE = CFG["training"]["batch_size"]
EPOCHS = CFG["training"]["epochs"]
LR = CFG["training"]["lr"]
WEIGHT_DECAY = CFG["training"]["weight_decay"]
T_MAX = CFG["training"]["T_max"]
ETA_MIN = CFG["training"]["eta_min"]
CLASS_WEIGHTS = CFG.get("class_weights", [1.0] * NUM_CLASSES)

WEIGHTS_DIR = Path("../models_weights")
OUTPUT_NAME = "m4_unet_thermal_insulation"


# ── 데이터셋 ──────────────────────────────
class ThermalDefectDataset(Dataset):
    """열화상 온도맵 + 세그멘테이션 마스크 데이터셋."""

    def __init__(self, thermal_dir: str, mask_dir: str, augment: bool = False):
        self.thermal_dir = Path(thermal_dir)
        self.mask_dir = Path(mask_dir)
        self.augment = augment
        self.files = sorted([
            f.name for f in self.thermal_dir.glob("*.npy")
            if (self.mask_dir / f.name.replace(".npy", "_mask.npy")).exists()
        ])

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        fname = self.files[idx]
        temp_map = np.load(self.thermal_dir / fname).astype(np.float32)
        mask = np.load(self.mask_dir / fname.replace(".npy", "_mask.npy")).astype(np.int64)

        if self.augment:
            temp_map, mask = self._augment(temp_map, mask)

        # 정규화 + 3채널 복제 (EfficientNet 호환)
        temp_norm = (temp_map - temp_map.mean()) / (temp_map.std() + 1e-6)
        input_3ch = np.stack([temp_norm] * 3, axis=0).astype(np.float32)

        return torch.from_numpy(input_3ch), torch.from_numpy(mask).long()

    @staticmethod
    def _augment(temp_map: np.ndarray, mask: np.ndarray):
        """열화상 전용 물리 기반 증강."""
        # 온도 오프셋 (계절별 외기온 시뮬레이션)
        temp_map = temp_map + np.random.uniform(-8.0, 8.0)

        # NETD 노이즈 (IRC-256CA ≈ 50mK)
        temp_map = temp_map + np.random.normal(0, 0.05, temp_map.shape).astype(np.float32)

        # 좌우 반전
        if np.random.random() > 0.5:
            temp_map = np.fliplr(temp_map).copy()
            mask = np.fliplr(mask).copy()

        # 회전 (±10도)
        h, w = temp_map.shape
        angle = np.random.uniform(-10, 10)
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        temp_map = cv2.warpAffine(temp_map, M, (w, h))
        mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST)

        # 온도 스케일 변동
        scale = np.random.uniform(-0.04, 0.04)
        temp_map = temp_map * (1.0 + scale)

        return temp_map, mask


# ── Loss: Dice + CrossEntropy 결합 ──
class DiceCELoss(nn.Module):
    def __init__(self, num_classes: int, class_weights: list):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(
            weight=torch.tensor(class_weights, dtype=torch.float32)
        )
        self.num_classes = num_classes

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce_loss = self.ce(pred, target)

        pred_soft = torch.softmax(pred, dim=1)
        target_oh = nn.functional.one_hot(target, self.num_classes)
        target_oh = target_oh.permute(0, 3, 1, 2).float()

        intersection = (pred_soft * target_oh).sum(dim=(2, 3))
        union = pred_soft.sum(dim=(2, 3)) + target_oh.sum(dim=(2, 3))
        dice = (2.0 * intersection + 1e-6) / (union + 1e-6)
        dice_loss = 1.0 - dice.mean()

        return 0.5 * ce_loss + 0.5 * dice_loss


def build_model() -> nn.Module:
    return smp.Unet(
        encoder_name=CFG["model"]["encoder"],
        encoder_weights=CFG["model"]["encoder_weights"],
        in_channels=3,
        classes=NUM_CLASSES,
        activation=None,
    )


def train():
    # GPU는 YOLO+ResNet이 사용 중 — 소규모 데이터(1.8K)라 CPU로 충분
    device = torch.device("cpu")
    print(f"[M4-UNet] Device: {device}")

    full_dataset = ThermalDefectDataset(
        CFG["data"]["thermal_dir"], CFG["data"]["mask_dir"], augment=True,
    )

    n = len(full_dataset)
    n_train = int(n * CFG["data"]["train_ratio"])
    n_val = int(n * CFG["data"]["val_ratio"])
    n_test = n - n_train - n_val

    train_ds, val_ds, _ = random_split(
        full_dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42),
    )

    # num_workers=0: Windows에서 multiprocessing fork 문제 방지
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=0, pin_memory=True)

    model = build_model().to(device)
    criterion = DiceCELoss(NUM_CLASSES, CLASS_WEIGHTS).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_MAX, eta_min=ETA_MIN)

    best_dice = 0.0
    run_dir = Path("runs/m4_thermal_unet")
    run_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(EPOCHS):
        model.train()
        for inputs, masks in train_loader:
            inputs, masks = inputs.to(device), masks.to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), masks)
            loss.backward()
            optimizer.step()
        scheduler.step()

        # 검증: mean Dice (background 제외)
        model.eval()
        dice_scores = []
        with torch.no_grad():
            for inputs, masks in val_loader:
                inputs, masks = inputs.to(device), masks.to(device)
                preds = model(inputs).argmax(dim=1)
                for cls_idx in range(1, NUM_CLASSES):
                    p = (preds == cls_idx).float()
                    g = (masks == cls_idx).float()
                    inter = (p * g).sum()
                    dice = (2 * inter + 1e-6) / (p.sum() + g.sum() + 1e-6)
                    dice_scores.append(dice.item())

        mean_dice = np.mean(dice_scores) if dice_scores else 0.0
        print(f"  Epoch {epoch+1:3d}/{EPOCHS} | Mean Dice: {mean_dice:.4f}")

        if mean_dice > best_dice:
            best_dice = mean_dice
            torch.save(model.state_dict(), run_dir / "best.pt")

    print(f"\n[M4-UNet] 최고 검증 Dice: {best_dice:.4f}")

    # ONNX 변환
    model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device, weights_only=True))
    model.eval()

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    onnx_path = WEIGHTS_DIR / f"{OUTPUT_NAME}.onnx"
    dummy = torch.randn(1, 3, INPUT_H, INPUT_W).to(device)
    torch.onnx.export(
        model, dummy, str(onnx_path), opset_version=17,
        input_names=["thermal_3ch"], output_names=["segmentation_mask"],
        dynamic_axes={"thermal_3ch": {0: "batch"}, "segmentation_mask": {0: "batch"}},
    )
    onnx.checker.check_model(onnx.load(str(onnx_path)))
    print(f"[M4-UNet] ONNX 저장 + 검증 완료: {onnx_path}")


if __name__ == "__main__":
    train()
