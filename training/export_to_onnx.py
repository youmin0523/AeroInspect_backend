# =============================================
# export_to_onnx.py
# 전 모델 일괄 ONNX 변환 유틸리티
# 기존 .pt 체크포인트를 models_weights/*.onnx로 변환
#
# 사용법:
#   cd backend/training
#   python export_to_onnx.py              # 전체 변환
#   python export_to_onnx.py --model m1   # 특정 모델만
# =============================================

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import onnx
import torch
from torchvision import models

WEIGHTS_DIR = Path("../models_weights")


# ── YOLO 모델 변환 ──────────────────────────
def export_yolo(pt_path: str, output_name: str):
    """ultralytics YOLO .pt → .onnx"""
    from ultralytics import YOLO

    model = YOLO(pt_path)
    model.export(format="onnx", opset=17, dynamic=True, simplify=True)

    onnx_src = pt_path.replace(".pt", ".onnx")
    onnx_dst = WEIGHTS_DIR / f"{output_name}.onnx"
    shutil.copy2(onnx_src, onnx_dst)
    print(f"  [YOLO] {onnx_dst}")


# ── ResNet 모델 변환 ─────────────────────────
def export_resnet(pt_path: str, output_name: str, num_classes: int):
    """PyTorch ResNet50 체크포인트 → .onnx"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(pt_path, map_location=device, weights_only=False)

    model = models.resnet50(weights=None)
    model.fc = torch.nn.Sequential(
        torch.nn.Dropout(0.3),
        torch.nn.Linear(model.fc.in_features, num_classes),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    dummy = torch.randn(1, 3, 224, 224).to(device)
    onnx_path = WEIGHTS_DIR / f"{output_name}.onnx"

    torch.onnx.export(
        model, dummy, str(onnx_path), opset_version=17,
        input_names=["image"], output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
    )
    onnx.checker.check_model(onnx.load(str(onnx_path)))
    print(f"  [ResNet] {onnx_path}")


# ── U-Net 모델 변환 ──────────────────────────
def export_unet(pt_path: str, output_name: str, num_classes: int = 5):
    """segmentation_models_pytorch U-Net → .onnx"""
    import segmentation_models_pytorch as smp

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = smp.Unet(
        encoder_name="efficientnet-b3", encoder_weights=None,
        in_channels=3, classes=num_classes, activation=None,
    )
    model.load_state_dict(torch.load(pt_path, map_location=device, weights_only=True))
    model.to(device).eval()

    dummy = torch.randn(1, 3, 192, 256).to(device)
    onnx_path = WEIGHTS_DIR / f"{output_name}.onnx"

    torch.onnx.export(
        model, dummy, str(onnx_path), opset_version=17,
        input_names=["thermal_3ch"], output_names=["segmentation_mask"],
        dynamic_axes={"thermal_3ch": {0: "batch"}, "segmentation_mask": {0: "batch"}},
    )
    onnx.checker.check_model(onnx.load(str(onnx_path)))
    print(f"  [U-Net] {onnx_path}")


# ── 전체 변환 매핑 ───────────────────────────
EXPORT_MAP = {
    "m1_yolo": ("runs/m1_structural/phase2_full/weights/best.pt", "m1_yolo_structural", "yolo"),
    "m1_resnet": ("runs/m1_resnet_crack/best.pt", "m1_resnet_crack_classifier", "resnet:2"),
    "m2_yolo": ("runs/m2_surface/phase2_full/weights/best.pt", "m2_yolo_surface", "yolo"),
    "m2_resnet": ("runs/m2_resnet_surface/best.pt", "m2_resnet_surface_classifier", "resnet:5"),
    "m3_yolo": ("runs/m3_floor_window/phase2_full/weights/best.pt", "m3_yolo_floor_window", "yolo"),
    "m3_resnet": ("runs/m3_resnet_floor_window/best.pt", "m3_resnet_floor_window_classifier", "resnet:4"),
    "m4_unet": ("runs/m4_thermal_unet/best.pt", "m4_unet_thermal_insulation", "unet:5"),
    "m5_seg": ("runs/m5_frame_seg/train/weights/best.pt", "m5_yolo_seg_frames", "yolo"),
}


def main():
    parser = argparse.ArgumentParser(description="전 모델 ONNX 일괄 변환")
    parser.add_argument("--model", type=str, default=None,
                        help="특정 모델만 변환 (예: m1, m2, m3, m4, m5)")
    args = parser.parse_args()

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    targets = EXPORT_MAP
    if args.model:
        targets = {k: v for k, v in EXPORT_MAP.items() if k.startswith(args.model)}

    print(f"=== ONNX 일괄 변환 ({len(targets)}개 모델) ===\n")

    for key, (pt_path, output_name, model_type) in targets.items():
        if not Path(pt_path).exists():
            print(f"  [SKIP] {key}: {pt_path} 없음")
            continue

        print(f"변환 중: {key}")
        if model_type == "yolo":
            export_yolo(pt_path, output_name)
        elif model_type.startswith("resnet:"):
            nc = int(model_type.split(":")[1])
            export_resnet(pt_path, output_name, nc)
        elif model_type.startswith("unet:"):
            nc = int(model_type.split(":")[1])
            export_unet(pt_path, output_name, nc)

    print("\n=== 변환 완료 ===")


if __name__ == "__main__":
    main()
