# =============================================
# app/services/wallpaper_classifier.py
# 역할: ResNet50 벽지 하자 분류 서비스 (19 클래스)
#       - torchvision ResNet50, ImageNet pretrained → Transfer Learning
#       - 체크포인트 dict 구조: {'model_state_dict', 'class_names', 'val_acc'}
#       - class_names가 하드코딩 WALLPAPER_CLASSES와 일치하는지 assert
#       - 입력 전처리: Resize(224) → ToTensor → ImageNet Normalize
#       - top1 + top3 softmax 확률 반환
#
# ⚠️ 중요: 'good' 클래스는 "정상"이 아니라 "터짐(Burst)" 하자 유형이다.
#          CLASS_DISPLAY_MAP으로 반드시 Burst/터짐으로 표시.
# =============================================

from __future__ import annotations

import os
from typing import List, Optional, Tuple

import numpy as np

from app.services.defect_taxonomy import (
    WALLPAPER_CLASSES,
    get_display_names,
)

# torch/torchvision은 상단 import 시 FastAPI 테스트 수집에서도 매번 로딩되므로
# lazy import 전략 (load_model 호출 시점에 import)
_torch = None
_F = None
_transforms = None
_models = None


def _lazy_import_torch():
    """torch/torchvision 지연 로드 (모델 로드 시점에만)."""
    global _torch, _F, _transforms, _models
    if _torch is None:
        import torch
        import torch.nn.functional as F
        from torchvision import models, transforms
        _torch = torch
        _F = F
        _transforms = transforms
        _models = models


class WallpaperClassifier:
    """ResNet50 기반 19-클래스 벽지 하자 분류 싱글톤."""

    INPUT_SIZE = 224
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    def __init__(self):
        self._model = None
        self._device = None
        self._preprocess = None
        self._val_acc: Optional[float] = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def device(self) -> str:
        return "cuda" if (self._device is not None and str(self._device).startswith("cuda")) else "cpu"

    @property
    def val_acc(self) -> Optional[float]:
        return self._val_acc

    def load_model(self, weights_path: str, device_pref: str = "auto") -> None:
        """
        ResNet50 체크포인트 로드 + WALLPAPER_CLASSES 검증.

        Args:
            weights_path: resnet50_wallpaper_best.pt 경로
            device_pref: 'auto' | 'cuda' | 'cpu'

        Raises:
            FileNotFoundError: 가중치 파일 없음
            AssertionError: class_names가 WALLPAPER_CLASSES와 불일치
        """
        if not os.path.exists(weights_path):
            raise FileNotFoundError(
                f"[Wallpaper] 가중치 파일 없음: {weights_path}. "
                f"weights/ 폴더에 resnet50_wallpaper_best.pt를 넣어주세요."
            )

        _lazy_import_torch()
        torch = _torch

        # 디바이스 선택
        if device_pref == "auto":
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(device_pref)

        # 체크포인트 로드
        ckpt = torch.load(weights_path, map_location=self._device, weights_only=False)

        if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
            raise ValueError(
                f"[Wallpaper] 체크포인트 형식 오류. "
                f"'model_state_dict' 키가 있는 dict가 필요하나, 받은 타입: {type(ckpt)}"
            )

        # 클래스명 검증 — WALLPAPER_CLASSES와 정확히 일치해야 함
        ckpt_classes = ckpt.get("class_names")
        if ckpt_classes is None:
            raise ValueError(
                "[Wallpaper] 체크포인트에 'class_names' 키가 없습니다."
            )
        assert list(ckpt_classes) == WALLPAPER_CLASSES, (
            f"[Wallpaper] class_names 불일치:\n"
            f"  체크포인트({len(ckpt_classes)}): {list(ckpt_classes)}\n"
            f"  하드코딩({len(WALLPAPER_CLASSES)}): {WALLPAPER_CLASSES}"
        )

        self._val_acc = ckpt.get("val_acc")

        # 모델 아키텍처 생성 (fc 레이어를 19-클래스로 교체)
        model = _models.resnet50(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, len(WALLPAPER_CLASSES))
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(self._device)
        model.eval()
        self._model = model

        # 전처리 파이프라인
        self._preprocess = _transforms.Compose([
            _transforms.ToPILImage(),
            _transforms.Resize((self.INPUT_SIZE, self.INPUT_SIZE)),
            _transforms.ToTensor(),
            _transforms.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD),
        ])

        print(
            f"[Wallpaper] ResNet50 로드 완료: device={self.device}, "
            f"val_acc={self._val_acc}, classes={len(WALLPAPER_CLASSES)}"
        )

    def classify(
        self,
        image_rgb: np.ndarray,
    ) -> Tuple[str, float, List[Tuple[str, float]]]:
        """
        RGB 이미지 → top1 클래스명, top1 확률, top3 리스트.

        Args:
            image_rgb: H,W,3 RGB uint8 numpy 배열

        Returns:
            (top1_class, top1_conf, [(class, conf), ...3개])
        """
        if self._model is None:
            raise RuntimeError("[Wallpaper] 모델이 로드되지 않았습니다.")

        torch = _torch
        F = _F

        tensor = self._preprocess(image_rgb).unsqueeze(0).to(self._device)

        with torch.no_grad():
            logits = self._model(tensor)
            probs = F.softmax(logits, dim=1)[0]  # [19]

        topk = torch.topk(probs, k=min(3, len(WALLPAPER_CLASSES)))
        top_indices = topk.indices.cpu().tolist()
        top_values = topk.values.cpu().tolist()

        top3 = [
            (WALLPAPER_CLASSES[idx], float(conf))
            for idx, conf in zip(top_indices, top_values)
        ]
        top1_class, top1_conf = top3[0]
        return top1_class, top1_conf, top3


# ── 모듈 레벨 싱글톤 ─────────────────────────
wallpaper_classifier = WallpaperClassifier()


__all__ = [
    "WallpaperClassifier",
    "wallpaper_classifier",
    "get_display_names",
]
