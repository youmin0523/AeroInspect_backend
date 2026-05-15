"""
services/furniture_inference.py
역할: 평면도 가구 검출 ML 추론 (YOLOv8) + 도형 기반 결과와의 하이브리드 결합.

흐름:
  1) ultralytics YOLO 가 models_weights/floorplan_furniture_yolo.pt 를 로드
  2) extract_walls_from_bytes() 가 호출 → 도형 기반 furniture 가 1차 후보
  3) 이 모듈이 ML 추론 결과로 보강 (NMS 결합)
  4) 가중치 파일 없으면 graceful pass-through (도형 기반만 반환)

영상 추론 라인의 furniture_aware (RGB 카메라 영상용) 와는 별개 모델.
이건 평면도 도면 심볼 전용.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np

# YOLO 가중치 위치 — 학습 후 train_floorplan_yolo.py 가 여기에 복사
DEFAULT_WEIGHTS = Path("models_weights/floorplan_furniture_yolo.pt")
CLASS_NAMES = ['rectangular', 'circular', 'small', 'unknown']


class FloorplanFurnitureDetector:
    """평면도 가구 YOLO 검출기. 가중치 없으면 비활성."""

    def __init__(self, weights_path: Optional[Path] = None, conf_threshold: float = 0.25):
        self.weights_path = Path(weights_path or DEFAULT_WEIGHTS)
        self.conf_threshold = conf_threshold
        self._model = None
        self._available = False
        self._try_load()

    def _try_load(self):
        if not self.weights_path.exists():
            return
        try:
            from ultralytics import YOLO
            self._model = YOLO(str(self.weights_path))
            self._available = True
        except ImportError:
            # ultralytics 미설치 → 도형 기반만 사용
            return
        except Exception as e:
            print(f"[FloorplanFurnitureDetector] 가중치 로드 실패: {e}")
            return

    @property
    def available(self) -> bool:
        return self._available

    def detect(self, img_bgr: np.ndarray) -> list[dict]:
        """
        이미지에서 가구 검출.

        Returns:
            [{cx, cy, w, h, angle, label, confidence}, ...]  — 정규화 0-1
        """
        if not self._available:
            return []
        H, W = img_bgr.shape[:2]
        try:
            results = self._model.predict(
                img_bgr, conf=self.conf_threshold, verbose=False
            )
        except Exception:
            return []

        out: list[dict] = []
        for r in results:
            if r.boxes is None:
                continue
            xyxy = r.boxes.xyxy.cpu().numpy()
            cls = r.boxes.cls.cpu().numpy().astype(int)
            confs = r.boxes.conf.cpu().numpy()
            for (x1, y1, x2, y2), c, conf in zip(xyxy, cls, confs):
                cx = (x1 + x2) / 2 / W
                cy = (y1 + y2) / 2 / H
                w = (x2 - x1) / W
                h = (y2 - y1) / H
                label = CLASS_NAMES[c] if 0 <= c < len(CLASS_NAMES) else 'unknown'
                out.append({
                    'cx': round(float(cx), 4), 'cy': round(float(cy), 4),
                    'w': round(float(w), 4), 'h': round(float(h), 4),
                    'angle': 0.0,
                    'label': label,
                    'confidence': round(float(conf), 3),
                })
        return out


# 모듈 레벨 싱글톤 — lazy 로드
_detector: Optional[FloorplanFurnitureDetector] = None


def get_detector() -> FloorplanFurnitureDetector:
    global _detector
    if _detector is None:
        _detector = FloorplanFurnitureDetector()
    return _detector


def merge_furniture_detections(
    shape_based: list[dict],
    ml_based: list[dict],
    iou_threshold: float = 0.4,
) -> list[dict]:
    """
    도형 기반 + ML 기반 가구 검출 결과 병합.
    - ML 우선 (confidence > 0.5 면 도형 기반 동위치 후보 교체)
    - 도형만 검출된 가구는 보존 (안전 마진 — 회피 우선)
    """
    if not ml_based:
        return shape_based
    if not shape_based:
        return ml_based

    # 모두 후보 풀에 넣고 NMS-style 머지 (ML 우선 정렬)
    candidates = []
    for f in ml_based:
        candidates.append({**f, '_priority': f.get('confidence', 0.5) + 1.0})
    for f in shape_based:
        candidates.append({**f, '_priority': 0.5})

    candidates.sort(key=lambda c: c['_priority'], reverse=True)

    def iou(a, b):
        ax1 = a['cx'] - a['w'] / 2; ay1 = a['cy'] - a['h'] / 2
        ax2 = a['cx'] + a['w'] / 2; ay2 = a['cy'] + a['h'] / 2
        bx1 = b['cx'] - b['w'] / 2; by1 = b['cy'] - b['h'] / 2
        bx2 = b['cx'] + b['w'] / 2; by2 = b['cy'] + b['h'] / 2
        iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
        ih = max(0.0, min(ay2, by2) - max(ay1, by1))
        inter = iw * ih
        if inter <= 0:
            return 0.0
        union = a['w'] * a['h'] + b['w'] * b['h'] - inter
        return inter / union if union > 0 else 0.0

    kept = []
    for c in candidates:
        if any(iou(c, k) > iou_threshold for k in kept):
            continue
        kept.append({k: v for k, v in c.items() if not k.startswith('_')})
    return kept
