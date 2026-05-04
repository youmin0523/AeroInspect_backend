# =============================================
# app/services/active_learning.py
# 역할: Hard Example Mining — 모델이 불확실해한 프레임 자동 수집
#       - 저신뢰 검출(conf 0.15~0.40): 모델이 "애매하다"고 판단한 영역
#       - PatchCore high + YOLO miss: 이상은 감지했으나 하자로 분류 못함
#       - Temporal reject: 투표 탈락(1/5)한 일시적 검출
#       → 이 프레임들이 재학습 시 가장 높은 학습 효과
#
# 저장: ./training/hard_examples/{date}/{model}/ (YOLO 학습 포맷)
# 용량: 프레임당 ~50KB JPEG, 2시간 검사 기준 ~3GB (디스크 관리 필요)
# =============================================

from __future__ import annotations

import os
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Deque, Dict, List, Optional

import cv2
import numpy as np


@dataclass
class HardExample:
    """수집된 Hard Example 메타데이터."""
    frame_id: int
    timestamp: float
    reason: str           # "low_conf" | "patchcore_miss" | "temporal_reject"
    model_source: str     # "m1" | "m2" | "m3" | ...
    detections: List[dict]
    image_path: Optional[str] = None  # 저장된 이미지 경로


class HardExampleMiner:
    """
    추론 파이프라인에서 불확실한 프레임을 수집하는 서비스.

    Args:
        output_dir: Hard example 저장 디렉토리
        low_conf_range: (min, max) — 이 범위의 conf를 가진 검출을 수집
        max_buffer: 메모리 버퍼 최대 크기 (디스크 기록 전 임시 보관)
        save_interval: 디스크 저장 주기 (초). 0이면 즉시 저장
        enabled: False면 수집 비활성화
    """

    def __init__(
        self,
        output_dir: str = "./training/hard_examples",
        low_conf_range: tuple = (0.15, 0.40),
        max_buffer: int = 500,
        save_interval: float = 30.0,
        enabled: bool = True,
    ):
        self.output_dir = output_dir
        self.low_conf_min = low_conf_range[0]
        self.low_conf_max = low_conf_range[1]
        self.max_buffer = max_buffer
        self.save_interval = save_interval
        self.enabled = enabled

        self._buffer: Deque[tuple] = deque(maxlen=max_buffer)  # (HardExample, frame_bgr)
        self._last_save_time = time.time()
        self._total_collected = 0
        self._total_saved = 0

    # ── 공개 API ─────────────────────────────

    def check_and_collect(
        self,
        frame_bgr: np.ndarray,
        detections: List[dict],
        frame_id: int,
        anomaly_score: Optional[float] = None,
    ) -> int:
        """
        검출 결과를 분석하여 hard example 여부 판단 → 수집.

        Returns:
            이번 프레임에서 수집된 hard example 수
        """
        if not self.enabled:
            return 0

        collected = 0

        # 1) 저신뢰 검출 (불확실 영역)
        low_conf_dets = [
            d for d in detections
            if self.low_conf_min <= d.get("conf", 0) <= self.low_conf_max
        ]
        if low_conf_dets:
            he = HardExample(
                frame_id=frame_id,
                timestamp=time.time(),
                reason="low_conf",
                model_source=low_conf_dets[0].get("defect_source", "unknown"),
                detections=low_conf_dets,
            )
            self._buffer.append((he, frame_bgr.copy()))
            self._total_collected += 1
            collected += 1

        # 2) PatchCore 이상 + YOLO 미탐지 (보이지 않는 하자)
        if anomaly_score is not None and anomaly_score > 0.5:
            has_yolo_det = any(
                d.get("conf", 0) > self.low_conf_max for d in detections
            )
            if not has_yolo_det:
                he = HardExample(
                    frame_id=frame_id,
                    timestamp=time.time(),
                    reason="patchcore_miss",
                    model_source="m6_patchcore",
                    detections=detections,
                )
                self._buffer.append((he, frame_bgr.copy()))
                self._total_collected += 1
                collected += 1

        # 주기적 디스크 저장
        if time.time() - self._last_save_time >= self.save_interval:
            self.flush_to_disk()

        return collected

    def collect_temporal_reject(
        self,
        frame_bgr: np.ndarray,
        rejected_dets: List[dict],
        frame_id: int,
    ) -> None:
        """Temporal filter에서 투표 탈락한 검출 수집."""
        if not self.enabled or not rejected_dets:
            return

        he = HardExample(
            frame_id=frame_id,
            timestamp=time.time(),
            reason="temporal_reject",
            model_source=rejected_dets[0].get("defect_source", "unknown"),
            detections=rejected_dets,
        )
        self._buffer.append((he, frame_bgr.copy()))
        self._total_collected += 1

    def flush_to_disk(self) -> int:
        """버퍼의 hard example을 디스크에 저장."""
        if not self._buffer:
            return 0

        date_str = datetime.now().strftime("%Y%m%d")
        saved = 0

        while self._buffer:
            he, frame_bgr = self._buffer.popleft()

            # 디렉토리: {output_dir}/{date}/{reason}/
            subdir = os.path.join(self.output_dir, date_str, he.reason)
            os.makedirs(subdir, exist_ok=True)

            # 이미지 저장
            filename = f"frame_{he.frame_id}_{int(he.timestamp)}.jpg"
            img_path = os.path.join(subdir, filename)
            cv2.imwrite(img_path, frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
            he.image_path = img_path

            # 메타데이터 저장 (YOLO 학습 시 활용)
            meta_path = img_path.replace(".jpg", ".txt")
            self._write_yolo_annotation(meta_path, he, frame_bgr.shape)

            saved += 1

        self._total_saved += saved
        self._last_save_time = time.time()
        return saved

    @property
    def stats(self) -> Dict[str, int]:
        return {
            "collected": self._total_collected,
            "saved": self._total_saved,
            "buffer_size": len(self._buffer),
        }

    def reset(self) -> None:
        """버퍼 초기화."""
        self._buffer.clear()
        self._total_collected = 0
        self._total_saved = 0

    # ── 내부 ─────────────────────────────────

    @staticmethod
    def _write_yolo_annotation(
        path: str, he: HardExample, img_shape: tuple,
    ) -> None:
        """YOLO 형식 어노테이션 파일 작성 (class_id cx cy w h)."""
        h, w = img_shape[:2]
        lines = []
        for det in he.detections:
            bbox = det.get("bbox_xyxy")
            if bbox is None or len(bbox) != 4:
                continue
            x1, y1, x2, y2 = bbox
            cx = ((x1 + x2) / 2) / w
            cy = ((y1 + y2) / 2) / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            # class_id는 클래스명을 그대로 기록 (후처리 시 매핑)
            cls_name = det.get("class", "unknown")
            conf = det.get("conf", 0)
            lines.append(f"{cls_name} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} {conf:.4f}")

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))


# ── 모듈 레벨 싱글톤 ─────────────────────────
hard_example_miner = HardExampleMiner(enabled=False)  # 기본 비활성, config에서 활성화


__all__ = ["HardExampleMiner", "HardExample", "hard_example_miner"]
