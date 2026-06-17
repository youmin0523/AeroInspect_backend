"""
tests/test_image_crop_roi.py
역할: image_utils.crop_roi 의 여백(padding)이 '박스 크기 비례' 인지 검증.
  회귀 방지: padding 을 정규화 좌표에 그대로 더해(전체 이미지의 padding*100%) 작은 박스가
  과도하게 크롭되던 버그. 이제 pad = box_edge * padding.
실행: pytest tests/test_image_crop_roi.py -v
"""

from __future__ import annotations

import numpy as np

from app.utils.image_utils import crop_roi


def test_padding_is_proportional_to_box_size():
    # 200x200 이미지, 중앙의 작은 박스 (정규화 w=h=0.1 → 20px)
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    cx, cy, bw, bh = 0.5, 0.5, 0.1, 0.1
    padding = 0.5  # 박스 변의 50%

    crop = crop_roi(frame, (cx, cy, bw, bh), padding=padding)
    assert crop is not None
    h, w = crop.shape[:2]

    # 기대: 변 = (bw + 2*bw*padding)*W = (0.1 + 0.1)*200 = 40px (±1 반올림 허용)
    assert abs(w - 40) <= 2, f"width {w} != ~40 (박스 비례 여백)"
    assert abs(h - 40) <= 2, f"height {h} != ~40 (박스 비례 여백)"
    # 회귀 가드: 옛 동작(이미지 비율 여백)이면 변 = (0.1 + 2*0.5)*200 = 220 → 클리핑 200
    assert w < 200 and h < 200


def test_zero_padding_matches_box():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    crop = crop_roi(frame, (0.5, 0.5, 0.2, 0.2), padding=0.0)
    assert crop is not None
    h, w = crop.shape[:2]
    assert abs(w - 20) <= 1 and abs(h - 20) <= 1
