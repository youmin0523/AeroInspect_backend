"""
tools/synthesize_korean_floorplans.py
역할: 실제 외부 평면도 데이터 수집이 라이선스/접근성 한계로 어려운 상황에서
      한국 아파트 패턴(발코니·주방·욕실·거실·침실 분리)을 모방한 합성 평면도 + GT
      를 다수 생성. 정확도 측정/ML 학습 보조 데이터.

생성 패턴 (실 한국 분양 평면도 모방):
  - 84A : 침실 3 + 거실 + 주방 + 욕실 2 + 발코니
  - 59B : 침실 2 + 거실/주방 결합 + 욕실 + 발코니
  - 110C: 침실 4 + 거실 + 주방 + 욕실 2 + 알파룸 + 발코니
  - studio: 원룸 + 욕실 + 미니주방
  - L_shape: ㄱ자 거실 평면

각 평면도마다:
  - 이미지 (PNG)
  - GT JSON (벽 + 가구 + outline 정규화 좌표 + 라벨)
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

DST = Path("datasets/synthetic_korean")


@dataclass
class GTRect:
    cx: float; cy: float; w: float; h: float; label: str


@dataclass
class GTLine:
    x1: float; y1: float; x2: float; y2: float


def _norm_rect(img, x1, y1, x2, y2, label) -> GTRect:
    H, W = img.shape[:2]
    return GTRect(
        cx=(x1 + x2) / 2 / W, cy=(y1 + y2) / 2 / H,
        w=abs(x2 - x1) / W, h=abs(y2 - y1) / H, label=label,
    )


def _norm_circle(img, cx, cy, r, label) -> GTRect:
    H, W = img.shape[:2]
    return GTRect(cx=cx / W, cy=cy / H, w=(2 * r) / W, h=(2 * r) / H, label=label)


# ── 평면도 패턴 ───────────────────────────────

def make_84A():
    """84㎡ 4 베이 — 침실3 + 거실 + 주방 + 욕실2 + 발코니."""
    W, H = 1800, 1200
    img = np.full((H, W, 3), 252, dtype=np.uint8)
    BLACK = (15, 15, 15); GRAY = (95, 95, 95); THK = 14
    # 외벽
    cv2.rectangle(img, (100, 100), (W - 100, H - 100), BLACK, THK)
    # 발코니 (남쪽)
    cv2.line(img, (100, 950), (W - 100, 950), BLACK, THK)
    # 침실3 분리 (북쪽)
    cv2.line(img, (100, 500), (1100, 500), BLACK, THK)
    cv2.line(img, (450, 100), (450, 500), BLACK, THK)
    cv2.line(img, (800, 100), (800, 500), BLACK, THK)
    # 거실/주방 분리 (동쪽 큰 공간)
    cv2.line(img, (1100, 100), (1100, 950), BLACK, THK)
    # 욕실2 (서쪽 + 안방)
    cv2.line(img, (100, 700), (450, 700), BLACK, THK)  # 작은 욕실 박스
    cv2.line(img, (100, 850), (450, 850), BLACK, THK)
    cv2.line(img, (450, 700), (450, 950), BLACK, THK)
    # 침실 가구
    rects = [
        (200, 200, 400, 380, 'rectangular'),    # 침실1 침대
        (550, 200, 750, 380, 'rectangular'),    # 침실2 침대
        (900, 200, 1080, 480, 'rectangular'),   # 안방 침대
        (200, 740, 380, 830, 'rectangular'),    # 욕실 욕조
    ]
    gt_rects = []
    for (x1, y1, x2, y2, lab) in rects:
        cv2.rectangle(img, (x1, y1), (x2, y2), GRAY, -1)
        gt_rects.append(_norm_rect(img, x1, y1, x2, y2, lab))
    # 거실 소파 + 식탁
    cv2.rectangle(img, (1200, 200), (1700, 320), GRAY, -1)
    gt_rects.append(_norm_rect(img, 1200, 200, 1700, 320, 'rectangular'))
    cv2.circle(img, (1450, 600), 110, GRAY, -1)
    gt_rects.append(_norm_circle(img, 1450, 600, 110, 'circular'))
    # 주방 (남쪽 책상형)
    cv2.rectangle(img, (1200, 800), (1700, 920), GRAY, -1)
    gt_rects.append(_norm_rect(img, 1200, 800, 1700, 920, 'rectangular'))
    return img, gt_rects, '84A_3bed'


def make_59B():
    """59㎡ 거실/주방 결합 — 침실2 + 욕실 + 발코니."""
    W, H = 1500, 1100
    img = np.full((H, W, 3), 252, dtype=np.uint8)
    BLACK = (15, 15, 15); GRAY = (90, 90, 90); THK = 12
    cv2.rectangle(img, (80, 80), (W - 80, H - 80), BLACK, THK)
    cv2.line(img, (80, 850), (W - 80, 850), BLACK, THK)  # 발코니
    cv2.line(img, (700, 80), (700, 600), BLACK, THK)     # 침실 분리
    cv2.line(img, (700, 600), (W - 80, 600), BLACK, THK)
    cv2.line(img, (1100, 600), (1100, 850), BLACK, THK)  # 욕실 분리

    rects = [
        (150, 150, 450, 380, 'rectangular'),    # 침실 침대
        (800, 150, 1080, 380, 'rectangular'),   # 안방 침대
        (1180, 650, 1380, 800, 'rectangular'),  # 욕실 욕조
    ]
    gt_rects = []
    for (x1, y1, x2, y2, lab) in rects:
        cv2.rectangle(img, (x1, y1), (x2, y2), GRAY, -1)
        gt_rects.append(_norm_rect(img, x1, y1, x2, y2, lab))
    cv2.rectangle(img, (200, 650, ), (550, 780), GRAY, -1)  # 소파
    gt_rects.append(_norm_rect(img, 200, 650, 550, 780, 'rectangular'))
    cv2.circle(img, (350, 520), 90, GRAY, -1)               # 식탁
    gt_rects.append(_norm_circle(img, 350, 520, 90, 'circular'))
    return img, gt_rects, '59B_2bed'


def make_110C():
    """110㎡ 4베이 — 침실4 + 알파룸."""
    W, H = 2000, 1200
    img = np.full((H, W, 3), 252, dtype=np.uint8)
    BLACK = (15, 15, 15); GRAY = (95, 95, 95); THK = 14
    cv2.rectangle(img, (100, 100), (W - 100, H - 100), BLACK, THK)
    cv2.line(img, (100, 600), (1300, 600), BLACK, THK)
    cv2.line(img, (450, 100), (450, 600), BLACK, THK)
    cv2.line(img, (800, 100), (800, 600), BLACK, THK)
    cv2.line(img, (1150, 100), (1150, 600), BLACK, THK)
    cv2.line(img, (1300, 100), (1300, H - 100), BLACK, THK)  # 거실
    cv2.line(img, (100, 900), (1300, 900), BLACK, THK)        # 욕실/알파룸

    rects = [
        (180, 200, 380, 480, 'rectangular'),
        (530, 200, 730, 480, 'rectangular'),
        (880, 200, 1080, 480, 'rectangular'),
        (1180, 200, 1280, 480, 'rectangular'),
        (200, 700, 400, 830, 'rectangular'),  # 욕조
    ]
    gt_rects = []
    for (x1, y1, x2, y2, lab) in rects:
        cv2.rectangle(img, (x1, y1), (x2, y2), GRAY, -1)
        gt_rects.append(_norm_rect(img, x1, y1, x2, y2, lab))
    cv2.rectangle(img, (1400, 200), (1900, 350), GRAY, -1)
    gt_rects.append(_norm_rect(img, 1400, 200, 1900, 350, 'rectangular'))
    cv2.circle(img, (1650, 700), 130, GRAY, -1)
    gt_rects.append(_norm_circle(img, 1650, 700, 130, 'circular'))
    return img, gt_rects, '110C_4bed'


def make_studio():
    """원룸 — 침대·미니주방·욕실."""
    W, H = 1200, 900
    img = np.full((H, W, 3), 252, dtype=np.uint8)
    BLACK = (15, 15, 15); GRAY = (90, 90, 90); THK = 12
    cv2.rectangle(img, (80, 80), (W - 80, H - 80), BLACK, THK)
    cv2.line(img, (700, 80), (700, 400), BLACK, THK)   # 욕실 분리
    cv2.line(img, (700, 400), (W - 80, 400), BLACK, THK)

    rects = [
        (150, 150, 450, 380, 'rectangular'),     # 침대
        (150, 500, 350, 580, 'rectangular'),     # 책상
        (800, 150, 1050, 350, 'rectangular'),    # 욕실 욕조
        (200, 700, 600, 800, 'rectangular'),     # 미니주방
    ]
    gt_rects = []
    for (x1, y1, x2, y2, lab) in rects:
        cv2.rectangle(img, (x1, y1), (x2, y2), GRAY, -1)
        gt_rects.append(_norm_rect(img, x1, y1, x2, y2, lab))
    return img, gt_rects, 'studio'


def make_l_shape():
    """ㄱ자 거실 평면."""
    W, H = 1600, 1200
    img = np.full((H, W, 3), 252, dtype=np.uint8)
    BLACK = (15, 15, 15); GRAY = (90, 90, 90); THK = 14
    # ㄱ자 외벽
    pts = np.array([[100, 100], [1500, 100], [1500, 700], [900, 700],
                    [900, 1100], [100, 1100], [100, 100]], dtype=np.int32)
    cv2.polylines(img, [pts], True, BLACK, THK)
    # 침실 분리
    cv2.line(img, (500, 100), (500, 600), BLACK, THK)
    cv2.line(img, (500, 600), (1100, 600), BLACK, THK)
    # 욕실
    cv2.line(img, (1100, 600), (1100, 700), BLACK, THK)

    rects = [
        (150, 200, 400, 480, 'rectangular'),
        (600, 200, 800, 380, 'rectangular'),
        (1000, 200, 1200, 380, 'rectangular'),
        (200, 800, 800, 950, 'rectangular'),     # 큰 소파
    ]
    gt_rects = []
    for (x1, y1, x2, y2, lab) in rects:
        cv2.rectangle(img, (x1, y1), (x2, y2), GRAY, -1)
        gt_rects.append(_norm_rect(img, x1, y1, x2, y2, lab))
    cv2.circle(img, (1300, 400), 100, GRAY, -1)  # 식탁
    gt_rects.append(_norm_circle(img, 1300, 400, 100, 'circular'))
    return img, gt_rects, 'L_shape'


CASES = [make_84A, make_59B, make_110C, make_studio, make_l_shape]


def main():
    DST.mkdir(parents=True, exist_ok=True)
    print(f"=== 한국 아파트 합성 평면도 생성 → {DST} ===\n")
    for fn in CASES:
        img, gts, name = fn()
        img_path = DST / f"{name}.png"
        gt_path = DST / f"{name}.json"
        cv2.imwrite(str(img_path), img)
        gt_data = {
            "name": name,
            "image": img_path.name,
            "image_width": img.shape[1],
            "image_height": img.shape[0],
            "furniture_gt": [asdict(g) for g in gts],
        }
        gt_path.write_text(json.dumps(gt_data, indent=2, ensure_ascii=False), encoding='utf-8')
        print(f"  {name:<14} → {img.shape[1]}x{img.shape[0]} px, GT 가구 {len(gts)}개  "
              f"({img_path.stat().st_size // 1024}KB)")
    print(f"\n총 {len(CASES)} 케이스 생성 완료")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
