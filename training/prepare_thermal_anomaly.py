# =============================================
# prepare_thermal_anomaly.py
# Thermal anomaly detection (PatchCore) 학습용 정상 패치 데이터 생성
#
# 전략: thermal_yolo의 train 이미지에서 라벨 영역을 제외한 정상 영역만
#       작은 패치로 cropping해 정상 데이터셋으로 사용.
#       - 라벨 박스(IoU>0)와 겹치지 않는 윈도우 패치 추출
#       - 패치 크기: 224×224 (PatchCore 표준)
#       - 이미지당 최대 N개, 총 ~2000장 목표 (MVTec AD 표준 400~2000)
#
# 출력: datasets/thermal_anomaly/good/  (정상 패치)
# =============================================

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

TRAIN = Path(__file__).resolve().parent
SRC_IMG = TRAIN / "datasets" / "thermal_yolo" / "images" / "train"
SRC_LBL = TRAIN / "datasets" / "thermal_yolo" / "labels" / "train"
OUT_DIR = TRAIN / "datasets" / "thermal_anomaly" / "good"

PATCH = 224
STRIDE = 112          # 50% overlap
PATCHES_PER_IMAGE = 6 # 이미지당 최대 6개 (다양성 vs 디스크)
TARGET_TOTAL = 2000
MIN_FREE_RATIO = 0.95 # 패치 영역의 95%가 라벨 박스와 안 겹쳐야 정상


def yolo_to_xyxy(line: str, W: int, H: int) -> Tuple[int, int, int, int] | None:
    """YOLO 라벨 한 줄 → xyxy 픽셀.

    포맷 호환:
    - bbox: cls cx cy w h
    - polygon (M5/M4 식): cls x1 y1 x2 y2 ... → 외접 박스로 변환
    """
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    try:
        nums = list(map(float, parts[1:]))
    except ValueError:
        return None

    if len(nums) == 4:
        cx, cy, w, h = nums
        x1 = int((cx - w / 2) * W)
        y1 = int((cy - h / 2) * H)
        x2 = int((cx + w / 2) * W)
        y2 = int((cy + h / 2) * H)
    else:
        # polygon 짝수쌍 → bounding box
        if len(nums) % 2 != 0:
            return None
        xs = nums[0::2]; ys = nums[1::2]
        x1 = int(min(xs) * W); y1 = int(min(ys) * H)
        x2 = int(max(xs) * W); y2 = int(max(ys) * H)
    return max(0, x1), max(0, y1), min(W, x2), min(H, y2)


def patch_safe(px1: int, py1: int, boxes: List[Tuple[int, int, int, int]]) -> bool:
    """패치 영역이 라벨 박스들과 충분히 안 겹치는지 (>=MIN_FREE_RATIO)."""
    px2, py2 = px1 + PATCH, py1 + PATCH
    patch_area = PATCH * PATCH
    overlap_total = 0
    for bx1, by1, bx2, by2 in boxes:
        ix1, iy1 = max(px1, bx1), max(py1, by1)
        ix2, iy2 = min(px2, bx2), min(py2, by2)
        if ix2 > ix1 and iy2 > iy1:
            overlap_total += (ix2 - ix1) * (iy2 - iy1)
    free_ratio = 1.0 - overlap_total / patch_area
    return free_ratio >= MIN_FREE_RATIO


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    img_paths = sorted([p for p in SRC_IMG.iterdir()
                        if p.suffix.lower() in (".jpg", ".jpeg", ".png")])
    print(f"[thermal_anomaly] {len(img_paths)} 소스 이미지 스캔")

    total = 0
    skipped = 0
    for img_path in img_paths:
        if total >= TARGET_TOTAL:
            break
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            continue
        H, W = img.shape[:2]
        if H < PATCH or W < PATCH:
            skipped += 1
            continue

        lbl_path = SRC_LBL / (img_path.stem + ".txt")
        boxes: List[Tuple[int, int, int, int]] = []
        if lbl_path.exists():
            for line in lbl_path.read_text(encoding="utf-8").splitlines():
                bb = yolo_to_xyxy(line, W, H)
                if bb:
                    boxes.append(bb)

        per_img = 0
        for y0 in range(0, H - PATCH + 1, STRIDE):
            if per_img >= PATCHES_PER_IMAGE or total >= TARGET_TOTAL:
                break
            for x0 in range(0, W - PATCH + 1, STRIDE):
                if per_img >= PATCHES_PER_IMAGE or total >= TARGET_TOTAL:
                    break
                if not patch_safe(x0, y0, boxes):
                    continue
                patch = img[y0:y0 + PATCH, x0:x0 + PATCH]
                # 너무 어두운/하얀 패치 제외 (열화상에서 의미 없는 영역)
                gray_mean = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY).mean()
                if gray_mean < 30 or gray_mean > 235:
                    continue
                out_name = f"{img_path.stem}_p{per_img:02d}.jpg"
                cv2.imwrite(str(OUT_DIR / out_name), patch,
                            [cv2.IMWRITE_JPEG_QUALITY, 92])
                per_img += 1
                total += 1
        if total % 200 == 0 and total > 0:
            print(f"[thermal_anomaly] 진행 {total}/{TARGET_TOTAL}")

    print(f"[thermal_anomaly] 완료: {total} 정상 패치 생성 (skip {skipped}) → {OUT_DIR}")


if __name__ == "__main__":
    main()
