# ADE20K segmentation → YOLO bbox 변환
# 5 클래스만 추출: wall, floor, ceiling, window, door
# 기존 m4_context 데이터셋에 추가 (옵션 A + 옵션 B 통합)
import sys
import shutil
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ADE_ROOT = Path("datasets/ade20k_download/ADEChallengeData2016")
DST = Path("datasets/m4_context")  # 기존 통합 데이터셋에 추가

# ADE20K 클래스 ID (1-indexed) → 우리 클래스 (0-indexed)
# 참고: objectInfo150.txt
ADE_CLASS_MAP = {
    1: 0,    # wall
    4: 2,    # floor
    6: 1,    # ceiling
    9: 3,    # window (windowpane)
    15: 4,   # door
    # 추가 후보:
    # 11: 0, # cabinet → 일반 wall(?) - 보류
    # 14: 0, # earth, ground → floor(?) - 보류
}

# bbox 최소 크기 (픽셀²)
MIN_BBOX_AREA = 1000   # 약 32x32px
MIN_BBOX_SIDE = 16     # 16px 이하면 제외


def mask_to_bboxes(mask: np.ndarray, ade_id: int) -> list:
    """특정 ADE class의 connected components → bbox 리스트.
    각 bbox는 (x_min, y_min, x_max, y_max) format."""
    binary = (mask == ade_id).astype(np.uint8)
    if binary.sum() < MIN_BBOX_AREA:
        return []

    # connected components (cv2 없으니 scipy 또는 직접)
    try:
        from scipy import ndimage
        labeled, num = ndimage.label(binary)
    except ImportError:
        # fallback: 전체 영역을 1 bbox로
        ys, xs = np.where(binary)
        if len(xs) == 0:
            return []
        return [(int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))]

    bboxes = []
    for i in range(1, num + 1):
        ys, xs = np.where(labeled == i)
        if len(xs) == 0:
            continue
        x_min, x_max = int(xs.min()), int(xs.max())
        y_min, y_max = int(ys.min()), int(ys.max())
        w, h = x_max - x_min, y_max - y_min
        if w < MIN_BBOX_SIDE or h < MIN_BBOX_SIDE:
            continue
        if w * h < MIN_BBOX_AREA:
            continue
        bboxes.append((x_min, y_min, x_max, y_max))
    return bboxes


def process_split(ade_split: str, our_split: str):
    """ADE training/validation → m4_context train/val 추가."""
    img_dir = ADE_ROOT / "images" / ade_split
    ann_dir = ADE_ROOT / "annotations" / ade_split
    if not img_dir.exists() or not ann_dir.exists():
        print(f"SKIP {ade_split} — paths not found")
        return 0

    dst_img = DST / "images" / our_split
    dst_lbl = DST / "labels" / our_split
    dst_img.mkdir(parents=True, exist_ok=True)
    dst_lbl.mkdir(parents=True, exist_ok=True)

    images = sorted(img_dir.glob("*.jpg"))
    print(f"\n[ADE {ade_split} → m4_context/{our_split}] {len(images)} images")

    saved = 0
    objs_total = 0
    start = time.time()
    for i, img_path in enumerate(images):
        ann_path = ann_dir / (img_path.stem + ".png")
        if not ann_path.exists():
            continue

        # annotation 읽기
        try:
            mask = np.array(Image.open(ann_path))
        except Exception as e:
            continue

        H, W = mask.shape[:2]

        # 5 클래스 bbox 추출
        yolo_lines = []
        for ade_id, our_cls in ADE_CLASS_MAP.items():
            for x1, y1, x2, y2 in mask_to_bboxes(mask, ade_id):
                cx = (x1 + x2) / 2 / W
                cy = (y1 + y2) / 2 / H
                bw = (x2 - x1) / W
                bh = (y2 - y1) / H
                yolo_lines.append(f"{our_cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        if not yolo_lines:
            continue

        # 이미지 + 라벨 저장 (prefix "ade_" 충돌 방지)
        new_stem = f"ade_{img_path.stem}"
        shutil.copy2(img_path, dst_img / (new_stem + ".jpg"))
        with open(dst_lbl / (new_stem + ".txt"), "w") as f:
            f.write("\n".join(yolo_lines))
        saved += 1
        objs_total += len(yolo_lines)

        if (i + 1) % 1000 == 0:
            elapsed = time.time() - start
            print(f"  {i+1}/{len(images)} ({saved} saved, {elapsed:.0f}s)")

    print(f"  Done: {saved} images, {objs_total} objects in {time.time()-start:.0f}s")
    return saved


def main():
    if not ADE_ROOT.exists():
        print(f"ERROR: ADE20K not found at {ADE_ROOT}")
        return

    print("=" * 60)
    print("ADE20K → YOLO 변환 (5 클래스: wall/ceiling/floor/window/door)")
    print("=" * 60)
    print(f"기존 m4_context 데이터셋에 추가합니다.")

    # ADE training → our train, ADE validation → our val
    train_n = process_split("training", "train")
    val_n = process_split("validation", "val")

    print(f"\n=== ADE20K 변환 완료 ===")
    print(f"  train 추가: {train_n} images")
    print(f"  val 추가: {val_n} images")

    # 최종 카운트
    print(f"\n=== 최종 m4_context 데이터셋 ===")
    for split in ["train", "val", "test"]:
        n = len(list((DST / "images" / split).glob("*"))) if (DST / "images" / split).exists() else 0
        print(f"  {split}: {n} images")
    size_mb = sum(f.stat().st_size for f in DST.rglob("*") if f.is_file()) / 1024 / 1024
    print(f"  size: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
