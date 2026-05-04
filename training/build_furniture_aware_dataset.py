# =============================================
# build_furniture_aware_dataset.py
# 빌트인 가구 인식 가능한 통합 데이터셋 구축
# - ADE20K (2만장) + 우리 frames + floor_window 통합
# - 클래스 10개: 부위 5 + 빌트인 가구 5
# 출력: datasets/furniture_aware/
# =============================================

import sys
import shutil
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DST = Path("datasets/furniture_aware")
ADE_ROOT = Path("datasets/ade20k_download/ADEChallengeData2016")
FRAMES = Path("datasets/frames")
FLOOR_WINDOW = Path("datasets/floor_window")

# 새 클래스 정의 (10 클래스)
NEW_CLASS_NAMES = [
    "wall",                # 0 — 시공 검사 대상
    "ceiling",             # 1 — 시공 검사 대상
    "floor",               # 2 — 시공 검사 대상
    "window",              # 3 — 시공 검사 대상
    "door",                # 4 — 시공 검사 대상
    "cabinet_builtin",     # 5 — 빌트인 수납장 (시공 대상)
    "kitchen_appliance",   # 6 — 냉장고/오븐/렌지 (시공 대상)
    "countertop_sink",     # 7 — 조리대/싱크 (시공 대상)
    "kitchen_island",      # 8 — 아일랜드 식탁 (시공 대상)
    "shelf",               # 9 — 선반/책장
]

# ADE20K 1-indexed class → 우리 0-indexed class
ADE_CLASS_MAP = {
    1: 0,    # wall → wall
    6: 1,    # ceiling → ceiling
    4: 2,    # floor → floor
    9: 3,    # windowpane, window → window
    15: 4,   # door → door
    11: 5,   # cabinet → cabinet_builtin
    36: 5,   # wardrobe → cabinet_builtin
    45: 5,   # chest of drawers → cabinet_builtin
    100: 5,  # buffet/sideboard → cabinet_builtin
    51: 6,   # refrigerator → kitchen_appliance
    72: 6,   # stove → kitchen_appliance
    119: 6,  # oven → kitchen_appliance
    125: 6,  # microwave → kitchen_appliance
    130: 6,  # dishwasher → kitchen_appliance
    46: 7,   # counter → countertop_sink
    48: 7,   # sink → countertop_sink
    71: 7,   # countertop → countertop_sink
    74: 8,   # kitchen island → kitchen_island
    25: 9,   # shelf → shelf
    63: 9,   # bookcase → shelf
}

# frames 데이터셋 (M5v2 원본): wall_edge=0, ceiling_edge=1, door_frame=2, window_frame=3
FRAMES_MAP = {
    0: 0,  # wall_edge → wall
    1: 1,  # ceiling_edge → ceiling
    2: 4,  # door_frame → door
    3: 3,  # window_frame → window
}

# floor_window (M3): floor_defect=0, glass_defect=1, frame_defect=2
FW_MAP = {
    0: 2,    # floor_defect → floor
    1: 3,    # glass_defect → window
    2: None, # frame_defect → 제외 (frames 데이터로 충분)
}

MIN_BBOX_AREA = 1000
MIN_BBOX_SIDE = 16


def mask_to_bboxes(mask: np.ndarray, ade_id: int) -> list:
    binary = (mask == ade_id).astype(np.uint8)
    if binary.sum() < MIN_BBOX_AREA:
        return []
    try:
        from scipy import ndimage
        labeled, num = ndimage.label(binary)
    except ImportError:
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


def remap_label_file(src_path: Path, dst_path: Path, mapping: dict) -> int:
    if not src_path.exists():
        return 0
    out_lines = []
    for line in src_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        try:
            cls = int(parts[0])
        except ValueError:
            continue
        new_cls = mapping.get(cls)
        if new_cls is None:
            continue
        out_lines.append(f"{new_cls} " + " ".join(parts[1:]) + "\n")
    if out_lines:
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dst_path, "w", encoding="utf-8") as f:
            f.writelines(out_lines)
        return len(out_lines)
    return 0


def copy_existing_split(src_root: Path, split: str, mapping: dict, prefix: str) -> int:
    src_img_dir = src_root / "images" / split
    src_lbl_dir = src_root / "labels" / split
    if not src_img_dir.exists():
        return 0
    dst_img_dir = DST / "images" / split
    dst_lbl_dir = DST / "labels" / split
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for img in src_img_dir.iterdir():
        if img.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        new_stem = f"{prefix}_{img.stem}"
        new_img = dst_img_dir / (new_stem + img.suffix)
        new_lbl = dst_lbl_dir / (new_stem + ".txt")
        src_lbl = src_lbl_dir / (img.stem + ".txt")
        n = remap_label_file(src_lbl, new_lbl, mapping)
        if n > 0:
            shutil.copy2(img, new_img)
            count += 1
    return count


def add_ade_split(ade_split: str, our_split: str) -> int:
    img_dir = ADE_ROOT / "images" / ade_split
    ann_dir = ADE_ROOT / "annotations" / ade_split
    if not img_dir.exists() or not ann_dir.exists():
        return 0

    dst_img = DST / "images" / our_split
    dst_lbl = DST / "labels" / our_split
    dst_img.mkdir(parents=True, exist_ok=True)
    dst_lbl.mkdir(parents=True, exist_ok=True)

    images = sorted(img_dir.glob("*.jpg"))
    print(f"\n[ADE {ade_split} → {our_split}] {len(images)} images")
    saved = 0
    objs = 0
    start = time.time()
    for i, img in enumerate(images):
        ann = ann_dir / (img.stem + ".png")
        if not ann.exists():
            continue
        try:
            mask = np.array(Image.open(ann))
        except Exception:
            continue
        H, W = mask.shape[:2]
        lines = []
        for ade_id, our_cls in ADE_CLASS_MAP.items():
            for x1, y1, x2, y2 in mask_to_bboxes(mask, ade_id):
                cx, cy = (x1+x2)/2/W, (y1+y2)/2/H
                bw, bh = (x2-x1)/W, (y2-y1)/H
                lines.append(f"{our_cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        if not lines:
            continue
        new_stem = f"ade_{img.stem}"
        shutil.copy2(img, dst_img / (new_stem + ".jpg"))
        with open(dst_lbl / (new_stem + ".txt"), "w") as f:
            f.write("\n".join(lines))
        saved += 1
        objs += len(lines)
        if (i+1) % 2000 == 0:
            print(f"  {i+1}/{len(images)} ({saved} saved, {time.time()-start:.0f}s)")
    print(f"  Done: {saved} images, {objs} objects in {time.time()-start:.0f}s")
    return saved


def main():
    if DST.exists():
        print(f"기존 {DST} 삭제")
        shutil.rmtree(DST)

    print("=" * 60)
    print("Furniture-Aware 통합 데이터셋 (10 클래스)")
    print("=" * 60)
    print(f"클래스: {NEW_CLASS_NAMES}")

    # frames 그대로 (wall/ceiling/door/window 보강)
    print("\n[frames 통합]")
    for split in ["train", "val", "test"]:
        n = copy_existing_split(FRAMES, split, FRAMES_MAP, "fr")
        print(f"  {split}: {n} images")

    # floor_window (floor 보강)
    print("\n[floor_window 통합]")
    for split in ["train", "val", "test"]:
        n = copy_existing_split(FLOOR_WINDOW, split, FW_MAP, "fw")
        print(f"  {split}: {n} images")

    # ADE20K (가구 + 부위 모두 추가)
    print("\n[ADE20K 통합]")
    add_ade_split("training", "train")
    add_ade_split("validation", "val")

    # data.yaml
    yaml_text = f"""# Furniture-aware dataset (10 classes)
path: {DST.resolve()}
train: images/train
val: images/val
test: images/test

nc: {len(NEW_CLASS_NAMES)}
names:
"""
    for i, n in enumerate(NEW_CLASS_NAMES):
        yaml_text += f"  {i}: {n}\n"
    (DST / "data.yaml").write_text(yaml_text, encoding="utf-8")

    # 최종
    print("\n=== 최종 ===")
    for split in ["train", "val", "test"]:
        n = len(list((DST / "images" / split).glob("*"))) if (DST / "images" / split).exists() else 0
        print(f"  {split}: {n} images")
    size_mb = sum(f.stat().st_size for f in DST.rglob("*") if f.is_file()) / 1024 / 1024
    print(f"  size: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
