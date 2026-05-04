# M5v2 v2 통합 데이터셋 (frames + ADE20K)
# - frames (M5v2 원본): wall_edge=0, ceiling_edge=1, door_frame=2, window_frame=3 그대로 유지
# - ADE20K: wall=1→0, ceiling=6→1, door=15→2, window=9→3
import sys
import shutil
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DST = Path("datasets/frames_ade")  # 새 통합 데이터셋
FRAMES = Path("datasets/frames")
ADE_ROOT = Path("datasets/ade20k_download/ADEChallengeData2016")

ADE_CLASS_MAP = {
    1: 0,    # wall → wall_edge
    6: 1,    # ceiling → ceiling_edge
    15: 2,   # door → door_frame
    9: 3,    # window → window_frame
}
NEW_CLASS_NAMES = ["wall_edge", "ceiling_edge", "door_frame", "window_frame"]
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
        if w < MIN_BBOX_SIDE or h < MIN_BBOX_SIDE or w * h < MIN_BBOX_AREA:
            continue
        bboxes.append((x_min, y_min, x_max, y_max))
    return bboxes


def copy_frames_split(split_in: str, split_out: str):
    """frames 데이터셋 그대로 복사 (라벨 변환 없이)."""
    src_img = FRAMES / "images" / split_in
    src_lbl = FRAMES / "labels" / split_in
    if not src_img.exists():
        return 0
    dst_img = DST / "images" / split_out
    dst_lbl = DST / "labels" / split_out
    dst_img.mkdir(parents=True, exist_ok=True)
    dst_lbl.mkdir(parents=True, exist_ok=True)

    n = 0
    for img in src_img.iterdir():
        if img.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        new_stem = f"fr_{img.stem}"
        new_img = dst_img / (new_stem + img.suffix)
        new_lbl = dst_lbl / (new_stem + ".txt")
        src_lbl_file = src_lbl / (img.stem + ".txt")
        if not src_lbl_file.exists():
            continue
        # 라벨 그대로 복사
        shutil.copy2(src_lbl_file, new_lbl)
        shutil.copy2(img, new_img)
        n += 1
    return n


def add_ade_split(ade_split: str, our_split: str):
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
        print(f"기존 {DST} 삭제 후 재생성")
        shutil.rmtree(DST)

    print("=" * 60)
    print("M5v2 v2 데이터셋 (frames + ADE20K, 4 클래스)")
    print("=" * 60)

    # frames 그대로
    print("\n[frames 복사]")
    for split_in, split_out in [("train", "train"), ("val", "val"), ("test", "test")]:
        n = copy_frames_split(split_in, split_out)
        print(f"  {split_in} → {split_out}: {n} images")

    # ADE 추가
    print("\n[ADE20K 추가]")
    add_ade_split("training", "train")
    add_ade_split("validation", "val")

    # data.yaml
    yaml_text = f"""# M5v2 v2 frames + ADE20K
path: /content/m5v2_train/frames_ade
train: images/train
val: images/val
test: images/test

nc: {len(NEW_CLASS_NAMES)}
names:
"""
    for i, n in enumerate(NEW_CLASS_NAMES):
        yaml_text += f"  {i}: {n}\n"
    (DST / "data.yaml").write_text(yaml_text, encoding="utf-8")

    print("\n=== 완료 ===")
    for split in ["train", "val", "test"]:
        n = len(list((DST / "images" / split).glob("*"))) if (DST / "images" / split).exists() else 0
        print(f"  {split}: {n} images")
    size_mb = sum(f.stat().st_size for f in DST.rglob("*") if f.is_file()) / 1024 / 1024
    print(f"  size: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
