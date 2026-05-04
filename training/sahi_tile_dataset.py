# =============================================
# sahi_tile_dataset.py
# SAHI 스타일 타일링: 큰 이미지를 작은 타일로 잘라 데이터 확장
# - 1280 이미지 → 640×640 타일 4개 (overlap 0.2)
# - 각 타일에 들어가는 bbox만 유지 (경계 50% 이상 포함된 것만)
# - 데이터 4~9배 확장
#
# 사용법: cd backend/training && python -u sahi_tile_dataset.py
# =============================================

import sys
import shutil
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from PIL import Image

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# 입력: 압축된 structural 데이터 사용 (1.6GB, max 1280)
SRC = Path("colab/upload_to_drive/structural_compressed")
DST = Path("datasets/structural_tiled")
TILE_SIZE = 640
OVERLAP = 0.2          # 타일 간 20% 오버랩
MIN_AREA_RATIO = 0.5   # bbox가 타일에 50% 이상 들어가야 유지
MIN_BBOX_SIDE = 12     # 너무 작은 bbox 제외 (픽셀)
WORKERS = 8


def calc_tile_positions(W: int, H: int, tile_size: int, overlap: float) -> list:
    """이미지 (W,H)를 tile_size로 자르는 시작 좌표 리스트."""
    step = int(tile_size * (1 - overlap))
    xs = list(range(0, W - tile_size + 1, step))
    ys = list(range(0, H - tile_size + 1, step))
    if not xs or xs[-1] + tile_size < W:
        xs.append(max(0, W - tile_size))
    if not ys or ys[-1] + tile_size < H:
        ys.append(max(0, H - tile_size))
    # 중복 제거
    xs = sorted(set(xs))
    ys = sorted(set(ys))
    return [(x, y) for x in xs for y in ys]


def yolo_to_pixel(box: list, W: int, H: int) -> tuple:
    """YOLO normalized → pixel (x1,y1,x2,y2)."""
    cx, cy, bw, bh = box[1] * W, box[2] * H, box[3] * W, box[4] * H
    return cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2


def process_image(args):
    img_path, lbl_path, dst_img_dir, dst_lbl_dir = args
    try:
        with Image.open(img_path) as im:
            im = im.convert("RGB")
            W, H = im.size

            # bbox 읽기
            bboxes = []
            if lbl_path.exists():
                for line in lbl_path.read_text(encoding="utf-8").splitlines():
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        try:
                            cls = int(parts[0])
                            box = [cls] + [float(p) for p in parts[1:5]]
                            bboxes.append(box)
                        except ValueError:
                            continue

            # 타일 위치
            positions = calc_tile_positions(W, H, TILE_SIZE, OVERLAP)
            saved = 0
            for tx, ty in positions:
                tile = im.crop((tx, ty, tx + TILE_SIZE, ty + TILE_SIZE))
                tw, th = tile.size

                # 타일 안의 bbox 추출
                tile_boxes = []
                for box in bboxes:
                    cls = box[0]
                    x1, y1, x2, y2 = yolo_to_pixel(box, W, H)
                    box_w = x2 - x1
                    box_h = y2 - y1
                    box_area = box_w * box_h
                    if box_area <= 0:
                        continue
                    # 타일과 bbox 교차
                    ix1 = max(x1, tx)
                    iy1 = max(y1, ty)
                    ix2 = min(x2, tx + TILE_SIZE)
                    iy2 = min(y2, ty + TILE_SIZE)
                    if ix2 <= ix1 or iy2 <= iy1:
                        continue
                    inter_area = (ix2 - ix1) * (iy2 - iy1)
                    if inter_area / box_area < MIN_AREA_RATIO:
                        continue  # 50% 미만이면 제외
                    new_w = ix2 - ix1
                    new_h = iy2 - iy1
                    if new_w < MIN_BBOX_SIDE or new_h < MIN_BBOX_SIDE:
                        continue
                    new_cx = (ix1 + ix2) / 2 - tx
                    new_cy = (iy1 + iy2) / 2 - ty
                    tile_boxes.append([
                        cls, new_cx / tw, new_cy / th, new_w / tw, new_h / th
                    ])

                # bbox가 있는 타일만 저장
                if not tile_boxes:
                    continue

                stem = f"{img_path.stem}_t{tx}_{ty}"
                tile.save(dst_img_dir / f"{stem}.jpg", "JPEG", quality=85, optimize=True)
                with open(dst_lbl_dir / f"{stem}.txt", "w", encoding="utf-8") as f:
                    for cls, cx, cy, bw, bh in tile_boxes:
                        f.write(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
                saved += 1
            return saved
    except Exception as e:
        return f"FAIL {img_path}: {e}"


def main():
    if not SRC.exists():
        print(f"ERROR: {SRC} 없음")
        return
    if DST.exists():
        print(f"기존 {DST} 삭제")
        shutil.rmtree(DST)

    print("=" * 60)
    print("SAHI Tile Dataset 생성")
    print(f"  src: {SRC}")
    print(f"  dst: {DST}")
    print(f"  tile_size: {TILE_SIZE}, overlap: {OVERLAP}")
    print("=" * 60)

    total_orig = 0
    total_tiled = 0
    start = time.time()

    for split in ["train", "val", "test"]:
        src_img = SRC / "images" / split
        src_lbl = SRC / "labels" / split
        if not src_img.exists():
            continue
        dst_img = DST / "images" / split
        dst_lbl = DST / "labels" / split
        dst_img.mkdir(parents=True, exist_ok=True)
        dst_lbl.mkdir(parents=True, exist_ok=True)

        # val/test는 타일링 X (평가 일관성 위해 원본 그대로)
        if split in ["val", "test"]:
            print(f"\n[{split}] 원본 그대로 복사 (평가용)")
            for img in src_img.iterdir():
                if img.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    shutil.copy2(img, dst_img / img.name)
                    lbl = src_lbl / (img.stem + ".txt")
                    if lbl.exists():
                        shutil.copy2(lbl, dst_lbl / lbl.name)
            n_split = len(list(dst_img.iterdir()))
            total_orig += n_split
            total_tiled += n_split
            print(f"  {n_split} images")
            continue

        # train만 타일링
        images = [f for f in src_img.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        print(f"\n[train] {len(images)} 원본 → 타일링 중...")
        total_orig += len(images)

        tasks = [
            (img, src_lbl / (img.stem + ".txt"), dst_img, dst_lbl)
            for img in images
        ]

        saved_total = 0
        with ProcessPoolExecutor(max_workers=WORKERS) as ex:
            for i, result in enumerate(ex.map(process_image, tasks, chunksize=20)):
                if isinstance(result, int):
                    saved_total += result
                else:
                    print(f"  {result}")
                if (i + 1) % 1000 == 0:
                    elapsed = time.time() - start
                    print(f"  {i+1}/{len(images)} ({saved_total} tiles, {elapsed:.0f}s)")
        print(f"  {len(images)} images → {saved_total} tiles")
        total_tiled += saved_total

    # data.yaml
    yaml_text = f"""# Tiled structural dataset (Plan A)
path: {DST.resolve()}
train: images/train
val: images/val
test: images/test

nc: 3
names:
  0: crack
  1: waterproof_defect
  2: caulking_defect
"""
    (DST / "data.yaml").write_text(yaml_text)

    elapsed = time.time() - start
    size_gb = sum(f.stat().st_size for f in DST.rglob("*") if f.is_file()) / 1024 / 1024 / 1024
    print(f"\n=== 완료 ===")
    print(f"  원본: {total_orig}장")
    print(f"  타일: {total_tiled}장 (배율 {total_tiled/total_orig:.1f}x)")
    print(f"  크기: {size_gb:.2f} GB")
    print(f"  시간: {elapsed/60:.1f}분")


if __name__ == "__main__":
    main()
