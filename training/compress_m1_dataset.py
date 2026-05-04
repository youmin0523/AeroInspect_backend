# M1 structural 데이터셋 압축 (코랩 업로드용)
# - 이미지 max 1280px 리사이즈
# - jpg quality 85
# - labels는 그대로 복사 (yolo bbox는 normalized라 imgsz 무관)
# 결과: 32GB → 약 6-9GB 추정
import sys
import shutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from PIL import Image

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SRC = Path("datasets/structural")
DST = Path("colab/upload_to_drive/structural_compressed")
TARGET_MAX = 1280
JPEG_Q = 85
WORKERS = 8


def process_image(args):
    src_path, dst_path = args
    try:
        with Image.open(src_path) as im:
            w, h = im.size
            if max(w, h) > TARGET_MAX:
                r = TARGET_MAX / max(w, h)
                im = im.resize((int(w * r), int(h * r)), Image.LANCZOS)
            im.convert("RGB").save(dst_path, "JPEG", quality=JPEG_Q, optimize=True)
        return True
    except Exception as e:
        print(f"FAIL {src_path}: {e}")
        return False


def main():
    if not SRC.exists():
        print(f"ERROR: {SRC} not found. Run from backend/training/")
        return
    DST.mkdir(parents=True, exist_ok=True)

    # data.yaml 생성 (코랩 경로 기준)
    yaml_text = """path: /content/m1_train/structural_compressed
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

    total = 0
    for split in ["train", "val", "test"]:
        src_img = SRC / "images" / split
        dst_img = DST / "images" / split
        if not src_img.exists():
            continue
        dst_img.mkdir(parents=True, exist_ok=True)

        # labels 그대로 복사
        src_lbl = SRC / "labels" / split
        dst_lbl = DST / "labels" / split
        if src_lbl.exists():
            if dst_lbl.exists():
                shutil.rmtree(dst_lbl)
            shutil.copytree(src_lbl, dst_lbl)

        # 이미지 처리 (병렬)
        tasks = []
        for img in src_img.iterdir():
            if img.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                tasks.append((img, dst_img / (img.stem + ".jpg")))
        print(f"[{split}] {len(tasks)} images")
        ok = 0
        with ProcessPoolExecutor(max_workers=WORKERS) as ex:
            for i, success in enumerate(ex.map(process_image, tasks, chunksize=50)):
                ok += int(success)
                if (i + 1) % 1000 == 0:
                    print(f"  {i+1}/{len(tasks)} ({ok} ok)")
        total += ok
        print(f"  {split}: {ok}/{len(tasks)} done")

    print(f"\nTOTAL: {total} images compressed -> {DST}")
    # 사이즈 출력
    size_bytes = sum(f.stat().st_size for f in DST.rglob("*") if f.is_file())
    print(f"Size: {size_bytes/1024/1024/1024:.2f} GB")


if __name__ == "__main__":
    main()
