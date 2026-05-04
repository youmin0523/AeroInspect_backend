# M2-ResNet crop 추출: YOLO bbox에서 surface 하자 crop
import io, sys, shutil, random
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
from pathlib import Path
import cv2, numpy as np

random.seed(42)

# M2 YOLO는 2클래스: 0=surface_defect_wall, 1=baseboard_defect
# ResNet은 5클래스: wallpaper_seam, wallpaper_bubble, paint_stain, scratch, baseboard_damage
# YOLO class 0 → ResNet에서 세부 분류 필요 → "surface_defect" 통합 폴더로 추출
# YOLO class 1 → baseboard_damage

YOLO_TO_RESNET = {
    0: "surface_defect",   # 세부 분류는 학습 시 자동
    1: "baseboard_damage",
}

for split in ["train", "val", "test"]:
    img_dir = Path(f"datasets/surface/images/{split}")
    lbl_dir = Path(f"datasets/surface/labels/{split}")
    if not img_dir.exists():
        continue

    for cls_name in set(YOLO_TO_RESNET.values()):
        (Path(f"datasets/surface_crops_v2/{split}/{cls_name}")).mkdir(parents=True, exist_ok=True)

    count = 0
    for img_path in sorted(img_dir.glob("*.jpg")):
        lbl_path = lbl_dir / img_path.with_suffix(".txt").name
        if not lbl_path.exists():
            continue

        buf = np.fromfile(str(img_path), dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is None:
            continue

        h, w = img.shape[:2]
        for i, line in enumerate(lbl_path.read_text().strip().split("\n")):
            if not line.strip():
                continue
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            cls_id = int(parts[0])
            if cls_id not in YOLO_TO_RESNET:
                continue

            cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            pad = 0.1
            x1 = max(0, int((cx - bw/2) * w - bw*w*pad))
            y1 = max(0, int((cy - bh/2) * h - bh*h*pad))
            x2 = min(w, int((cx + bw/2) * w + bw*w*pad))
            y2 = min(h, int((cy + bh/2) * h + bh*h*pad))

            if x2-x1 < 10 or y2-y1 < 10:
                continue

            crop = img[y1:y2, x1:x2]
            cls_name = YOLO_TO_RESNET[cls_id]
            out = Path(f"datasets/surface_crops_v2/{split}/{cls_name}/{img_path.stem}_crop{i}.jpg")
            cv2.imwrite(str(out), crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
            count += 1

    print(f"{split}: {count} crops")

# 결과 확인
for split in ["train", "val", "test"]:
    base = Path(f"datasets/surface_crops_v2/{split}")
    for cls in sorted(base.iterdir()):
        if cls.is_dir():
            n = len(list(cls.glob("*.jpg")))
            print(f"  {split}/{cls.name}: {n}")
