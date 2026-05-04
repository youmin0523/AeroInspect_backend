# =============================================
# extract_resnet_crops.py
# M1/M3 ResNet 학습용 crop 이미지 자동 추출
# YOLO 라벨(bbox)에서 ROI를 crop하여 클래스별 폴더에 저장
#
# 사용법:
#   cd backend/training
#   python extract_resnet_crops.py --model m3
#   python extract_resnet_crops.py --model m1
#   python extract_resnet_crops.py --all
# =============================================

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import cv2
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


# ── 설정: YOLO 데이터셋 → ResNet crop 매핑 ──

CONFIGS = {
    "m1": {
        "yolo_dir": "datasets/structural",
        "crop_dir": "datasets/structural_crops",
        # YOLO class_id → ResNet 폴더명
        "class_map": {
            0: "crack_indicator",
            1: "waterproof_defect",      # data.yaml 순서
            2: "caulking_indicator",
        },
        "padding": 0.1,  # bbox 패딩 비율
    },
    "m3": {
        "yolo_dir": "datasets/floor_window",
        "crop_dir": "datasets/floor_window_crops",
        "class_map": {
            0: "floor_defect",
            1: "glass_defect",
            2: "frame_defect",
        },
        "padding": 0.1,
    },
}


def extract_crops(model_key: str) -> dict:
    """YOLO bbox에서 crop 이미지 추출."""
    cfg = CONFIGS[model_key]
    yolo_dir = Path(cfg["yolo_dir"])
    crop_dir = Path(cfg["crop_dir"])
    class_map = cfg["class_map"]
    padding = cfg["padding"]

    stats = {cls_name: 0 for cls_name in class_map.values()}

    for split in ["train", "val", "test"]:
        img_dir = yolo_dir / "images" / split
        lbl_dir = yolo_dir / "labels" / split

        if not img_dir.exists():
            print(f"  [{model_key}] {split}: 이미지 디렉토리 없음 — 스킵")
            continue

        # crop 출력 디렉토리 생성
        for cls_name in class_map.values():
            (crop_dir / split / cls_name).mkdir(parents=True, exist_ok=True)

        count = 0
        for img_path in sorted(img_dir.glob("*.jpg")):
            lbl_path = lbl_dir / img_path.with_suffix(".txt").name
            if not lbl_path.exists():
                continue

            # 한글 경로 대응 imread
            buf = np.fromfile(str(img_path), dtype=np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if img is None:
                continue

            h, w = img.shape[:2]

            for line_no, line in enumerate(lbl_path.read_text().strip().split("\n")):
                if not line.strip():
                    continue
                parts = line.strip().split()
                if len(parts) < 5:
                    continue

                cls_id = int(parts[0])
                if cls_id not in class_map:
                    continue

                cls_name = class_map[cls_id]
                cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])

                # 정규화 좌표 → 픽셀 좌표 + 패딩
                pw = bw * w * padding
                ph = bh * h * padding
                x1 = max(0, int((cx - bw / 2) * w - pw))
                y1 = max(0, int((cy - bh / 2) * h - ph))
                x2 = min(w, int((cx + bw / 2) * w + pw))
                y2 = min(h, int((cy + bh / 2) * h + ph))

                if x2 <= x1 or y2 <= y1 or (x2 - x1) < 10 or (y2 - y1) < 10:
                    continue

                crop = img[y1:y2, x1:x2]
                out_name = f"{img_path.stem}_crop{line_no}.jpg"
                out_path = crop_dir / split / cls_name / out_name
                cv2.imwrite(str(out_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
                stats[cls_name] += 1
                count += 1

        print(f"  [{model_key}] {split}: {count} crops 추출")

    return stats


def main():
    parser = argparse.ArgumentParser(description="YOLO bbox → ResNet crop 추출")
    parser.add_argument("--model", type=str, default=None, help="m1 또는 m3 (생략 시 전체)")
    parser.add_argument("--all", action="store_true", help="전체 모델 추출")
    args = parser.parse_args()

    models = list(CONFIGS.keys()) if args.all or args.model is None else [args.model]

    for model_key in models:
        if model_key not in CONFIGS:
            print(f"지원하지 않는 모델: {model_key}")
            continue

        print(f"\n{'=' * 50}")
        print(f"[{model_key.upper()}] ResNet crop 추출 시작")
        print(f"{'=' * 50}")

        stats = extract_crops(model_key)

        print(f"\n  클래스별 추출 결과:")
        total = 0
        for cls_name, count in stats.items():
            print(f"    {cls_name:30s}: {count:5d}장")
            total += count
        print(f"    {'합계':30s}: {total:5d}장")


if __name__ == "__main__":
    main()
