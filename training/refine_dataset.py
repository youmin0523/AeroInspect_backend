# =============================================
# refine_dataset.py
# 데이터 정제 (Active Learning)
# - best.pt로 train 데이터 전수 추론
# - 노이즈 레이블 자동 검출:
#   1) GT bbox와 모델 예측 IoU < 0.3 → 잘못된 GT (제거 또는 수정)
#   2) 모델 conf > 0.85인데 매칭 GT 없음 → 놓친 GT (추가 candidate)
#   3) 한 이미지에 GT 0개인데 모델이 5개+ 검출 → 어노테이션 누락 의심
# - 결과: refined_dataset/ 디렉토리에 정제된 데이터셋 저장
#
# 사용법: cd backend/training && python refine_dataset.py --model M1 또는 M2 또는 M5 또는 M4
# =============================================

import sys
import shutil
import argparse
from pathlib import Path

import numpy as np
from ultralytics import YOLO

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# 모델별 설정
MODEL_CONFIGS = {
    "M1": {
        "best_pt_candidates": [
            "../models_weights/m1_aggressive_best.pt",  # 코랩 결과 (다운로드 후)
            "runs/detect/runs/m1_yolo_structural_960/finetune/weights/best.pt",
        ],
        "data_yaml": "configs/structural.yaml",
        "src_dataset": "datasets/structural",
        "dst_dataset": "datasets/structural_refined",
        "nc": 3,
        "names": ["crack", "waterproof_defect", "caulking_defect"],
    },
    "M2": {
        "best_pt_candidates": [
            "../models_weights/m2_aggressive_best.pt",
            "../models_weights/m2_best.pt",
        ],
        "data_yaml": "configs/surface.yaml",
        "src_dataset": "datasets/surface",
        "dst_dataset": "datasets/surface_refined",
        "nc": 2,
        "names": ["surface_defect_wall", "baseboard_defect"],
    },
    "M5": {
        "best_pt_candidates": [
            "../models_weights/m5v2_v2_best.pt",
            "runs/detect/runs/m5_frame_seg/train/weights/best.pt",
        ],
        "data_yaml": "configs/frame_seg.yaml",
        "src_dataset": "datasets/frames_ade",  # ADE 통합본
        "dst_dataset": "datasets/frames_refined",
        "nc": 4,
        "names": ["wall_edge", "ceiling_edge", "door_frame", "window_frame"],
    },
    "M4": {
        "best_pt_candidates": [
            "runs/detect/runs/m4_context/train/weights/best.pt",
        ],
        "data_yaml": "datasets/m4_context/data.yaml",
        "src_dataset": "datasets/m4_context",
        "dst_dataset": "datasets/m4_context_refined",
        "nc": 5,
        "names": ["wall", "ceiling", "floor", "window", "door"],
    },
}

# 노이즈 검출 임계값
IOU_BAD = 0.3            # IoU < 0.3 → 잘못된 GT
CONF_HIGH = 0.85         # conf > 0.85인데 매칭 GT 없음 → 놓친 GT
MAX_MISSED_PER_IMG = 5   # 이미지당 놓친 GT 5개 이상 → 의심


def find_best_pt(candidates):
    for c in candidates:
        p = Path(c)
        if p.exists():
            print(f"[best.pt] {p} ({p.stat().st_size/1024/1024:.1f}MB)")
            return p
    return None


def yolo_to_xyxy(box, w, h):
    """YOLO 형식 [cls, cx, cy, bw, bh] (normalized) → [x1, y1, x2, y2] (pixel)."""
    cx, cy, bw, bh = box[1] * w, box[2] * h, box[3] * w, box[4] * h
    return [cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2]


def iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0
    a_area = (a[2] - a[0]) * (a[3] - a[1])
    b_area = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (a_area + b_area - inter)


def read_label_file(path):
    if not path.exists():
        return []
    boxes = []
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            try:
                boxes.append([int(parts[0])] + [float(p) for p in parts[1:5]])
            except ValueError:
                continue
    return boxes


def write_label_file(path, boxes):
    if not boxes:
        path.unlink(missing_ok=True)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for b in boxes:
            f.write(f"{b[0]} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}\n")


def refine(model_name: str):
    cfg = MODEL_CONFIGS[model_name]
    best_pt = find_best_pt(cfg["best_pt_candidates"])
    if best_pt is None:
        print(f"[ERROR] {model_name} best.pt 없음. 후보: {cfg['best_pt_candidates']}")
        return

    src = Path(cfg["src_dataset"])
    dst = Path(cfg["dst_dataset"])

    if dst.exists():
        print(f"[clean] 기존 {dst} 삭제")
        shutil.rmtree(dst)

    print(f"\n=== {model_name} 데이터 정제 시작 ===")
    print(f"  src: {src}")
    print(f"  dst: {dst}")

    model = YOLO(str(best_pt))
    stats = {"total": 0, "bad_removed": 0, "missed_added": 0, "kept": 0}

    for split in ["train", "val", "test"]:
        src_img = src / "images" / split
        src_lbl = src / "labels" / split
        if not src_img.exists():
            continue

        dst_img = dst / "images" / split
        dst_lbl = dst / "labels" / split
        dst_img.mkdir(parents=True, exist_ok=True)
        dst_lbl.mkdir(parents=True, exist_ok=True)

        # val/test는 그대로 복사 (정제하면 평가 일관성 깨짐)
        if split in ["val", "test"]:
            print(f"  [{split}] 그대로 복사 (정제 대상 아님)")
            for img in src_img.iterdir():
                if img.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    shutil.copy2(img, dst_img / img.name)
                    lbl = src_lbl / (img.stem + ".txt")
                    if lbl.exists():
                        shutil.copy2(lbl, dst_lbl / (img.stem + ".txt"))
            continue

        # train 정제
        images = [f for f in src_img.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        print(f"  [{split}] {len(images)} images, 정제 중...")

        # 배치 추론 (속도 위해)
        BATCH = 16
        for i in range(0, len(images), BATCH):
            batch_imgs = images[i:i+BATCH]
            results = model.predict(
                source=[str(im) for im in batch_imgs],
                conf=0.25, iou=0.5, imgsz=960, verbose=False, save=False,
            )

            for img_path, res in zip(batch_imgs, results):
                stats["total"] += 1
                gt_boxes = read_label_file(src_lbl / (img_path.stem + ".txt"))
                pred_xyxy = res.boxes.xyxy.cpu().numpy() if len(res.boxes) > 0 else np.empty((0, 4))
                pred_conf = res.boxes.conf.cpu().numpy() if len(res.boxes) > 0 else np.empty(0)
                pred_cls = res.boxes.cls.cpu().numpy().astype(int) if len(res.boxes) > 0 else np.empty(0, dtype=int)

                W, H = res.orig_shape[1], res.orig_shape[0]

                # 1. GT 검증: 잘못된 GT 제거
                cleaned_gt = []
                for gt in gt_boxes:
                    gt_xyxy = yolo_to_xyxy(gt, W, H)
                    # 같은 클래스 예측 중 IoU 가장 높은 것
                    same_cls_mask = pred_cls == gt[0]
                    if same_cls_mask.sum() == 0:
                        # 모델이 이 클래스 검출 못함 → GT 일단 유지 (놓친 검출일 수 있음)
                        cleaned_gt.append(gt)
                        continue
                    ious = np.array([iou(gt_xyxy, pred_xyxy[j]) for j in np.where(same_cls_mask)[0]])
                    if ious.max() >= IOU_BAD:
                        cleaned_gt.append(gt)
                    else:
                        # IoU 너무 낮음 → 잘못된 GT 가능성 (제거)
                        stats["bad_removed"] += 1

                # 2. 놓친 GT 추가: conf > CONF_HIGH인데 매칭 GT 없음
                missed_count = 0
                for j in range(len(pred_xyxy)):
                    if pred_conf[j] < CONF_HIGH:
                        continue
                    matched = False
                    for gt in cleaned_gt:
                        if int(gt[0]) != pred_cls[j]:
                            continue
                        gt_xyxy = yolo_to_xyxy(gt, W, H)
                        if iou(gt_xyxy, pred_xyxy[j]) >= IOU_BAD:
                            matched = True
                            break
                    if not matched and missed_count < MAX_MISSED_PER_IMG:
                        # 새 GT 추가
                        x1, y1, x2, y2 = pred_xyxy[j]
                        cx = (x1 + x2) / 2 / W
                        cy = (y1 + y2) / 2 / H
                        bw = (x2 - x1) / W
                        bh = (y2 - y1) / H
                        cleaned_gt.append([int(pred_cls[j]), cx, cy, bw, bh])
                        stats["missed_added"] += 1
                        missed_count += 1

                # GT가 1개 이상 남으면 저장 (빈 라벨 이미지는 negative sample로 유지하려면 변경)
                shutil.copy2(img_path, dst_img / img_path.name)
                write_label_file(dst_lbl / (img_path.stem + ".txt"), cleaned_gt)
                stats["kept"] += 1

            if (i + BATCH) % 200 == 0 or i + BATCH >= len(images):
                print(f"    {min(i+BATCH, len(images))}/{len(images)} (bad: {stats['bad_removed']}, missed: {stats['missed_added']})")

    # data.yaml 작성
    yaml_text = f"""# {model_name} refined dataset
path: {dst.resolve()}
train: images/train
val: images/val
test: images/test

nc: {cfg['nc']}
names:
"""
    for i, n in enumerate(cfg["names"]):
        yaml_text += f"  {i}: {n}\n"
    (dst / "data.yaml").write_text(yaml_text)

    print(f"\n=== {model_name} 정제 완료 ===")
    print(f"  total: {stats['total']}")
    print(f"  bad_removed: {stats['bad_removed']}")
    print(f"  missed_added: {stats['missed_added']}")
    print(f"  kept: {stats['kept']}")
    print(f"  결과: {dst}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODEL_CONFIGS.keys()),
                        help="정제할 모델 (M1/M2/M5/M4)")
    args = parser.parse_args()
    refine(args.model)


if __name__ == "__main__":
    main()
