# =============================================
# train_m4v2_local.py
# M4 Context 정제 + Hard Mining + 재학습 (로컬 RTX 5070, 자동)
#
# 사용법:
#   cd backend/training
#   python -u train_m4v2_local.py
#
# 입력:
#   - datasets/m4_context (옵션 B 통합 데이터셋, 이미 로컬에 있음)
#   - runs/detect/runs/m4_context/train/weights/best.pt (현재 진행 중 학습 결과)
# 출력:
#   - datasets/m4_context_refined (정제 + Hard Mining 데이터셋)
#   - runs/m4v2/stage1/weights/best.pt
#   - runs/m4v2/stage2/weights/best.pt
#   - ../models_weights/m4_yolo_context_elements.onnx (자동 덮어쓰기)
# =============================================

import sys
import shutil
import time
from pathlib import Path

import numpy as np
from ultralytics import YOLO

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# 경로 (cwd=backend/training 기준)
DATA_ROOT = Path("datasets/m4_context")
REFINED = Path("datasets/m4_context_refined")
# ultralytics가 project=str(PROJECT.parent), name="m4v2/stage1" 옵션을 받으면
# 실제 출력은 runs/detect/runs/m4v2/stage1/ 에 생성됨. resume 감지를 위해 실제 경로 사용.
PROJECT = Path("../../runs/detect/runs/m4v2")
PROJECT_PARENT_FOR_TRAIN = Path("runs")  # ultralytics에 넘기는 project= 인자
WEIGHTS_DIR = Path("../models_weights")
BEST_PT_CANDIDATES = [
    Path("../../runs/detect/runs/m4_context/train/weights/best.pt"),
    Path("../../runs/m4_context/train/weights/best.pt"),
    Path("runs/detect/runs/m4_context/train/weights/best.pt"),
    Path("runs/m4_context/train/weights/best.pt"),
]

# 정제 임계값
IOU_BAD = 0.3
CONF_HIGH = 0.85
MAX_MISSED = 5
HARD_RATIO = 0.10        # 상위 10%를 hard sample로
HARD_WEIGHT = 4          # 4배 추가 복사 (5배 가중)


def yolo_to_xyxy(b, w, h):
    cx, cy, bw, bh = b[1] * w, b[2] * h, b[3] * w, b[4] * h
    return [cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2]


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0
    aa = (a[2] - a[0]) * (a[3] - a[1])
    bb = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (aa + bb - inter)


def read_lbl(p):
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) >= 5:
            try:
                out.append([int(parts[0])] + [float(x) for x in parts[1:5]])
            except ValueError:
                pass
    return out


def write_lbl(p, boxes):
    if not boxes:
        if p.exists():
            p.unlink()
        return
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(
        f"{b[0]} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}" for b in boxes
    ))


def find_best_pt():
    for c in BEST_PT_CANDIDATES:
        if c.exists():
            print(f"[best.pt] {c} ({c.stat().st_size/1024/1024:.1f}MB)")
            return c
    raise FileNotFoundError(f"M4 best.pt 못 찾음. 후보: {BEST_PT_CANDIDATES}")


# ─────────────────────────────────────────────
# Step 1: 데이터 정제
# ─────────────────────────────────────────────
def refine_dataset(model: YOLO):
    print("\n" + "=" * 60)
    print("Step 1: 데이터 정제 (Active Learning)")
    print("=" * 60)
    if REFINED.exists():
        shutil.rmtree(REFINED)

    stats = {"total": 0, "bad": 0, "missed": 0}

    for split in ["train", "val", "test"]:
        src_img = DATA_ROOT / "images" / split
        src_lbl = DATA_ROOT / "labels" / split
        if not src_img.exists():
            continue
        dst_img = REFINED / "images" / split
        dst_lbl = REFINED / "labels" / split
        dst_img.mkdir(parents=True, exist_ok=True)
        dst_lbl.mkdir(parents=True, exist_ok=True)

        images = [f for f in src_img.iterdir()
                  if f.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        print(f"\n[{split}] {len(images)} images")

        if split in ["val", "test"]:
            for im in images:
                shutil.copy2(im, dst_img / im.name)
                lbl = src_lbl / (im.stem + ".txt")
                if lbl.exists():
                    shutil.copy2(lbl, dst_lbl / lbl.name)
            continue

        BATCH = 16
        for i in range(0, len(images), BATCH):
            batch = images[i:i + BATCH]
            results = model.predict(
                source=[str(b) for b in batch],
                conf=0.25, iou=0.5, imgsz=960, verbose=False, save=False, device=0,
            )
            for img_path, res in zip(batch, results):
                stats["total"] += 1
                gt = read_lbl(src_lbl / (img_path.stem + ".txt"))
                pxy = res.boxes.xyxy.cpu().numpy() if len(res.boxes) > 0 else np.empty((0, 4))
                pcf = res.boxes.conf.cpu().numpy() if len(res.boxes) > 0 else np.empty(0)
                pcl = res.boxes.cls.cpu().numpy().astype(int) if len(res.boxes) > 0 else np.empty(0, dtype=int)
                W, H = res.orig_shape[1], res.orig_shape[0]

                cleaned = []
                for g in gt:
                    gxy = yolo_to_xyxy(g, W, H)
                    same = pcl == g[0]
                    if same.sum() == 0:
                        cleaned.append(g)
                        continue
                    ious = np.array([iou(gxy, pxy[j]) for j in np.where(same)[0]])
                    if ious.max() >= IOU_BAD:
                        cleaned.append(g)
                    else:
                        stats["bad"] += 1

                missed = 0
                for j in range(len(pxy)):
                    if pcf[j] < CONF_HIGH:
                        continue
                    matched = False
                    for g in cleaned:
                        if int(g[0]) != pcl[j]:
                            continue
                        if iou(yolo_to_xyxy(g, W, H), pxy[j]) >= IOU_BAD:
                            matched = True
                            break
                    if not matched and missed < MAX_MISSED:
                        x1, y1, x2, y2 = pxy[j]
                        cleaned.append([
                            int(pcl[j]),
                            (x1 + x2) / 2 / W, (y1 + y2) / 2 / H,
                            (x2 - x1) / W, (y2 - y1) / H,
                        ])
                        stats["missed"] += 1
                        missed += 1

                shutil.copy2(img_path, dst_img / img_path.name)
                write_lbl(dst_lbl / (img_path.stem + ".txt"), cleaned)
            if (i + BATCH) % 320 == 0 or i + BATCH >= len(images):
                print(f"  {min(i+BATCH, len(images))}/{len(images)} (bad:{stats['bad']}, missed:{stats['missed']})")

    # data.yaml
    yaml_text = f"""path: {REFINED.resolve()}
train: images/train
val: images/val
test: images/test
nc: 5
names:
  0: wall
  1: ceiling
  2: floor
  3: window
  4: door
"""
    (REFINED / "data.yaml").write_text(yaml_text)
    print(f"\n정제 완료: total={stats['total']}, bad={stats['bad']}, missed={stats['missed']}")


# ─────────────────────────────────────────────
# Step 2: Hard Sample Mining
# ─────────────────────────────────────────────
def hard_sample_mining(model: YOLO):
    print("\n" + "=" * 60)
    print("Step 2: Hard Sample Mining")
    print("=" * 60)

    train_imgs = sorted([f for f in (REFINED / "images" / "train").iterdir()
                         if f.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    losses = []
    BATCH = 8
    for i in range(0, len(train_imgs), BATCH):
        batch = train_imgs[i:i + BATCH]
        results = model.predict(
            source=[str(b) for b in batch],
            conf=0.05, iou=0.5, imgsz=960, verbose=False, save=False, device=0,
        )
        for img_path, res in zip(batch, results):
            gt = read_lbl(REFINED / "labels" / "train" / (img_path.stem + ".txt"))
            if not gt:
                losses.append((img_path, 0.0))
                continue
            if len(res.boxes) == 0:
                losses.append((img_path, 1.0))
                continue
            pxy = res.boxes.xyxy.cpu().numpy()
            pcf = res.boxes.conf.cpu().numpy()
            pcl = res.boxes.cls.cpu().numpy().astype(int)
            W, H = res.orig_shape[1], res.orig_shape[0]
            confs = []
            for g in gt:
                gxy = yolo_to_xyxy(g, W, H)
                best = 0.0
                for j in range(len(pxy)):
                    if pcl[j] == g[0] and iou(gxy, pxy[j]) >= 0.3:
                        best = max(best, float(pcf[j]))
                confs.append(best)
            losses.append((img_path, 1.0 - sum(confs) / len(confs)))
        if (i + BATCH) % 320 == 0:
            print(f"  {min(i+BATCH, len(train_imgs))}/{len(train_imgs)}")

    losses.sort(key=lambda x: x[1], reverse=True)
    n_hard = max(1, int(len(losses) * HARD_RATIO))
    print(f"\nHard samples: {n_hard} / {len(losses)} ({HARD_RATIO*100:.0f}%)")
    for img_path, _ in losses[:n_hard]:
        lbl = REFINED / "labels" / "train" / (img_path.stem + ".txt")
        for k in range(HARD_WEIGHT):
            shutil.copy2(img_path, REFINED / "images" / "train" / f"{img_path.stem}_h{k}{img_path.suffix}")
            if lbl.exists():
                shutil.copy2(lbl, REFINED / "labels" / "train" / f"{img_path.stem}_h{k}.txt")
    print(f"복사 완료. 새 train: {len(list((REFINED / 'images' / 'train').glob('*')))}")


# ─────────────────────────────────────────────
# Step 3: Multi-stage 학습
# ─────────────────────────────────────────────
def train_multi_stage():
    data_yaml = str(REFINED / "data.yaml")

    # Stage 1: 정제 데이터 일반 학습
    stage1_last = PROJECT / "stage1" / "weights" / "last.pt"
    stage1_best = PROJECT / "stage1" / "weights" / "best.pt"
    print("\n" + "=" * 60)
    if stage1_last.exists() and not stage1_best.exists():
        # last는 있는데 best 없는 비정상 케이스 (드물게 best 미생성)
        print(f"Step 3-1: Stage 1 RESUME (from {stage1_last})")
        print("=" * 60)
        s1_model = YOLO(str(stage1_last))
        s1_model.train(resume=True)
    elif stage1_last.exists() and stage1_best.exists():
        # 정상 진행 중 중단 → resume
        print(f"Step 3-1: Stage 1 RESUME (from {stage1_last})")
        print("=" * 60)
        s1_model = YOLO(str(stage1_last))
        s1_model.train(resume=True)
    else:
        print("Step 3-1: Stage 1 학습 (yolov11m + 960 + 50ep + lr=1e-3)")
        print("=" * 60)
        s1_model = YOLO("yolo11m.pt")
        s1_model.train(
            data=data_yaml,
            epochs=50,
            batch=4,
            imgsz=960,
            cache="disk",
            workers=4,
            optimizer="AdamW",
            lr0=1e-3,
            lrf=0.01,
            cos_lr=True,
            patience=15,
            warmup_epochs=3,
            close_mosaic=15,
            freeze=0,
            hsv_h=0.015, hsv_s=0.5, hsv_v=0.4,
            degrees=5.0, translate=0.1, scale=0.5,
            shear=2.0, perspective=0.001,
            flipud=0.0, fliplr=0.5,
            mosaic=1.0, mixup=0.1, copy_paste=0.3,
            erasing=0.0,
            multi_scale=0.2,
            save_period=5,
            plots=True,
            project=str(PROJECT_PARENT_FOR_TRAIN),
            name="m4v2/stage1",
            exist_ok=True,
        )

    print(f"\nStage 1 best: {stage1_best}")

    # Stage 2: fine-tune
    stage2_last = PROJECT / "stage2" / "weights" / "last.pt"
    print("\n" + "=" * 60)
    if stage2_last.exists():
        print(f"Step 3-2: Stage 2 RESUME (from {stage2_last})")
        print("=" * 60)
        s2_model = YOLO(str(stage2_last))
        s2_model.train(resume=True)
    else:
        print("Step 3-2: Stage 2 fine-tune (lr=1e-5 + freeze=10 + 15ep)")
        print("=" * 60)
        s2_model = YOLO(str(stage1_best))
        s2_model.train(
            data=data_yaml,
            epochs=15,
            batch=4,
            imgsz=960,
            cache="disk",
            workers=4,
            optimizer="AdamW",
            lr0=1e-5,
            lrf=0.01,
            cos_lr=True,
            patience=8,
            warmup_epochs=1,
            close_mosaic=10,
            freeze=10,
            mosaic=0.5, mixup=0.0, copy_paste=0.2,
            save_period=5,
            plots=True,
            project=str(PROJECT_PARENT_FOR_TRAIN),
            name="m4v2/stage2",
            exist_ok=True,
        )


# ─────────────────────────────────────────────
# Step 4: ONNX export + 평가
# ─────────────────────────────────────────────
def _read_best_map_from_csv(csv_path: Path) -> float:
    """results.csv 파일에서 column 8(mAP50)의 max 값을 읽음. 없거나 빈 파일이면 -1."""
    if not csv_path.exists():
        return -1.0
    best = -1.0
    try:
        for i, line in enumerate(csv_path.read_text(encoding="utf-8").splitlines()):
            if i == 0:
                continue  # header
            parts = line.split(",")
            if len(parts) < 8:
                continue
            try:
                v = float(parts[7])  # column 8 (0-indexed 7) = metrics/mAP50(B)
                if v > best:
                    best = v
            except ValueError:
                continue
    except Exception as e:
        print(f"[WARN] results.csv 파싱 실패: {csv_path} — {e}")
        return -1.0
    return best


def export_and_evaluate():
    print("\n" + "=" * 60)
    print("Step 4: ONNX export + 평가")
    print("=" * 60)
    stage1_best = PROJECT / "stage1" / "weights" / "best.pt"
    stage2_best = PROJECT / "stage2" / "weights" / "best.pt"
    stage1_csv = PROJECT / "stage1" / "results.csv"
    stage2_csv = PROJECT / "stage2" / "results.csv"

    # Stage 1 vs Stage 2 best.pt 중 mAP 높은 쪽 선택
    s1_map = _read_best_map_from_csv(stage1_csv) if stage1_best.exists() else -1.0
    s2_map = _read_best_map_from_csv(stage2_csv) if stage2_best.exists() else -1.0

    print(f"  Stage 1 best mAP50: {s1_map:.4f} ({'exists' if stage1_best.exists() else 'missing'})")
    print(f"  Stage 2 best mAP50: {s2_map:.4f} ({'exists' if stage2_best.exists() else 'missing'})")

    if s1_map < 0 and s2_map < 0:
        print("[ERROR] best.pt 둘 다 없음 — export 불가")
        return

    if s2_map >= s1_map and stage2_best.exists():
        best_path = stage2_best
        print(f"  → Stage 2 채택 (mAP50 {s2_map:.4f} ≥ {s1_map:.4f})")
    else:
        best_path = stage1_best
        print(f"  → Stage 1 채택 (mAP50 {s1_map:.4f} > {s2_map:.4f})")

    if not best_path.exists():
        print(f"[ERROR] 선택된 best.pt 없음: {best_path}")
        return

    best_model = YOLO(str(best_path))
    best_model.export(format="onnx", opset=17, dynamic=True, simplify=True)
    onnx_path = best_path.with_suffix(".onnx")

    metrics = best_model.val(data=str(REFINED / "data.yaml"), imgsz=960, batch=4)
    print("\n=== M4v2 최종 결과 ===")
    print(f"  mAP50:    {metrics.box.map50:.4f}")
    print(f"  mAP50-95: {metrics.box.map:.4f}")
    print(f"  precision: {metrics.box.mp:.4f}")
    print(f"  recall:    {metrics.box.mr:.4f}")
    print(f"  0.9 도달? {'YES ✅' if metrics.box.map50 >= 0.9 else 'NO'}")

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    dst = WEIGHTS_DIR / "m4_yolo_context_elements.onnx"
    shutil.copy2(onnx_path, dst)
    print(f"\nONNX 저장: {dst} ({dst.stat().st_size/1024/1024:.1f}MB)")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    start = time.time()
    print("=" * 60)
    print("M4v2 Local 자동 학습 (정제 → Hard Mining → multi-stage)")
    print("=" * 60)

    # Resume 감지: REFINED 데이터 + Stage 1 last.pt가 있으면 데이터 정제/Hard Mining 스킵
    stage1_last = PROJECT / "stage1" / "weights" / "last.pt"
    stage2_last = PROJECT / "stage2" / "weights" / "last.pt"
    refined_ready = (
        (REFINED / "data.yaml").exists()
        and (REFINED / "images" / "train").exists()
        and len(list((REFINED / "images" / "train").glob("*"))) > 100
    )

    if (stage1_last.exists() or stage2_last.exists()) and refined_ready:
        which = "Stage 2" if stage2_last.exists() else "Stage 1"
        print(f"\n[RESUME] {which} last.pt 감지 → 데이터 정제/Hard Mining 스킵")
        print(f"  REFINED: {REFINED.resolve()}")
    else:
        best_pt_path = find_best_pt()
        model = YOLO(str(best_pt_path))
        refine_dataset(model)
        hard_sample_mining(model)

    train_multi_stage()
    export_and_evaluate()

    elapsed = time.time() - start
    print(f"\n총 소요 시간: {elapsed/3600:.1f}시간")


if __name__ == "__main__":
    main()
