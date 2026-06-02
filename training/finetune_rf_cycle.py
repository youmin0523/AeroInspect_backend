"""
Roboflow 데이터셋 fine-tuning 순환 파이프라인 (2026-06-01, 사용자 지시).

방식 (메모리 project_roboflow_finetune_program 확정):
  남의 .pt 다운 불가 → Roboflow 데이터셋(CC BY 4.0 등) 다운로드 → 라벨을 우리 클래스로 리매핑
  → 우리 train 데이터와 병합 → 우리 base best.pt에서 fine-tune(빠른 epoch) → ONNX export+배치
  → Roboflow 데이터 삭제 → 다음 모델. (로컬 순차, 디스크 절약, OOM 안전)

실행 (모델 1개):
  backend/venv/Scripts/python.exe backend/training/finetune_rf_cycle.py --model THERMAL
사전: Roboflow 데이터셋은 rfenv로 미리 다운로드(download_rf_dataset.py). 이 스크립트는 venv(GPU 학습)에서 실행.
"""
from __future__ import annotations
import argparse, os, shutil, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BT = ROOT / "backend/training"
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── 모델별 설정 ──
# base_pt: fine-tune 시작점(우리 best.pt). our_data: 우리 train/val 데이터셋 dir.
# rf_dir: 다운된 Roboflow 데이터셋. remap: {rf_class_id: our_class_id or None(drop)}.
# names: 우리 클래스(순서=our_class_id). imgsz/epochs.
CONFIGS = {
    "THERMAL": {
        "base_pt": ROOT / "runs/detect/runs/thermal_v11/weights/best.pt",
        "our_data": BT / "datasets/thermal_yolo",
        "rf_dir": BT / "rf_downloads/thermal_idt_v3",
        # RF idt: 0 air-infil,1 air-leak,2 delam,3 hollow,4 insulation,5 moisture
        # 우리: 0 Crack,1 Moisture,2 delamination
        "remap": {0: 2, 1: 2, 2: 2, 3: 2, 4: 2, 5: 1},  # 단열계열→delam, moisture→Moisture
        "names": ["Crack", "Moisture", "delamination"],
        "imgsz": 640, "epochs": 25, "our_sample": 800, "our_sample_val": 200,
        "onnx_out": "thermal_yolo.onnx", "onnx_imgsz": 960, "task": "detect",
    },
    "THERMAL2": {
        # 이미 1차 fine-tune된 thermal에 ScanX moisture 데이터 추가 보강.
        "base_pt": ROOT / "runs/detect/runs/ft_thermal_rf/weights/best.pt",
        "our_data": BT / "datasets/thermal_yolo",
        "rf_dir": BT / "rf_downloads/thermal_scanx_v2",
        # ScanX: 0 Moisture, 1 moisture → 우리 1 Moisture
        "remap": {0: 1, 1: 1},
        "names": ["Crack", "Moisture", "delamination"],
        "imgsz": 640, "epochs": 20, "our_sample": 800, "our_sample_val": 200,
        "onnx_out": "thermal_yolo.onnx", "onnx_imgsz": 960, "task": "detect",
    },
    "M2": {
        "base_pt": ROOT / "runs/detect/runs/m2_surface/phase1_freeze/weights/best.pt",
        "our_data": BT / "datasets/surface",
        "rf_dir": BT / "rf_downloads/m2_builddef_v4",
        "rf_project": ("builddef2", "building-defect-on-walls", 4),
        # RF: 0 mold,1 peeling_paint,2 stairstep_crack,3 water_seepage,4 crack → 전부 표면결함(0)
        "remap": {0: 0, 1: 0, 2: 0, 3: 0, 4: 0},
        "names": ["surface_defect_wall", "baseboard_defect"],
        "imgsz": 640, "epochs": 30,
        "onnx_out": "m2_yolo_surface.onnx", "onnx_imgsz": 640, "task": "detect",
    },
    "M3": {
        "base_pt": ROOT / "runs/detect/runs/m3_floor_window/phase1_freeze/weights/best.pt",
        "our_data": BT / "datasets/floor_window",
        "rf_dir": BT / "rf_downloads/m3_glass_v2",
        "rf_project": ("roboflow-100", "glass-defect-detection-fvbcu", 2),
        # RF: 0 defect,1 glass → glass_defect(1)
        "remap": {0: 1, 1: 1},
        "names": ["floor_defect", "glass_defect", "frame_defect"],
        "imgsz": 640, "epochs": 30,
        "onnx_out": "m3_yolo_floor_window.onnx", "onnx_imgsz": 960, "task": "detect",
    },
    "M4": {
        "base_pt": ROOT / "runs/detect/runs/m4_context/train/weights/best.pt",
        "our_data": BT / "datasets/m4_context",
        "rf_dir": BT / "rf_downloads/m4_wcf_v1",
        "rf_project": ("wall-detection", "wall-ceiling-floor-m6bao", 1),
        # RF: 0 ceiling,1 wall,2 floor → 우리 0 wall,1 ceiling,2 floor
        "remap": {0: 1, 1: 0, 2: 2},
        "names": ["wall", "ceiling", "floor", "window", "door"],
        "imgsz": 640, "epochs": 30,
        "onnx_out": "m4_yolo_context_elements.onnx", "onnx_imgsz": 960, "task": "detect",
    },
}


def remap_labels(src_labels: Path, dst_labels: Path, remap: dict):
    """RF 라벨을 우리 class_id로 리매핑하며 복사. drop=None인 클래스는 제외."""
    dst_labels.mkdir(parents=True, exist_ok=True)
    n_box = 0
    for lf in src_labels.glob("*.txt"):
        out_lines = []
        for line in lf.read_text(encoding="utf-8", errors="ignore").splitlines():
            p = line.strip().split()
            if len(p) < 5:
                continue
            try:
                cid = int(float(p[0]))
            except Exception:
                continue
            mapped = remap.get(cid)
            if mapped is None:
                continue
            out_lines.append(" ".join([str(mapped)] + p[1:]))
            n_box += 1
        (dst_labels / lf.name).write_text("\n".join(out_lines), encoding="utf-8")
    return n_box


def link_or_copy_images(src_images: Path, dst_images: Path, prefix: str):
    """이미지를 dst로 복사(prefix 부여로 파일명 충돌 방지). 라벨 파일명도 동일 prefix 필요."""
    dst_images.mkdir(parents=True, exist_ok=True)
    cnt = 0
    for im in list(src_images.glob("*.jpg")) + list(src_images.glob("*.png")):
        shutil.copy2(im, dst_images / (prefix + im.name))
        cnt += 1
    return cnt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(CONFIGS.keys()))
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()
    cfg = CONFIGS[args.model]
    epochs = args.epochs or cfg["epochs"]

    base_pt = cfg["base_pt"]
    if not base_pt.exists():
        print(f"❌ base.pt 없음: {base_pt}"); return 1
    rf_dir = cfg["rf_dir"]
    if not rf_dir.exists():
        print(f"❌ RF 데이터 없음: {rf_dir} (먼저 download)"); return 1

    # ── 병합 데이터셋 구성 (우리 train+val + RF train+val 리매핑) ──
    merged = BT / f"datasets/_merged_{args.model.lower()}"
    if merged.exists():
        shutil.rmtree(merged)
    for split in ("train", "val"):
        (merged / "images" / split).mkdir(parents=True, exist_ok=True)
        (merged / "labels" / split).mkdir(parents=True, exist_ok=True)

    our = cfg["our_data"]
    # 1) 우리 데이터 복사 (prefix our_) — RF 위주 빠른 fine-tune이므로 소량 앵커만 샘플링.
    #    전체 재학습은 epoch당 89분(thermal 59h)이라 비현실 → 우리 데이터는 과적합 방지 앵커로만.
    #    sampling은 균등 stride (정렬 후 일정 간격)로 deterministic. (Math.random 회피)
    our_sample = cfg.get("our_sample", 800)       # train 앵커 수
    our_sample_val = cfg.get("our_sample_val", 200)
    our_box = 0
    for split, cap in (("train", our_sample), ("val", our_sample_val)):
        si = our / "images" / split
        sl = our / "labels" / split
        if not si.exists():
            continue
        imgs = sorted(list(si.glob("*.jpg")) + list(si.glob("*.png")))
        if cap and len(imgs) > cap:
            stride = len(imgs) / cap
            imgs = [imgs[int(i * stride)] for i in range(cap)]
        for im in imgs:
            shutil.copy2(im, merged / "images" / split / ("our_" + im.name))
            lf = sl / (im.stem + ".txt")
            if lf.exists():
                shutil.copy2(lf, merged / "labels" / split / ("our_" + im.stem + ".txt"))
                our_box += 1
    # 2) RF 데이터 리매핑 복사 (prefix rf_). RF는 train/valid 폴더.
    rf_split_map = {"train": "train", "val": "valid"}
    rf_imgs = 0
    rf_box = 0
    for our_split, rf_split in rf_split_map.items():
        si = rf_dir / rf_split / "images"
        sl = rf_dir / rf_split / "labels"
        if not si.exists():
            continue
        rf_imgs += link_or_copy_images(si, merged / "images" / our_split, "rf_")
        # 라벨 리매핑
        tmp = merged / "labels" / our_split
        for lf in sl.glob("*.txt"):
            out_lines = []
            for line in lf.read_text(encoding="utf-8", errors="ignore").splitlines():
                p = line.strip().split()
                if len(p) < 5:
                    continue
                try:
                    cid = int(float(p[0]))
                except Exception:
                    continue
                m = cfg["remap"].get(cid)
                if m is None:
                    continue
                out_lines.append(" ".join([str(m)] + p[1:]))
                rf_box += 1
            (tmp / ("rf_" + lf.name)).write_text("\n".join(out_lines), encoding="utf-8")

    # data.yaml
    names = cfg["names"]
    (merged / "data.yaml").write_text(
        "train: ./images/train\nval: ./images/val\n"
        f"nc: {len(names)}\nnames: {names}\n", encoding="utf-8")
    print(f"[merge] 우리 {our_box}라벨 + RF {rf_imgs}img/{rf_box}box → {merged}", flush=True)

    # ── fine-tune (우리 best.pt에서 이어서) ──
    from ultralytics import YOLO
    model = YOLO(str(base_pt))
    run_name = f"ft_{args.model.lower()}_rf"
    t0 = time.time()
    model.train(
        data=str(merged / "data.yaml"),
        epochs=epochs, imgsz=cfg["imgsz"], batch=8,
        patience=15, optimizer="AdamW", lr0=2e-4, lrf=0.01,
        warmup_epochs=2,
        # thermal: 약한 색aug (feedback_thermal_weak_augmentation)
        hsv_s=0.3, hsv_v=0.3, degrees=0.0, mixup=0.0, mosaic=0.5,
        project=str(ROOT / "runs/detect/runs"), name=run_name, exist_ok=True,
        verbose=False,
    )
    print(f"[train] {epochs}ep 완료 {(time.time()-t0)/60:.1f}분", flush=True)

    # ── ONNX export + 배치 ──
    best = ROOT / f"runs/detect/runs/{run_name}/weights/best.pt"
    m2 = YOLO(str(best))
    onnx_imgsz = cfg.get("onnx_imgsz", cfg["imgsz"])
    p = m2.export(format="onnx", opset=14, dynamic=(cfg["task"] == "segment"),
                  simplify=True, imgsz=onnx_imgsz)
    dst = ROOT / "backend/models_weights" / cfg["onnx_out"]
    if dst.exists():
        shutil.copy2(dst, dst.with_name(dst.stem + "_pre_rfft.onnx"))
    shutil.copy2(p, dst)
    print(f"[export] → {dst} (imgsz {onnx_imgsz}, _pre_rfft 백업)", flush=True)

    # ── RF 데이터 + 병합 삭제 (디스크 절약) ──
    shutil.rmtree(rf_dir, ignore_errors=True)
    shutil.rmtree(merged, ignore_errors=True)
    print(f"[cleanup] RF+merged 삭제 완료. {args.model} fine-tune 끝.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
