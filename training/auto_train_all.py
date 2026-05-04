"""
자동 ML 학습 파이프라인 — 모니터링 + 순차 학습 + ONNX 변환
한 번 실행하면 모든 학습을 자동으로 이어서 진행합니다.
5분마다 로그를 auto_train_log.txt에 기록합니다.
"""
import os
import sys
import time
import shutil
import json
from pathlib import Path
from datetime import datetime

# ── 경로 설정 ──
TRAIN_DIR = Path(__file__).parent.resolve()
PROJECT_DIR = TRAIN_DIR.parent.parent
WEIGHTS_DIR = TRAIN_DIR.parent / "models_weights"
WEIGHTS_DIR.mkdir(exist_ok=True)

LOG_FILE = TRAIN_DIR / "auto_train_log.txt"

def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")

def log_separator(title):
    log(f"{'='*60}")
    log(f"  {title}")
    log(f"{'='*60}")

# ── GPU 확인 ──
import torch
log(f"PyTorch: {torch.__version__}")
log(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    log(f"GPU: {torch.cuda.get_device_name(0)}")

# ══════════════════════════════════════════════
# 1. M1 YOLO — structural 균열+방수 (GPU)
# ══════════════════════════════════════════════
def train_m1_yolo():
    log_separator("M1 YOLO: structural (crack + waterproof) — GPU")
    from ultralytics import YOLO

    os.chdir(str(TRAIN_DIR))
    model = YOLO("yolov8m.pt")

    results = model.train(
        data="datasets/structural/data.yaml",
        epochs=100,
        imgsz=640,
        batch=16,
        patience=20,
        project=str(TRAIN_DIR / "runs" / "m1_structural"),
        name="yolov8m",
        device=0,
        optimizer="AdamW",
        lr0=0.001,
        cos_lr=True,
        save=True,
        exist_ok=True,
    )

    metrics = model.val()
    log(f"M1 YOLO 완료 — mAP@0.5: {metrics.box.map50:.4f}, mAP@0.5:0.95: {metrics.box.map:.4f}")

    # ONNX
    model.export(format="onnx", imgsz=640, dynamic=True, simplify=True)
    best_onnx = TRAIN_DIR / "runs" / "m1_structural" / "yolov8m" / "weights" / "best.onnx"
    dst = WEIGHTS_DIR / "m1_yolo_structural.onnx"
    if best_onnx.exists():
        shutil.copy(best_onnx, dst)
        log(f"M1 ONNX 저장: {dst}")

    return metrics

# ══════════════════════════════════════════════
# 2. M2 YOLO — surface 마감하자 (GPU)
# ══════════════════════════════════════════════
def train_m2_yolo():
    log_separator("M2 YOLO: surface (surface_defect_wall + baseboard) — GPU")
    from ultralytics import YOLO

    os.chdir(str(TRAIN_DIR))
    model = YOLO("yolov8m.pt")

    results = model.train(
        data="datasets/surface/data.yaml",
        epochs=100,
        imgsz=640,
        batch=16,
        patience=20,
        project=str(TRAIN_DIR / "runs" / "m2_surface"),
        name="yolov8m",
        device=0,
        optimizer="AdamW",
        lr0=0.001,
        cos_lr=True,
        save=True,
        exist_ok=True,
    )

    metrics = model.val()
    log(f"M2 YOLO 완료 — mAP@0.5: {metrics.box.map50:.4f}, mAP@0.5:0.95: {metrics.box.map:.4f}")

    model.export(format="onnx", imgsz=640, dynamic=True, simplify=True)
    best_onnx = TRAIN_DIR / "runs" / "m2_surface" / "yolov8m" / "weights" / "best.onnx"
    dst = WEIGHTS_DIR / "m2_yolo_surface.onnx"
    if best_onnx.exists():
        shutil.copy(best_onnx, dst)
        log(f"M2 YOLO ONNX 저장: {dst}")

    return metrics

# ══════════════════════════════════════════════
# 3. M2 ResNet — surface_crops 5클래스 분류 (CPU)
# ══════════════════════════════════════════════
def train_m2_resnet():
    log_separator("M2 ResNet: surface_crops 5클래스 분류 — CPU")
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from torchvision import datasets, models, transforms

    os.chdir(str(TRAIN_DIR))
    device = torch.device("cpu")

    train_tf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.RandomGrayscale(0.05),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_ds = datasets.ImageFolder("datasets/surface_crops/train", train_tf)
    val_ds = datasets.ImageFolder("datasets/surface_crops/val", val_tf)
    CLASS_NAMES = train_ds.classes
    NUM_CLASSES = len(CLASS_NAMES)
    log(f"Classes({NUM_CLASSES}): {CLASS_NAMES}")
    log(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=2)

    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Sequential(nn.Dropout(0.4), nn.Linear(model.fc.in_features, NUM_CLASSES))
    model = model.to(device)

    class_weights = torch.tensor([3.93, 2.25, 1.0, 5.02, 7.53], dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-6)

    save_dir = TRAIN_DIR / "runs" / "m2_resnet_surface"
    save_dir.mkdir(parents=True, exist_ok=True)
    best_val_acc = 0.0
    start = time.time()

    for epoch in range(30):
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            optimizer.zero_grad()
            out = model(images)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            correct += (out.argmax(1) == labels).sum().item()
            total += labels.size(0)
        scheduler.step()

        model.eval()
        vc, vt = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                _, pred = model(images).max(1)
                vt += labels.size(0)
                vc += pred.eq(labels).sum().item()
        val_acc = vc / vt

        if (epoch + 1) % 5 == 0 or val_acc > best_val_acc:
            log(f"  Epoch {epoch+1:3d}/30 | Loss: {train_loss/len(train_loader):.4f} | TrainAcc: {correct/total:.4f} | ValAcc: {val_acc:.4f} | {time.time()-start:.0f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({"model_state_dict": model.state_dict(), "class_names": CLASS_NAMES, "val_acc": val_acc}, save_dir / "best.pt")

    log(f"M2 ResNet 완료 — Best Val Acc: {best_val_acc:.4f} ({time.time()-start:.0f}s)")

    # ONNX
    ckpt = torch.load(save_dir / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    dummy = torch.randn(1, 3, 224, 224)
    onnx_path = WEIGHTS_DIR / "m2_resnet_surface_classifier.onnx"
    torch.onnx.export(model, dummy, str(onnx_path), opset_version=17,
                       input_names=["image"], output_names=["logits"],
                       dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}})
    log(f"M2 ResNet ONNX 저장: {onnx_path}")

    return best_val_acc

# ══════════════════════════════════════════════
# 4. 열화상 YOLO — thermal_yolo 3클래스 (GPU)
# ══════════════════════════════════════════════
def train_thermal_yolo():
    log_separator("열화상 YOLO: thermal_yolo (Crack+Moisture+delamination) — GPU")
    from ultralytics import YOLO

    os.chdir(str(TRAIN_DIR))
    model = YOLO("yolov8m.pt")

    results = model.train(
        data="datasets/thermal_yolo/data.yaml",
        epochs=100,
        imgsz=640,
        batch=16,
        patience=20,
        project=str(TRAIN_DIR / "runs" / "thermal"),
        name="yolov8m",
        device=0,
        optimizer="AdamW",
        lr0=0.001,
        cos_lr=True,
        hsv_h=0.0,  # 열화상 특화: 색상 증강 약화
        hsv_s=0.3,
        hsv_v=0.3,
        save=True,
        exist_ok=True,
    )

    metrics = model.val()
    log(f"열화상 YOLO 완료 — mAP@0.5: {metrics.box.map50:.4f}, mAP@0.5:0.95: {metrics.box.map:.4f}")

    model.export(format="onnx", imgsz=640, dynamic=True, simplify=True)
    best_onnx = TRAIN_DIR / "runs" / "thermal" / "yolov8m" / "weights" / "best.onnx"
    dst = WEIGHTS_DIR / "thermal_yolo.onnx"
    if best_onnx.exists():
        shutil.copy(best_onnx, dst)
        log(f"열화상 ONNX 저장: {dst}")

    return metrics

# ══════════════════════════════════════════════
# 5. M3 YOLO — floor_window 창호 (CPU, 소량)
# ══════════════════════════════════════════════
def train_m3_yolo_cpu():
    log_separator("M3 YOLO: floor_window (frame_defect) — CPU")
    from ultralytics import YOLO

    os.chdir(str(TRAIN_DIR))
    model = YOLO("yolov8s.pt")  # 작은 모델 (CPU용)

    # yolov8s.pt 없으면 다운로드됨
    results = model.train(
        data="datasets/floor_window/data.yaml",
        epochs=100,
        imgsz=640,
        batch=8,
        patience=30,
        project=str(TRAIN_DIR / "runs" / "m3_floor_window"),
        name="yolov8s",
        device="cpu",
        optimizer="AdamW",
        lr0=0.001,
        cos_lr=True,
        save=True,
        exist_ok=True,
    )

    metrics = model.val()
    log(f"M3 YOLO 완료 — mAP@0.5: {metrics.box.map50:.4f}")

    model.export(format="onnx", imgsz=640, dynamic=True, simplify=True)
    best_onnx = TRAIN_DIR / "runs" / "m3_floor_window" / "yolov8s" / "weights" / "best.onnx"
    dst = WEIGHTS_DIR / "m3_yolo_floor_window.onnx"
    if best_onnx.exists():
        shutil.copy(best_onnx, dst)
        log(f"M3 ONNX 저장: {dst}")

    return metrics


# ══════════════════════════════════════════════
# 메인 파이프라인
# ══════════════════════════════════════════════
if __name__ == "__main__":
    log_separator("자동 ML 학습 파이프라인 시작")
    log(f"작업 디렉토리: {TRAIN_DIR}")
    log(f"가중치 저장: {WEIGHTS_DIR}")

    results = {}

    # ── 이미 진행 중인 학습 대기 ──
    # 기존 M1 YOLO (PID 28876)과 M2 ResNet (PID 15628)이 돌고 있을 수 있음
    # psutil 없이 간단하게 체크
    import subprocess

    def is_pid_alive(pid):
        try:
            r = subprocess.run(["tasklist", "/FI", f"PID eq {pid}"], capture_output=True, text=True)
            return str(pid) in r.stdout
        except:
            return False

    # 기존 M1 YOLO 대기
    m1_pid = 28876
    if is_pid_alive(m1_pid):
        log(f"기존 M1 YOLO (PID {m1_pid}) 진행 중 — 완료 대기...")
        while is_pid_alive(m1_pid):
            # 5분마다 로그
            m1_csv = Path(r"c:/Users/Codelab/Desktop/PROJECT/TEAM_PROJECT_2_Drone_project/runs/detect/runs/m1_structural/yolov8m/results.csv")
            if m1_csv.exists():
                lines = m1_csv.read_text().strip().split("\n")
                if len(lines) > 1:
                    last = lines[-1].split(",")
                    epoch = last[0].strip()
                    map50 = last[6].strip() if len(last) > 6 else "?"
                    log(f"  M1 YOLO: epoch {epoch}/100, mAP@0.5={map50}")
            time.sleep(300)  # 5분 대기
        log("기존 M1 YOLO 완료 감지")

        # M1 결과 확인
        m1_best = Path(r"c:/Users/Codelab/Desktop/PROJECT/TEAM_PROJECT_2_Drone_project/runs/detect/runs/m1_structural/yolov8m/weights/best.pt")
        if m1_best.exists():
            dst = WEIGHTS_DIR / "m1_yolo_structural.pt"
            shutil.copy(m1_best, dst)
            log(f"M1 best.pt 복사: {dst}")
            # ONNX 변환
            from ultralytics import YOLO
            m = YOLO(str(m1_best))
            m.export(format="onnx", imgsz=640, dynamic=True, simplify=True)
            onnx_src = m1_best.with_suffix(".onnx")
            if onnx_src.exists():
                shutil.copy(onnx_src, WEIGHTS_DIR / "m1_yolo_structural.onnx")
                log(f"M1 ONNX 저장 완료")
    else:
        # M1이 안 돌고 있으면 새로 시작
        log("M1 YOLO 프로세스 없음 — 새로 학습 시작")
        try:
            train_m1_yolo()
        except Exception as e:
            log(f"M1 YOLO 에러: {e}")

    # ── M2 ResNet 대기 (CPU) ──
    m2r_pid = 15628
    if is_pid_alive(m2r_pid):
        log(f"기존 M2 ResNet (PID {m2r_pid}) CPU 진행 중 — 백그라운드 대기 (GPU 학습과 병렬)")

    # ── M2 YOLO surface (GPU) ──
    try:
        train_m2_yolo()
        results["m2_yolo"] = "완료"
    except Exception as e:
        log(f"M2 YOLO 에러: {e}")
        results["m2_yolo"] = f"에러: {e}"

    # ── 열화상 YOLO (GPU) ──
    try:
        train_thermal_yolo()
        results["thermal_yolo"] = "완료"
    except Exception as e:
        log(f"열화상 YOLO 에러: {e}")
        results["thermal_yolo"] = f"에러: {e}"

    # ── M2 ResNet 완료 대기 ──
    if is_pid_alive(m2r_pid):
        log("M2 ResNet CPU 완료 대기 중...")
        while is_pid_alive(m2r_pid):
            time.sleep(300)
        log("M2 ResNet 완료 감지")
    else:
        # 안 돌고 있으면 새로 시작
        try:
            train_m2_resnet()
            results["m2_resnet"] = "완료"
        except Exception as e:
            log(f"M2 ResNet 에러: {e}")

    # ── M3 YOLO floor_window (CPU) ──
    try:
        train_m3_yolo_cpu()
        results["m3_yolo"] = "완료"
    except Exception as e:
        log(f"M3 YOLO 에러: {e}")
        results["m3_yolo"] = f"에러: {e}"

    # ── 최종 리포트 ──
    log_separator("전체 학습 파이프라인 완료")

    # models_weights/ 확인
    log("생성된 ONNX 모델:")
    for f in sorted(WEIGHTS_DIR.glob("*.onnx")):
        size_mb = f.stat().st_size / 1024 / 1024
        log(f"  {f.name}: {size_mb:.1f} MB")

    log(f"\n결과 요약: {json.dumps(results, ensure_ascii=False, indent=2)}")
    log("내일 auto_train_log.txt 확인해주세요.")
