# =============================================
# finetune_960.py
# 기존 best.pt → imgsz=960 fine-tune (소형 객체 대응)
# copy_paste + multi_scale + 50ep
# 전 YOLO 모델 순차 실행
#
# 사용법:
#   cd backend/training
#   python -u finetune_960.py
# =============================================

import io
import shutil
import sys
from pathlib import Path

from ultralytics import YOLO

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

WEIGHTS_DIR = Path("../models_weights")


def find_best_pt(pattern: str) -> str:
    """기존 학습에서 저장된 best.pt 찾기."""
    import glob
    root = Path(__file__).resolve().parent.parent.parent
    for search_root in [Path("."), Path(".."), root]:
        for g in glob.glob(str(search_root / pattern), recursive=True):
            p = Path(g).resolve()
            if p.exists():
                print(f"  Found: {p} ({p.stat().st_size/1024/1024:.1f}MB)")
                return str(p)
    return None


def finetune(
    name: str,
    best_pt: str,
    data_yaml: str,
    output_name: str,
    epochs: int = 25,    # 30 → 25 (추가 단축)
    batch: int = 4,
    imgsz: int = 960,
    patience: int = 8,   # 10 → 8 (더 빠른 plateau 종료)
    copy_paste: float = 0.3,
    multi_scale: float = 0.2,  # 0.5 → 0.2 (입력 변동 줄여 ep 시간 안정화)
    freeze: int = 10,    # 백본 freeze로 fine-tune 안정화
):
    """기존 best.pt에서 imgsz=960으로 fine-tune.

    0.9+ 달성 전략:
    - lr0=1e-4 (10배 키움): 1e-5는 너무 보수적이라 baseline 회복도 못 함
    - freeze=10: 백본 안정화, head만 적극 학습
    - close_mosaic=20: 마지막 20ep는 mosaic 끔 → 안정 수렴
    - mixup 0.1→0.05: 안정성 우선
    """
    print(f"\n{'='*60}")
    print(f"[{name}] 960 Fine-tune 시작 (0.9+ 전략)")
    print(f"  base: {best_pt}")
    print(f"  imgsz={imgsz}, batch={batch}, epochs={epochs}, lr0=1e-4, freeze={freeze}")
    print(f"{'='*60}")

    model = YOLO(best_pt)
    model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        cache="disk",        # 디스크 캐싱: 1ep 후 npy로 저장 → 2ep부터 빠름
        workers=4,           # 2 → 4 (데이터 로딩 병렬화)
        optimizer="AdamW",
        lr0=1e-4,            # 1e-5 → 1e-4 (핵심 변경)
        lrf=0.01,
        patience=patience,
        warmup_epochs=2,     # 3 → 2 (단축)
        close_mosaic=15,     # 20 → 15 (epochs=25에 맞춤)
        freeze=freeze,
        hsv_h=0.015, hsv_s=0.5, hsv_v=0.4,
        degrees=5.0, translate=0.1, scale=0.5,
        shear=2.0, perspective=0.001,
        flipud=0.0, fliplr=0.5,
        mosaic=0.8, mixup=0.0,  # mosaic 1.0 → 0.8, mixup OFF (속도 +)
        erasing=0.0,
        copy_paste=copy_paste,
        multi_scale=multi_scale,
        save_period=5,       # 매 5ep마다만 저장 (IO 감소)
        plots=False,         # matplotlib 시각화 OFF (속도 +)
        project=f"runs/{output_name}_960",
        name="finetune",
        exist_ok=True,
    )

    # ONNX export
    import glob
    root = Path(__file__).resolve().parent.parent.parent
    pattern = f"**/{output_name}_960/finetune/weights/best.pt"
    ft_best = None
    for sr in [Path("."), Path(".."), root]:
        for g in glob.glob(str(sr / pattern), recursive=True):
            ft_best = str(Path(g).resolve())
            break
        if ft_best:
            break

    if ft_best:
        best_model = YOLO(ft_best)
        best_model.export(format="onnx", opset=17, dynamic=True, simplify=True)
        onnx_src = ft_best.replace(".pt", ".onnx")
        onnx_dst = WEIGHTS_DIR / f"{output_name}.onnx"
        WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(onnx_src, onnx_dst)
        print(f"[{name}] ONNX 저장 완료: {onnx_dst}")
    else:
        print(f"[{name}] best.pt 못 찾음 — ONNX 미생성")


def main():
    # M3-YOLO는 04/29 09:39 finetune 완료 (m3_yolo_floor_window.onnx 저장됨) → 제외
    # M5-v2는 코랩 T4에서 별도 학습 (병렬화로 시간 단축)
    # 로컬: M1 → M2 순서
    models = [
        {
            "name": "M1-YOLO",
            "pattern": "**/m1_structural/phase2_full/weights/best.pt",
            "data_yaml": "configs/structural.yaml",
            "output_name": "m1_yolo_structural",
            "epochs": 25,
            "copy_paste": 0.3,
            "multi_scale": 0.2,
        },
        {
            "name": "M2-YOLO",
            "pattern": None,  # fresh train (yolov8m.pt → 25ep)
            "data_yaml": "configs/surface.yaml",
            "output_name": "m2_yolo_surface",
            "epochs": 25,
            "copy_paste": 0.3,
            "multi_scale": 0.2,
        },
    ]

    for m in models:
        if m["pattern"]:
            best_pt = find_best_pt(m["pattern"])
            if not best_pt:
                print(f"[{m['name']}] best.pt 없음 — 스킵")
                continue
        else:
            # Fresh train (M2-YOLO)
            best_pt = "yolov8m.pt"
            print(f"[{m['name']}] Fresh train from pretrained")

        finetune(
            name=m["name"],
            best_pt=best_pt,
            data_yaml=m["data_yaml"],
            output_name=m["output_name"],
            epochs=m["epochs"],
            copy_paste=m["copy_paste"],
            multi_scale=m["multi_scale"],
        )

    print(f"\n{'='*60}")
    print("전체 960 Fine-tune 완료!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
