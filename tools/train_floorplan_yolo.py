"""
tools/train_floorplan_yolo.py
역할: CubiCasa5K (또는 합성 평면도 + GT) → YOLOv8 가구 검출 모델 학습.

핵심 흐름:
  1) CubiCasa5K SVG → YOLO bbox 라벨 변환 (svg2yolo)
  2) datasets/yolo_floorplan/ 에 images/{train,val} + labels/{train,val} 구조 생성
  3) ultralytics YOLO 학습 (CPU/GPU 자동 감지)
  4) 학습 후 best.pt → app/services/floorplan_furniture_yolo.py 가 로드해 추론

요구 사항:
  - ultralytics (pip install ultralytics)
  - PyTorch (학습은 GPU 권장 — CPU 도 가능하나 매우 느림)

실행:
  python tools/train_floorplan_yolo.py --data datasets/cubicasa5k --epochs 30
  python tools/train_floorplan_yolo.py --use-synthetic --epochs 5  # 합성 데이터로 빠른 시연 학습
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from random import shuffle

import cv2

# 클래스 정의 — 평면도 가구 4 종 (도형 기반과 호환)
CLASS_NAMES = ['rectangular', 'circular', 'small', 'unknown']
CLASS_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}


def _ensure_yolo_layout(root: Path):
    for sub in ('images/train', 'images/val', 'labels/train', 'labels/val'):
        (root / sub).mkdir(parents=True, exist_ok=True)


def _gt_to_yolo_label(gt_furniture: list[dict]) -> list[str]:
    """GT 가구 리스트 → YOLO 라벨 라인. cx,cy,w,h 가 이미 정규화."""
    lines = []
    for f in gt_furniture:
        cls = CLASS_TO_ID.get(f.get('label', 'rectangular'), 0)
        lines.append(f"{cls} {f['cx']:.6f} {f['cy']:.6f} {f['w']:.6f} {f['h']:.6f}")
    return lines


def build_dataset_from_synthetic(out_root: Path, korean_dir: Path = Path("datasets/synthetic_korean")):
    """한국 합성 + GT JSON → YOLO 데이터셋. 학습 5장 + val 1장 (시연용)."""
    if not korean_dir.exists():
        print(f"[ERR] {korean_dir} 없음 — 먼저 tools/synthesize_korean_floorplans.py 실행")
        sys.exit(1)

    _ensure_yolo_layout(out_root)
    cases = []
    for json_path in sorted(korean_dir.glob("*.json")):
        gt = json.loads(json_path.read_text(encoding='utf-8'))
        img_path = korean_dir / gt['image']
        if img_path.exists():
            cases.append((img_path, gt))

    shuffle(cases)
    n_val = max(1, len(cases) // 5)
    val_cases = cases[:n_val]
    train_cases = cases[n_val:]

    for split, items in [('train', train_cases), ('val', val_cases)]:
        for img_path, gt in items:
            dst_img = out_root / 'images' / split / img_path.name
            shutil.copy(img_path, dst_img)
            label_path = out_root / 'labels' / split / (img_path.stem + '.txt')
            label_path.write_text('\n'.join(_gt_to_yolo_label(gt['furniture_gt'])), encoding='utf-8')

    print(f"  train: {len(train_cases)} 장 / val: {len(val_cases)} 장")
    return train_cases, val_cases


def build_dataset_from_cubicasa5k(cubi_root: Path, out_root: Path, max_samples: int | None = None):
    """CubiCasa5K SVG → YOLO 라벨 변환. SVG 파싱은 svgpathtools / svg.path 필요."""
    try:
        from xml.etree import ElementTree as ET
    except ImportError:
        print("[ERR] xml 표준 라이브러리 누락 (불가능한 상황)")
        sys.exit(1)

    if not cubi_root.exists():
        print(f"[ERR] CubiCasa5K 없음: {cubi_root}")
        print("  먼저 다운로드: python tools/fetch_cubicasa5k.py")
        sys.exit(1)

    _ensure_yolo_layout(out_root)
    # CubiCasa5K 의 폴더 구조: high_quality_architectural/<id>/F1_scaled.png + model.svg
    samples = []
    for sub in ('high_quality_architectural', 'high_quality', 'colorful'):
        d = cubi_root / sub
        if not d.exists():
            continue
        for case_dir in d.iterdir():
            png = case_dir / 'F1_scaled.png'
            svg = case_dir / 'model.svg'
            if png.exists() and svg.exists():
                samples.append((png, svg))
                if max_samples and len(samples) >= max_samples:
                    break
        if max_samples and len(samples) >= max_samples:
            break

    if not samples:
        print(f"[ERR] {cubi_root} 안에 학습 가능한 샘플 없음")
        sys.exit(1)

    print(f"  발견 샘플: {len(samples)} 장")
    shuffle(samples)
    n_val = max(1, len(samples) // 10)
    train_samples = samples[n_val:]
    val_samples = samples[:n_val]

    # CubiCasa SVG 의 가구 클래스 매핑 (단순화)
    CUBI_LABEL_MAP = {
        'Bed': 'rectangular', 'Sofa': 'rectangular', 'Table': 'circular',
        'Chair': 'small', 'Toilet': 'small', 'Sink': 'small',
        'Bath': 'rectangular', 'Refrigerator': 'rectangular',
        'Stove': 'rectangular', 'Closet': 'rectangular',
    }

    converted = 0
    for split, items in [('train', train_samples), ('val', val_samples)]:
        for png, svg in items:
            try:
                img = cv2.imread(str(png))
                if img is None:
                    continue
                H, W = img.shape[:2]
                tree = ET.parse(svg)
                root = tree.getroot()
                lines = []
                for elem in root.iter():
                    cls_name = elem.attrib.get('class', '')
                    label = CUBI_LABEL_MAP.get(cls_name)
                    if not label:
                        continue
                    # SVG 가구 노드는 transform/d 속성이 다양 — bbox 단순화
                    # 정확한 변환은 svg.path 라이브러리 필요. 여기는 transform 의 translate 만
                    # 추출하는 휴리스틱 (학습 시 정밀 변환 필요).
                    transform = elem.attrib.get('transform', '')
                    if 'translate(' not in transform:
                        continue
                    inside = transform.split('translate(')[1].split(')')[0]
                    parts = [p.strip() for p in inside.replace(',', ' ').split() if p.strip()]
                    if len(parts) < 2:
                        continue
                    try:
                        tx, ty = float(parts[0]), float(parts[1])
                    except ValueError:
                        continue
                    # bbox 추정 (가구별 표준 크기 — 실제 학습 시 path d 파싱 필요)
                    fw, fh = 80, 60
                    cx_n = (tx + fw/2) / W
                    cy_n = (ty + fh/2) / H
                    if 0 <= cx_n <= 1 and 0 <= cy_n <= 1:
                        cls_id = CLASS_TO_ID[label]
                        lines.append(f"{cls_id} {cx_n:.6f} {cy_n:.6f} {fw/W:.6f} {fh/H:.6f}")
                if not lines:
                    continue
                shutil.copy(png, out_root / 'images' / split / png.name)
                (out_root / 'labels' / split / (png.stem + '.txt')).write_text(
                    '\n'.join(lines), encoding='utf-8')
                converted += 1
            except Exception as e:
                print(f"  [skip] {png.name}: {type(e).__name__}: {e}")

    print(f"  변환 완료: {converted} 장 (train+val)")
    if converted == 0:
        print("[ERR] 변환 가능한 샘플 0 — SVG 형식 변경 가능. svg.path 라이브러리 도입 필요")
        sys.exit(1)


def write_data_yaml(yolo_root: Path):
    yaml_path = yolo_root / 'data.yaml'
    yaml_path.write_text(
        f"path: {yolo_root.resolve().as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"nc: {len(CLASS_NAMES)}\n"
        f"names: {CLASS_NAMES}\n",
        encoding='utf-8',
    )
    return yaml_path


def train(yaml_path: Path, epochs: int, imgsz: int, model_name: str = "yolov8n.pt"):
    """ultralytics YOLO 학습."""
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERR] ultralytics 미설치. 설치:")
        print("  pip install ultralytics")
        sys.exit(1)

    model = YOLO(model_name)
    results = model.train(
        data=str(yaml_path),
        epochs=epochs,
        imgsz=imgsz,
        batch=8,
        project=str(yaml_path.parent / 'runs'),
        name='floorplan_furniture',
        exist_ok=True,
    )
    best_pt = Path(results.save_dir) / 'weights' / 'best.pt'
    if best_pt.exists():
        target = Path('models_weights') / 'floorplan_furniture_yolo.pt'
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(best_pt, target)
        print(f"\n학습 완료 → {target}")
    return results


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("datasets/cubicasa5k"))
    p.add_argument("--out", type=Path, default=Path("datasets/yolo_floorplan"))
    p.add_argument("--use-synthetic", action="store_true",
                   help="CubiCasa 대신 합성 한국 평면도 (datasets/synthetic_korean) 사용")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--build-only", action="store_true",
                   help="데이터셋 변환만 하고 학습은 스킵")
    args = p.parse_args()

    print("=" * 60)
    print(" YOLOv8 평면도 가구 검출 학습 인프라")
    print("=" * 60)

    if args.use_synthetic:
        print("\n[1/3] 합성 한국 평면도 → YOLO 데이터셋")
        build_dataset_from_synthetic(args.out)
    else:
        print(f"\n[1/3] CubiCasa5K → YOLO 데이터셋 (max={args.max_samples})")
        build_dataset_from_cubicasa5k(args.data, args.out, args.max_samples)

    print(f"\n[2/3] data.yaml 작성")
    yaml_path = write_data_yaml(args.out)
    print(f"  {yaml_path}")

    if args.build_only:
        print("\n--build-only 지정 — 학습 스킵")
        return

    print(f"\n[3/3] YOLOv8n 학습 시작 (epochs={args.epochs}, imgsz={args.imgsz})")
    train(yaml_path, args.epochs, args.imgsz)


if __name__ == "__main__":
    main()
