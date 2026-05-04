# =============================================
# prepare_instructor_datasets.py
# 역할: 강사 제출용 dataset 정리 + zip 묶기
#       - 각 학습용 데이터셋 (train/val/test) 정리
#       - data.yaml + README 동봉
#       - Drive 업로드용 zip 분할 (10GB 단위)
#       - 메타데이터 (각 split의 이미지/라벨 개수, 클래스 정보)
#
# 사용:
#   cd backend/training
#   python prepare_instructor_datasets.py --out instructor_submission --zip
# =============================================

from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Dict, List

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# 강사 제출용 데이터셋 (학습용 핵심 데이터셋만)
DATASETS_TO_INCLUDE = [
    {
        "name": "structural",
        "description": "M1: 구조 균열·방수 하자 (YOLO bbox)",
        "model": "M1 YOLO + ResNet 분류",
        "purpose": "균열, 방수 결함, 코킹 결함 탐지",
        "train_command": "python train_m1_yolo_structural.py",
    },
    {
        "name": "surface",
        "description": "M2: 마감·표면 하자 (YOLO bbox)",
        "model": "M2 YOLO + ResNet 분류",
        "purpose": "도배/도색/걸레받이 결함 탐지",
        "train_command": "python train_m2_yolo_surface.py",
    },
    {
        "name": "floor_window",
        "description": "M3: 바닥·창호 하자 (YOLO bbox)",
        "model": "M3 YOLO + ResNet 분류",
        "purpose": "바닥/유리/창호 결함 탐지",
        "train_command": "python train_m3_yolo_floor_window.py",
    },
    {
        "name": "frames",
        "description": "M5: 창/문 프레임 segmentation",
        "model": "M5 YOLOv8m-seg",
        "purpose": "수직도/수평도/직각도 측정용 프레임 추출",
        "train_command": "python train_m5_frame_seg.py",
    },
    {
        "name": "m4_context",
        "description": "M4 Context: 환경 컨텍스트 (wall/ceiling/floor/window/door)",
        "model": "M4 Context YOLO",
        "purpose": "geometric gating용 (검출 위치 검증)",
        "train_command": "python train_m4v2_local.py",
    },
    {
        "name": "structural_crops",
        "description": "M1 ResNet 학습용 균열 분류 크롭 (5 클래스)",
        "model": "M1 ResNet50 classifier",
        "purpose": "crack 검출 후 sub-type 분류",
        "train_command": "python train_m1_resnet_crack.py",
    },
    {
        "name": "surface_crops",
        "description": "M2 ResNet 학습용 표면 분류 크롭 (19 클래스)",
        "model": "M2 ResNet50 classifier",
        "purpose": "surface_defect_wall 검출 후 sub-type 분류",
        "train_command": "python train_m2_resnet_surface.py",
    },
    {
        "name": "floor_window_crops",
        "description": "M3 ResNet 학습용 바닥/창호 분류 크롭 (3 클래스)",
        "model": "M3 ResNet50 classifier",
        "purpose": "floor/glass/frame 검출 후 sub-type 분류",
        "train_command": "python train_m3_resnet_floor_window.py",
    },
    {
        "name": "thermal_yolo",
        "description": "M4 U-Net 학습용 열화상 segmentation",
        "model": "M4 U-Net thermal",
        "purpose": "단열 결함 영역 segmentation",
        "train_command": "python train_m4_thermal_unet.py",
    },
    {
        "name": "furniture_aware",
        "description": "furniture_aware: 빌트인 가구 + 환경 인식 (10 클래스)",
        "model": "yolov11l 10-class",
        "purpose": "FP 차단 (가구 위 false positive 제거)",
        "train_command": "Colab notebook: furniture_aware_train.ipynb",
    },
]


def count_files(d: Path, exts: set) -> int:
    if not d.exists():
        return 0
    return sum(1 for f in d.rglob("*") if f.is_file() and f.suffix.lower() in exts)


def make_dataset_metadata(ds_dir: Path, info: dict) -> dict:
    """데이터셋 통계 수집."""
    img_exts = {".jpg", ".jpeg", ".png"}
    lbl_exts = {".txt"}

    metadata = {
        "name": info["name"],
        "description": info["description"],
        "model": info["model"],
        "purpose": info["purpose"],
        "train_command": info["train_command"],
        "splits": {},
    }

    # YOLO format (images/labels 분리)
    for split in ["train", "val", "valid", "test"]:
        img_dir = ds_dir / "images" / split
        lbl_dir = ds_dir / "labels" / split
        n_img = count_files(img_dir, img_exts)
        n_lbl = count_files(lbl_dir, lbl_exts)
        if n_img > 0 or n_lbl > 0:
            metadata["splits"][split] = {
                "images": n_img,
                "labels": n_lbl,
                "img_path": str(img_dir.relative_to(ds_dir)) if img_dir.exists() else None,
                "lbl_path": str(lbl_dir.relative_to(ds_dir)) if lbl_dir.exists() else None,
            }

    # ImageFolder format (split별로 폴더, ResNet crops)
    for split in ["train", "val", "valid", "test"]:
        sd = ds_dir / split
        if sd.exists() and sd.is_dir():
            n = count_files(sd, img_exts)
            classes = sorted([c.name for c in sd.iterdir() if c.is_dir()])
            if n > 0:
                metadata["splits"][split] = {
                    "images": n,
                    "classes": classes,
                    "format": "ImageFolder",
                }

    # data.yaml 존재 여부
    yaml_path = ds_dir / "data.yaml"
    metadata["data_yaml"] = yaml_path.exists()

    # 총 용량
    total_bytes = 0
    if ds_dir.exists():
        for f in ds_dir.rglob("*"):
            if f.is_file():
                total_bytes += f.stat().st_size
    metadata["total_size_mb"] = round(total_bytes / (1024 * 1024), 1)

    return metadata


def render_readme(metadata: dict) -> str:
    lines = []
    lines.append(f"# {metadata['name']}")
    lines.append(f"\n**설명**: {metadata['description']}")
    lines.append(f"**모델**: {metadata['model']}")
    lines.append(f"**목적**: {metadata['purpose']}")
    lines.append(f"**학습 명령**: `{metadata['train_command']}`")
    lines.append(f"**총 용량**: {metadata['total_size_mb']} MB")
    lines.append(f"**data.yaml 동봉**: {'예' if metadata['data_yaml'] else '아니오'}\n")

    lines.append("## Splits\n")
    if metadata["splits"]:
        for split, info in metadata["splits"].items():
            lines.append(f"### {split}")
            if "format" in info and info["format"] == "ImageFolder":
                lines.append(f"- 형식: ImageFolder (PyTorch)")
                lines.append(f"- 이미지: {info['images']}")
                lines.append(f"- 클래스: {info.get('classes', [])}")
            else:
                lines.append(f"- 이미지: {info['images']}")
                lines.append(f"- 라벨: {info['labels']}")
                lines.append(f"- 형식: YOLO (xyxy normalized)")
            lines.append("")
    else:
        lines.append("(splits 없음)\n")

    return "\n".join(lines)


def prepare_dataset(
    src_root: Path, ds_info: dict, out_root: Path, dry_run: bool = False,
) -> Optional[dict]:
    """데이터셋 1개 정리 + 메타데이터 + README 생성."""
    src = src_root / ds_info["name"]
    if not src.exists():
        print(f"  [SKIP] {ds_info['name']}: 디렉토리 없음 ({src})")
        return None

    dst = out_root / ds_info["name"]
    metadata = make_dataset_metadata(src, ds_info)

    if dry_run:
        print(f"  [DRY-RUN] {ds_info['name']}: {metadata['total_size_mb']} MB")
        return metadata

    print(f"  [COPY] {ds_info['name']} → {dst.name}/ ({metadata['total_size_mb']} MB)")
    if dst.exists():
        shutil.rmtree(dst)
    # 복사 (단, .cache, __pycache__ 같은 캐시 제외)
    def ignore_patterns(_dir, files):
        ignore = []
        for f in files:
            if f in {"__pycache__", ".cache", ".DS_Store"} or f.endswith(".cache"):
                ignore.append(f)
        return ignore
    shutil.copytree(src, dst, ignore=ignore_patterns)

    # README + metadata.json 동봉
    (dst / "README.md").write_text(render_readme(metadata), encoding="utf-8")
    (dst / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    return metadata


def make_zip_archives(out_root: Path, max_size_mb: int = 10000) -> List[Path]:
    """out_root 디렉토리의 각 dataset을 zip으로 묶음.
    크기 큰 데이터셋은 단독, 작은 것들은 합쳐서 묶음 가능 (단순화 위해 단독 zip)."""
    archives = []
    for ds_dir in sorted(out_root.iterdir()):
        if not ds_dir.is_dir():
            continue
        zip_path = out_root / f"{ds_dir.name}.zip"
        print(f"  [ZIP] {ds_dir.name}.zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in ds_dir.rglob("*"):
                if f.is_file():
                    zf.write(f, f.relative_to(out_root))
        archives.append(zip_path)
    return archives


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, default="datasets")
    parser.add_argument("--out", type=str, default="instructor_submission")
    parser.add_argument("--zip", action="store_true", help="zip 아카이브 생성")
    parser.add_argument("--dry-run", action="store_true", help="크기만 측정 (복사 X)")
    args = parser.parse_args()

    cwd = Path.cwd()
    src_root = cwd / args.src
    out_root = cwd / args.out

    if not src_root.exists():
        print(f"[ERROR] src not found: {src_root}")
        return 1

    if not args.dry_run:
        out_root.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("강사 제출용 dataset 정리")
    print(f"  src: {src_root}")
    print(f"  out: {out_root}")
    print(f"  dry_run: {args.dry_run}")
    print("=" * 60)

    all_metadata: List[dict] = []
    total_mb = 0
    for ds_info in DATASETS_TO_INCLUDE:
        md = prepare_dataset(src_root, ds_info, out_root, dry_run=args.dry_run)
        if md:
            all_metadata.append(md)
            total_mb += md["total_size_mb"]

    print(f"\n총 데이터셋: {len(all_metadata)}개")
    print(f"총 용량: {total_mb:.1f} MB ({total_mb/1024:.2f} GB)")

    if not args.dry_run:
        # 종합 README 생성
        readme = ["# 드론 하자검출 학습 데이터셋 모음\n"]
        readme.append(f"\n총 {len(all_metadata)}개 데이터셋, {total_mb/1024:.2f} GB\n")
        readme.append("\n## 데이터셋 목록\n")
        readme.append("| 이름 | 모델 | 용량 (MB) | 목적 |")
        readme.append("|------|------|----------|------|")
        for md in all_metadata:
            readme.append(
                f"| {md['name']} | {md['model']} | {md['total_size_mb']} | {md['purpose']} |"
            )
        readme.append("\n## 학습 재현 방법\n")
        readme.append("각 데이터셋 디렉토리의 README.md 참조. backend/training/ 디렉토리에서 train_*.py 실행.")
        readme.append("\n## 라이선스\n")
        readme.append("연구·교육 목적, 강사님 검토용. 외부 배포 금지.\n")
        (out_root / "README.md").write_text("\n".join(readme), encoding="utf-8")

        # 종합 metadata.json
        (out_root / "datasets_index.json").write_text(
            json.dumps(all_metadata, indent=2, ensure_ascii=False), encoding="utf-8",
        )

        if args.zip:
            print("\n[ZIP] 아카이브 생성 중...")
            archives = make_zip_archives(out_root)
            print(f"  {len(archives)}개 zip 생성")
            for a in archives:
                size_mb = a.stat().st_size / (1024 * 1024)
                print(f"  - {a.name}: {size_mb:.1f} MB")

    return 0


if __name__ == "__main__":
    from typing import Optional  # noqa: F401
    sys.exit(main())
