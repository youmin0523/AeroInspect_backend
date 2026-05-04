# =============================================
# download_and_organize.py
# Google Drive에서 데이터 다운로드 → 자동 분류 → 학습 데이터셋 구성
#
# 기능:
#   1. gdown으로 Google Drive 폴더 다운로드
#   2. 다운로드된 폴더/파일 구조 분석
#   3. 이미지와 라벨(txt/json)을 20종 하자 카테고리에 매핑
#   4. train/val/test 자동 분할
#   5. YOLO 포맷 + ImageFolder 포맷으로 정리
#
# 사용법:
#   cd backend/training
#   python download_and_organize.py --url "https://drive.google.com/drive/folders/XXXXX"
#   python download_and_organize.py --local ./gdrive_raw   # 이미 다운로드된 경우
# =============================================

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2

try:
    import gdown
except ImportError:
    gdown = None


# ── 20종 하자 키워드 매핑 (폴더/파일명에서 하자 유형 추론) ──
KEYWORD_MAP: Dict[str, Tuple[str, str]] = {
    # (키워드): (YOLO 모델 그룹, class_name)
    # M1: 구조·방수
    "crack": ("structural", "crack"),
    "균열": ("structural", "crack"),
    "caulking": ("structural", "caulking_defect"),
    "코킹": ("structural", "caulking_defect"),
    "waterproof": ("structural", "waterproof_defect"),
    "방수": ("structural", "waterproof_defect"),
    "누수": ("structural", "waterproof_defect"),
    "leak": ("structural", "waterproof_defect"),
    "moisture": ("structural", "waterproof_defect"),

    # M2: 마감·표면
    "wallpaper": ("surface", "surface_defect_wall"),
    "도배": ("surface", "surface_defect_wall"),
    "seam": ("surface", "surface_defect_wall"),
    "이음매": ("surface", "surface_defect_wall"),
    "bubble": ("surface", "surface_defect_wall"),
    "들뜸": ("surface", "surface_defect_wall"),
    "paint": ("surface", "surface_defect_wall"),
    "도색": ("surface", "surface_defect_wall"),
    "scratch": ("surface", "surface_defect_wall"),
    "스크래치": ("surface", "surface_defect_wall"),
    "찍힘": ("surface", "surface_defect_wall"),
    "baseboard": ("surface", "baseboard_defect"),
    "걸레받이": ("surface", "baseboard_defect"),
    "오염": ("surface", "surface_defect_wall"),
    "pollution": ("surface", "surface_defect_wall"),
    "mold": ("surface", "surface_defect_wall"),
    "곰팡이": ("surface", "surface_defect_wall"),
    "stain": ("surface", "surface_defect_wall"),

    # M3: 바닥·창호
    "floor": ("floor_window", "floor_defect"),
    "바닥": ("floor_window", "floor_defect"),
    "grout": ("floor_window", "floor_defect"),
    "줄눈": ("floor_window", "floor_defect"),
    "tile": ("floor_window", "floor_defect"),
    "타일": ("floor_window", "floor_defect"),
    "glass": ("floor_window", "glass_defect"),
    "유리": ("floor_window", "glass_defect"),
    "window": ("floor_window", "glass_defect"),
    "창호": ("floor_window", "glass_defect"),
    "frame": ("floor_window", "frame_defect"),
    "문틀": ("floor_window", "frame_defect"),
    "창틀": ("floor_window", "frame_defect"),

    # M4: 열화상 (thermal 폴더명)
    "thermal": ("thermal", "thermal"),
    "열화상": ("thermal", "thermal"),
    "insulation": ("thermal", "thermal"),
    "단열": ("thermal", "thermal"),

    # M5: 기하학 (door/window frame)
    "alignment": ("frames", "wall_edge"),
    "수직": ("frames", "wall_edge"),
    "수평": ("frames", "wall_edge"),
    "직각": ("frames", "door_frame"),

    # 정상 (PatchCore 학습용)
    "normal": ("normal", "good"),
    "good": ("normal", "good"),
    "정상": ("normal", "good"),
}

# ── train/val/test 분할 비율 ──
SPLIT_RATIOS = {"train": 0.7, "val": 0.15, "test": 0.15}


def download_from_gdrive(url: str, output_dir: str) -> str:
    """Google Drive 폴더 다운로드."""
    if gdown is None:
        raise ImportError("gdown 미설치. pip install gdown")

    os.makedirs(output_dir, exist_ok=True)
    print(f"[Download] Google Drive 다운로드 시작 → {output_dir}")
    gdown.download_folder(url, output=output_dir, quiet=False)
    print(f"[Download] 완료")
    return output_dir


def analyze_folder(root_dir: str) -> Dict[str, List[dict]]:
    """
    다운로드된 폴더 구조 분석.
    Returns: {group: [{img_path, label_path(optional), class_name}]}
    """
    root = Path(root_dir)
    categorized: Dict[str, List[dict]] = {
        "structural": [], "surface": [], "floor_window": [],
        "thermal": [], "frames": [], "normal": [], "unknown": [],
    }

    all_images: List[Path] = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        all_images.extend(root.rglob(ext))

    print(f"\n[Analyze] 총 이미지 발견: {len(all_images)}")

    for img_path in all_images:
        # 폴더 경로 + 파일명에서 키워드 추론
        full_path_str = str(img_path).lower()
        group, class_name = _classify_by_keywords(full_path_str)

        # 라벨 파일 탐색 (YOLO txt 또는 COCO json)
        label_path = _find_label(img_path)

        categorized[group].append({
            "img_path": str(img_path),
            "label_path": str(label_path) if label_path else None,
            "class_name": class_name,
        })

    # 요약 출력
    print("\n[Analyze] 분류 결과:")
    for group, items in categorized.items():
        labeled = sum(1 for i in items if i["label_path"])
        print(f"  {group:15s}: {len(items):5d} 이미지 ({labeled} 라벨 보유)")

    return categorized


def _classify_by_keywords(path_str: str) -> Tuple[str, str]:
    """경로 문자열에서 키워드 매칭으로 그룹/클래스 추론."""
    for keyword, (group, class_name) in KEYWORD_MAP.items():
        if keyword in path_str:
            return group, class_name
    return "unknown", "unknown"


def _find_label(img_path: Path) -> Optional[Path]:
    """이미지에 대응하는 라벨 파일 탐색."""
    stem = img_path.stem

    # 같은 폴더에 .txt (YOLO 포맷)
    txt_path = img_path.with_suffix(".txt")
    if txt_path.exists():
        return txt_path

    # labels/ 폴더에 .txt
    labels_dir = img_path.parent.parent / "labels" / img_path.parent.name
    txt_in_labels = labels_dir / f"{stem}.txt"
    if txt_in_labels.exists():
        return txt_in_labels

    # 같은 폴더 레벨의 labels/ 폴더
    for labels_candidate in img_path.parent.parent.glob("*label*"):
        if labels_candidate.is_dir():
            for sub in labels_candidate.rglob(f"{stem}.txt"):
                return sub

    return None


def organize_dataset(
    categorized: Dict[str, List[dict]],
    output_base: str = "datasets",
):
    """
    분류된 데이터를 학습 디렉토리 구조로 정리.
    YOLO 포맷: datasets/{group}/images/{split}/, labels/{split}/
    """
    output = Path(output_base)

    for group, items in categorized.items():
        if not items or group == "unknown":
            if items and group == "unknown":
                print(f"\n[Organize] ⚠ {len(items)}개 이미지 분류 불가 — unknown/ 에 보관")
                unknown_dir = output / "unknown"
                unknown_dir.mkdir(parents=True, exist_ok=True)
                for item in items:
                    shutil.copy2(item["img_path"], unknown_dir)
            continue

        print(f"\n[Organize] {group} ({len(items)} 이미지)")

        # 셔플 후 분할
        random.seed(42)
        random.shuffle(items)

        n = len(items)
        n_train = int(n * SPLIT_RATIOS["train"])
        n_val = int(n * SPLIT_RATIOS["val"])

        splits = {
            "train": items[:n_train],
            "val": items[n_train : n_train + n_val],
            "test": items[n_train + n_val :],
        }

        for split_name, split_items in splits.items():
            img_dir = output / group / "images" / split_name
            lbl_dir = output / group / "labels" / split_name
            img_dir.mkdir(parents=True, exist_ok=True)
            lbl_dir.mkdir(parents=True, exist_ok=True)

            for item in split_items:
                # 이미지 복사
                src_img = Path(item["img_path"])
                dst_img = img_dir / src_img.name
                shutil.copy2(src_img, dst_img)

                # 라벨 복사 (있으면)
                if item["label_path"]:
                    src_lbl = Path(item["label_path"])
                    dst_lbl = lbl_dir / src_img.with_suffix(".txt").name
                    shutil.copy2(src_lbl, dst_lbl)

            print(f"  {split_name}: {len(split_items)} 이미지")

    print(f"\n[Organize] 완료! 데이터셋 경로: {output.resolve()}")


def generate_report(categorized: Dict[str, List[dict]], output_path: str = "data_report.json"):
    """데이터 분석 리포트 생성."""
    report = {}
    for group, items in categorized.items():
        sample_imgs = [i["img_path"] for i in items[:5]]
        has_labels = sum(1 for i in items if i["label_path"])
        classes = list(set(i["class_name"] for i in items))
        report[group] = {
            "total_images": len(items),
            "labeled": has_labels,
            "unlabeled": len(items) - has_labels,
            "classes": classes,
            "sample_files": sample_imgs,
        }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n[Report] 리포트 저장: {output_path}")
    return report


def main():
    parser = argparse.ArgumentParser(description="Google Drive 데이터 다운로드 + 자동 분류")
    parser.add_argument("--url", type=str, default=None,
                        help="Google Drive 폴더 공유 링크")
    parser.add_argument("--local", type=str, default=None,
                        help="이미 다운로드된 로컬 폴더 경로")
    parser.add_argument("--output", type=str, default="datasets",
                        help="정리된 데이터셋 출력 경로")
    args = parser.parse_args()

    # Step 1: 데이터 확보
    if args.local:
        raw_dir = args.local
        print(f"[Main] 로컬 폴더 사용: {raw_dir}")
    elif args.url:
        raw_dir = download_from_gdrive(args.url, "./gdrive_raw")
    else:
        print("--url 또는 --local 중 하나를 지정하세요.")
        return

    # Step 2: 폴더 분석
    categorized = analyze_folder(raw_dir)

    # Step 3: 리포트 생성
    generate_report(categorized)

    # Step 4: 데이터셋 구성
    organize_dataset(categorized, args.output)

    print("\n" + "=" * 60)
    print("다음 단계:")
    print("  1. data_report.json 확인 → unknown 이미지 수동 분류")
    print("  2. 라벨 없는 이미지 → CVAT/Roboflow로 어노테이션")
    print("  3. python train_m1_yolo_structural.py 등으로 학습 시작")
    print("=" * 60)


if __name__ == "__main__":
    main()
