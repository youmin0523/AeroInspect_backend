# =============================================
# validate_m4_seg_labels.py
# M4 seg 학습 실패 원인 파악 — 라벨 무결성 검사
#
# 검사 항목:
#   1. class_id 범위 (0 <= cid < nc)
#   2. 좌표 개수 (polygon은 짝수쌍 ≥ 3쌍 = 6개)
#   3. 좌표 정규화 범위 (0 <= x,y <= 1)
#   4. polygon 점 수 (ultralytics seg는 일반적으로 ≥3 점)
#   5. 이미지-라벨 매칭 (image filename → label exists)
# =============================================

from __future__ import annotations

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

TRAIN = Path(__file__).resolve().parent
ROOT = TRAIN / "datasets" / "m4_context"
NC = 5  # data.yaml: 0=wall 1=ceiling 2=floor 3=window 4=door


def validate(split: str) -> dict:
    img_dir = ROOT / "images" / split
    lbl_dir = ROOT / "labels" / split

    stats = {
        "total_labels": 0,
        "total_polygons": 0,
        "bad_cls": 0,
        "bad_coord_count": 0,
        "out_of_range": 0,
        "too_few_points": 0,
        "missing_image": 0,
        "samples_bad": [],
    }

    if not lbl_dir.exists():
        print(f"  {split}: 라벨 디렉토리 없음 — {lbl_dir}")
        return stats

    img_stems = set()
    if img_dir.exists():
        for p in img_dir.iterdir():
            if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"):
                img_stems.add(p.stem)

    for lbl_file in lbl_dir.glob("*.txt"):
        stats["total_labels"] += 1
        text = lbl_file.read_text(encoding="utf-8")

        if lbl_file.stem not in img_stems:
            stats["missing_image"] += 1
            if len(stats["samples_bad"]) < 5:
                stats["samples_bad"].append(f"missing_image: {lbl_file.name}")
            continue

        for ln, line in enumerate(text.splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            stats["total_polygons"] += 1

            parts = line.split()
            try:
                cid = int(parts[0])
                coords = list(map(float, parts[1:]))
            except (ValueError, IndexError):
                stats["bad_coord_count"] += 1
                if len(stats["samples_bad"]) < 10:
                    stats["samples_bad"].append(f"parse: {lbl_file.name}:{ln}")
                continue

            if not (0 <= cid < NC):
                stats["bad_cls"] += 1
                if len(stats["samples_bad"]) < 10:
                    stats["samples_bad"].append(f"bad_cls={cid}: {lbl_file.name}:{ln}")
                continue

            if len(coords) % 2 != 0:
                stats["bad_coord_count"] += 1
                if len(stats["samples_bad"]) < 10:
                    stats["samples_bad"].append(f"odd_coords({len(coords)}): {lbl_file.name}:{ln}")
                continue

            n_points = len(coords) // 2
            if n_points < 3:
                stats["too_few_points"] += 1
                if len(stats["samples_bad"]) < 10:
                    stats["samples_bad"].append(f"only_{n_points}_pts: {lbl_file.name}:{ln}")
                continue

            for v in coords:
                if not (0.0 <= v <= 1.0):
                    stats["out_of_range"] += 1
                    if len(stats["samples_bad"]) < 10:
                        stats["samples_bad"].append(f"oor({v:.3f}): {lbl_file.name}:{ln}")
                    break

    return stats


def main():
    print(f"[validate_m4_seg] {ROOT}")
    print(f"[validate_m4_seg] nc={NC}")

    overall_bad = 0
    for split in ("train", "val", "test"):
        print(f"\n=== {split} ===")
        s = validate(split)
        bad = s["bad_cls"] + s["bad_coord_count"] + s["out_of_range"] + s["too_few_points"] + s["missing_image"]
        overall_bad += bad

        print(f"  라벨 파일: {s['total_labels']}")
        print(f"  polygon 총: {s['total_polygons']}")
        print(f"  잘못된 cls (out of [0,{NC-1}]): {s['bad_cls']}")
        print(f"  좌표 개수 이상 (홀수/파싱실패): {s['bad_coord_count']}")
        print(f"  좌표 범위 초과 [0,1]: {s['out_of_range']}")
        print(f"  point 수 < 3 (seg 학습 불가): {s['too_few_points']}")
        print(f"  이미지 없음: {s['missing_image']}")
        if s["samples_bad"]:
            print(f"  샘플 (최대 10):")
            for sm in s["samples_bad"]:
                print(f"    - {sm}")

    print(f"\n[validate_m4_seg] 총 문제 라벨: {overall_bad}")
    if overall_bad == 0:
        print("  ✅ 라벨 무결성 양호 — 다른 원인 의심 (모델 OOM/CUDA 등)")
    else:
        print("  ⚠ 문제 라벨 존재 — seg 학습 실패 원인 가능")


if __name__ == "__main__":
    main()
