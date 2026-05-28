# =============================================
# convert_m4_bbox_to_polygon.py
# M4 라벨: bbox(cx cy w h) → 4꼭짓점 polygon 변환
#
# 배경:
#   - M4 dataset 라벨 104K 중 80% (83K)가 bbox 형식
#   - 4꼭짓점 polygon은 ultralytics seg가 학습 가능 (rectangular mask)
#   - polygon 라벨은 그대로 유지 (≥3점 짝수쌍)
#
# 안전장치:
#   - 원본 라벨은 labels_bbox_backup/<split>/ 으로 백업 후 변환
#   - 이미 변환됐는지 확인 (좌표 8개 이상이면 polygon으로 간주)
#
# 사용법:
#   python convert_m4_bbox_to_polygon.py          # dry-run
#   python convert_m4_bbox_to_polygon.py --apply  # 실제 변환
# =============================================

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

TRAIN = Path(__file__).resolve().parent
ROOT = TRAIN / "datasets" / "m4_context"
SPLITS = ("train", "val", "test")


def bbox_to_polygon(cx: float, cy: float, w: float, h: float) -> list[float]:
    """bbox 중심+크기 → 4꼭짓점 (시계방향)."""
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    # 좌상→우상→우하→좌하 (clockwise)
    return [x1, y1, x2, y1, x2, y2, x1, y2]


def clip01(v: float) -> float:
    return max(0.0, min(1.0, v))


def convert_file(lbl_path: Path) -> tuple[int, int]:
    """단일 라벨 파일 변환. returns (converted_lines, kept_lines)."""
    text = lbl_path.read_text(encoding="utf-8")
    out_lines: list[str] = []
    converted = 0
    kept = 0

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        try:
            cid = int(parts[0])
            coords = list(map(float, parts[1:]))
        except (ValueError, IndexError):
            continue

        n = len(coords)
        if n == 4:
            # bbox → polygon
            cx, cy, w, h = coords
            poly = bbox_to_polygon(cx, cy, w, h)
            poly = [clip01(v) for v in poly]
            out_lines.append(f"{cid} " + " ".join(f"{v:.6f}" for v in poly))
            converted += 1
        elif n >= 6 and n % 2 == 0:
            # 이미 polygon (≥3 점)
            poly = [clip01(v) for v in coords]
            out_lines.append(f"{cid} " + " ".join(f"{v:.6f}" for v in poly))
            kept += 1
        # else: 스킵 (홀수 좌표 등)

    if out_lines:
        lbl_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    return converted, kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="실제 변환 (없으면 dry-run)")
    args = ap.parse_args()

    backup_root = ROOT / "labels_bbox_backup"

    for split in SPLITS:
        lbl_dir = ROOT / "labels" / split
        if not lbl_dir.exists():
            print(f"[{split}] 디렉토리 없음 — 스킵")
            continue

        files = list(lbl_dir.glob("*.txt"))
        print(f"[{split}] 라벨 파일 {len(files)}")

        if not args.apply:
            # dry-run: 통계만
            tot_conv = tot_kept = 0
            for f in files[:200]:  # 샘플 200
                c, k = 0, 0
                for line in f.read_text(encoding="utf-8").splitlines():
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        try:
                            ncoord = len(list(map(float, parts[1:])))
                            if ncoord == 4:
                                c += 1
                            elif ncoord >= 6 and ncoord % 2 == 0:
                                k += 1
                        except ValueError:
                            pass
                tot_conv += c
                tot_kept += k
            print(f"  [dry-run 샘플 200] bbox 변환 후보 {tot_conv} | polygon 유지 {tot_kept}")
            continue

        # 백업 (이미 있으면 스킵)
        bk_split = backup_root / split
        if bk_split.exists():
            print(f"  백업 폴더 이미 존재 — 스킵: {bk_split}")
        else:
            print(f"  원본 백업 중 → {bk_split}")
            shutil.copytree(lbl_dir, bk_split)

        # 변환
        total_c = total_k = 0
        for f in files:
            c, k = convert_file(f)
            total_c += c
            total_k += k
        print(f"  [{split}] 변환 {total_c} bbox → polygon | polygon 유지 {total_k}")

    if args.apply:
        print(f"\n[✅ 완료] 원본 백업: {backup_root}")


if __name__ == "__main__":
    main()
