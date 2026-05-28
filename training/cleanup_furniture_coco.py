# =============================================
# cleanup_furniture_coco.py
# furniture_aware 학습 후 coco_* 보강 파일 제거 (디스크 회수)
#
# coco_furniture_supplement.py 로 추가된 파일은 학습 완료 후 삭제 가능.
# 약 7000+ 파일, 수 GB 회수 예상.
#
# 안전장치:
#   - dry-run 기본 (실제 삭제는 --apply 플래그 필요)
#   - 삭제 대상은 'coco_' 접두사 파일만
#   - images/train + labels/train 만 (val/test는 건드리지 않음)
#
# 사용법:
#   python cleanup_furniture_coco.py          # dry-run (목록만)
#   python cleanup_furniture_coco.py --apply  # 실제 삭제
# =============================================

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

TRAIN = Path(__file__).resolve().parent
FURN = TRAIN / "datasets" / "furniture_aware"
IMG_DIR = FURN / "images" / "train"
LBL_DIR = FURN / "labels" / "train"


def gather(dir_path: Path) -> list[Path]:
    if not dir_path.exists():
        return []
    return [p for p in dir_path.iterdir() if p.is_file() and p.name.startswith("coco_")]


def humanbytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="실제 삭제 (없으면 dry-run)")
    args = ap.parse_args()

    imgs = gather(IMG_DIR)
    lbls = gather(LBL_DIR)
    total_size = sum(p.stat().st_size for p in imgs + lbls)

    print(f"[cleanup_furniture_coco] 대상:")
    print(f"  이미지: {len(imgs)} (in {IMG_DIR})")
    print(f"  라벨:   {len(lbls)} (in {LBL_DIR})")
    print(f"  총 용량: {humanbytes(total_size)}")

    if not args.apply:
        print("\n[dry-run] --apply 플래그 없음 — 실제 삭제 X")
        print("샘플 5건:")
        for p in (imgs + lbls)[:5]:
            print(f"  {p.name}")
        return

    deleted = 0
    failed = 0
    for p in imgs + lbls:
        try:
            p.unlink()
            deleted += 1
        except Exception as e:
            print(f"  [실패] {p.name}: {e}")
            failed += 1
    print(f"\n[cleanup] 완료: 삭제 {deleted}건, 실패 {failed}건, 회수 {humanbytes(total_size)}")


if __name__ == "__main__":
    main()
