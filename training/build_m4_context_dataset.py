# M4 Context 데이터셋 구축 (옵션 A: 우리 데이터 통합)
# - frames (M5v2)에서 wall/ceiling/door/window 추출
# - floor_window (M3)에서 floor 추출
# 결과 클래스: 0=wall, 1=ceiling, 2=floor, 3=window, 4=door
import sys
import shutil
import random
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
random.seed(42)

DST = Path("datasets/m4_context")
FRAMES = Path("datasets/frames")           # wall_edge=0, ceiling_edge=1, door_frame=2, window_frame=3
FLOOR_WINDOW = Path("datasets/floor_window")  # floor_defect=0, glass_defect=1, frame_defect=2

# frames 클래스 매핑: original_id → new_id (None = 제외)
FRAMES_MAP = {
    0: 0,  # wall_edge → wall
    1: 1,  # ceiling_edge → ceiling
    2: 4,  # door_frame → door
    3: 3,  # window_frame → window
}

# floor_window 클래스 매핑: original_id → new_id (None = 제외)
FW_MAP = {
    0: 2,    # floor_defect → floor
    1: 3,    # glass_defect → window (유리도 창의 일부)
    2: None, # frame_defect → 제외 (이미 frames에 frame 종류 다 있음)
}

NEW_CLASS_NAMES = ["wall", "ceiling", "floor", "window", "door"]


def remap_label_file(src_path: Path, dst_path: Path, mapping: dict) -> int:
    """label 파일 클래스 매핑하면서 복사. 매핑된 줄 수 반환."""
    if not src_path.exists():
        return 0
    out_lines = []
    with open(src_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            try:
                cls = int(parts[0])
            except ValueError:
                continue
            new_cls = mapping.get(cls)
            if new_cls is None:
                continue
            out_lines.append(f"{new_cls} " + " ".join(parts[1:]) + "\n")
    if out_lines:
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dst_path, "w", encoding="utf-8") as f:
            f.writelines(out_lines)
        return len(out_lines)
    return 0


def copy_split(src_root: Path, split: str, mapping: dict, prefix: str, stats: dict) -> int:
    """src_root/images/{split}와 src_root/labels/{split}을 통합 데이터셋에 복사."""
    src_img_dir = src_root / "images" / split
    src_lbl_dir = src_root / "labels" / split
    if not src_img_dir.exists():
        return 0
    dst_img_dir = DST / "images" / split
    dst_lbl_dir = DST / "labels" / split
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for img in src_img_dir.iterdir():
        if img.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        # 충돌 방지를 위해 prefix 붙이기
        new_stem = f"{prefix}_{img.stem}"
        new_img = dst_img_dir / (new_stem + img.suffix)
        new_lbl = dst_lbl_dir / (new_stem + ".txt")

        # 라벨 매핑 결과가 있을 때만 이미지 복사
        src_lbl = src_lbl_dir / (img.stem + ".txt")
        n_objs = remap_label_file(src_lbl, new_lbl, mapping)
        if n_objs > 0:
            shutil.copy2(img, new_img)
            count += 1
            stats["objects"] = stats.get("objects", 0) + n_objs

    return count


def main():
    if DST.exists():
        print(f"기존 {DST} 삭제 후 재생성")
        shutil.rmtree(DST)

    print("=== M4 Context 데이터셋 통합 시작 ===\n")

    # frames 데이터셋 처리
    if FRAMES.exists():
        print("[frames] 처리 중...")
        for split in ["train", "val", "test"]:
            stats = {"objects": 0}
            n = copy_split(FRAMES, split, FRAMES_MAP, "fr", stats)
            print(f"  {split}: {n} images, {stats['objects']} objects")
    else:
        print(f"[frames] SKIP — {FRAMES} 없음")

    # floor_window 데이터셋 처리
    if FLOOR_WINDOW.exists():
        print("\n[floor_window] 처리 중...")
        for split in ["train", "val", "test"]:
            stats = {"objects": 0}
            n = copy_split(FLOOR_WINDOW, split, FW_MAP, "fw", stats)
            print(f"  {split}: {n} images, {stats['objects']} objects")
    else:
        print(f"[floor_window] SKIP — {FLOOR_WINDOW} 없음")

    # data.yaml 생성
    yaml_text = f"""# M4 Context 통합 데이터셋
# 출처: frames (M5v2) + floor_window (M3)
path: {DST.resolve()}
train: images/train
val: images/val
test: images/test

nc: {len(NEW_CLASS_NAMES)}
names:
"""
    for i, n in enumerate(NEW_CLASS_NAMES):
        yaml_text += f"  {i}: {n}\n"
    (DST / "data.yaml").write_text(yaml_text, encoding="utf-8")

    # 최종 카운트
    print("\n=== 통합 데이터셋 완료 ===")
    print(f"위치: {DST.resolve()}")
    for split in ["train", "val", "test"]:
        n = len(list((DST / "images" / split).glob("*"))) if (DST / "images" / split).exists() else 0
        print(f"  {split}: {n} images")
    size_mb = sum(f.stat().st_size for f in DST.rglob("*") if f.is_file()) / 1024 / 1024
    print(f"  size: {size_mb:.1f} MB")
    print(f"\n클래스: {NEW_CLASS_NAMES}")


if __name__ == "__main__":
    main()
