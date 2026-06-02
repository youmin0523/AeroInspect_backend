# 데이터 보강 우선순위용: 각 모델 데이터셋의 클래스별 라벨 분포 분석
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import yaml

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

TRAIN = Path(__file__).resolve().parent

DATASETS = {
    "Thermal (Crack/Moisture/delam)": "datasets/thermal_yolo/data.yaml",
    "M4_Context (wall~door)": "datasets/m4_context/data.yaml",
    "furniture (10 cls)": "datasets/furniture_aware/data.yaml",
    "M5_FrameSeg": "configs/frame_seg.yaml",
}


def resolve_base(yp: Path, cfg: dict) -> Path:
    base = cfg.get("path")
    if base:
        b = Path(base)
        return b if b.is_absolute() else (yp.parent / b).resolve()
    return yp.parent


def count_split(base: Path, split_rel: str, names) -> tuple[int, Counter]:
    # split_rel 예: images/train → labels/train
    img_dir = (base / split_rel).resolve()
    lbl_dir = Path(str(img_dir).replace("images", "labels", 1))
    if not lbl_dir.exists():
        # 라벨이 images 옆에 있을 수도
        lbl_dir = img_dir
    counter = Counter()
    n_img = 0
    for txt in lbl_dir.rglob("*.txt"):
        n_img += 1
        for line in txt.read_text(errors="ignore").splitlines():
            p = line.split()
            if p:
                try:
                    counter[int(p[0])] += 1
                except ValueError:
                    pass
    return n_img, counter


def main():
    for name, rel in DATASETS.items():
        yp = TRAIN / rel
        print("=" * 60)
        print(f"[{name}]  ({rel})")
        if not yp.exists():
            print("  data.yaml 없음")
            continue
        cfg = yaml.safe_load(yp.read_text(encoding="utf-8"))
        names = cfg.get("names")
        if isinstance(names, dict):
            names = [names[k] for k in sorted(names)]
        base = resolve_base(yp, cfg)
        print(f"  base={base}")
        print(f"  classes(nc={cfg.get('nc')}): {names}")
        for split_key in ("train", "val"):
            rel_split = cfg.get(split_key)
            if not rel_split:
                continue
            rel_split = rel_split.lstrip("./")
            n_img, counter = count_split(base, rel_split, names)
            total = sum(counter.values())
            print(f"  [{split_key}] 라벨파일 {n_img} / 총 instance {total}")
            if names and total:
                for cid in sorted(counter):
                    cname = names[cid] if cid < len(names) else f"id{cid}"
                    pct = 100 * counter[cid] / total
                    bar = "#" * int(pct / 3)
                    print(f"      {cid:2d} {cname:22s} {counter[cid]:7d} ({pct:5.1f}%) {bar}")


if __name__ == "__main__":
    main()
