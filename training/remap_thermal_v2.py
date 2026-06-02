# thermal_yolo → thermal_yolo_v2 재매핑
# Crack(class 0) 제거, Moisture(1→0), delamination(2→1)
# 이미지는 junction으로 공유(디스크 0), 라벨만 재생성
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

TRAIN = Path(__file__).resolve().parent
SRC = TRAIN / "datasets" / "thermal_yolo"
DST = TRAIN / "datasets" / "thermal_yolo_v2"
REMAP = {1: 0, 2: 1}  # Crack(0) drop

for split in ("train", "val", "test"):
    src_lbl = SRC / "labels" / split
    if not src_lbl.exists():
        continue
    dst_lbl = DST / "labels" / split
    dst_lbl.mkdir(parents=True, exist_ok=True)

    kept = removed = files = polyfix = 0
    for lbl in src_lbl.glob("*.txt"):
        files += 1
        out = []
        for ln in lbl.read_text(errors="ignore").splitlines():
            p = ln.split()
            if not p:
                continue
            cid = int(p[0])
            if cid == 0:        # Crack 제거
                removed += 1
                continue
            coords = [float(x) for x in p[1:]]
            if len(coords) == 4:
                cx, cy, w, h = coords
            elif len(coords) >= 6 and len(coords) % 2 == 0:
                # polygon → bbox (detection 학습용)
                xs, ys = coords[0::2], coords[1::2]
                x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
                cx, cy, w, h = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
                polyfix += 1
            else:
                continue  # 형식 이상 라벨 스킵
            # 정규화 범위 클램프
            cx, cy = min(max(cx, 0.0), 1.0), min(max(cy, 0.0), 1.0)
            w, h = min(max(w, 0.0), 1.0), min(max(h, 0.0), 1.0)
            if w <= 0 or h <= 0:
                continue
            out.append(f"{REMAP[cid]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            kept += 1
        (dst_lbl / lbl.name).write_text("\n".join(out) + ("\n" if out else ""), encoding="utf-8")

    # images junction (디스크 0)
    src_img = (SRC / "images" / split).resolve()
    dst_img = DST / "images" / split
    if not dst_img.exists():
        dst_img.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["cmd", "/c", "mklink", "/J", str(dst_img), str(src_img)], check=True)

    print(f"[{split}] 파일 {files} / 유지 instance {kept} / Crack 제거 {removed} / polygon→bbox {polyfix}")

(DST / "data.yaml").write_text(
    "train: ./images/train\nval: ./images/val\ntest: ./images/test\n\nnc: 2\nnames: ['Moisture', 'delamination']\n",
    encoding="utf-8",
)
print(f"완료: {DST}")
