# =============================================
# coco_furniture_supplement.py
# COCO에서 kitchen_appliance/countertop_sink 보강 → furniture_aware
# refrigerator/oven/microwave/toaster → 6(kitchen_appliance), sink → 7(countertop_sink)
# 추가 파일은 *_coco* 접미사 → furniture 학습 완료 후 삭제 가능
# 무인 실행: 네트워크 에러 재시도, 부분 성공 허용
# =============================================
from __future__ import annotations

import sys
import time
import urllib.request
import zipfile
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

TRAIN = Path(__file__).resolve().parent
WORK = TRAIN / "datasets" / "_coco_tmp"          # 다운로드 임시 (학습 후 삭제)
FURN = TRAIN / "datasets" / "furniture_aware"
ANN_ZIP_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

# COCO category name → furniture class id
COCO_MAP = {
    "refrigerator": 6, "oven": 6, "microwave": 6, "toaster": 6,  # kitchen_appliance
    "sink": 7,                                                    # countertop_sink
}
MAX_PER_CAT = 700   # 카테고리당 이미지 제한 (디스크 관리)


def log(msg):
    print(f"[coco] {time.strftime('%H:%M:%S')} {msg}", flush=True)


def download(url, dst, retries=3):
    for i in range(retries):
        try:
            urllib.request.urlretrieve(url, dst)
            return True
        except Exception as e:
            log(f"다운로드 실패({i+1}/{retries}) {url}: {e}")
            time.sleep(5)
    return False


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    ann_dir = WORK / "annotations"

    # 1) annotation 다운로드 (없으면)
    if not (ann_dir / "instances_train2017.json").exists():
        zip_path = WORK / "ann.zip"
        if not zip_path.exists():
            log("annotation zip 다운로드 중 (~250MB)...")
            if not download(ANN_ZIP_URL, zip_path):
                log("annotation 다운로드 최종 실패 — 중단")
                return
        log("annotation 압축 해제...")
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(WORK)

    from pycocotools.coco import COCO
    coco = COCO(str(ann_dir / "instances_train2017.json"))

    # 2) 카테고리 id 매핑
    cat_ids = {}
    for name in COCO_MAP:
        ids = coco.getCatIds(catNms=[name])
        if ids:
            cat_ids[ids[0]] = (name, COCO_MAP[name])

    img_out = FURN / "images" / "train"
    lbl_out = FURN / "labels" / "train"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    # 3) 카테고리별 이미지 수집 (중복 이미지 통합)
    img_to_anns = {}  # img_id → list[(furn_cls, bbox)]
    for cat_id, (name, furn_cls) in cat_ids.items():
        ann_ids = coco.getAnnIds(catIds=[cat_id])
        anns = coco.loadAnns(ann_ids)
        per_cat_imgs = set()
        for a in anns:
            if len(per_cat_imgs) >= MAX_PER_CAT and a["image_id"] not in img_to_anns:
                continue
            per_cat_imgs.add(a["image_id"])
            img_to_anns.setdefault(a["image_id"], []).append((furn_cls, a["bbox"]))
        log(f"{name}: {len(per_cat_imgs)} 이미지")

    log(f"총 {len(img_to_anns)} 이미지 다운로드 + 변환 시작")
    done = 0
    for img_id, items in img_to_anns.items():
        info = coco.loadImgs([img_id])[0]
        W, H = info["width"], info["height"]
        fname = f"coco_{info['file_name']}"
        img_path = img_out / fname
        if not img_path.exists():
            if not download(info["coco_url"], img_path, retries=2):
                continue
        # YOLO 라벨 (해당 클래스 bbox만)
        lines = []
        for furn_cls, bbox in items:
            x, y, w, h = bbox
            cx, cy = (x + w / 2) / W, (y + h / 2) / H
            nw, nh = w / W, h / H
            if nw <= 0 or nh <= 0:
                continue
            lines.append(f"{furn_cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
        (lbl_out / f"coco_{Path(info['file_name']).stem}.txt").write_text(
            "\n".join(lines) + "\n", encoding="utf-8")
        done += 1
        if done % 200 == 0:
            log(f"진행 {done}/{len(img_to_anns)}")

    log(f"완료: {done} 이미지 보강 (furniture_aware/train에 coco_* 추가)")
    log("학습 완료 후 'coco_*' 파일 삭제로 원복 가능")


if __name__ == "__main__":
    main()
