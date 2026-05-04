"""
건물 열화상(Thermal) 추가 데이터셋 다운로드 스크립트
Roboflow Universe에서 실제 건물 열화상 검사 데이터 수집
"""
import os, shutil, json
from pathlib import Path
from datetime import datetime

TRAIN_DIR = Path(__file__).parent.resolve()
DOWNLOAD_DIR = TRAIN_DIR / "gdrive_raw" / "thermal_building_additional"
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

API_KEY = "nuC9Lxr51Ds7c1IwN4Gy"

LOG_FILE = TRAIN_DIR / "download_thermal_log.txt"

def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")

# ── 다운로드 대상 데이터셋 목록 ──
DATASETS = [
    {
        "name": "ScanX Thermal Building Inspection",
        "workspace": "scanx-datasets",
        "project": "thermal-imaging-in-building-inspection-nmh6j",
        "version": 1,
        "format": "yolov8",
        "expected_images": 137,
        "classes": ["Moisture"],
        "license": "CC BY 4.0",
        "folder": "scanx_thermal_building",
    },
    {
        "name": "Buildsys2022 Thermal Leak Detection",
        "workspace": "buildsys2022",
        "project": "thermal-leak-detection-acteabot",
        "version": 1,
        "format": "yolov8",
        "expected_images": 465,
        "classes": ["thermal-leak"],
        "license": "CC BY 4.0",
        "folder": "buildsys_thermal_leak",
    },
    {
        "name": "Water Leak Thermal Detection",
        "workspace": "school-yfphj",
        "project": "water-leak",
        "version": 1,
        "format": "yolov8",
        "expected_images": 127,
        "classes": ["water-leak"],
        "license": "CC BY 4.0",
        "folder": "water_leak_thermal",
    },
    {
        "name": "Cerejo Thermal Defects",
        "workspace": "cerejo",
        "project": "thermal-defects-e1irw",
        "version": 1,
        "format": "yolov8",
        "expected_images": 200,
        "classes": ["thermal-defect"],
        "license": "CC BY 4.0",
        "folder": "cerejo_thermal_defects",
    },
    {
        "name": "Thermal Building v1 (Additional)",
        "workspace": "thermal-7r5p9",
        "project": "thermal-hev2u",
        "version": 1,
        "format": "yolov8",
        "expected_images": 100,
        "classes": ["thermal"],
        "license": "CC BY 4.0",
        "folder": "thermal_building_v1",
    },
]


def download_dataset(ds_info):
    """Roboflow API로 데이터셋 다운로드"""
    from roboflow import Roboflow

    name = ds_info["name"]
    folder = DOWNLOAD_DIR / ds_info["folder"]

    if folder.exists() and any(folder.rglob("*.jpg")):
        existing = len(list(folder.rglob("*.jpg"))) + len(list(folder.rglob("*.png")))
        log(f"  [{name}] 이미 다운로드됨 ({existing}장) — 스킵")
        return existing

    log(f"  [{name}] 다운로드 시작...")
    try:
        rf = Roboflow(api_key=API_KEY)
        ws = rf.workspace(ds_info["workspace"])
        proj = ws.project(ds_info["project"])

        # 최신 버전 시도
        version = ds_info["version"]
        try:
            ver = proj.version(version)
        except Exception:
            # 버전 1이 안되면 다른 버전 시도
            for v in range(1, 10):
                try:
                    ver = proj.version(v)
                    version = v
                    break
                except Exception:
                    continue
            else:
                log(f"  [{name}] 사용 가능한 버전 없음 — 스킵")
                return 0

        dataset = ver.download(
            ds_info["format"],
            location=str(folder),
            overwrite=True,
        )
        # 이미지 수 카운트
        img_count = 0
        for ext in ["*.jpg", "*.png", "*.jpeg", "*.bmp"]:
            img_count += len(list(folder.rglob(ext)))

        log(f"  [{name}] v{version} 다운로드 완료 — {img_count}장")
        return img_count

    except Exception as e:
        log(f"  [{name}] 에러: {e}")
        return 0


def integrate_to_thermal_yolo(download_dir, dataset_dir):
    """
    다운로드된 YOLO 포맷 데이터를 기존 thermal_yolo에 통합
    클래스 매핑: 모든 열화상 하자 → {0: Crack, 1: Moisture, 2: delamination}
    - thermal-leak, water-leak, moisture → 1 (Moisture)
    - delamination, insulation → 2 (delamination)
    - crack, hotspot → 0 (Crack)
    """
    thermal_yolo = dataset_dir / "thermal_yolo"
    added_total = 0

    CLASS_MAP_RULES = {
        # source class name → target class id
        "moisture": 1,
        "Moisture": 1,
        "thermal-leak": 1,
        "water-leak": 1,
        "leak": 1,
        "air_infiltration": 1,
        "air_leakage": 1,
        "insulation": 2,
        "delamination": 2,
        "hollow": 2,
        "crack": 0,
        "Crack": 0,
        "hotspot": 0,
        "Hotspot": 0,
        "thermal-defect": 1,
        "thermal": 1,
    }

    for ds_folder in download_dir.iterdir():
        if not ds_folder.is_dir():
            continue

        log(f"  통합 처리: {ds_folder.name}")

        # data.yaml에서 클래스 정보 읽기
        yaml_path = ds_folder / "data.yaml"
        src_classes = {}
        if yaml_path.exists():
            import yaml
            with open(yaml_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            names = cfg.get("names", [])
            if isinstance(names, dict):
                src_classes = {int(k): v for k, v in names.items()}
            elif isinstance(names, list):
                src_classes = {i: n for i, n in enumerate(names)}
            log(f"    원본 클래스: {src_classes}")

        # 각 split (train/valid/test) 처리
        for split_name in ["train", "valid", "val", "test"]:
            img_dir = ds_folder / split_name / "images"
            lbl_dir = ds_folder / split_name / "labels"

            if not img_dir.exists():
                img_dir = ds_folder / "images" / split_name
                lbl_dir = ds_folder / "labels" / split_name

            if not img_dir.exists():
                continue

            # 대상 split 결정
            target_split = "train" if split_name in ["train"] else "val"
            if split_name == "test":
                target_split = "val"  # test도 val에 합침

            dst_img = thermal_yolo / "images" / target_split
            dst_lbl = thermal_yolo / "labels" / target_split
            dst_img.mkdir(parents=True, exist_ok=True)
            dst_lbl.mkdir(parents=True, exist_ok=True)

            count = 0
            for img_path in img_dir.iterdir():
                if img_path.suffix.lower() not in [".jpg", ".png", ".jpeg", ".bmp"]:
                    continue

                # 라벨 파일 찾기
                lbl_path = lbl_dir / (img_path.stem + ".txt")
                if not lbl_path.exists():
                    continue

                # 라벨 변환
                new_lines = []
                with open(lbl_path, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) < 5:
                            continue
                        src_cls_id = int(parts[0])
                        src_cls_name = src_classes.get(src_cls_id, "")

                        # 클래스 매핑
                        target_cls = None
                        if src_cls_name in CLASS_MAP_RULES:
                            target_cls = CLASS_MAP_RULES[src_cls_name]
                        else:
                            # 이름 기반 매칭
                            name_lower = src_cls_name.lower()
                            for key, val in CLASS_MAP_RULES.items():
                                if key.lower() in name_lower or name_lower in key.lower():
                                    target_cls = val
                                    break

                        if target_cls is None:
                            target_cls = 1  # 기본: Moisture

                        # 전체 이미지 bbox 필터링 (0.5 0.5 1.0 1.0)
                        x, y, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                        if w > 0.95 and h > 0.95:
                            continue  # 전체 이미지 bbox 제외

                        new_lines.append(f"{target_cls} {' '.join(parts[1:])}")

                if not new_lines:
                    continue

                # 파일명 중복 방지
                prefix = ds_folder.name[:8]
                new_name = f"thm_{prefix}_{img_path.name}"

                shutil.copy2(img_path, dst_img / new_name)
                with open(dst_lbl / (Path(new_name).stem + ".txt"), "w") as f:
                    f.write("\n".join(new_lines) + "\n")

                count += 1

            if count > 0:
                log(f"    {split_name} → {target_split}: {count}장 통합")
                added_total += count

    return added_total


if __name__ == "__main__":
    log("=" * 60)
    log("건물 열화상 추가 데이터 수집 시작")
    log("=" * 60)

    total_downloaded = 0
    for ds in DATASETS:
        count = download_dataset(ds)
        total_downloaded += count

    log(f"\n총 다운로드: {total_downloaded}장")

    # 통합
    log("\n" + "=" * 60)
    log("thermal_yolo 데이터셋 통합 시작")
    log("=" * 60)

    datasets_dir = TRAIN_DIR / "datasets"
    added = integrate_to_thermal_yolo(DOWNLOAD_DIR, datasets_dir)

    # 최종 카운트
    final_count = 0
    thermal_img = datasets_dir / "thermal_yolo" / "images"
    for split in ["train", "val", "test"]:
        d = thermal_img / split
        if d.exists():
            c = len(list(d.glob("*")))
            log(f"  {split}: {c}장")
            final_count += c

    log(f"\n최종 thermal_yolo: {final_count}장 (추가: {added}장)")
    log("=" * 60)
    log("완료")
