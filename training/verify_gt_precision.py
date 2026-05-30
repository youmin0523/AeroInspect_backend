# =============================================
# verify_gt_precision.py
# CONFIRMED 등급 검출의 실제 Precision/Recall — GT 라벨 비교
#
# 배경:
#   - verify_test_mode.py는 검출 카운트만 측정 (분모 없음, "검출됐는가"만)
#   - 과검출 의심 (ext_glass 장당 11.7건) — CONFIRMED가 진짜 결함인가?
#   - roboflow test/labels GT polygon → bbox 변환 후 IoU 매칭
#
# 측정:
#   - Precision = (GT와 IoU >= 0.5인 CONFIRMED) / 전체 CONFIRMED
#   - Recall    = (CONFIRMED와 매칭된 GT) / 전체 GT
#   - 클래스 무시 (위치만). roboflow GT는 1~2 클래스라 우리 20종과 매핑 어려움
#
# 원인 분석:
#   - 카테고리별 매칭률 → 어느 카테고리가 약한지
#   - 매칭 안 된 CONFIRMED 검출의 source 분포 → 어느 모델이 과민한지
# =============================================

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

TRAIN = Path(__file__).resolve().parent
TEST_DIR = TRAIN / "test_external"
OUT_BASE = TRAIN / "runs" / "verify_gt_precision"


def load_gt_polygons(label_path: Path) -> list:
    """roboflow polygon 라벨 → bbox 리스트 (정규화 좌표)."""
    boxes = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        try:
            coords = list(map(float, parts[1:]))
        except ValueError:
            continue
        if len(coords) == 4:
            cx, cy, w, h = coords
            x1, y1 = cx - w / 2, cy - h / 2
            x2, y2 = cx + w / 2, cy + h / 2
        elif len(coords) >= 6 and len(coords) % 2 == 0:
            xs = coords[0::2]; ys = coords[1::2]
            x1, y1 = min(xs), min(ys)
            x2, y2 = max(xs), max(ys)
        else:
            continue
        boxes.append([max(0, x1), max(0, y1), min(1, x2), min(1, y2)])
    return boxes


def iou_xyxy(a: list, b: list) -> float:
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    aa = (a[2] - a[0]) * (a[3] - a[1])
    bb = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (aa + bb - inter + 1e-6)


def main():
    if not TEST_DIR.exists():
        print(f"[gt-verify] test_external 없음")
        return

    os.chdir(str(TRAIN.parent))  # backend/로 cwd (settings 상대경로 정상화)
    sys.path.insert(0, str(TRAIN.parent))

    try:
        from app.services.inference_pipeline_20 import pipeline20
        import cv2
        pipeline20.load_models()
    except Exception as e:
        print(f"[gt-verify] pipeline20 로드 실패: {e}")
        return

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = OUT_BASE / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    categories = sorted(p for p in TEST_DIR.iterdir() if p.is_dir())
    print(f"[gt-verify] {len(categories)} 카테고리 시작 → {out_dir}")

    summary = {
        "timestamp": ts,
        "iou_threshold": 0.5,
        "categories": {},
        "fp_sources": Counter(),  # false positive 검출의 source 분포
    }

    for cat_dir in categories:
        cat_name = cat_dir.name
        img_dir = cat_dir / "test" / "images"
        lbl_dir = cat_dir / "test" / "labels"
        if not img_dir.exists():
            continue

        # CPU fallback 환경 — 카테고리당 12장으로 제한 (시간 대비 sample). 7 카테고리 = 84장 ~15분.
        images = sorted([p for p in img_dir.iterdir()
                        if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")])[:12]
        print(f"  [{cat_name}] {len(images)}장", flush=True)

        cat_stats = {
            "images": len(images),
            "gt_total": 0,
            "confirmed_total": 0,
            "tp": 0,  # CONFIRMED ↔ GT 매칭 성공
            "fp": 0,  # CONFIRMED인데 GT 매칭 실패 (과검출)
            "fn": 0,  # GT인데 CONFIRMED 매칭 실패 (놓침)
            "fp_sources": Counter(),
        }

        for img_path in images:
            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if img is None:
                continue
            H, W = img.shape[:2]

            # 추론
            result = pipeline20.detect(img, tier=3)
            confirmed_dets = [d for d in result.detections if d.grade == "CONFIRMED"]
            cat_stats["confirmed_total"] += len(confirmed_dets)

            # GT 로드
            lbl_path = lbl_dir / (img_path.stem + ".txt")
            gt_norm = load_gt_polygons(lbl_path)
            gt_pixel = [[b[0]*W, b[1]*H, b[2]*W, b[3]*H] for b in gt_norm]
            cat_stats["gt_total"] += len(gt_pixel)

            # 매칭 (greedy IoU >= 0.5)
            gt_matched = [False] * len(gt_pixel)
            for det in confirmed_dets:
                if not det.bbox_xyxy:
                    cat_stats["fp"] += 1  # bbox 없으면 FP 취급
                    cat_stats["fp_sources"][det.defect_source or "unknown"] += 1
                    continue
                best_iou = 0; best_gi = -1
                for gi, gtb in enumerate(gt_pixel):
                    if gt_matched[gi]:
                        continue
                    iou = iou_xyxy(det.bbox_xyxy, gtb)
                    if iou > best_iou:
                        best_iou = iou; best_gi = gi
                if best_iou >= 0.5:
                    cat_stats["tp"] += 1
                    gt_matched[best_gi] = True
                else:
                    cat_stats["fp"] += 1
                    cat_stats["fp_sources"][det.defect_source or "unknown"] += 1

            cat_stats["fn"] += gt_matched.count(False) if gt_pixel else 0

        # 메트릭 계산
        tp = cat_stats["tp"]; fp = cat_stats["fp"]; fn = cat_stats["fn"]
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        cat_stats["precision"] = round(precision, 4)
        cat_stats["recall"] = round(recall, 4)
        cat_stats["fp_sources"] = dict(cat_stats["fp_sources"])

        verdict = "✅" if precision >= 0.9 else "⚠"
        print(f"    GT {cat_stats['gt_total']} | CONFIRMED {cat_stats['confirmed_total']} | "
              f"TP {tp} FP {fp} FN {fn} | Precision {precision:.3f} Recall {recall:.3f} {verdict}")
        if cat_stats["fp_sources"]:
            print(f"    FP source: {cat_stats['fp_sources']}")

        summary["categories"][cat_name] = cat_stats
        summary["fp_sources"].update(cat_stats["fp_sources"])

    # 전체 집계
    total_tp = sum(c["tp"] for c in summary["categories"].values())
    total_fp = sum(c["fp"] for c in summary["categories"].values())
    total_fn = sum(c["fn"] for c in summary["categories"].values())
    overall_p = total_tp / max(total_tp + total_fp, 1)
    overall_r = total_tp / max(total_tp + total_fn, 1)
    summary["totals"] = {
        "tp": total_tp, "fp": total_fp, "fn": total_fn,
        "precision": round(overall_p, 4),
        "recall": round(overall_r, 4),
    }
    summary["fp_sources"] = dict(summary["fp_sources"])

    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 60)
    print("[gt-verify] 종합")
    print("=" * 60)
    print(f"  TP {total_tp} | FP {total_fp} | FN {total_fn}")
    print(f"  Precision: {overall_p:.3f} (목표 ≥0.90)")
    print(f"  Recall:    {overall_r:.3f} (목표 ≥0.99)")
    print(f"  FP source 분포: {summary['fp_sources']}")
    print(f"\n  결과: {out_dir}/summary.json")


if __name__ == "__main__":
    main()
