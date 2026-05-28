# =============================================
# verify_test_mode.py
# 학습된 모델 통합 후 test mode 검증 — Recall·Precision 측정
#
# 동작:
#   1. test_external/ 카테고리별 이미지 추론
#   2. 등급별 검출 시각화 (빨강 CONFIRMED / 노랑 REVIEW / 점선 회색 REFERENCE)
#   3. 통계 JSON 저장 (등급별 개수, 카테고리별 검출률)
#   4. 사람이 결과 이미지 보고 놓침/오탐 판정
#
# 출력:
#   runs/verify_test_mode/<timestamp>/
#     vis/<category>/<image>_result.jpg   결과 시각화
#     summary.json                         전체 통계
#     review_required.txt                  검증 필요 케이스 리스트
#
# Recall 절대 우선 정책 ([[feedback_recall_priority_paid_service]]):
#   - 통과 조건: 카테고리별 최소 1건 이상 검출 (등급 무관)
#   - 0건 검출 카테고리는 즉시 약한 모델 보완 사이클 트리거
# =============================================

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

TRAIN = Path(__file__).resolve().parent
TEST_DIR = TRAIN / "test_external"
OUT_BASE = TRAIN / "runs" / "verify_test_mode"

GRADE_COLORS = {
    "CONFIRMED": (0, 0, 255),     # 빨강 BGR
    "REVIEW": (0, 215, 255),      # 주황/노랑 BGR
    "REFERENCE": (180, 180, 180), # 회색 BGR
}
GRADE_THICKNESS = {
    "CONFIRMED": 3,
    "REVIEW": 2,
    "REFERENCE": 1,
}


def draw_detection(img, det) -> None:
    """단일 검출 bbox + 라벨 그리기."""
    import cv2
    grade = getattr(det, "grade", "REVIEW")
    color = GRADE_COLORS.get(grade, (255, 255, 255))
    thickness = GRADE_THICKNESS.get(grade, 1)

    x1, y1, x2, y2 = [int(v) for v in det.bbox_xyxy]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

    label = f"[{grade[0]}] {getattr(det, 'class_display_ko', det.class_)} {det.conf:.2f}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(img, (x1, y1 - th - 4), (x1 + tw + 4, y1), color, -1)
    cv2.putText(img, label, (x1 + 2, y1 - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)


def process_image(img_path: Path, pipeline20, vis_dir: Path) -> dict:
    """단일 이미지 추론 + 시각화. 통계 dict 반환."""
    import cv2
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        return {"error": "load_failed", "path": str(img_path)}

    result = pipeline20.detect(img, tier=3)

    by_grade = {"CONFIRMED": 0, "REVIEW": 0, "REFERENCE": 0}
    for det in result.detections:
        by_grade[det.grade] = by_grade.get(det.grade, 0) + 1
        draw_detection(img, det)
    for ins in result.insulation:
        by_grade[ins.grade] = by_grade.get(ins.grade, 0) + 1
        draw_detection(img, ins)
    for al in result.alignment:
        by_grade[al.grade] = by_grade.get(al.grade, 0) + 1
        draw_detection(img, al)

    vis_dir.mkdir(parents=True, exist_ok=True)
    out_path = vis_dir / f"{img_path.stem}_result.jpg"
    cv2.imwrite(str(out_path), img, [cv2.IMWRITE_JPEG_QUALITY, 90])

    return {
        "path": img_path.name,
        "total": result.defect_count,
        "by_grade": by_grade,
        "by_class": {
            d.class_: d.conf for d in result.detections
        },
    }


def main():
    if not TEST_DIR.exists():
        print(f"[verify] test_external 폴더 없음: {TEST_DIR}")
        return

    sys.path.insert(0, str(TRAIN.parent))
    try:
        from app.services.inference_pipeline_20 import pipeline20
        pipeline20.load_models()
    except Exception as e:
        print(f"[verify] pipeline20 로드 실패: {e}")
        return

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = OUT_BASE / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = out_dir / "vis"

    categories = sorted(p for p in TEST_DIR.iterdir() if p.is_dir())
    print(f"[verify] {len(categories)} 카테고리 검증 시작 → {out_dir}")

    summary: dict = {
        "timestamp": ts,
        "categories": {},
        "totals": {"CONFIRMED": 0, "REVIEW": 0, "REFERENCE": 0, "no_detection_imgs": 0},
        "review_required": [],
    }

    for cat_dir in categories:
        cat_name = cat_dir.name
        cat_stats = {"images": 0, "with_detection": 0, "no_detection": 0,
                     "CONFIRMED": 0, "REVIEW": 0, "REFERENCE": 0}

        images = sorted([p for p in cat_dir.iterdir()
                         if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")])
        print(f"  [{cat_name}] {len(images)}장 처리 중...")

        for img_path in images:
            cat_vis = vis_dir / cat_name
            r = process_image(img_path, pipeline20, cat_vis)
            cat_stats["images"] += 1
            if r.get("total", 0) > 0:
                cat_stats["with_detection"] += 1
            else:
                cat_stats["no_detection"] += 1
                # 놓침 의심 (사람이 직접 봐야 함)
                summary["review_required"].append(f"{cat_name}/{r['path']}")
                summary["totals"]["no_detection_imgs"] += 1

            for g in ("CONFIRMED", "REVIEW", "REFERENCE"):
                cat_stats[g] += r.get("by_grade", {}).get(g, 0)
                summary["totals"][g] += r.get("by_grade", {}).get(g, 0)

        summary["categories"][cat_name] = cat_stats
        recall_proxy = cat_stats["with_detection"] / max(cat_stats["images"], 1)
        verdict = "✅" if recall_proxy >= 0.95 else "⚠ 보완 필요"
        print(f"    검출률(proxy): {recall_proxy:.2%} "
              f"[{cat_stats['with_detection']}/{cat_stats['images']}] {verdict} | "
              f"C={cat_stats['CONFIRMED']} R={cat_stats['REVIEW']} F={cat_stats['REFERENCE']}")

    # 결과 저장
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "review_required.txt").write_text(
        "\n".join(summary["review_required"]), encoding="utf-8")

    # 최종 verdict
    print("\n" + "=" * 60)
    print("[verify] 종합 결과")
    print("=" * 60)
    print(f"  CONFIRMED: {summary['totals']['CONFIRMED']} (보고서 등재)")
    print(f"  REVIEW:    {summary['totals']['REVIEW']} (점검자 추가확인)")
    print(f"  REFERENCE: {summary['totals']['REFERENCE']} (참고용)")
    print(f"  미검출 이미지: {summary['totals']['no_detection_imgs']}")

    overall_recall = 1 - (summary["totals"]["no_detection_imgs"] /
                          max(sum(c["images"] for c in summary["categories"].values()), 1))
    target = 0.99
    if overall_recall >= target:
        print(f"  ✅ Recall(proxy) {overall_recall:.2%} ≥ 목표 {target:.0%} — 통과")
    else:
        print(f"  ⚠ Recall(proxy) {overall_recall:.2%} < 목표 {target:.0%} — 약한 모델 보완 사이클 필요")
        print(f"  검증 필요 케이스: {out_dir}/review_required.txt")

    print(f"\n  결과: {out_dir}")
    print(f"  시각화: {vis_dir}/<category>/")


if __name__ == "__main__":
    main()
