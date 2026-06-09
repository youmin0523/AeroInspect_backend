# =============================================
# training/eval/compare_vlm_vs_onnx.py
# 역할: 기존 ONNX(3-모델) vs 비전 LLM(VLM) 검출 병행 비교 평가
#       - test_external GT(roboflow YOLO 라벨)로 image-level Precision/Recall 산출
#       - grounding 모드면 IoU 매칭 localization P/R 추가 (verify_gt_precision 재사용)
#       - 두 방식 agreement, 평균 latency, 추정 비용/이미지 리포트
#
# 배경:
#   학습 모델 검출률 저하(M4 mAP 0.503, GT Precision 0.535) → VLM이 더 나은지 정량 확인.
#   이 수치로 "VLM 단독 전환 / 하이브리드 / ONNX 유지" 최종 의사결정.
#
# 사용:
#   # 환경변수: GOOGLE_API_KEY=... (VLM 호출), VLM_DETECTION_ENABLED=true
#   python training/eval/compare_vlm_vs_onnx.py --provider gemini --mode classify --limit 8
#
# 주의: VLM 호출은 실비 발생. --limit 으로 카테고리당 장수 제한 (비용 가드).
# =============================================

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

EVAL_DIR = Path(__file__).resolve().parent
TRAIN = EVAL_DIR.parent              # training/
BACKEND = TRAIN.parent               # backend/
TEST_DIR = TRAIN / "test_external"
OUT_BASE = EVAL_DIR / "runs" / "compare_vlm_vs_onnx"

# verify_gt_precision 의 GT 로더/IoU 재사용
sys.path.insert(0, str(TRAIN))
from verify_gt_precision import load_gt_polygons, iou_xyxy  # noqa: E402

# ── 추정 비용표 ($/이미지, 입력 이미지 1장+짧은 프롬프트 기준 대략치) ──
# 정확한 비용은 실제 청구서로 확인. 여기선 방식 간 상대 비교용 근사치.
PRICE_PER_IMAGE_USD = {
    "gemini-2.5-flash": 0.0004,
    "gemini-2.0-flash": 0.0003,
    "gemini-1.5-pro": 0.0020,
    "claude-opus-4-5": 0.0120,
    "claude-sonnet-4-6": 0.0040,
    "gpt-4o": 0.0050,
    "gpt-4o-mini": 0.0006,
}


def _f1(p: float, r: float) -> float:
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


async def main(provider: str, mode: str, limit: int, only_cats: list[str] | None) -> None:
    if not TEST_DIR.exists():
        print(f"[compare] test_external 없음: {TEST_DIR}")
        return

    os.chdir(str(BACKEND))
    sys.path.insert(0, str(BACKEND))

    import cv2
    from app.services.vlm_detector import detect_vlm_async, VLMQuotaExceeded, vlm_detector

    # ── ONNX 3-모델 파이프라인 로드 (실패 시 VLM 단독 모드) ──
    onnx_ok = False
    try:
        from app.services.inference_pipeline import pipeline
        pipeline.load_models()
        onnx_ok = pipeline.is_loaded
    except Exception as e:
        print(f"[compare] ONNX 파이프라인 로드 실패 — VLM 단독 평가로 진행: {e}")

    model = vlm_detector.stats()["model"]
    price = PRICE_PER_IMAGE_USD.get(model, 0.0)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = OUT_BASE / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    categories = sorted(p for p in TEST_DIR.iterdir() if p.is_dir())
    if only_cats:
        categories = [c for c in categories if c.name in only_cats]
    print(f"[compare] provider={provider} mode={mode} model={model} "
          f"onnx={'ON' if onnx_ok else 'OFF'} | {len(categories)} 카테고리 → {out_dir}")

    summary = {
        "timestamp": ts, "provider": provider, "mode": mode, "model": model,
        "onnx_available": onnx_ok, "limit_per_cat": limit,
        "categories": {},
    }
    # 전역 image-level 혼동행렬 (vs GT has_defect)
    G = {
        "onnx": {"tp": 0, "fp": 0, "fn": 0, "tn": 0},
        "vlm": {"tp": 0, "fp": 0, "fn": 0, "tn": 0},
        "agree": 0, "total_imgs": 0,
        "vlm_calls": 0, "vlm_latency_sum": 0.0,
        "loc_tp": 0, "loc_fp": 0, "loc_fn": 0,  # grounding IoU 매칭
    }

    for cat_dir in categories:
        img_dir = cat_dir / "test" / "images"
        lbl_dir = cat_dir / "test" / "labels"
        if not img_dir.exists():
            continue
        images = sorted([p for p in img_dir.iterdir()
                         if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")])[:limit]
        if not images:
            continue
        print(f"  [{cat_dir.name}] {len(images)}장", flush=True)

        cs = {"images": len(images),
              "onnx": {"tp": 0, "fp": 0, "fn": 0, "tn": 0},
              "vlm": {"tp": 0, "fp": 0, "fn": 0, "tn": 0},
              "agree": 0}

        for img_path in images:
            raw = img_path.read_bytes()
            img = cv2.imdecode(
                __import__("numpy").frombuffer(raw, dtype="uint8"), cv2.IMREAD_COLOR)
            if img is None:
                continue
            H, W = img.shape[:2]

            gt_norm = load_gt_polygons(lbl_dir / (img_path.stem + ".txt"))
            gt_has = len(gt_norm) > 0
            gt_pixel = [[b[0] * W, b[1] * H, b[2] * W, b[3] * H] for b in gt_norm]

            # ── ONNX ──
            onnx_has = False
            if onnx_ok:
                try:
                    r = pipeline.detect(img, None, False)
                    onnx_has = bool(r.has_defect)
                except Exception as e:
                    print(f"    ONNX 오류 {img_path.name}: {e}")

            # ── VLM ──
            vlm_has = False
            vlm_boxes = []
            try:
                vr = await detect_vlm_async(raw, mode=mode, provider=provider)
                vlm_has = bool(vr.has_defect)
                G["vlm_calls"] += 1 if not vr.cached else 0
                G["vlm_latency_sum"] += vr.latency_ms
                if mode == "grounding":
                    vlm_boxes = [d.bbox_xyxy for d in vr.detections
                                 if d.localization == "bbox" and d.bbox_xyxy]
            except VLMQuotaExceeded as e:
                print(f"    [중단] {e}")
                break
            except Exception as e:
                print(f"    VLM 오류 {img_path.name}: {e}")

            # image-level 혼동행렬 누적
            for key, has in (("onnx", onnx_has), ("vlm", vlm_has)):
                if has and gt_has:
                    cs[key]["tp"] += 1
                elif has and not gt_has:
                    cs[key]["fp"] += 1
                elif not has and gt_has:
                    cs[key]["fn"] += 1
                else:
                    cs[key]["tn"] += 1
            if onnx_has == vlm_has:
                cs["agree"] += 1

            # grounding localization IoU 매칭 (VLM 박스 vs GT)
            if mode == "grounding" and gt_pixel:
                matched = [False] * len(gt_pixel)
                for box in vlm_boxes:
                    best, gi_best = 0.0, -1
                    for gi, gtb in enumerate(gt_pixel):
                        if matched[gi]:
                            continue
                        iou = iou_xyxy(box, gtb)
                        if iou > best:
                            best, gi_best = iou, gi
                    if best >= 0.5:
                        G["loc_tp"] += 1
                        matched[gi_best] = True
                    else:
                        G["loc_fp"] += 1
                G["loc_fn"] += matched.count(False)

        # 카테고리 메트릭
        for key in ("onnx", "vlm"):
            d = cs[key]
            p = d["tp"] / max(d["tp"] + d["fp"], 1)
            r = d["tp"] / max(d["tp"] + d["fn"], 1)
            d["precision"] = round(p, 3)
            d["recall"] = round(r, 3)
            d["f1"] = round(_f1(p, r), 3)
            for m in ("tp", "fp", "fn", "tn"):
                G[key][m] += d[m]
        G["agree"] += cs["agree"]
        G["total_imgs"] += cs["images"]
        summary["categories"][cat_dir.name] = cs
        print(f"    ONNX P{cs['onnx']['precision']} R{cs['onnx']['recall']} F1 {cs['onnx']['f1']} | "
              f"VLM P{cs['vlm']['precision']} R{cs['vlm']['recall']} F1 {cs['vlm']['f1']} | "
              f"agree {cs['agree']}/{cs['images']}")

    # ── 전역 집계 ──
    def metrics(d):
        p = d["tp"] / max(d["tp"] + d["fp"], 1)
        r = d["tp"] / max(d["tp"] + d["fn"], 1)
        return {"tp": d["tp"], "fp": d["fp"], "fn": d["fn"], "tn": d["tn"],
                "precision": round(p, 3), "recall": round(r, 3), "f1": round(_f1(p, r), 3)}

    totals = {"onnx_image_level": metrics(G["onnx"]), "vlm_image_level": metrics(G["vlm"]),
              "agreement": round(G["agree"] / max(G["total_imgs"], 1), 3),
              "total_images": G["total_imgs"],
              "vlm_avg_latency_ms": round(G["vlm_latency_sum"] / max(G["vlm_calls"], 1), 1),
              "vlm_api_calls": G["vlm_calls"],
              "vlm_est_cost_usd": round(G["vlm_calls"] * price, 4),
              "vlm_price_per_image_usd": price}
    if mode == "grounding":
        lp = G["loc_tp"] / max(G["loc_tp"] + G["loc_fp"], 1)
        lr = G["loc_tp"] / max(G["loc_tp"] + G["loc_fn"], 1)
        totals["vlm_localization"] = {
            "tp": G["loc_tp"], "fp": G["loc_fp"], "fn": G["loc_fn"],
            "precision": round(lp, 3), "recall": round(lr, 3), "f1": round(_f1(lp, lr), 3)}
    summary["totals"] = totals

    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 64)
    print("[compare] 종합 (image-level: 이미지에 하자 존재 여부 vs GT)")
    print("=" * 64)
    o, v = totals["onnx_image_level"], totals["vlm_image_level"]
    print(f"  ONNX : P {o['precision']} | R {o['recall']} | F1 {o['f1']}  (TP{o['tp']} FP{o['fp']} FN{o['fn']} TN{o['tn']})")
    print(f"  VLM  : P {v['precision']} | R {v['recall']} | F1 {v['f1']}  (TP{v['tp']} FP{v['fp']} FN{v['fn']} TN{v['tn']})")
    if mode == "grounding":
        L = totals["vlm_localization"]
        print(f"  VLM 위치(IoU≥0.5): P {L['precision']} | R {L['recall']} | F1 {L['f1']}")
    print(f"  두 방식 일치율: {totals['agreement']}  | 총 {totals['total_images']}장")
    print(f"  VLM 평균 지연: {totals['vlm_avg_latency_ms']}ms | API 호출 {totals['vlm_api_calls']}회 "
          f"| 추정 비용 ${totals['vlm_est_cost_usd']} (${price}/장, 근사치)")
    print(f"\n  결과: {out_dir / 'summary.json'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="ONNX vs VLM 검출 병행 비교 평가")
    ap.add_argument("--provider", default=None, help="gemini|claude|openai (기본: settings)")
    ap.add_argument("--mode", default=None, help="classify|grounding (기본: settings)")
    ap.add_argument("--limit", type=int, default=8, help="카테고리당 최대 장수 (비용 가드)")
    ap.add_argument("--categories", default=None, help="쉼표구분 카테고리 필터")
    args = ap.parse_args()

    from app.config import settings
    prov = args.provider or settings.VLM_PROVIDER
    md = args.mode or settings.VLM_MODE
    cats = args.categories.split(",") if args.categories else None
    asyncio.run(main(prov, md, args.limit, cats))
