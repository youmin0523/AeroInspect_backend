"""
우리 ONNX 모델 × Roboflow 보조모델 WBF Ensemble — before/after 실측 (venv py3.13, GPU/CPU).

목적(사용자 핵심: "놓치는 것 없이 정확한 검출"):
  - 주지표 = class-agnostic **Recall**(GT 놓침 여부) + **FP**(오탐 비용). Recall 우선 정책.
  - 보조지표 = class-aware mAP50.
  - before(우리 단독) vs after(우리+Roboflow WBF) 비교 → 모델별 채택/기각 판단(실측).

전제: Roboflow 검출 JSON은 rfenv에서 roboflow_adapter.py로 미리 생성
      (동일 test 이미지 basename 키, pixel xyxy + mapped class + conf).

실행:
  # 1단계 (rfenv): 보조검출 JSON 생성
  backend/rfenv/Scripts/python.exe backend/training/roboflow_adapter.py \
      "backend/training/datasets/structural/images/test/*.jpg" \
      backend/training/eval/results/rf_M1.json  crack-bphdr/2
  # 2단계 (venv): WBF before/after 실측
  backend/venv/Scripts/python.exe backend/training/eval/ensemble_eval_roboflow.py \
      --target M1 --rf-json backend/training/eval/results/rf_M1.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml
from ensemble_boxes import weighted_boxes_fusion

ROOT = Path(__file__).resolve().parents[3]
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# target → 우리 ONNX + dataset + 추론 imgsz
MODELS = {
    "M1":         {"onnx": ROOT / "backend/models_weights/m1_yolo_structural.onnx",       "imgsz": 640},
    "M2":         {"onnx": ROOT / "backend/models_weights/m2_yolo_surface.onnx",          "imgsz": 480},
    "M3":         {"onnx": ROOT / "backend/models_weights/m3_yolo_floor_window.onnx",     "imgsz": 960},
    "M4_CONTEXT": {"onnx": ROOT / "backend/models_weights/m4_yolo_context_elements.onnx", "imgsz": 960},
    "M5":         {"onnx": ROOT / "backend/models_weights/m5_yolo_seg_frames.onnx",       "imgsz": 640},
    "THERMAL":    {"onnx": ROOT / "backend/models_weights/thermal_yolo.onnx",             "imgsz": 960},
    "FURNITURE":  {"onnx": ROOT / "backend/models_weights/furniture_aware.onnx",          "imgsz": 640},
}
DATASETS = {
    "M1":         ROOT / "backend/training/datasets/structural/data.yaml",
    "M2":         ROOT / "backend/training/datasets/surface/data.yaml",
    "M3":         ROOT / "backend/training/datasets/floor_window/data.yaml",
    "M4_CONTEXT": ROOT / "backend/training/datasets/m4_context/data.yaml",
    "M5":         ROOT / "backend/training/datasets/frames/data.yaml",
    "THERMAL":    ROOT / "backend/training/datasets/thermal_yolo/data.yaml",
    "FURNITURE":  ROOT / "backend/training/datasets/furniture_aware/data.yaml",
}

# WBF 가중치: 우리 모델 우대(feedback_postprocess_strength_policy), Roboflow 보조는 낮게
W_OURS = 2.0
W_RF = 1.0
# Recall 우선: 약한 검출도 살림 (feedback_recall_priority_paid_service)
SKIP_BOX_THR = 0.0001
IOU_THR = 0.55
CONF_OURS = 0.05      # 우리 모델도 낮게 → 놓침 최소
MATCH_IOU = 0.5       # GT 매칭 기준


def iou(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3]); inter = max(0, x2-x1)*max(0, y2-y1)
    aa = (a[2]-a[0])*(a[3]-a[1]); bb = (b[2]-b[0])*(b[3]-b[1])
    return inter / (aa + bb - inter + 1e-6)


def load_yolo_labels(label_path, img_w, img_h):
    """class-agnostic GT 박스 (위치만)."""
    if not label_path.exists():
        return []
    out = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        try:
            cx = float(parts[1]); cy = float(parts[2]); w = float(parts[3]); h = float(parts[4])
            out.append([(cx-w/2)*img_w, (cy-h/2)*img_h, (cx+w/2)*img_w, (cy+h/2)*img_h])
        except Exception:
            continue
    return out


def recall_fp(all_preds_px, all_gts_px, match_iou=MATCH_IOU):
    """class-agnostic Recall + FP. preds_px/gts_px = [[x1,y1,x2,y2,...]] per image."""
    total_gt = sum(len(g) for g in all_gts_px)
    matched_gt = 0
    total_pred = 0
    fp = 0
    for preds, gts in zip(all_preds_px, all_gts_px):
        used = [False] * len(gts)
        total_pred += len(preds)
        # conf 내림차순 가정 안 함 — 단순 매칭(위치 기준)
        for pb in preds:
            best, bj = 0.0, -1
            for j, gb in enumerate(gts):
                if used[j]:
                    continue
                v = iou(pb[:4], gb)
                if v > best:
                    best, bj = v, j
            if best >= match_iou and bj >= 0:
                used[bj] = True
            else:
                fp += 1
        matched_gt += sum(used)
    recall = matched_gt / (total_gt + 1e-9)
    precision = (total_pred - fp) / (total_pred + 1e-9)
    return {"total_gt": total_gt, "matched_gt": matched_gt, "recall": recall,
            "total_pred": total_pred, "fp": fp, "precision": precision}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, choices=list(MODELS.keys()))
    ap.add_argument("--rf-json", required=True, help="roboflow_adapter.py 산출 JSON")
    ap.add_argument("--max-images", type=int, default=200)
    args = ap.parse_args()

    cfg_path = DATASETS[args.target]
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    base = cfg_path.parent
    test_dir = base / "images" / "test"
    if not test_dir.exists():
        test_dir = base / "images" / "val"
    img_files = sorted(test_dir.glob("*.jpg")) + sorted(test_dir.glob("*.png"))
    if args.max_images and len(img_files) > args.max_images:
        img_files = img_files[: args.max_images]
    print(f"=== Ensemble eval (target={args.target}, {len(img_files)} imgs) ===", flush=True)

    onnx = MODELS[args.target]["onnx"]
    if not onnx.exists():
        print(f"❌ ONNX 없음: {onnx}")
        return 1

    rf_path = Path(args.rf_json)
    rf_map = json.loads(rf_path.read_text(encoding="utf-8")) if rf_path.exists() else {}
    print(f"  Roboflow JSON: {rf_path.name} ({sum(len(v) for v in rf_map.values())} dets)", flush=True)

    # GT + shapes
    all_gts, shapes = [], []
    for p in img_files:
        im = cv2.imread(str(p))
        if im is None:
            all_gts.append([]); shapes.append((0, 0)); continue
        h, w = im.shape[:2]; shapes.append((w, h))
        lbl = base / "labels" / p.parent.name / (p.stem + ".txt")
        all_gts.append(load_yolo_labels(lbl, w, h))

    # 우리 ONNX 추론
    from ultralytics import YOLO
    model = YOLO(str(onnx), task="detect")
    ours_px = []          # before: 우리 단독 (pixel boxes)
    ours_norm = []        # WBF용 normalized
    t0 = time.time()
    imgsz = MODELS[args.target]["imgsz"]
    for p, (w, h) in zip(img_files, shapes):
        try:
            res = model(str(p), imgsz=imgsz, conf=CONF_OURS, iou=0.6, verbose=False)[0]
            if res.boxes is None or len(res.boxes) == 0:
                ours_px.append([]); ours_norm.append(([], [], [])); continue
            xyxy = res.boxes.xyxy.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()
            ours_px.append([[*b, c] for b, c in zip(xyxy.tolist(), confs.tolist())])
            nb = (xyxy / np.array([w, h, w, h])).clip(0, 1)
            ours_norm.append((nb.tolist(), confs.tolist(), [0]*len(confs)))  # class-agnostic label 0
        except Exception:
            ours_px.append([]); ours_norm.append(([], [], []))
    print(f"  우리 ONNX 추론 {time.time()-t0:.1f}s", flush=True)

    # before 지표
    before = recall_fp(ours_px, all_gts)

    # after: WBF(우리 + Roboflow), class-agnostic(label 0)
    after_px = []
    for p, (w, h), (ob, os_, ol) in zip(img_files, shapes, ours_norm):
        rf_dets = rf_map.get(p.name, [])
        rb, rs, rl = [], [], []
        for d in rf_dets:
            x1, y1, x2, y2 = d["bbox_xyxy"]
            rb.append([min(max(x1/w, 0), 1), min(max(y1/h, 0), 1),
                       min(max(x2/w, 0), 1), min(max(y2/h, 0), 1)])
            rs.append(d["conf"]); rl.append(0)
        boxes_l, scores_l, labels_l, weights = [], [], [], []
        if ob:
            boxes_l.append(ob); scores_l.append(os_); labels_l.append(ol); weights.append(W_OURS)
        if rb:
            boxes_l.append(rb); scores_l.append(rs); labels_l.append(rl); weights.append(W_RF)
        if not boxes_l:
            after_px.append([]); continue
        fb, fs, fl = weighted_boxes_fusion(
            boxes_l, scores_l, labels_l, weights=weights,
            iou_thr=IOU_THR, skip_box_thr=SKIP_BOX_THR,
        )
        after_px.append([[bx[0]*w, bx[1]*h, bx[2]*w, bx[3]*h, sc] for bx, sc in zip(fb, fs)])

    after = recall_fp(after_px, all_gts)

    # 보고
    print("\n----- 결과 (class-agnostic, GT 매칭 IoU>=0.5) -----", flush=True)
    print(f"  BEFORE 우리 단독 : Recall={before['recall']:.4f} "
          f"({before['matched_gt']}/{before['total_gt']}) "
          f"FP={before['fp']} preds={before['total_pred']} P={before['precision']:.4f}", flush=True)
    print(f"  AFTER  +Roboflow : Recall={after['recall']:.4f} "
          f"({after['matched_gt']}/{after['total_gt']}) "
          f"FP={after['fp']} preds={after['total_pred']} P={after['precision']:.4f}", flush=True)
    d_recall = after['recall'] - before['recall']
    d_fp = after['fp'] - before['fp']
    print(f"  Δ Recall={d_recall:+.4f}  Δ FP={d_fp:+d}", flush=True)
    verdict = "채택(Recall↑)" if d_recall > 0.001 and d_fp <= max(5, before['fp']*0.2) else (
        "기각(FP 급증)" if d_recall > 0 else "효과미미")
    print(f"  >>> 판정: {verdict}", flush=True)

    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out = out_dir / f"ensemble_rf_{args.target}_{ts}.json"
    out.write_text(json.dumps({
        "target": args.target, "n_images": len(img_files),
        "rf_json": rf_path.name, "weights": {"ours": W_OURS, "rf": W_RF},
        "iou_thr": IOU_THR, "skip_box_thr": SKIP_BOX_THR,
        "before": before, "after": after,
        "delta_recall": d_recall, "delta_fp": d_fp, "verdict": verdict,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n결과: {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
