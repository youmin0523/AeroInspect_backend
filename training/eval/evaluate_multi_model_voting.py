"""
Multi-model voting eval — 같은 이미지에 여러 모델 동시 추론 후
cross_model_spatial_boost 진짜 효과 측정.

각 모델은 다른 클래스 taxonomy를 가지므로 평가 전략:
- 각 모델 dataset의 GT는 그 모델 클래스만 포함
- 다른 모델 검출은 보조 신호 (cross-model boost 발동)
- 최종 mAP 계산은 dataset의 클래스 taxonomy 기준
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[3]
BACKEND = ROOT / "backend"
sys.path.insert(0, str(BACKEND))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from app.services.ensemble import (
    cross_model_nms,
    cross_model_spatial_boost,
)


# 모든 ONNX 모델 (cross-model 신호 위해 전부 추론)
MODELS = [
    {"key": "M1", "onnx": ROOT / "backend/models_weights/m1_yolo_structural.onnx", "imgsz": 640},
    {"key": "M2", "onnx": ROOT / "backend/models_weights/m2_yolo_surface.onnx", "imgsz": 480},
    {"key": "M3", "onnx": ROOT / "backend/models_weights/m3_yolo_floor_window.onnx", "imgsz": 960},
    {"key": "M4_CONTEXT", "onnx": ROOT / "backend/models_weights/m4_yolo_context_elements.onnx", "imgsz": 960},
    {"key": "M5", "onnx": ROOT / "backend/models_weights/m5_yolo_seg_frames.onnx", "imgsz": 640},
]

DATASETS = {
    "M1": ROOT / "backend/training/datasets/structural/data.yaml",
    "M2": ROOT / "backend/training/datasets/surface/data.yaml",
    "M3": ROOT / "backend/training/datasets/floor_window/data.yaml",
    "M4_CONTEXT": ROOT / "backend/training/datasets/m4_context/data.yaml",
    "M5": ROOT / "backend/training/datasets/frames/data.yaml",
}


def iou(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3]); inter = max(0, x2-x1)*max(0, y2-y1)
    aa = (a[2]-a[0])*(a[3]-a[1]); bb = (b[2]-b[0])*(b[3]-b[1])
    return inter / (aa + bb - inter + 1e-6)


def compute_map50(all_preds, all_gts, target_classes=None):
    """target_classes: 평가할 클래스 set (다른 모델 검출은 무시)."""
    classes = set()
    for gts in all_gts:
        for g in gts: classes.add(g["class"])
    if target_classes:
        classes = classes & target_classes

    aps, ps, rs = [], [], []
    for c in classes:
        cls_preds = []; cls_gts = []
        for i, preds in enumerate(all_preds):
            for p in preds:
                if p["class"] == c:
                    cls_preds.append((i, p["conf"], p["bbox_xyxy"]))
        for i, gts in enumerate(all_gts):
            for g in gts:
                if g["class"] == c:
                    cls_gts.append([i, g["bbox_xyxy"], False])
        if not cls_gts: continue
        cls_preds.sort(key=lambda x: -x[1])
        tp_arr, fp_arr = [], []
        for img_idx, conf, bbox in cls_preds:
            best_iou, best_j = 0.0, -1
            for j, (gi, gb, matched) in enumerate(cls_gts):
                if gi != img_idx or matched: continue
                ii = iou(bbox, gb)
                if ii > best_iou: best_iou, best_j = ii, j
            if best_iou >= 0.5 and best_j >= 0:
                cls_gts[best_j][2] = True
                tp_arr.append(1); fp_arr.append(0)
            else:
                tp_arr.append(0); fp_arr.append(1)
        if not tp_arr: aps.append(0.0); ps.append(0.0); rs.append(0.0); continue
        tp_cum = np.cumsum(tp_arr); fp_cum = np.cumsum(fp_arr)
        precision = tp_cum / (tp_cum + fp_cum + 1e-6)
        recall = tp_cum / (len(cls_gts) + 1e-6)
        ap = 0.0
        for t in np.linspace(0, 1, 11):
            mask = recall >= t
            ap += precision[mask].max() / 11 if mask.any() else 0.0
        aps.append(ap); ps.append(precision[-1]); rs.append(recall[-1])
    return (float(np.mean(aps)) if aps else 0.0,
            float(np.mean(ps)) if ps else 0.0,
            float(np.mean(rs)) if rs else 0.0)


def load_yolo_labels(label_path, img_w, img_h, name_map):
    if not label_path.exists(): return []
    out = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 5: continue
        try:
            cid = int(parts[0])
            cx = float(parts[1]); cy = float(parts[2])
            w = float(parts[3]); h = float(parts[4])
            x1 = (cx - w/2) * img_w; y1 = (cy - h/2) * img_h
            x2 = (cx + w/2) * img_w; y2 = (cy + h/2) * img_h
            out.append({"class": name_map.get(cid, str(cid)), "bbox_xyxy": [x1, y1, x2, y2]})
        except: continue
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="M3", help="평가 대상 dataset (M1|M2|M3|M4_CONTEXT|M5)")
    parser.add_argument("--max-images", type=int, default=200)
    args = parser.parse_args()

    target_data = DATASETS[args.target]
    cfg = yaml.safe_load(target_data.read_text(encoding="utf-8"))
    base = target_data.parent
    test_dir = base / "images" / "test"
    if not test_dir.exists(): test_dir = base / "images" / "val"
    img_files = sorted(test_dir.glob("*.jpg")) + sorted(test_dir.glob("*.png"))
    if args.max_images and len(img_files) > args.max_images:
        img_files = img_files[: args.max_images]
    name_map = {i: n for i, n in enumerate(cfg.get("names", []))}
    target_classes = set(name_map.values())
    print(f"=== Multi-model voting eval (target={args.target}, {len(img_files)} images) ===")
    print(f"  Target classes: {sorted(target_classes)}")

    # GT 로드 (target 클래스만)
    print("GT 로딩...")
    all_gts = []
    img_shapes = []
    for p in img_files:
        im = cv2.imread(str(p))
        if im is None: all_gts.append([]); img_shapes.append((0,0)); continue
        h, w = im.shape[:2]; img_shapes.append((w, h))
        lbl = base / "labels" / p.parent.name / (p.stem + ".txt")
        all_gts.append(load_yolo_labels(lbl, w, h, name_map))

    # 각 모델 추론 (모든 모델, target 이미지에)
    print("\n전체 모델 추론...")
    from ultralytics import YOLO
    per_model_preds = {}  # key → [per-image preds]
    t0 = time.time()
    for m in MODELS:
        if not m["onnx"].exists():
            print(f"  ❌ {m['key']}: missing ONNX"); continue
        try:
            model = YOLO(str(m["onnx"]), task="detect")
        except Exception as e:
            print(f"  ❌ {m['key']}: load fail {e}"); continue

        # 모델별 클래스 이름 가져오기
        m_data = DATASETS[m["key"]]
        m_cfg = yaml.safe_load(m_data.read_text(encoding="utf-8"))
        m_name_map = {i: n for i, n in enumerate(m_cfg.get("names", []))}

        per_img = []
        for p in img_files:
            try:
                res = model(str(p), imgsz=m["imgsz"], conf=0.001, iou=0.6, verbose=False)[0]
                preds = []
                if res.boxes is not None:
                    xyxy = res.boxes.xyxy.cpu().numpy()
                    confs = res.boxes.conf.cpu().numpy()
                    clss = res.boxes.cls.cpu().numpy().astype(int)
                    for box, cf, cl in zip(xyxy, confs, clss):
                        preds.append({
                            "class": m_name_map.get(int(cl), str(int(cl))),
                            "conf": float(cf),
                            "bbox_xyxy": [float(box[0]), float(box[1]), float(box[2]), float(box[3])],
                            "defect_source": m["key"],
                        })
                per_img.append(preds)
            except Exception:
                per_img.append([])
        per_model_preds[m["key"]] = per_img
        print(f"  {m['key']}: {sum(len(p) for p in per_img)} detections")
    print(f"추론 완료: {time.time()-t0:.1f}s")

    # Stage A: target 모델 단독 (baseline)
    print("\n[A] Target 모델 단독 ── baseline")
    target_only = per_model_preds.get(args.target, [[] for _ in img_files])
    m, p, r = compute_map50(target_only, all_gts, target_classes)
    print(f"    {args.target} alone:                 mAP50={m:.4f} P={p:.4f} R={r:.4f}")
    a_score = m

    # Stage B: 모든 모델 합치기 (cross_model_nms)
    print("\n[B] 전체 모델 detections 합치기 + cross_model_nms")
    combined = []
    for i in range(len(img_files)):
        all_d = []
        for k, per_img in per_model_preds.items():
            if i < len(per_img): all_d.extend(per_img[i])
        deduped = cross_model_nms(all_d, iou_threshold=0.4)
        combined.append(deduped)
    m, p, r = compute_map50(combined, all_gts, target_classes)
    print(f"    + cross_model_nms (target classes 만): mAP50={m:.4f} P={p:.4f} R={r:.4f}")
    b_score = m

    # Stage C: cross_model_spatial_boost 적용
    print("\n[C] + cross_model_spatial_boost")
    boosted = [cross_model_spatial_boost(list(c), iou_threshold=0.25, boost_factor=0.50)
               for c in combined]
    m, p, r = compute_map50(boosted, all_gts, target_classes)
    print(f"    + cross_model_spatial_boost:           mAP50={m:.4f} P={p:.4f} R={r:.4f}")
    c_score = m

    # 결과 저장
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_json = out_dir / f"multi_model_voting_{args.target}_{ts}.json"
    out_md = out_dir / f"multi_model_voting_{args.target}_{ts}.md"
    out_json.write_text(json.dumps({
        "target": args.target, "n_images": len(img_files),
        "models": list(per_model_preds.keys()),
        "stages": [
            {"stage": "A_alone", "mAP50": a_score},
            {"stage": "B_cross_nms", "mAP50": b_score},
            {"stage": "C_spatial_boost", "mAP50": c_score},
        ],
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    out_md.write_text(
        f"# Multi-model voting eval (target={args.target})\n\n"
        f"- 이미지 수: {len(img_files)}, 사용 모델: {list(per_model_preds.keys())}\n\n"
        f"| Stage | mAP50 | Δ |\n"
        f"|-------|-------|---|\n"
        f"| A) Target 단독 | {a_score:.4f} | baseline |\n"
        f"| B) + cross_model_nms | {b_score:.4f} | {b_score-a_score:+.4f} |\n"
        f"| C) + cross_model_spatial_boost | {c_score:.4f} | {c_score-a_score:+.4f} |\n",
        encoding="utf-8",
    )
    print(f"\n결과: {out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
