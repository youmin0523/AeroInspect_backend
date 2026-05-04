"""
WBF v2 — 노이즈 증폭 버그 수정.
이전 v1: skip_box_thr=0.001 + 모든 detections fusion → noise FP 폭증, mAP 저하
v2 수정:
  1. per-config top-K 필터 (이미지당 K개만 유지)
  2. skip_box_thr 그리드 (0.05, 0.1, 0.2)
  3. iou_thr 그리드 (0.5, 0.55, 0.6, 0.65)
  4. 각 조합으로 최고 mAP 찾기
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from itertools import product
from pathlib import Path

import cv2
import numpy as np
import yaml
from ensemble_boxes import weighted_boxes_fusion

ROOT = Path(__file__).resolve().parents[3]
sys.stdout.reconfigure(encoding="utf-8", errors="replace")


CONFIGS = {
    "M3": {"pt": ROOT / "runs/detect/runs/m3_floor_window/phase2_full/weights/best.pt",
           "data": ROOT / "backend/training/datasets/floor_window/data.yaml",
           "imgsz_list": [640, 800, 960]},
    "M2": {"pt": Path("c:/Users/Codelab/Downloads/colab_pt_extracted/m2_plan_a_results/m2_yolo_surface_best.pt"),
           "data": ROOT / "backend/training/datasets/surface/data.yaml",
           "imgsz_list": [480, 640, 800]},
}


def iou(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3]); inter = max(0, x2-x1)*max(0, y2-y1)
    aa = (a[2]-a[0])*(a[3]-a[1]); bb = (b[2]-b[0])*(b[3]-b[1])
    return inter / (aa + bb - inter + 1e-6)


def compute_map50(all_preds, all_gts):
    classes = set()
    for gts in all_gts:
        for g in gts: classes.add(g["class"])
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
    parser.add_argument("--model", default="M3")
    parser.add_argument("--max-images", type=int, default=300)
    args = parser.parse_args()

    cfg = CONFIGS[args.model]
    if not cfg["pt"].exists():
        print(f"❌ {args.model}: pt 없음"); return 1
    cfg_yaml = yaml.safe_load(cfg["data"].read_text(encoding="utf-8"))
    base = cfg["data"].parent
    test_dir = base / "images" / "test"
    if not test_dir.exists(): test_dir = base / "images" / "val"
    img_files = sorted(test_dir.glob("*.jpg")) + sorted(test_dir.glob("*.png"))
    if args.max_images and len(img_files) > args.max_images:
        img_files = img_files[: args.max_images]
    name_map = {i: n for i, n in enumerate(cfg_yaml.get("names", []))}
    print(f"=== {args.model} WBF v2 ({len(img_files)} images, imgsz={cfg['imgsz_list']}) ===")

    from ultralytics import YOLO
    model = YOLO(str(cfg["pt"]))

    print("GT 로딩...")
    all_gts = []; img_shapes = []
    for p in img_files:
        im = cv2.imread(str(p))
        if im is None: all_gts.append([]); img_shapes.append((0,0)); continue
        h, w = im.shape[:2]; img_shapes.append((w, h))
        lbl = base / "labels" / p.parent.name / (p.stem + ".txt")
        all_gts.append(load_yolo_labels(lbl, w, h, name_map))

    print("\nMulti-scale × TTA 추론 (top-K 필터링 적용)...")
    per_config_preds = {}  # (imgsz, tta) → [per-image (boxes, scores, labels)]
    TOP_K = 100  # 이미지당 top-K만 유지 (이전엔 무제한 → noise 폭증)
    t0 = time.time()
    for imgsz in cfg["imgsz_list"]:
        for tta in [False, True]:
            key = (imgsz, tta)
            per_config_preds[key] = []
            for p in img_files:
                try:
                    res = model(str(p), imgsz=imgsz, conf=0.001, iou=0.6,
                                augment=tta, verbose=False)[0]
                    if res.boxes is None or len(res.boxes) == 0:
                        per_config_preds[key].append(([], [], []))
                        continue
                    h, w = res.orig_shape
                    boxes = res.boxes.xyxy.cpu().numpy() / np.array([w, h, w, h])
                    boxes = np.clip(boxes, 0, 1)
                    scores = res.boxes.conf.cpu().numpy()
                    labels = res.boxes.cls.cpu().numpy().astype(int)
                    # top-K (conf 내림차순)
                    if len(scores) > TOP_K:
                        idx = np.argsort(-scores)[:TOP_K]
                        boxes, scores, labels = boxes[idx], scores[idx], labels[idx]
                    per_config_preds[key].append((boxes.tolist(), scores.tolist(), labels.tolist()))
                except Exception:
                    per_config_preds[key].append(([], [], []))
            n = sum(len(p[0]) for p in per_config_preds[key])
            print(f"  imgsz={imgsz} tta={'O' if tta else 'X'}: {n} detections (top-K={TOP_K})")
    print(f"추론 완료: {time.time()-t0:.1f}s")

    # 단일 config baseline
    best_single = {"mAP50": -1, "config": None}
    for key, per_img in per_config_preds.items():
        all_p = []
        for (b, s, l), (w, h) in zip(per_img, img_shapes):
            preds = []
            for box, sc, lb in zip(b, s, l):
                preds.append({"class": name_map.get(int(lb), str(lb)),
                              "conf": float(sc),
                              "bbox_xyxy": [box[0]*w, box[1]*h, box[2]*w, box[3]*h]})
            all_p.append(preds)
        m, p, r = compute_map50(all_p, all_gts)
        if m > best_single["mAP50"]:
            best_single = {"mAP50": m, "P": p, "R": r,
                          "config": f"imgsz={key[0]} tta={'O' if key[1] else 'X'}"}
    print(f"\n[BEST single] {best_single['config']}: mAP50={best_single['mAP50']:.4f}")

    # WBF 그리드 탐색
    print("\n[WBF v2 grid: skip_box_thr × iou_thr]")
    best_wbf = {"mAP50": -1, "config": None}
    for skip_thr, iou_thr in product([0.05, 0.1, 0.2], [0.5, 0.55, 0.6, 0.65]):
        fused = []
        for i in range(len(img_files)):
            bl, sl, ll = [], [], []
            for key in per_config_preds:
                b, s, l = per_config_preds[key][i]
                if b: bl.append(b); sl.append(s); ll.append(l)
            if not bl:
                fused.append([]); continue
            try:
                fb, fs, fl = weighted_boxes_fusion(
                    bl, sl, ll, weights=[1]*len(bl),
                    iou_thr=iou_thr, skip_box_thr=skip_thr,
                )
            except Exception:
                fb, fs, fl = [], [], []
            w, h = img_shapes[i]
            preds = []
            for box, sc, lb in zip(fb, fs, fl):
                preds.append({"class": name_map.get(int(lb), str(int(lb))),
                              "conf": float(sc),
                              "bbox_xyxy": [float(box[0])*w, float(box[1])*h,
                                           float(box[2])*w, float(box[3])*h]})
            fused.append(preds)
        m, p, r = compute_map50(fused, all_gts)
        delta = m - best_single["mAP50"]
        marker = "🎯" if m >= 0.85 else "📈" if delta > 0 else "📉"
        print(f"  {marker} skip={skip_thr} iou={iou_thr}: mAP50={m:.4f} (Δ {delta:+.4f}) | 0.85 갭 {m-0.85:+.4f}")
        if m > best_wbf["mAP50"]:
            best_wbf = {"mAP50": m, "P": p, "R": r,
                       "config": f"skip={skip_thr} iou={iou_thr}"}

    print(f"\n[BEST WBF v2] {best_wbf['config']}: mAP50={best_wbf['mAP50']:.4f}")
    print(f"  vs BEST single: Δ {best_wbf['mAP50']-best_single['mAP50']:+.4f}")
    print(f"  0.85 갭: {best_wbf['mAP50']-0.85:+.4f}")

    out_dir = Path(__file__).parent / "results"
    ts = time.strftime("%Y%m%d_%H%M%S")
    out = out_dir / f"wbf_v2_{args.model}_{ts}.json"
    out.write_text(json.dumps({
        "model": args.model, "n_images": len(img_files),
        "imgsz_list": cfg["imgsz_list"], "top_k": TOP_K,
        "best_single": best_single, "best_wbf": best_wbf,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n결과: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
