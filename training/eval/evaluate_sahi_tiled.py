"""
SAHI Tiled inference eval - 작은 객체 검출용.
이미지를 640x640 타일로 분할 + 추론 + cross-tile NMS.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[3]
BACKEND = ROOT / "backend"
sys.path.insert(0, str(BACKEND))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")


CONFIGS = {
    "M1": {"onnx": ROOT / "backend/models_weights/m1_yolo_structural.onnx",
           "data": ROOT / "backend/training/datasets/structural/data.yaml"},
    "M2": {"onnx": ROOT / "backend/models_weights/m2_yolo_surface.onnx",
           "data": ROOT / "backend/training/datasets/surface/data.yaml"},
    "M3": {"onnx": ROOT / "backend/models_weights/m3_yolo_floor_window.onnx",
           "data": ROOT / "backend/training/datasets/floor_window/data.yaml"},
    "M5": {"onnx": ROOT / "backend/models_weights/m5_yolo_seg_frames.onnx",
           "data": ROOT / "backend/training/datasets/frames/data.yaml"},
    "furniture": {"onnx": ROOT / "backend/models_weights/furniture_aware.onnx",
                  "data": ROOT / "backend/training/datasets/furniture_aware/data.yaml"},
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


def generate_tiles(h, w, tile=640, overlap=0.2):
    stride = int(tile * (1 - overlap))
    if h <= tile and w <= tile: return [(0, 0, w, h)]
    n_rows = max(1, math.ceil((h - tile) / stride) + 1)
    n_cols = max(1, math.ceil((w - tile) / stride) + 1)
    tiles = []
    for r in range(n_rows):
        for c in range(n_cols):
            y1 = min(r * stride, h - tile); x1 = min(c * stride, w - tile)
            y1 = max(0, y1); x1 = max(0, x1)
            y2 = min(y1 + tile, h); x2 = min(x1 + tile, w)
            tiles.append((x1, y1, x2, y2))
    return tiles


def cross_tile_nms(dets, iou_t=0.5):
    if len(dets) <= 1: return dets
    by_class = {}
    for d in dets:
        by_class.setdefault(d["class"], []).append(d)
    out = []
    for cls, ds in by_class.items():
        ds.sort(key=lambda d: -d["conf"])
        keep = []
        for d in ds:
            dup = False
            for k in keep:
                if iou(d["bbox_xyxy"], k["bbox_xyxy"]) >= iou_t:
                    dup = True; break
            if not dup: keep.append(d)
        out.extend(keep)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="M5")
    parser.add_argument("--max-images", type=int, default=200)
    parser.add_argument("--tile", type=int, default=640)
    parser.add_argument("--overlap", type=float, default=0.2)
    args = parser.parse_args()

    cfg = CONFIGS[args.model]
    if not cfg["onnx"].exists():
        print(f"❌ {args.model}: ONNX not found"); return 1

    data_yaml = cfg["data"]
    if not data_yaml.exists():
        print(f"❌ {args.model}: data yaml not found"); return 1

    cfg_yaml = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    base = data_yaml.parent
    test_dir = base / "images" / "test"
    if not test_dir.exists(): test_dir = base / "images" / "val"
    img_files = sorted(test_dir.glob("*.jpg")) + sorted(test_dir.glob("*.png"))
    if args.max_images and len(img_files) > args.max_images:
        img_files = img_files[: args.max_images]
    name_map = {i: n for i, n in enumerate(cfg_yaml.get("names", []))}
    print(f"=== {args.model} SAHI Tiled (tile={args.tile}, overlap={args.overlap}, {len(img_files)} images) ===")

    from ultralytics import YOLO
    model = YOLO(str(cfg["onnx"]), task="detect")

    # GT 로드
    print("GT 로딩...")
    all_gts = []
    img_shapes = []
    for p in img_files:
        im = cv2.imread(str(p))
        if im is None: all_gts.append([]); img_shapes.append((0,0)); continue
        h, w = im.shape[:2]; img_shapes.append((w, h))
        lbl = base / "labels" / p.parent.name / (p.stem + ".txt")
        all_gts.append(load_yolo_labels(lbl, w, h, name_map))

    # Whole-image baseline
    print("\n[A] Whole-image baseline 추론...")
    all_baseline = []
    t0 = time.time()
    for p in img_files:
        try:
            res = model(str(p), imgsz=args.tile, conf=0.001, iou=0.6, verbose=False)[0]
            preds = []
            if res.boxes is not None:
                xyxy = res.boxes.xyxy.cpu().numpy()
                confs = res.boxes.conf.cpu().numpy()
                clss = res.boxes.cls.cpu().numpy().astype(int)
                for box, cf, cl in zip(xyxy, confs, clss):
                    preds.append({"class": name_map.get(int(cl), str(int(cl))),
                                  "conf": float(cf),
                                  "bbox_xyxy": [float(box[0]), float(box[1]), float(box[2]), float(box[3])]})
            all_baseline.append(preds)
        except: all_baseline.append([])
    bm, bp, br = compute_map50(all_baseline, all_gts)
    print(f"  baseline: mAP50={bm:.4f} P={bp:.4f} R={br:.4f} ({time.time()-t0:.1f}s)")

    # Tiled
    print(f"\n[B] Tiled 추론...")
    all_tiled = []
    t0 = time.time()
    for img_path, (w, h) in zip(img_files, img_shapes):
        if w == 0: all_tiled.append([]); continue
        im = cv2.imread(str(img_path))
        tiles = generate_tiles(h, w, args.tile, args.overlap)
        all_dets = []
        for (x1, y1, x2, y2) in tiles:
            tile_img = im[y1:y2, x1:x2]
            try:
                res = model(tile_img, imgsz=args.tile, conf=0.001, iou=0.6, verbose=False)[0]
                if res.boxes is None: continue
                xyxy = res.boxes.xyxy.cpu().numpy()
                confs = res.boxes.conf.cpu().numpy()
                clss = res.boxes.cls.cpu().numpy().astype(int)
                for box, cf, cl in zip(xyxy, confs, clss):
                    all_dets.append({
                        "class": name_map.get(int(cl), str(int(cl))),
                        "conf": float(cf),
                        "bbox_xyxy": [float(box[0])+x1, float(box[1])+y1, float(box[2])+x1, float(box[3])+y1],
                    })
            except: pass
        deduped = cross_tile_nms(all_dets, iou_t=0.5)
        all_tiled.append(deduped)
    tm, tp, tr = compute_map50(all_tiled, all_gts)
    print(f"  tiled: mAP50={tm:.4f} P={tp:.4f} R={tr:.4f} ({time.time()-t0:.1f}s)")

    delta = tm - bm
    marker = "🎯" if tm >= 0.85 else "📈" if delta > 0 else "📉"
    print(f"\n{marker} {args.model}: baseline {bm:.4f} → tiled {tm:.4f} (Δ {delta:+.4f}) | 0.85 갭 {tm-0.85:+.4f}")

    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_json = out_dir / f"sahi_{args.model}_{ts}.json"
    out_md = out_dir / f"sahi_{args.model}_{ts}.md"
    out_json.write_text(json.dumps({
        "model": args.model, "n_images": len(img_files),
        "tile": args.tile, "overlap": args.overlap,
        "baseline": {"mAP50": bm, "P": bp, "R": br},
        "tiled": {"mAP50": tm, "P": tp, "R": tr},
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    out_md.write_text(
        f"# {args.model} SAHI Tiled eval\n\n"
        f"- 이미지 수: {len(img_files)}, tile={args.tile}, overlap={args.overlap}\n\n"
        f"| 방법 | mAP50 | P | R | Δ | 0.85 갭 |\n"
        f"|------|-------|---|---|---|---------|\n"
        f"| Whole-image baseline | {bm:.4f} | {bp:.4f} | {br:.4f} | - | {bm-0.85:+.4f} |\n"
        f"| **Tiled** | **{tm:.4f}** | {tp:.4f} | {tr:.4f} | {delta:+.4f} | {tm-0.85:+.4f} |\n",
        encoding="utf-8",
    )
    print(f"결과: {out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
