"""
"최대 효과" 그리드 서치 — 우리모델 + 자가앙상블(형제버전) + Roboflow 결합 최적점 탐색.

사용자 요청: "roboflow + 내가 훈련한 것의 결과물이 최대 효과".
이전 실패 원인: RF를 낮은 conf(0.01)로 단순 WBF 합산 → FP만 폭증.
개선 가설:
  A) base = 우리 주모델 + 우리 형제버전 WBF (자가앙상블, 도메인 일치 → recall↑ 입증됨)
  B) +RF: RF는 "고신뢰만 보충"(union) — base와 IoU<thr인 RF 고conf 박스만 추가 → 놓침 보강, FP 억제
그리드: RF conf_min × union_iou × (WBF weight 조합)
지표: recall, incremental_recall(놓친 GT 중 새로 채운 수), FP.

실행: backend/venv/Scripts/python.exe backend/training/eval/max_effect_grid.py --target M1
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import cv2, numpy as np, yaml
from ensemble_boxes import weighted_boxes_fusion

ROOT = Path(__file__).resolve().parents[3]
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# target → 우리 형제 ckpts(자가앙상블) + dataset + RF json
GROUPS = {
    "M1": {
        "ckpts": [("m1_yolo_structural", 640), ("m1_structural_v3", 640), ("m1_structural_v4s", 640)],
        "data": ROOT / "backend/training/datasets/structural/data.yaml",
        "rf": ROOT / "backend/training/eval/results/rf_M1.json",
    },
    "M3": {
        "ckpts": [("m3_yolo_floor_window", 960), ("m3_floor_window_v3", 640), ("m3_floor_window_v4s", 640)],
        "data": ROOT / "backend/training/datasets/floor_window/data.yaml",
        "rf": ROOT / "backend/training/eval/results/rf_M3.json",
    },
    "THERMAL": {
        "ckpts": [("thermal_yolo", 960), ("thermal_v3", 640), ("thermal_yolo_prev", 640)],
        "data": ROOT / "backend/training/datasets/thermal_yolo/data.yaml",
        "rf": ROOT / "backend/training/eval/results/rf_THERMAL.json",
    },
}

CONF_OURS = 0.05
MATCH_IOU = 0.5


def iou(a, b):
    x1=max(a[0],b[0]);y1=max(a[1],b[1]);x2=min(a[2],b[2]);y2=min(a[3],b[3])
    inter=max(0,x2-x1)*max(0,y2-y1)
    aa=(a[2]-a[0])*(a[3]-a[1]);bb=(b[2]-b[0])*(b[3]-b[1])
    return inter/(aa+bb-inter+1e-6)


def load_gts(label_path, w, h):
    if not label_path.exists(): return []
    out=[]
    for line in label_path.read_text(encoding="utf-8").splitlines():
        p=line.strip().split()
        if len(p)<5: continue
        try:
            cx,cy,bw,bh=float(p[1]),float(p[2]),float(p[3]),float(p[4])
            out.append([(cx-bw/2)*w,(cy-bh/2)*h,(cx+bw/2)*w,(cy+bh/2)*h])
        except: continue
    return out


def metrics(preds_all, gts_all, miou=MATCH_IOU):
    tg=sum(len(g) for g in gts_all); mg=0; tp_pred=0; fp=0
    for preds,gts in zip(preds_all,gts_all):
        used=[False]*len(gts); tp_pred+=len(preds)
        for pb in preds:
            best,bj=0.0,-1
            for j,gb in enumerate(gts):
                if used[j]: continue
                v=iou(pb[:4],gb)
                if v>best: best,bj=v,j
            if best>=miou and bj>=0: used[bj]=True
            else: fp+=1
        mg+=sum(used)
    return {"gt":tg,"matched":mg,"recall":mg/(tg+1e-9),"pred":tp_pred,"fp":fp,
            "prec":(tp_pred-fp)/(tp_pred+1e-9)}


def wbf_combine(per_ver_norm, weights, shapes, iou_thr=0.55, skip=0.0001):
    """형제버전 WBF → pixel 박스 리스트(이미지별)."""
    out=[]
    for i,(w,h) in enumerate(shapes):
        bl,sl,ll,wl=[],[],[],[]
        for vi,pv in enumerate(per_ver_norm):
            if pv is None: continue
            b,s=pv[i]
            if b: bl.append(b); sl.append(s); ll.append([0]*len(s)); wl.append(weights[vi])
        if not bl: out.append([]); continue
        fb,fs,fl=weighted_boxes_fusion(bl,sl,ll,weights=wl,iou_thr=iou_thr,skip_box_thr=skip)
        out.append([[bx[0]*w,bx[1]*h,bx[2]*w,bx[3]*h,sc] for bx,sc in zip(fb,fs)])
    return out


def add_rf_supplement(base_px, rf_map, img_names, rf_conf_min, union_iou):
    """RF 고신뢰 박스 중 base와 IoU<union_iou(겹치지 않는)만 보충 추가."""
    out=[]
    added=0
    for base, name in zip(base_px, img_names):
        rf_dets=[d for d in rf_map.get(name,[]) if d["conf"]>=rf_conf_min]
        merged=list(base)
        for d in rf_dets:
            bb=d["bbox_xyxy"]
            # base의 어떤 박스와도 안 겹치면 = 우리가 놓친 영역일 가능성 → 보충
            if all(iou(bb, mb[:4])<union_iou for mb in merged):
                merged.append([bb[0],bb[1],bb[2],bb[3],d["conf"]])
                added+=1
        out.append(merged)
    return out, added


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--target",required=True,choices=list(GROUPS.keys()))
    ap.add_argument("--max-images",type=int,default=200)
    args=ap.parse_args()
    g=GROUPS[args.target]

    cfg=yaml.safe_load(g["data"].read_text(encoding="utf-8"))
    base_dir=g["data"].parent
    tdir=base_dir/"images"/"test"
    if not tdir.exists(): tdir=base_dir/"images"/"val"
    imgs=sorted(tdir.glob("*.jpg"))+sorted(tdir.glob("*.png"))
    if args.max_images and len(imgs)>args.max_images: imgs=imgs[:args.max_images]
    names=[p.name for p in imgs]
    print(f"=== Max-effect grid (target={args.target}, {len(imgs)} imgs) ===",flush=True)

    gts=[]; shapes=[]
    for p in imgs:
        im=cv2.imread(str(p))
        if im is None: gts.append([]); shapes.append((0,0)); continue
        h,w=im.shape[:2]; shapes.append((w,h))
        gts.append(load_gts(base_dir/"labels"/p.parent.name/(p.stem+".txt"),w,h))

    rf_map=json.loads(g["rf"].read_text(encoding="utf-8")) if g["rf"].exists() else {}
    print(f"  RF json: {g['rf'].name} ({sum(len(v) for v in rf_map.values())} dets)",flush=True)

    # 각 형제버전 추론 (normalized)
    from ultralytics import YOLO
    per_ver=[]
    for name,imgsz in g["ckpts"]:
        onnx=ROOT/f"backend/models_weights/{name}.onnx"
        if not onnx.exists(): per_ver.append(None); print(f"  skip {name}",flush=True); continue
        model=YOLO(str(onnx),task="detect")
        per_img=[]
        for p,(w,h) in zip(imgs,shapes):
            try:
                r=model(str(p),imgsz=imgsz,conf=CONF_OURS,iou=0.6,verbose=False)[0]
                if r.boxes is None or len(r.boxes)==0: per_img.append(([],[])); continue
                xy=r.boxes.xyxy.cpu().numpy(); cf=r.boxes.conf.cpu().numpy()
                nb=(xy/np.array([w,h,w,h])).clip(0,1)
                per_img.append((nb.tolist(),cf.tolist()))
            except Exception: per_img.append(([],[]))
        per_ver.append(per_img)
        print(f"  {name}: {sum(len(x[1]) for x in per_img)} dets",flush=True)

    rows=[]
    # baseline: 주모델 단독
    primary=per_ver[0]
    base_solo=[]
    for (b,s),(w,h) in zip(primary,shapes):
        base_solo.append([[bx[0]*w,bx[1]*h,bx[2]*w,bx[3]*h,sc] for bx,sc in zip(b,s)])
    m=metrics(base_solo,gts)
    rows.append(("주모델_단독", m, 0))

    # 자가앙상블 (형제 WBF, weight 2/1/1)
    W=[2.0]+[1.0]*(len(per_ver)-1)
    self_ens=wbf_combine(per_ver,W,shapes)
    m=metrics(self_ens,gts)
    rows.append(("자가앙상블_WBF", m, 0))

    # 자가앙상블 + RF 보충 (그리드)
    best=None
    for rf_conf in [0.15,0.25,0.40,0.60]:
        for u_iou in [0.3,0.45]:
            comb, added = add_rf_supplement(self_ens, rf_map, names, rf_conf, u_iou)
            m=metrics(comb,gts)
            tag=f"자가앙상블+RF(conf>={rf_conf},uiou={u_iou})"
            rows.append((tag, m, added))
            # 최대효과 기준: recall 최대, 단 FP가 자가앙상블 대비 +20% 이내
            base_fp=rows[1][1]["fp"]
            if m["fp"]<=max(base_fp*1.2, base_fp+10):
                if best is None or m["recall"]>best[1]["recall"]:
                    best=(tag,m,added)

    print("\n----- 결과 (class-agnostic, GT IoU>=0.5) -----",flush=True)
    for tag,m,added in rows:
        print(f"  {tag:42s} R={m['recall']:.4f} ({m['matched']}/{m['gt']}) FP={m['fp']} P={m['prec']:.3f} +RF={added}",flush=True)
    if best:
        print(f"\n  >>> 최대효과(FP가드 통과): {best[0]} R={best[1]['recall']:.4f} FP={best[1]['fp']}",flush=True)

    out_dir=Path(__file__).parent/"results"; out_dir.mkdir(exist_ok=True)
    ts=time.strftime("%Y%m%d_%H%M%S")
    res=out_dir/f"max_effect_{args.target}_{ts}.json"
    res.write_text(json.dumps({"target":args.target,"n":len(imgs),
        "rows":[{"cfg":t,"m":m,"rf_added":a} for t,m,a in rows],
        "best":({"cfg":best[0],"m":best[1],"rf_added":best[2]} if best else None)},
        indent=2,ensure_ascii=False),encoding="utf-8")
    print(f"\n결과: {res}",flush=True)
    return 0


if __name__=="__main__":
    sys.exit(main())
