"""
자가앙상블 실측 — 우리 모델의 여러 버전을 WBF fuse (도메인 일치).
Roboflow ensemble이 효과 없던 것과 대조: 같은 도메인 학습 버전끼리는 놓침 보강 가능성.

지표: class-agnostic Recall(GT 놓침) + FP. before(현 배포 단일) vs after(버전 WBF).
실행: backend/venv/Scripts/python.exe backend/training/eval/self_ensemble_eval.py --target THERMAL
"""
from __future__ import annotations
import argparse, glob, json, sys, time
from pathlib import Path
import cv2, numpy as np, yaml
from ensemble_boxes import weighted_boxes_fusion

ROOT = Path(__file__).resolve().parents[3]
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# target → [현배포(primary), 보조버전...] + dataset + imgsz
GROUPS = {
    "THERMAL": {
        "ckpts": [
            ("thermal_yolo", 960),       # v11 현배포 (고정입력 960)
            ("thermal_v3", 640),
            ("thermal_yolo_prev", 640),  # v1
        ],
        "data": ROOT / "backend/training/datasets/thermal_yolo/data.yaml",
    },
    "M1": {
        "ckpts": [("m1_yolo_structural", 640), ("m1_structural_v3", 640), ("m1_structural_v4s", 640)],
        "data": ROOT / "backend/training/datasets/structural/data.yaml",
    },
    "M3": {
        "ckpts": [("m3_yolo_floor_window", 960), ("m3_floor_window_v3", 640), ("m3_floor_window_v4s", 640)],
        "data": ROOT / "backend/training/datasets/floor_window/data.yaml",
    },
}

IOU_THR = 0.55
SKIP_BOX_THR = 0.0001  # Recall 우선
CONF = 0.05
MATCH_IOU = 0.5
W_PRIMARY = 2.0
W_AUX = 1.0


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


def recall_fp(preds_all, gts_all, miou=MATCH_IOU):
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
    return {"total_gt":tg,"matched_gt":mg,"recall":mg/(tg+1e-9),
            "total_pred":tp_pred,"fp":fp,"precision":(tp_pred-fp)/(tp_pred+1e-9)}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--target",required=True,choices=list(GROUPS.keys()))
    ap.add_argument("--max-images",type=int,default=200)
    args=ap.parse_args()

    g=GROUPS[args.target]
    cfg=yaml.safe_load(g["data"].read_text(encoding="utf-8"))
    base=g["data"].parent
    tdir=base/"images"/"test"
    if not tdir.exists(): tdir=base/"images"/"val"
    imgs=sorted(tdir.glob("*.jpg"))+sorted(tdir.glob("*.png"))
    if args.max_images and len(imgs)>args.max_images: imgs=imgs[:args.max_images]
    print(f"=== Self-ensemble (target={args.target}, {len(imgs)} imgs, {len(g['ckpts'])} versions) ===",flush=True)

    gts=[]; shapes=[]
    for p in imgs:
        im=cv2.imread(str(p))
        if im is None: gts.append([]); shapes.append((0,0)); continue
        h,w=im.shape[:2]; shapes.append((w,h))
        gts.append(load_gts(base/"labels"/p.parent.name/(p.stem+".txt"),w,h))

    from ultralytics import YOLO
    # 각 버전 추론 → normalized boxes per image
    per_ver=[]  # list over versions: [(boxes_norm, scores) per image]
    for name,imgsz in g["ckpts"]:
        onnx=ROOT/f"backend/models_weights/{name}.onnx"
        if not onnx.exists(): print(f"  skip {name}: missing",flush=True); per_ver.append(None); continue
        try: model=YOLO(str(onnx),task="detect")
        except Exception as e: print(f"  skip {name}: {str(e)[:60]}",flush=True); per_ver.append(None); continue
        t0=time.time(); per_img=[]
        for p,(w,h) in zip(imgs,shapes):
            try:
                r=model(str(p),imgsz=imgsz,conf=CONF,iou=0.6,verbose=False)[0]
                if r.boxes is None or len(r.boxes)==0: per_img.append(([],[])); continue
                xy=r.boxes.xyxy.cpu().numpy(); cf=r.boxes.conf.cpu().numpy()
                nb=(xy/np.array([w,h,w,h])).clip(0,1)
                per_img.append((nb.tolist(),cf.tolist()))
            except Exception: per_img.append(([],[]))
        per_ver.append(per_img)
        print(f"  {name}(imgsz{imgsz}): {sum(len(x[1]) for x in per_img)} dets, {time.time()-t0:.1f}s",flush=True)

    # before = primary 단독 (per_ver[0])
    primary=per_ver[0]
    before_px=[]
    for (b,s),(w,h) in zip(primary,shapes):
        before_px.append([[bx[0]*w,bx[1]*h,bx[2]*w,bx[3]*h,sc] for bx,sc in zip(b,s)])
    before=recall_fp(before_px,gts)

    # after = 모든 버전 WBF
    weights=[W_PRIMARY]+[W_AUX]*(len(per_ver)-1)
    after_px=[]
    for i,(w,h) in enumerate(shapes):
        bl,sl,ll,wl=[],[],[],[]
        for vi,pv in enumerate(per_ver):
            if pv is None: continue
            b,s=pv[i]
            if b: bl.append(b); sl.append(s); ll.append([0]*len(s)); wl.append(weights[vi])
        if not bl: after_px.append([]); continue
        fb,fs,fl=weighted_boxes_fusion(bl,sl,ll,weights=wl,iou_thr=IOU_THR,skip_box_thr=SKIP_BOX_THR)
        after_px.append([[bx[0]*w,bx[1]*h,bx[2]*w,bx[3]*h,sc] for bx,sc in zip(fb,fs)])
    after=recall_fp(after_px,gts)

    dR=after["recall"]-before["recall"]; dFP=after["fp"]-before["fp"]
    print("\n----- 결과 (class-agnostic, GT IoU>=0.5) -----",flush=True)
    print(f"  BEFORE primary 단독: R={before['recall']:.4f} ({before['matched_gt']}/{before['total_gt']}) FP={before['fp']} P={before['precision']:.4f}",flush=True)
    print(f"  AFTER  버전 WBF    : R={after['recall']:.4f} ({after['matched_gt']}/{after['total_gt']}) FP={after['fp']} P={after['precision']:.4f}",flush=True)
    print(f"  dR={dR:+.4f} dFP={dFP:+d}",flush=True)

    out_dir=Path(__file__).parent/"results"; out_dir.mkdir(exist_ok=True)
    ts=time.strftime("%Y%m%d_%H%M%S")
    res=out_dir/f"self_ensemble_{args.target}_{ts}.json"
    res.write_text(json.dumps({"target":args.target,"n_images":len(imgs),
        "ckpts":[n for n,_ in g["ckpts"]],"before":before,"after":after,
        "delta_recall":dR,"delta_fp":dFP},indent=2,ensure_ascii=False),encoding="utf-8")
    print(f"\n결과: {res}",flush=True)
    return 0


if __name__=="__main__":
    sys.exit(main())
