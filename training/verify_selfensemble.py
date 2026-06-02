"""자가앙상블 최종 검증 — 운영 파이프라인 tier1(단독) vs tier3(자가앙상블 WBF). 결과 파일 기록."""
import sys, glob, os
sys.path.insert(0, '.')
import cv2
from app.services.inference_pipeline_20 import pipeline20

def iou(a, b):
    x1=max(a[0],b[0]);y1=max(a[1],b[1]);x2=min(a[2],b[2]);y2=min(a[3],b[3])
    inter=max(0,x2-x1)*max(0,y2-y1)
    return inter/((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter+1e-6)

def load_gt(lp, w, h):
    out=[]
    if not os.path.exists(lp): return out
    for line in open(lp, encoding='utf-8', errors='ignore'):
        p=line.split()
        if len(p)<5: continue
        cx,cy,bw,bh=map(float,p[1:5])
        out.append([(cx-bw/2)*w,(cy-bh/2)*h,(cx+bw/2)*w,(cy+bh/2)*h])
    return out

def evalset(runfn, base, n=150):
    imgs=sorted(glob.glob(base+'/images/test/*.jpg'))[:n]
    tg=mg=fp=0
    for ip in imgs:
        im=cv2.imread(ip)
        if im is None: continue
        h,w=im.shape[:2]
        gts=load_gt(os.path.join(base,'labels/test',os.path.splitext(os.path.basename(ip))[0]+'.txt'),w,h)
        preds=[d['bbox_xyxy'] for d in runfn(im)]
        tg+=len(gts); used=[False]*len(gts)
        for pb in preds:
            best,bj=0,-1
            for j,gb in enumerate(gts):
                if used[j]: continue
                v=iou(pb,gb)
                if v>best: best,bj=v,j
            if best>=0.5 and bj>=0: used[bj]=True
            else: fp+=1
        mg+=sum(used)
    return mg,tg,fp

def main():
    pipeline20.load_models()
    out=[]
    for tag,base,fn1,fn3 in [
      ('M1','training/datasets/structural', lambda im:pipeline20._run_m1(im,tier=1), lambda im:pipeline20._run_m1(im,tier=3)),
      ('M3','training/datasets/floor_window', lambda im:pipeline20._run_m3(im,tier=1), lambda im:pipeline20._run_m3(im,tier=3)),
    ]:
        m1,t1,f1=evalset(fn1,base); m3,t3,f3=evalset(fn3,base)
        r1=m1/(t1+1e-9); r3=m3/(t3+1e-9)
        out.append('%s SOLO recall=%.3f(%d/%d) FP=%d | ENSEMBLE recall=%.3f(%d/%d) FP=%d | dRecall=%+.3f dFP=%+d'%(
            tag,r1,m1,t1,f1, r3,m3,t3,f3, r3-r1, f3-f1))
    open('training/_selfens_result.txt','w',encoding='utf-8').write('\n'.join(out))
    print('DONE')

if __name__=='__main__':
    main()
