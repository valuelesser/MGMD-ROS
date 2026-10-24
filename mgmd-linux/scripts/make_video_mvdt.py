"""scripts/make_video_mvdt.py — MVDT 四序列可视化视频（SPEC §8.27）。

MVDT 是点 GT，画法是：
    绿色十字/小圈 = GT 点
    红色粗框     = 最终检测
    黄色细框     = MFE 原始候选

右侧放大图用于看清像素级目标。HUD 显示帧号/距离/命中状态。
"""
from __future__ import annotations
import argparse, csv, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import warnings; warnings.filterwarnings("ignore")
import cv2, numpy as np

from mgmd import config as C
from mgmd import viz
from mgmd.mfe import MFEConfig, enhance
from mgmd.lac import LacClassifier, crop_box
from mgmd.lad import LadDetector

ROOT = Path("/home/gaoduan/tracking/data_raw/mvdt")
GT_DIR = Path("/home/gaoduan/毕业设计/tracking/data_derived/gt")
SEQS = {"D1": ("D1/cam2/cam2.mp4", "D1_cam2.csv", 3884),
        "D2": ("D2/cam1/cam1.mp4", "D2_cam1.csv", 500),
        "D3": ("D3/cam3/cam3.mp4", "D3_cam3.csv", 3884),
        "D4": ("D4/cam3/cam3.mp4", "D4_cam3.csv", 500)}

GREEN=(0,255,0); RED=(0,0,255); YELLOW=(0,255,255); CYAN=(255,255,0); WHITE=(255,255,255)

def load_gt(csvf):
    out={}
    with open(GT_DIR/csvf, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                fr=int(r["frame"]); vis=int(r["visible"] or 0)
                cx=float(r["cx"]) if r["cx"] else None
                cy=float(r["cy"]) if r["cy"] else None
            except (KeyError,ValueError): continue
            out[fr]=(cx,cy,vis)
    return out

def draw_hud(canvas, seq, t, dist, hit, info, n_cand, n_det):
    h,w=canvas.shape[:2]
    bar=np.full((78,w,3), 28, np.uint8)
    cv2.putText(bar, f"{seq}  frame {t}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, WHITE, 2, cv2.LINE_AA)
    status="HIT" if hit else "MISS"
    cv2.putText(bar, status, (250,30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, GREEN if hit else RED, 2, cv2.LINE_AA)
    dtxt=f"dist={dist:.1f}px" if dist is not None else "dist=-"
    cv2.putText(bar, dtxt, (340,30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, YELLOW, 2, cv2.LINE_AA)
    cv2.putText(bar, f"CAND={n_cand}  DET={n_det}", (520,30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, CYAN, 2, cv2.LINE_AA)
    cv2.line(bar,(10,48),(28,48),GREEN,2); cv2.putText(bar,"GT point",(34,53),cv2.FONT_HERSHEY_SIMPLEX,0.45,WHITE,1,cv2.LINE_AA)
    cv2.rectangle(bar,(120,42),(136,56),RED,-1); cv2.putText(bar,"detection",(142,53),cv2.FONT_HERSHEY_SIMPLEX,0.45,WHITE,1,cv2.LINE_AA)
    cv2.rectangle(bar,(230,42),(246,56),YELLOW,1); cv2.putText(bar,"MFE cand",(252,53),cv2.FONT_HERSHEY_SIMPLEX,0.45,WHITE,1,cv2.LINE_AA)
    cv2.putText(bar, info, (350,53), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (170,170,170), 1, cv2.LINE_AA)
    return np.vstack([bar, canvas])

def main(argv=None):
    ap=argparse.ArgumentParser()
    ap.add_argument("--seqs", nargs="+", default=["D1","D2","D3","D4"])
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--threshold", type=int, default=2)
    ap.add_argument("--min-area", type=int, default=5)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--outdir", default=str(C.FIGURES))
    args=ap.parse_args(argv)

    cfg=MFEConfig(k=args.k, mode="3frame_or").apply_recommended()
    cfg.threshold=args.threshold; cfg.min_area=args.min_area; cfg.k=args.k
    clf=LacClassifier(device=args.device); lad=LadDetector(device=args.device)
    k=cfg.k
    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    for seq in args.seqs:
        mp4, csvf, start = SEQS[seq]
        gt=load_gt(csvf)
        cap=cv2.VideoCapture(str(ROOT/mp4))
        if not cap.isOpened():
            print(f"[{seq}] 打不开"); continue
        total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)

        # 下采样比例：统一输出到约 960 宽
        w0=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        out_w=960; scale=out_w/w0
        out_h=int(int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))*scale)
        zoom=420
        out=f"{args.outdir}/{seq}_mvdt_viz.mp4"
        vw=cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), args.fps,
                           (out_w+zoom, out_h+78))
        if not vw.isOpened():
            print(f"[{seq}] 建视频失败"); cap.release(); continue

        buf={}; n=0; nvis=0; nhit=0; t0=time.perf_counter()
        while n < args.frames + 2*k:
            ok,fr=cap.read()
            if not ok: break
            idx=start+n; n+=1
            buf[idx]=fr
            t=idx-k
            if t<start+k or (t-k) not in buf or (t+k) not in buf: continue

            res=enhance(buf[t-k],buf[t],buf[t+k],cfg,t)
            cand=[(float(x),float(y),float(x+w),float(y+h)) for (x,y,w,h) in res.boxes]
            pool=cand
            if pool:
                crops=[crop_box(buf[t],b) for b in pool]
                pairs=[(b,c) for b,c in zip(pool,crops) if c is not None]
                if pairs:
                    ps=clf.mav_prob_batch([c for _,c in pairs])
                    pool=[b for (b,_),p in zip(pairs,ps) if p>=0.5]
                else: pool=[]
            if pool:
                rbs=lad.refine_batch(buf[t],[((b[0]+b[2])/2,(b[1]+b[3])/2) for b in pool],batch_size=8)
                pool=[rb.box for rb in rbs if rb is not None and rb.conf>=0.5]

            g=gt.get(t); gpt=None; dist=None
            if g and g[2]==1 and g[0] is not None:
                gpt=(g[0],g[1]); nvis+=1
                if pool:
                    dist=min((( (b[0]+b[2])/2-gpt[0])**2+((b[1]+b[3])/2-gpt[1])**2)**0.5 for b in pool)
                    if dist<=50: nhit+=1
            hit = dist is not None and dist<=50

            # 全图（缩放）
            full=cv2.resize(buf[t],(out_w,out_h),interpolation=cv2.INTER_AREA)
            def sc(b): return (b[0]*scale,b[1]*scale,b[2]*scale,b[3]*scale)
            for b in cand: full=viz.draw_boxes(full,[sc(b)],YELLOW,1,None)
            for b in pool: full=viz.draw_boxes(full,[sc(b)],RED,2,None)
            if gpt:
                p=(int(gpt[0]*scale),int(gpt[1]*scale))
                cv2.drawMarker(full,p,GREEN,cv2.MARKER_CROSS,14,2)
                cv2.circle(full,p,8,GREEN,1)

            # 放大
            patch=100
            cxy=gpt if gpt else ((pool[0][0]+pool[0][2])/2,(pool[0][1]+pool[0][3])/2) if pool else (w0/2,1080/2)
            cx,cy=int(cxy[0]),int(cxy[1])
            x1=max(0,min(cx-patch//2,max(0,w0-patch))); y1=max(0,min(cy-patch//2,max(0,int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))-patch)))
            crop=buf[t][y1:y1+patch, x1:x1+patch]
            if crop.size:
                sc2=zoom/max(1,patch); z=cv2.resize(crop,(zoom,zoom),interpolation=cv2.INTER_NEAREST)
                def zc(b): return ((b[0]-x1)*sc2,(b[1]-y1)*sc2,(b[2]-x1)*sc2,(b[3]-y1)*sc2)
                for b in cand:
                    if b[2]>x1 and b[0]<x1+patch and b[3]>y1 and b[1]<y1+patch:
                        z=viz.draw_boxes(z,[zc(b)],YELLOW,1,None)
                for b in pool:
                    if b[2]>x1 and b[0]<x1+patch and b[3]>y1 and b[1]<y1+patch:
                        z=viz.draw_boxes(z,[zc(b)],RED,2,None)
                if gpt and x1<=gpt[0]<x1+patch and y1<=gpt[1]<y1+patch:
                    p=(int((gpt[0]-x1)*sc2),int((gpt[1]-y1)*sc2))
                    cv2.drawMarker(z,p,GREEN,cv2.MARKER_CROSS,18,2)
                    cv2.circle(z,p,10,GREEN,1)
            else:
                z=np.zeros((zoom,zoom,3),np.uint8)
            if z.shape[0]<out_h: z=cv2.copyMakeBorder(z,0,out_h-z.shape[0],0,0,cv2.BORDER_CONSTANT,value=(20,20,20))
            combo=np.hstack([full,z[:out_h]])
            info=f"th={cfg.threshold} k={cfg.k} median={cfg.median_blur}"
            vw.write(draw_hud(combo,seq,t,dist,hit,info,len(cand),len(pool)))
            for kk in [b for b in buf if b< t-2*k-2]: del buf[kk]

        vw.release(); cap.release()
        dt=time.perf_counter()-t0
        print(f"[{seq}] {n}帧 命中{nhit}/{nvis}={100*nhit/max(1,nvis):.1f}%  "
              f"{dt/max(1,n)*1000:.0f}ms/帧 → {out}", flush=True)

if __name__=="__main__":
    raise SystemExit(main())
