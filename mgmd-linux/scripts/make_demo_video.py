"""scripts/make_demo_video.py — 生成带检测标注的演示视频（SPEC §8.31）。

用途：没有真实无人机可拍时，用 ARD-MAV 数据生成"像实时追踪"的演示视频。
这是**离线的**，但输出与 live_track.py 的 --display 一致（同样的 HUD、轨迹、框）。

用法：
    python3 scripts/make_demo_video.py --video phantom70 --frames 600
    python3 scripts/make_demo_video.py --video phantom63 --frames 300 --preset mvdt
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import warnings; warnings.filterwarnings("ignore")
import cv2, numpy as np
from mgmd import config as C
from mgmd.mfe import MFEConfig
from scripts.live_track import LiveTracker, draw

ARD = C.ARD_VIDEOS

def main(argv=None):
    ap=argparse.ArgumentParser()
    ap.add_argument("--video", default="phantom70")
    ap.add_argument("--frames", type=int, default=600)
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--preset", default="recommended", choices=["paper","recommended","mvdt"])
    ap.add_argument("--threshold", type=int, default=5)
    ap.add_argument("--min-area", type=int, default=15)
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--lad-batch", type=int, default=8)
    ap.add_argument("--lad-trt-engine", default=None)
    ap.add_argument("--out", default=None)
    args=ap.parse_args(argv)

    cfg=MFEConfig(k=args.k, mode="3frame_or")
    if args.preset=="recommended": cfg.apply_recommended()
    elif args.preset=="mvdt":
        cfg.apply_recommended(); cfg.threshold,cfg.min_area,cfg.k=2,5,2
    if args.threshold!=5: cfg.threshold=args.threshold
    if args.min_area!=15: cfg.min_area=args.min_area
    cfg.k=args.k

    src=ARD/f"{args.video}.mp4"
    if not src.is_file():
        print(f"[mgmd] E_DATA_MISSING: {src}", file=sys.stderr); return C.EXIT_DATA
    cap=cv2.VideoCapture(str(src))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0,args.start-1))
    w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out=Path(args.out) if args.out else (C.FIGURES/f"demo_{args.video}_{args.frames}.mp4")
    out.parent.mkdir(parents=True, exist_ok=True)
    vw=cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w,h+58))
    if not vw.isOpened():
        print(f"[mgmd] 无法创建 {out}", file=sys.stderr); cap.release(); return C.EXIT_RUNTIME

    tr=LiveTracker(cfg,args.device,args.lad_batch,args.lad_trt_engine)
    print(f"[mgmd] {args.video} 从帧 {args.start}，{args.frames} 帧，preset={args.preset}")
    hits=0; n=0; t0=time.perf_counter()
    while n<args.frames:
        ok,fr=cap.read()
        if not ok: break
        r=tr.process(fr)
        vw.write(draw(fr,r,tr,False,False))
        if r["boxes"]: hits+=1
        n+=1
        if n%100==0:
            print(f"    {n}/{args.frames} 帧, 检出 {hits}, {tr.fps():.1f} FPS", flush=True)
    vw.release(); cap.release()
    dt=time.perf_counter()-t0
    print(f"\n[mgmd] 输出 {out} ({out.stat().st_size/1e6:.1f} MB)")
    print(f"[mgmd] {n} 帧，检出 {hits} ({100*hits/max(1,n):.1f}%)，"
          f"处理 {dt:.1f}s ({n/dt:.1f} FPS)")
    return C.EXIT_OK

if __name__=="__main__":
    raise SystemExit(main())
