"""scripts/run_mvdt.py — 在 MVDT 上跑当前 pipeline（SPEC §8.26）。

MVDT 与 ARD-MAV 的差异（必须在报告中声明）：
    1. 静态地面相机（ARD-MAV 是机载运动相机）
    2. 点 GT（cx,cy），无框 → 用「检测框中心到 GT 点距离」匹配，非 IoU
    3. 分辨率/帧率各异（D1 50fps, D2 29fps, D3 25fps 1440p, D4 4K 21fps）
    4. D4 为 4K，需降采样

因此本结果不能与 ARD-MAV 的 F1 直接比较，只能看「方法在另一数据集上的可行性」。
"""
from __future__ import annotations
import argparse, csv, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import warnings; warnings.filterwarnings("ignore")
import cv2, numpy as np

from mgmd.mfe import MFEConfig
from mgmd.pipeline import parse_modules

ROOT = Path("/home/gaoduan/tracking/data_raw/mvdt")
GT_DIR = Path("/home/gaoduan/毕业设计/tracking/data_derived/gt")
SEQS = {"D1": ("D1/cam2/cam2.mp4", "D1_cam2.csv"),
        "D2": ("D2/cam1/cam1.mp4", "D2_cam1.csv"),
        "D3": ("D3/cam3/cam3.mp4", "D3_cam3.csv"),
        "D4": ("D4/cam3/cam3.mp4", "D4_cam3.csv")}
RADII = (10, 20, 50, 100)

def load_gt(csv_path):
    """返回 {frame(0基): (cx,cy,visible)}。"""
    out = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                fr = int(row["frame"])
                vis = int(row["visible"] or 0)
                cx = float(row["cx"]) if row["cx"] else None
                cy = float(row["cy"]) if row["cy"] else None
            except (KeyError, ValueError):
                continue
            out[fr] = (cx, cy, vis)
    return out

def run_seq(name, mods, cfg, downscale, limit, device, lad_batch, start=0):
    from mgmd.mfe import enhance
    from mgmd.lac import LacClassifier, crop_box
    from mgmd.lad import LadDetector

    mp4, csvf = SEQS[name]
    gt = load_gt(GT_DIR / csvf)
    clf = LacClassifier(device=device) if "lac" in mods else None
    lad = LadDetector(device=device) if "lad" in mods else None

    cap = cv2.VideoCapture(str(ROOT / mp4))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开 {mp4}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n = min(total, limit) if limit else total
    if start > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)   # 必须先 seek，否则 idx 与实际帧号错位

    buf = {}
    k = cfg.k
    hits = {R: 0 for R in RADII}
    nvis = 0
    nframes = 0
    t0 = time.perf_counter()

    for idx in range(start, n):
        ok, frame = cap.read()
        if not ok:
            break
        if downscale != 1.0:
            frame = cv2.resize(frame, None, fx=downscale, fy=downscale,
                               interpolation=cv2.INTER_AREA)
        buf[idx] = frame
        t = idx - k          # 中心帧：需 t-k 和 t+k 都已读入
        if t < k or (t - k) not in buf or (t + k) not in buf:
            continue
        res = enhance(buf[t-k], buf[t], buf[t+k] if cfg.needs_next else None, cfg, t)
        boxes = [(float(x), float(y), float(x+w), float(y+h)) for (x,y,w,h) in res.boxes]
        pool = boxes

        if clf is not None and pool:
            crops = [crop_box(buf[t], b) for b in pool]
            pairs = [(b, c) for b, c in zip(pool, crops) if c is not None]
            if pairs:
                ps = clf.mav_prob_batch([c for _, c in pairs])
                pool = [b for (b, _), p in zip(pairs, ps) if p >= 0.5]
            else:
                pool = []

        if lad is not None and pool:
            rbs = lad.refine_batch(buf[t], [((b[0]+b[2])/2, (b[1]+b[3])/2) for b in pool],
                                   batch_size=lad_batch)
            pool = [rb.box for rb in rbs if rb is not None and rb.conf >= 0.5]

        g = gt.get(t)
        nframes += 1
        if g and g[2] == 1 and g[0] is not None:
            nvis += 1
            gx, gy = g[0] * downscale, g[1] * downscale
            # 每个检测框中心到 GT 点的距离
            best = None
            for b in pool:
                cx, cy = (b[0]+b[2])/2, (b[1]+b[3])/2
                d = ((cx-gx)**2 + (cy-gy)**2) ** 0.5 / downscale
                best = d if best is None else min(best, d)
            if best is not None:
                for R in RADII:
                    if best <= R:
                        hits[R] += 1
        if nframes % 500 == 0:
            print(f"    {name}: {nframes}/{n} 帧, {time.perf_counter()-t0:.0f}s", flush=True)

    cap.release()
    dt = time.perf_counter() - t0
    return {"seq": name, "frames": nframes, "visible": nvis,
            "recall": {R: (hits[R]/nvis if nvis else 0.0) for R in RADII},
            "elapsed_s": round(dt,1), "ms_per_frame": round(dt/max(1,nframes)*1000,1),
            "total_video_frames": total}

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqs", nargs="+", default=["D1","D2","D3","D4"])
    ap.add_argument("--modules", default="mfe,lac,lad")
    ap.add_argument("--mfe-preset", default="recommended",
                    choices=["paper","official","recommended","mvdt",""])
    ap.add_argument("--threshold", type=int, default=None, help="覆盖二值化阈值")
    ap.add_argument("--k", type=int, default=1, help="帧间隔")
    ap.add_argument("--limit", type=int, default=600, help="每序列帧数上限")
    ap.add_argument("--start", type=int, default=0, help="起始帧（0基）")
    ap.add_argument("--downscale", type=float, default=1.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--lad-batch", type=int, default=8)
    ap.add_argument("--out", default="outputs/logs/mvdt_results.json")
    args = ap.parse_args(argv)

    mods = parse_modules(args.modules)
    cfg = MFEConfig(k=args.k, mode="3frame_or")
    if args.mfe_preset == "recommended": cfg.apply_recommended()
    elif args.mfe_preset == "official": cfg.apply_official()
    elif args.mfe_preset == "mvdt":
        # MVDT 专用（SPEC §8.26）：静态地面相机，无视差；
        # 帧间 diff 均值仅 0.43，ARD-MAV 的参数（th=5）会漏光
        cfg.apply_recommended()
        cfg.threshold = 2          # 实测 th=2,k=2 召回 98.2%
        cfg.min_area = 5           # 静态相机目标更小
        cfg.k = 2
    if args.threshold is not None:
        cfg.threshold = args.threshold
    cfg.k = args.k if args.k != 1 else cfg.k

    print(f"[config] preset={args.mfe_preset or 'paper'} modules={args.modules} "
          f"limit={args.limit} downscale={args.downscale} device={args.device}", flush=True)
    rows = []
    for s in args.seqs:
        ds = args.downscale
        if s == "D4" and ds == 1.0:
            ds = 0.5
            print(f"  [{s}] 4K 视频，自动降采样 0.5", flush=True)
        print(f"  [{s}] 开始...", flush=True)
        # D3 前 764 帧无目标，且可见帧分 40 段；用最长连续段 3884-4692（809帧）
        st = 3884 if s == "D3" else args.start
        r = run_seq(s, mods, cfg, ds, args.limit + st if args.limit else args.limit,
                    args.device, args.lad_batch, start=st)
        rows.append(r)
        rc = r["recall"]
        print(f"  [{s}] {r['frames']}帧/{r['visible']}可见  "
              f"R@10={rc[10]*100:.1f}% R@20={rc[20]*100:.1f}% "
              f"R@50={rc[50]*100:.1f}% R@100={rc[100]*100:.1f}%  "
              f"{r['ms_per_frame']:.0f}ms/帧", flush=True)

    print("\n" + "="*76)
    print(f"{'seq':<6}{'帧数':>7}{'可见':>7}{'R@10':>8}{'R@20':>8}{'R@50':>8}{'R@100':>8}{'ms/帧':>8}")
    print("-"*76)
    for r in rows:
        rc = r["recall"]
        print(f"{r['seq']:<6}{r['frames']:>7}{r['visible']:>7}"
              f"{rc[10]*100:>7.1f}%{rc[20]*100:>7.1f}%{rc[50]*100:>7.1f}%{rc[100]*100:>7.1f}%"
              f"{r['ms_per_frame']:>8.0f}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"config":{"preset":args.mfe_preset,
        "modules":args.modules,"limit":args.limit,"device":args.device},
        "rows":rows}, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    print(f"\n已写入 {args.out}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
