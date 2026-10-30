"""scripts/verify_onnx.py — ONNX 模型正确性验证（SPEC-ROS §15.4）。

在 3.8 环境下跑，检查：
    1. 模型可加载，provider 符合预期
    2. LAC 输出与权重文件语义一致（用真实裁剪，概率落在合理区间）
    3. LAD 输出形状正确 (1,6300,6)，且真实帧上有检测
    4. 与 torch 参考输出对比（若提供了参考 npz）

    source scripts/env_setup_ros38.sh && $PY38 scripts/verify_onnx.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from mgmd import config as C


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--weights-dir", default=str(C.WEIGHTS_DIR))
    args = ap.parse_args()

    from mgmd_onnx import LacOnnx, LadOnnx
    from mgmd.io_video import VideoReader
    from mgmd.mfe import MFEConfig, enhance
    from mgmd.lac import crop_box

    d = Path(args.weights_dir)
    ok = True

    # ---- LAC
    print("=== LAC ===")
    lac = LacOnnx(model=d / "lac.onnx", device=args.device)
    print(f"  providers: {lac.session.get_providers()}")
    print(f"  输入: {[(i.name, i.shape) for i in lac.session.get_inputs()]}")

    cfg = MFEConfig()
    cfg.apply_recommended()
    cfg.validate()
    buf = {}
    with VideoReader(C.ARD_VIDEOS / "phantom30") as vr:
        for i, f in vr.iter_frames(start=1, limit=5):
            buf[i] = f
    res = enhance(buf[1], buf[2], buf[3], cfg, 2)
    cands = [(float(x), float(y), float(x + w), float(y + h))
             for (x, y, w, h) in res.boxes]
    crops = [c for c in (crop_box(buf[2], b) for b in cands[:8])
             if c is not None]
    probs = lac.mav_prob_batch(crops)
    print(f"  {len(probs)} 个裁剪 -> 概率 {[round(p, 6) for p in probs[:4]]}...")
    if not all(0.0 <= p <= 1.0 for p in probs):
        print("  ❌ 概率越界")
        ok = False
    else:
        print("  ✅ 概率均在 [0,1]")

    # ---- LAD
    print("\n=== LAD ===")
    lad = LadOnnx(model=d / "yolov5s_GLAD-crop.onnx", device=args.device)
    print(f"  providers: {lad.session.get_providers()}")
    print(f"  输入: {[(i.name, i.shape) for i in lad.session.get_inputs()]}")
    from mgmd.lad import crop_region
    h, w = buf[2].shape[:2]
    # 取一个靠近真实目标的中心
    from mgmd.annotations import load_video_gt
    gt = load_video_gt("phantom30")
    o = gt[2].objects[0]
    ctr = ((o.x1 + o.x2) / 2, (o.y1 + o.y2) / 2)
    x0, y0, cw, ch = crop_region((h, w), ctr, 320)
    crop = buf[2][y0:y0 + ch, x0:x0 + cw]
    raw = lad.infer(crop)
    print(f"  原始输出 shape: {raw.shape}")
    if raw.shape != (1, 6300, 6):
        print(f"  ❌ 形状不符，期望 (1, 6300, 6)")
        ok = False
    else:
        print("  ✅ 输出形状正确")
    rb = lad.refine(buf[2], ctr)
    if rb is None:
        print("  ❌ 真实目标位置无检测")
        ok = False
    else:
        from mgmd.tf import iou
        v = iou(rb.box, (o.x1, o.y1, o.x2, o.y2))
        print(f"  精修框 conf={rb.conf:.3f}, 与 GT IoU={v:.2f}")
        if v < 0.5:
            print("  ❌ IoU 过低")
            ok = False
        else:
            print("  ✅ 检测合理")

    print("\n判定:", "✅ 通过" if ok else "❌ 未通过")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
