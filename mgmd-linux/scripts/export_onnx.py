#!/usr/bin/env python3
"""scripts/export_onnx.py — 导出 LAC / LAD 的 ONNX 模型（SPEC-ROS §15.4）。

为什么必须有这个脚本：ONNX 模型原本散落在 `/home/gaoduan/tracking/data_raw/`，
换机器就没了。此脚本让模型**可从权重重新生成**，判据是导出后数值一致。

    # 在 3.11 + torch 环境执行（需要 torch 2.5 和 yolov5 源码）
    source scripts/env_setup.sh
    $PY scripts/export_onnx.py

产物（写入 `mgmd.config.WEIGHTS_DIR`）：
    lac.onnx                  242 KB，动态 batch
    yolov5s_GLAD-crop.onnx    28 MB，固定 [1,3,320,320]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mgmd import config as C


def export_lac(out: Path) -> None:
    """LAC：小 CNN，动态 batch。"""
    import torch
    from mgmd.lac import LacClassifier

    lac = LacClassifier(device="cpu")
    net = lac.net.eval()
    torch.onnx.export(
        net, torch.randn(1, 3, lac.input_size, lac.input_size), str(out),
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=12, dynamo=False,
    )
    print(f"  LAC -> {out}  ({out.stat().st_size / 1024:.0f} KB)")


def export_lad(out: Path, imgsz: int = 320) -> None:
    """LAD：YOLOv5s。用官方 export.py 的等价逻辑（含 onnxsim 简化）。

    注意：**ONNX 输出是原始预测 (1,6300,6)，不含 NMS**。
    NMS 由 `mgmd_onnx.LadOnnx.refine` 在 Python 侧完成（SPEC-ROS §13.2）。
    """
    from mgmd.lad import LadDetector, yolov5_dir, _inject_syspath
    _inject_syspath(yolov5_dir())
    import torch

    lad = LadDetector(device="cpu")
    net = lad.model.eval()
    torch.onnx.export(
        net, torch.randn(1, 3, imgsz, imgsz), str(out),
        input_names=["images"], output_names=["output"],
        opset_version=12, dynamo=False,
    )
    print(f"  LAD -> {out}  ({out.stat().st_size / 1024 / 1024:.1f} MB)")

    # onnxsim：把训练模式的输出头简化（不装也能用，只是更大更慢）
    try:
        import onnx
        from onnxsim import simplify
        m = onnx.load(str(out))
        ms, ok = simplify(m)
        if ok:
            onnx.save(ms, str(out))
            print(f"  LAD onnxsim 简化完成 ({out.stat().st_size / 1024 / 1024:.1f} MB)")
        else:
            print("  LAD onnxsim 简化失败（保留未简化模型）")
    except ImportError:
        print("  ⚠️ 未装 onnx-simplifier，跳过简化（pip install onnx-simplifier）")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(C.WEIGHTS_DIR))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    d = Path(args.out_dir)
    d.mkdir(parents=True, exist_ok=True)
    targets = [("lac.onnx", export_lac), ("yolov5s_GLAD-crop.onnx", export_lad)]

    print(f"导出目录: {d}")
    for name, fn in targets:
        p = d / name
        if p.exists() and not args.force:
            print(f"  {name} 已存在，跳过（--force 覆盖）")
            continue
        fn(p)
    print("\n完成。下一步验证：")
    print("  source scripts/env_setup_ros38.sh && $PY38 scripts/verify_onnx.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
