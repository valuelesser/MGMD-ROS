"""scripts/build_trt_engine.py — 构建 LAD 的 TensorRT engine（SPEC §8.28）。

TRT engine 绑定 GPU 架构与 TRT 版本，**不能跨机器移植**。
但导出脚本可随代码交付：目标机器一条命令重建（约 3 分钟）。

用法：
    python3 scripts/build_trt_engine.py                    # 默认 batch 1-16
    python3 scripts/build_trt_engine.py --max-batch 8
    python3 scripts/build_trt_engine.py --fp32             # 不用 FP16

产出：
    outputs/lad_trt_fp16.engine

注意（本项目踩过的坑）：
    1. 必须导出**单输出** ONNX（Detect 层返回 (cat(z,1), x)，多输出会让 TRT 建 4 个）
    2. 推理时输出缓冲必须 torch.zeros，torch.empty 会读到脏值
    3. 需设 LD_LIBRARY_PATH 指向 site-packages/tensorrt_cu12_libs 与 nvidia/*/lib
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn

from mgmd import config as C
from mgmd.lad import LadDetector


class _OnlyPred(nn.Module):
    """只返回 pred，丢弃 Detect 层的中间输出元组。"""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        o = self.m(x)
        return o[0] if isinstance(o, (tuple, list)) else o


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=None, help="LAD 权重路径")
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--min-batch", type=int, default=1)
    ap.add_argument("--max-batch", type=int, default=16)
    ap.add_argument("--fp32", action="store_true", help="禁用 FP16")
    ap.add_argument("--workspace-gb", type=float, default=2.0)
    ap.add_argument("--out", default=str(C.PROJECT_ROOT / "outputs" / "lad_trt_fp16.engine"))
    args = ap.parse_args(argv)

    try:
        import tensorrt as trt
    except ImportError:
        print("[mgmd] E_DEP_MISSING: tensorrt 未安装 → pip install tensorrt-cu12==10.7.0",
              file=sys.stderr)
        return C.EXIT_RUNTIME

    weights = Path(args.weights) if args.weights else (C.WEIGHTS_DIR / "yolov5s_GLAD-crop.pt")
    print(f"[1/4] 加载 PyTorch 模型: {weights.name}")
    lad = LadDetector(weights=weights, imgsz=args.imgsz)
    wrapped = _OnlyPred(lad.model).eval()

    onnx_path = C.PROJECT_ROOT / "outputs" / f"lad_{args.imgsz}_predonly.onnx"
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    bs = max(args.min_batch, 1)
    dummy = torch.randn(bs, 3, args.imgsz, args.imgsz).to(lad.device)
    print(f"[2/4] 导出单输出 ONNX (batch={bs}): {onnx_path.name}")
    t0 = time.perf_counter()
    with torch.no_grad():
        torch.onnx.export(
            wrapped, dummy, str(onnx_path), opset_version=12,
            input_names=["images"], output_names=["pred"],
            dynamic_axes={"images": {0: "batch"}, "pred": {0: "batch"}},
        )
    print(f"      完成 {time.perf_counter()-t0:.1f}s, {onnx_path.stat().st_size/1e6:.1f} MB")

    print(f"[3/4] 构建 TRT engine (batch {args.min_batch}–{args.max_batch}, "
          f"{'FP32' if args.fp32 else 'FP16'})")
    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    net = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(net, logger)
    if not parser.parse(onnx_path.read_bytes()):
        for i in range(parser.num_errors):
            print(f"      parse error: {parser.get_error(i)}", file=sys.stderr)
        return C.EXIT_RUNTIME

    cfg = builder.create_builder_config()
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(args.workspace_gb * (1 << 30)))
    if not args.fp32 and builder.platform_has_fast_fp16:
        cfg.set_flag(trt.BuilderFlag.FP16)

    prof = builder.create_optimization_profile()
    inp = net.get_input(0)
    s = args.imgsz
    prof.set_shape(inp.name, (args.min_batch, 3, s, s),
                   (min(8, args.max_batch), 3, s, s), (args.max_batch, 3, s, s))
    cfg.add_optimization_profile(prof)

    t0 = time.perf_counter()
    serialized = builder.build_serialized_network(net, cfg)
    if serialized is None:
        print("[mgmd] engine 构建失败", file=sys.stderr)
        return C.EXIT_RUNTIME
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(bytes(serialized))
    print(f"      完成 {time.perf_counter()-t0:.0f}s, {out.stat().st_size/1e6:.1f} MB")

    print(f"[4/4] 验证 engine 输出与 PyTorch 一致")
    eng = trt.Runtime(logger).deserialize_cuda_engine(out.read_bytes())
    ctx = eng.create_execution_context()
    names = [eng.get_tensor_name(i) for i in range(eng.num_io_tensors)]
    in_n = next(n for n in names if eng.get_tensor_mode(n) == trt.TensorIOMode.INPUT)
    out_n = next(n for n in names if eng.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT)

    x = torch.randn(bs, 3, s, s)
    with torch.no_grad():
        pt = wrapped(x.to(lad.device)).cpu().numpy()
    ctx.set_input_shape(in_n, tuple(x.shape))
    d_in = x.cuda()
    ctx.set_tensor_address(in_n, d_in.data_ptr())
    d_out = torch.zeros(tuple(ctx.get_tensor_shape(out_n)), dtype=torch.float32, device="cuda")
    ctx.set_tensor_address(out_n, d_out.data_ptr())
    ctx.execute_async_v3(stream_handle=torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    o = d_out.cpu().numpy()
    conf_err = float(abs(pt[..., 4].max() - o[..., 4].max()))
    print(f"      conf 最大误差 = {conf_err:.6f}  {'✅ 通过' if conf_err < 0.01 else '❌ 偏大'}")
    print(f"\n使用方式：LadDetector(trt_engine='{out}')")
    return C.EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
