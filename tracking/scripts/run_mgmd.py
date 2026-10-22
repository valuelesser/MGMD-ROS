"""scripts/run_mgmd.py — MGMD 单序列端到端运行（SPEC §4.2；issue-018 骨架 / issue-022 完整实现）。

用法：
    python scripts/run_mgmd.py --config configs/mgmd_D1.yaml \
        --gt data_derived/gt/D1_cam2.csv --out outputs/runs_mgmd/D1_full \
        [--limit N] [--modules mfe,tf,lac,lad] [--backend numpy]

错误格式 ``[mgmd] <code>: <msg> → <remediation>``；
退出码：0 成功 / 2 配置非法 / 3 数据缺失 / 4 运行时 / 130 中断（SPEC §4.3）。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mgmd.config import (EXIT_CONFIG_INVALID, EXIT_DATA, EXIT_INTERRUPT, EXIT_OK,
                         EXIT_RUNTIME, MgmdError, load_config)  # noqa: E402
from mgmd import pipeline  # noqa: E402
from spa2d import records as R  # noqa: E402


def _err(code: str, msg: str, remediation: str) -> None:
    print(f"[mgmd] {code}: {msg} → {remediation}", file=sys.stderr)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="run_mgmd", description="MGMD × MVDT 端到端运行（SPEC §4.2）")
    parser.add_argument("--config", required=True, help="configs/mgmd_*.yaml")
    parser.add_argument("--gt", required=True, help="data_derived/gt/D*_cam*.csv")
    parser.add_argument("--out", required=True, help="输出目录，如 outputs/runs_mgmd/D1_full")
    parser.add_argument("--limit", type=int, default=None, help="帧上限（覆盖 runtime.limit）")
    parser.add_argument("--start", type=int, default=0, help="起始帧（前 N 帧仅解码不产出）")
    parser.add_argument("--modules", default="mfe,tf,lac,lad",
                        help="模块子集（消融四档：mfe / mfe,tf / mfe,tf,lac / mfe,tf,lac,lad）")
    parser.add_argument("--backend", default=None, help="覆盖 config backend（记录标记）")
    parser.add_argument("--viz", action="store_true", help="启用实时 OpenCV 可视化窗口")
    parser.add_argument("--viz-every", type=int, default=1,
                        help="每 N 帧渲染一次（默认 1）")
    parser.add_argument("--viz-buffer", type=int, default=200,
                        help="回看缓冲帧数（10–2000，默认 200）")
    parser.add_argument("--viz-width", type=int, default=1280,
                        help="显示帧宽度（320–3840，默认 1280）")
    parser.add_argument("--viz-scan-max", type=int, default=5000,
                        help="向前扫描失败帧上限（默认 5000）")
    args = parser.parse_args(argv)

    if args.viz_every < 1:
        _err("E_CONFIG_INVALID", f"--viz-every 必须 ≥ 1，实际 {args.viz_every}",
             "修正 --viz-every")
        return EXIT_CONFIG_INVALID
    if not (10 <= args.viz_buffer <= 2000):
        _err("E_CONFIG_INVALID",
             f"--viz-buffer 必须在 10..2000，实际 {args.viz_buffer}",
             "修正 --viz-buffer")
        return EXIT_CONFIG_INVALID
    if not (320 <= args.viz_width <= 3840):
        _err("E_CONFIG_INVALID",
             f"--viz-width 必须在 320..3840，实际 {args.viz_width}",
             "修正 --viz-width")
        return EXIT_CONFIG_INVALID
    if args.viz_scan_max < 1:
        _err("E_CONFIG_INVALID",
             f"--viz-scan-max 必须 ≥ 1，实际 {args.viz_scan_max}",
             "修正 --viz-scan-max")
        return EXIT_CONFIG_INVALID

    try:
        cfg = load_config(args.config)
        modules = pipeline.parse_modules(args.modules)
    except MgmdError as exc:
        print(exc.format(), file=sys.stderr)
        return exc.exit_code

    gt_path = Path(args.gt)
    if not gt_path.is_file():
        _err("E_DATA_MISSING", f"GT 文件不存在: {gt_path}",
             "先运行 scripts/make_derived_gt.py 生成 data_derived/gt/*.csv")
        return EXIT_DATA

    if args.limit is not None and args.limit < 1:
        _err("E_CONFIG_INVALID", f"--limit 必须 ≥ 1，实际 {args.limit}", "修正 --limit")
        return EXIT_CONFIG_INVALID
    limit = args.limit if args.limit is not None else cfg.runtime.limit
    backend = args.backend or cfg.backend

    # 外观模块启用时才检查权重存在性（懒加载原则，FR-11）
    for mod, weights in (("lac", cfg.lac.weights), ("lad", cfg.lad.weights)):
        if mod in modules and not Path(weights).is_file():
            _err("E_WEIGHTS_MISSING", f"{mod.upper()} 权重不存在: {weights}",
                 "bash scripts/fetch_mgmd_weights.sh 下载并校验官方权重")
            return EXIT_DATA

    visualizer = None
    try:
        if args.viz:
            try:
                from mgmd.viz import Visualizer
            except ImportError as exc:  # pragma: no cover - opencv 在 requirements 中
                _err("E_DEP_MISSING", f"可视化依赖缺失: {exc.name}",
                     "安装 opencv-python（env/requirements.txt）")
                return EXIT_RUNTIME
            try:
                gt_frames = R.read_gt_csv(gt_path)
            except (ValueError, OSError, KeyError) as exc:
                _err("E_DATA_MISSING", f"GT 文件不可读或列不匹配: {gt_path}: {exc}",
                     "先运行 scripts/make_derived_gt.py 生成 data_derived/gt/*.csv")
                return EXIT_DATA
            n_total = pipeline.frame_count(cfg)
            viz_start = max(0, min(args.start, max(0, n_total - 1)))
            n_frames = (min(limit, n_total - viz_start) if limit
                        else max(0, n_total - viz_start))
            try:
                visualizer = Visualizer(
                    cfg, modules, gt_frames, total=max(0, n_frames),
                    start=viz_start, every=args.viz_every,
                    buffer=args.viz_buffer, width=args.viz_width,
                    scan_max=args.viz_scan_max)
            except RuntimeError as exc:
                _err("E_VIZ_UNAVAILABLE", str(exc),
                     "确认 DISPLAY/WSLg 或 X server 可用；或去掉 --viz 重跑")
                return EXIT_RUNTIME
        rc = pipeline.run(cfg, modules, gt_path, Path(args.out), limit, backend,
                          start=args.start,
                          on_frame=visualizer.update if visualizer is not None else None)
        if visualizer is not None:
            s = visualizer.summary()
            print(f"[mgmd] viz: viewed={s['viewed']} jumps={s['jumps']} "
                  f"fp={s['fp']} miss={s['miss']} aborted={s['aborted']}")
        return rc
    except MgmdError as exc:
        print(exc.format(), file=sys.stderr)
        return exc.exit_code
    except NotImplementedError as exc:
        _err("E_NOT_IMPLEMENTED", str(exc),
             "按 .autoresearch/issues/ 依赖顺序推进（issue-019..022）")
        return EXIT_RUNTIME
    except KeyboardInterrupt:
        _err("E_INTERRUPT", "用户中断", "重新运行；已写出部分不受影响")
        return EXIT_INTERRUPT
    except Exception as exc:  # noqa: BLE001
        name, msg = type(exc).__name__, str(exc)
        if "OutOfMemory" in name or "out of memory" in msg.lower():
            _err("E_RUNTIME", f"显存不足: {name}: {msg}",
                 "调小 config runtime.downscale（如 0.5→0.33）后重试（SPEC §8.3）")
            return EXIT_RUNTIME
        _err("E_RUNTIME", f"未处理异常: {name}: {msg}",
             "带 --limit 小样例复现后提 issue")
        return EXIT_RUNTIME
    finally:
        if visualizer is not None:
            visualizer.close()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]) if len(sys.argv) > 1 else main(["--help"]) or EXIT_OK)
