"""scripts/make_synthetic.py — 生成合成数据（SPEC §4.1 / §5.11）。

用法：
    python scripts/make_synthetic.py --suite sanity --out data_raw/synthetic [--seqs 30] [--seed 0]
    python scripts/make_synthetic.py --suite benchmark --out data_raw/synthetic [--seed 0]
可选：--frames N 覆盖帧数，--width/--height 覆盖画布，--overwrite 允许覆盖已有 manifest。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spa2d import synthetic  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="make_synthetic.py",
        description="生成确定性合成序列（sanity / benchmark）")
    parser.add_argument("--suite", choices=["sanity", "benchmark"], required=True)
    parser.add_argument("--out", required=True, help="输出目录，例如 data_raw/synthetic")
    parser.add_argument("--seqs", type=int, default=30, help="sanity 序列条数（默认 30）")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--frames", type=int, default=None, help="覆盖每条序列帧数")
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    canvas = None
    if args.width is not None or args.height is not None:
        if not args.width or not args.height:
            print("[spa2d] E_SYNTHETIC: --width 与 --height 必须同时提供", file=sys.stderr)
            return 2
        canvas = (args.height, args.width)

    try:
        manifest = synthetic.generate(
            args.suite, args.out, seqs=args.seqs, seed=args.seed,
            n_frames=args.frames, canvas=canvas, overwrite=args.overwrite)
    except synthetic.SyntheticError as exc:
        print(f"[spa2d] E_SYNTHETIC: {exc.message}", file=sys.stderr)
        return exc.exit_code

    total_frames = sum(s["n_frames"] for s in manifest["sequences"])
    print(f"suite={manifest['suite']} seed={manifest['seed']} "
          f"sequences={manifest['n_sequences']} frames={total_frames} out={args.out}")
    for seq in manifest["sequences"][:5]:
        print(f"  {seq['seq_id']}: w={seq['width_px']}px snr={seq['snr']} "
              f"v=({seq['velocity']['vx']},{seq['velocity']['vy']}) "
              f"{seq['resolution'][0]}x{seq['resolution'][1]} "
              f"neg={seq['negatives']}")
    if len(manifest["sequences"]) > 5:
        print(f"  ... (+{len(manifest['sequences']) - 5} more)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
