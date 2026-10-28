#!/usr/bin/env bash
# scripts/pack_deliverables.sh — 演示视频单独打包（SPEC §0 V6）。
#
# 用户决策（2026-10-28）：演示视频不进 git，改为打包到 deliverables/ 单独分发。
# 本脚本列出所有演示视频、生成清单（含生成命令），并打成 tar.gz。
#
# 用法：
#   bash scripts/pack_deliverables.sh            # 打包
#   bash scripts/pack_deliverables.sh --list     # 只列清单，不打包
set -euo pipefail

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FIG="$PROJECT/outputs/figures"
OUTDIR="$PROJECT/../deliverables"
STAMP="$(date +%Y%m%d)"
NAME="mgmd_demo_videos_${STAMP}"
LIST_ONLY=0
[ "${1:-}" = "--list" ] && LIST_ONLY=1

if [ ! -d "$FIG" ]; then
  echo "错误：找不到 $FIG" >&2
  exit 1
fi

mkdir -p "$OUTDIR"
MANIFEST="$OUTDIR/${NAME}_MANIFEST.md"

{
  echo "# MGMD 演示视频清单"
  echo
  echo "> 打包时间：$(date '+%Y-%m-%d %H:%M:%S')"
  echo "> 源码 commit：$(git -C "$PROJECT" rev-parse --short HEAD 2>/dev/null || echo unknown)"
  echo "> SPEC 版本：$(grep -m1 'spec_version:' "$PROJECT/docs/SPEC.md" | tr -d '`' | sed 's/spec_version: //')"
  echo
  echo "自动生成，请勿手改。重新生成：\`bash scripts/pack_deliverables.sh\`"
  echo
  echo "| 文件 | 大小 | 说明 |"
  echo "|---|---:|---|"
} > "$MANIFEST"

declare -A DESC=(
  ["multi_phantom20.mp4"]="多目标诊断：phantom20 帧24-55，四栏并列展示双目标丢失全过程（SPEC §10.8）"
  ["multi_phantom20_gap.mp4"]="多目标诊断慢放：帧34-43，展示 36-39 帧 GT0 连续 4 帧无候选"
  ["vcam_track.mp4"]="实时追踪：虚拟摄像头输入（SPEC §8.33）"
  ["vcam_full.mp4"]="实时追踪完整版"
  ["demo_phantom70_400.mp4"]="主结果演示：phantom70 400 帧完整流水线"
  ["phantom40_full_compressed.mp4"]="完整流水线长序列演示"
  ["phantom76_300_200_mfe-lac-lad.mp4"]="MFE+LAC+LAD 全链路"
  ["phantom76_300_60_mfe-lac-lad.mp4"]="MFE+LAC+LAD 短样例"
  ["phantom76_373_30_slowmiss.mp4"]="小目标漏检样例（SPEC §8.25）"
  ["D1_mvdt_viz.mp4"]="MVDT 跨域测试 D1"
  ["D2_mvdt_viz.mp4"]="MVDT 跨域测试 D2"
  ["D3_mvdt_viz.mp4"]="MVDT 跨域测试 D3"
  ["D4_mvdt_viz.mp4"]="MVDT 跨域测试 D4"
  ["live_sim.mp4"]="实时链路模拟"
)

n=0
total=0
while IFS= read -r f; do
  base="$(basename "$f")"
  sz=$(stat -c %s "$f")
  human=$(du -h "$f" | cut -f1)
  d="${DESC[$base]:-（无说明）}"
  printf '| `%s` | %s | %s |\n' "$base" "$human" "$d" >> "$MANIFEST"
  n=$((n+1))
  total=$((total+sz))
done < <(find "$FIG" -maxdepth 1 -name '*.mp4' -type f | sort)

{
  echo
  echo "**共 $n 个视频，合计 $(numfmt --to=iec "$total" 2>/dev/null || echo "${total}B")**"
  echo
  echo "## 复现命令"
  echo
  echo '```bash'
  echo 'cd mgmd-linux && source scripts/env_setup.sh'
  echo '# 多目标诊断'
  echo '$PY scripts/make_multi_video.py'
  echo '$PY scripts/make_multi_video.py --start 34 --frames 10 --fps 2 --out outputs/figures/multi_phantom20_gap.mp4'
  echo '# 其他视频见 scripts/make_video.py / make_demo_video.py / live_track.py --record'
  echo '```'
} >> "$MANIFEST"

echo "清单：$MANIFEST"
echo "视频：$n 个，合计 $(numfmt --to=iec "$total" 2>/dev/null || echo "${total}B")"

if [ "$LIST_ONLY" = "1" ]; then
  cat "$MANIFEST"
  exit 0
fi

PKG="$OUTDIR/$NAME"
rm -rf "$PKG"
mkdir -p "$PKG"
cp "$FIG"/*.mp4 "$PKG"/ 2>/dev/null || true
cp "$MANIFEST" "$PKG"/
tar -czf "$OUTDIR/${NAME}.tar.gz" -C "$OUTDIR" "$NAME"
rm -rf "$PKG"
echo "打包完成：$OUTDIR/${NAME}.tar.gz ($(du -h "$OUTDIR/${NAME}.tar.gz" | cut -f1))"
