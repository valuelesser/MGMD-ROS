#!/usr/bin/env bash
# scripts/fetch_mgmd_weights.sh — 下载 MGMD 官方预训练权重（SPEC §3.3/§3.4）。
#
# 用法：
#   bash scripts/fetch_mgmd_weights.sh [--with-global] [--check] [dest]
#     --with-global  额外下载 yolov5s_GLAD.pt（可选全局检测器基线）
#     --check        仅校验已有文件（大小 + sha256），不下载
#     dest           目标目录（默认 data_raw/mgmd_weights）
#
# 来源：https://github.com/WindyLab/Global-Local-MAV-Detection（MIT license）
# 走 GitHub contents API（Accept: application/vnd.github.raw）绕开 raw 限速；
# 已知坑：该通道可能静默截断（SPEC §3.4 实测），故先验大小再验 sha256。
# 退出码：0 成功 / 3 数据缺失或校验失败（SPEC §4.3）。
set -uo pipefail
cd "$(dirname "$0")/.."

WITH_GLOBAL=0
CHECK_ONLY=0
DEST="data_raw/mgmd_weights"
for arg in "$@"; do
  case "$arg" in
    --with-global) WITH_GLOBAL=1 ;;
    --check) CHECK_ONLY=1 ;;
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
    *) DEST="$arg" ;;
  esac
done

BASE="https://api.github.com/repos/WindyLab/Global-Local-MAV-Detection/contents/weights"

# name size_bytes sha256 required(1/0)
ITEMS=(
  "Net_best.pth 248841 0d3a4631060339ee72551a89e95887fc49a2d50f9f3473ef354cf3db2830faf5 1"
  "yolov5s_GLAD-crop.pt 14454645 031813a6c7ef8446080528e198b4f4e44ee936f066d1b12612adc4ef8af0d8a0 1"
  "yolov5s_GLAD.pt 14359541 9e8edde080cf9e1eac0fcd68c12d7073bb23dac4059f2929ed3a06f301537a0c 0"
)

err() { echo "[mgmd] $1: $2 → $3" >&2; }

verify() { # path want_size want_sha -> 0 ok
  local f="$1" want_size="$2" want_sha="$3"
  [ -f "$f" ] || return 1
  local got_size; got_size=$(stat -c%s "$f" 2>/dev/null || echo 0)
  if [ "$got_size" != "$want_size" ]; then
    echo "[bad-size] $f got=$got_size want=$want_size"; return 1
  fi
  local got_sha; got_sha=$(sha256sum "$f" | cut -d' ' -f1)
  if [ "$got_sha" != "$want_sha" ]; then
    echo "[bad-sha256] $f got=$got_sha"; return 1
  fi
  return 0
}

mkdir -p "$DEST"
rc=0
fetched=0
for item in "${ITEMS[@]}"; do
  read -r name size sha required <<<"$item"
  [ "$required" = "1" ] || [ "$WITH_GLOBAL" = "1" ] || continue
  out="$DEST/$name"
  if verify "$out" "$size" "$sha"; then
    echo "[ok] $name ($size)"
    continue
  fi
  if [ "$CHECK_ONLY" = "1" ]; then
    err E_WEIGHTS_MISSING "$name 缺失或校验失败" "bash scripts/fetch_mgmd_weights.sh 重新下载"
    rc=3; continue
  fi
  # 下载（残留坏文件先移开，避免续传叠加脏数据）
  [ -f "$out" ] && mv -f "$out" "$out.bad"
  echo "[dl] $name (want $size)"
  curl -sL --max-time 900 --retry 3 -C - \
    -H "Accept: application/vnd.github.raw" \
    -o "$out" "$BASE/$name"
  if verify "$out" "$size" "$sha"; then
    echo "[ok] $name"
    rm -f "$out.bad"
    fetched=1
  else
    err E_WEIGHTS_HASH "$name 下载后校验失败（已知静默截断问题）" \
      "rm -f $out $out.bad 后重试；仍失败则检查网络/稍后再试（API 限流 60/h）"
    rc=3
  fi
done

# README（来源/license/hash 台账；US-001 AC4）
if [ "$CHECK_ONLY" = "0" ] && [ "$rc" = "0" ]; then
  {
    echo "# MGMD 官方预训练权重"
    echo
    echo "- 来源: https://github.com/WindyLab/Global-Local-MAV-Detection （weights/ 目录）"
    echo "- License: MIT（官方仓库）；仅本地下载使用，不入 git"
    echo "- 获取脚本: scripts/fetch_mgmd_weights.sh；获取日期: $(date +%F)"
    echo
    echo "| 文件 | 大小 (bytes) | sha256 | 用途 |"
    echo "|---|---|---|---|"
    echo "| Net_best.pth | 248841 | 0d3a4631060339ee72551a89e95887fc49a2d50f9f3473ef354cf3db2830faf5 | LAC 32×32 CNN |"
    echo "| yolov5s_GLAD-crop.pt | 14454645 | 031813a6c7ef8446080528e198b4f4e44ee936f066d1b12612adc4ef8af0d8a0 | LAD 320×320 裁剪检测 |"
    echo "| yolov5s_GLAD.pt | 14359541 | 9e8edde080cf9e1eac0fcd68c12d7073bb23dac4059f2929ed3a06f301537a0c | 可选全局检测器基线（--with-global） |"
  } > "$DEST/README.md"
  echo "[ok] README.md -> $DEST/README.md"
fi

echo "fetch_mgmd_weights rc=$rc -> $DEST"
exit $rc
