#!/usr/bin/env bash
# scripts/unpack_mvdt.sh — 解包 MVDT 分卷 zip 归档为 mp4（7z 多卷），落到 configs 期望的 D 布局。
#
# 用法：bash scripts/unpack_mvdt.sh [archives_dir]
#   归档：data_raw/mvdt/_archives/datasetN/camM/camM.z01..zip
#   输出：data_raw/mvdt/D{n}/cam{m}/{cam}.mp4（D1/cam2, D2/cam1, D3/cam3, D4/cam3）
set -uo pipefail
cd "$(dirname "$0")/.."
SRC="${1:-data_raw/mvdt/_archives}"
DEST="data_raw/mvdt"
rc=0
for spec in "dataset1:D1:cam2" "dataset2:D2:cam1" "dataset3:D3:cam3" "dataset4:D4:cam3"; do
  IFS=: read -r ds D cam <<< "$spec"
  adir="$SRC/$ds/$cam"
  [ -d "$adir" ] || { echo "[skip] 无归档 $adir"; continue; }
  if [ ! -f "$adir/$cam.zip" ]; then echo "[skip] 缺 $adir/$cam.zip"; rc=1; continue; fi
  tmp="$(mktemp -d)"; echo "[unpack] $ds/$cam -> $DEST/$D/$cam"
  7z x -y -o"$tmp" "$adir/$cam.zip" >/dev/null 2>&1 || { echo "[FAIL] 7z $adir/$cam.zip"; rm -rf "$tmp"; rc=1; continue; }
  mp4="$(find "$tmp" -type f -name '*.mp4' | head -1)"
  if [ -z "$mp4" ]; then echo "[FAIL] 无 mp4 in $adir/$cam.zip"; rm -rf "$tmp"; rc=1; continue; fi
  mkdir -p "$DEST/$D/$cam"
  mv "$mp4" "$DEST/$D/$cam/$cam.mp4"
  rm -rf "$tmp"
  echo "[ok] $DEST/$D/$cam/$cam.mp4 ($(du -h "$DEST/$D/$cam/$cam.mp4" | cut -f1))"
done
echo "unpack_mvdt rc=$rc"
exit $rc
