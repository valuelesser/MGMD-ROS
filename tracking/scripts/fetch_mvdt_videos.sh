#!/usr/bin/env bash
# scripts/fetch_mvdt_videos.sh — 抓取 MVDT 分卷 zip 归档（aria2 多连接 + 断点续传）。
#
# 用法：bash scripts/fetch_mvdt_videos.sh [dest]   # 默认 data_raw/mvdt/_archives
# 需要的四条序列：dataset1/cam2, dataset2/cam1, dataset3/cam3, dataset4/cam3。
# 走 GitHub contents API（Accept: application/vnd.github.raw，支持 Range）以绕开 raw 限速。
set -uo pipefail
cd "$(dirname "$0")/.."
DEST="${1:-data_raw/mvdt/_archives}"
BASE="https://api.github.com/repos/CenekAlbl/drone-tracking-datasets/contents"
CONN="${SPA2D_ARIA_CONN:-4}"
rate_remaining() { curl -s --max-time 30 https://api.github.com/rate_limit | python3 -c "import sys,json;print(json.load(sys.stdin)['rate']['remaining'])" 2>/dev/null || echo 0; }
rate_wait() { local need="$1"; while [ "$(rate_remaining)" -lt "$need" ]; do echo "[rate] remaining=$(rate_remaining) < $need, wait 60s"; sleep 60; done; }

download() { # dataset/cam relpath want
  local dscam="$1" rel="$2" want="$3"
  local dir="$DEST/$dscam"; mkdir -p "$dir"
  local out="$dir/$rel"
  if [ -f "$out" ] && [ "$(stat -c%s "$out")" = "$want" ]; then
    echo "[ok] $dscam/$rel ($want)"; return 0
  fi
  echo "[dl] $dscam/$rel (want $want)"
  rate_wait $((CONN+1))
  aria2c -x"$CONN" -s"$CONN" -k1M --continue=true --file-allocation=none --allow-overwrite=true \
    --header="Accept: application/vnd.github.raw" --summary-interval=0 \
    -o "$out" "$BASE/$dscam/$rel" >/dev/null 2>&1 || true
  local got; got=$(stat -c%s "$out" 2>/dev/null || echo 0)
  if [ "$got" = "$want" ]; then echo "[ok] $dscam/$rel"; else echo "[FAIL] $dscam/$rel got=$got want=$want"; return 1; fi
}

rc=0
download dataset1/cam2 cam2.z01 104857600 || rc=1
download dataset1/cam2 cam2.zip 97719789 || rc=1
download dataset2/cam1 cam1.zip 46491738 || rc=1
for v in 01 02 03 04 05 06 07 08; do download dataset3/cam3 "cam3.z$v" 104857600 || rc=1; done
download dataset3/cam3 cam3.zip 12815805 || rc=1
for v in 01 02 03 04 05 06 07; do download dataset4/cam3 "cam3.z$v" 104857600 || rc=1; done
download dataset4/cam3 cam3.zip 29504825 || rc=1
echo "fetch_mvdt_videos rc=$rc -> $DEST"
exit $rc
