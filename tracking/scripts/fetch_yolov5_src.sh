#!/usr/bin/env bash
# scripts/fetch_yolov5_src.sh — 获取 YOLOv5 v6.0 源码（SPEC §1.3 D3 修正）。
#
# 用法：bash scripts/fetch_yolov5_src.sh [--check] [dest]
#   --check  仅校验已解压源码与 tarball sha256，不下载
#   dest     目标目录（默认 third_party/yolov5-v6.0）
#
# 背景：yolov5 v6.0 tag 仓库无 setup.py/pyproject.toml，不能 pip 安装；
# yolov5s_GLAD-crop.pt 的 pickle 引用顶层 models.*/utils.* 模块路径，
# PyPI 版 yolov5（7.x，yolov5.models.* 路径）unpickle 不兼容。
# 故经 codeload tarball（sha256 预 pin）落地 third_party/，运行时 sys.path 注入。
# 退出码：0 成功 / 3 下载或校验失败。
set -uo pipefail
cd "$(dirname "$0")/.."

CHECK_ONLY=0
DEST="third_party/yolov5-v6.0"
for arg in "$@"; do
  case "$arg" in
    --check) CHECK_ONLY=1 ;;
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
    *) DEST="$arg" ;;
  esac
done

URL="https://codeload.github.com/ultralytics/yolov5/tar.gz/refs/tags/v6.0"
SHA256="4c4036cf96fcb1700e9ec5f92ccee3a92f35003715d5b869d09bf474cc6f9204"
TARBALL="third_party/_yolov5-v6.0.tar.gz"

err() { echo "[mgmd] $1: $2 → $3" >&2; }

verify_tree() { # dest -> 0 ok
  [ -f "$1/models/experimental.py" ] && [ -f "$1/utils/general.py" ] \
    && [ -f "$1/utils/augmentations.py" ] && [ -f "$1/LICENSE" ]
}

if verify_tree "$DEST"; then
  echo "[ok] $DEST（v6.0 源码已就绪）"
  exit 0
fi
if [ "$CHECK_ONLY" = "1" ]; then
  err E_DEP_MISSING "yolov5 v6.0 源码缺失: $DEST" "bash scripts/fetch_yolov5_src.sh"
  exit 3
fi

mkdir -p third_party "$(dirname "$DEST")"
echo "[dl] $URL"
curl -sL --max-time 600 --retry 3 -C - -o "$TARBALL" "$URL"
got=$(sha256sum "$TARBALL" | cut -d' ' -f1)
if [ "$got" != "$SHA256" ]; then
  err E_DEP_MISSING "yolov5 tarball sha256 不匹配（got=$got）" \
    "rm -f $TARBALL 后重试；GitHub codeload 打包字节若漂移需更新脚本内 SHA256"
  exit 3
fi
tmp="third_party/_yolov5_extract"
rm -rf "$tmp" && mkdir -p "$tmp"
tar xzf "$TARBALL" -C "$tmp"
rm -rf "$DEST"
mv "$tmp/yolov5-6.0" "$DEST"
rm -rf "$tmp" "$TARBALL"
if verify_tree "$DEST"; then
  echo "[ok] yolov5 v6.0 -> $DEST"
  exit 0
fi
err E_DEP_MISSING "解压后源码树不完整" "重跑 bash scripts/fetch_yolov5_src.sh"
exit 3
