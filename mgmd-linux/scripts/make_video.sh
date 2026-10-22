#!/usr/bin/env bash
# 一键生成可视化视频（自动加载环境）
set -e
cd "$(dirname "$0")/.."
source scripts/env_setup.sh
exec $PY scripts/make_video.py "$@"
