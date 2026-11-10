#!/usr/bin/env bash
# 用 .venv38 跑 ONNX 推理冒烟（带 nvidia 库路径）
set -eo pipefail
ROOT=/home/gaodu/毕业设计/mgmd-linux
SP=$ROOT/.venv38/lib/python3.8/site-packages
NV=$(find "$SP/nvidia" -maxdepth 3 -type d -name lib 2>/dev/null | tr '\n' ':')
export LD_LIBRARY_PATH="${NV}${LD_LIBRARY_PATH:-}"
export MGMD_YOLOV5_DIR="$ROOT/third_party/yolov5-v6.0"
export MGMD_ARD_ROOT="/mnt/c/Users/gaodu/Desktop/毕业设计/mgmd/data/videos/视频/ARD-MAV"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
echo "NV libs: $(echo "$NV" | tr ':' '\n' | grep -c .) 个目录"
exec "$ROOT/.venv38/bin/python" "$ROOT/.dsh-run/07_smoke_onnx.py"
