#!/usr/bin/env bash
# Python 3.8 + ONNX Runtime 环境（ROS1 同版本，无跨解释器）
# 用法：source scripts/env_setup_ros38.sh
#
# 关键：onnxruntime-gpu 需要 nvidia/*/lib 在 LD_LIBRARY_PATH（pip 装的不自动加）

ROOT=/home/gaoduan/毕业设计/mgmd-linux
export PY38="$ROOT/.venv38/bin/python"

_NVLIBS=$(find "$ROOT/.venv38/lib/python3.8/site-packages/nvidia" -maxdepth 2 -type d -name lib 2>/dev/null | tr '\n' ':')
export LD_LIBRARY_PATH="${_NVLIBS}${LD_LIBRARY_PATH}"

unset PYTHONPATH
echo "[env38] $($PY38 -V 2>&1)  LD_LIBRARY_PATH 含 $(echo "$_NVLIBS" | tr ':' '\n' | grep -c .) 个 nvidia 库目录"
