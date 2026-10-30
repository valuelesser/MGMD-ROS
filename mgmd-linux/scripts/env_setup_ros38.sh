#!/usr/bin/env bash
# Python 3.8 + ONNX Runtime 环境（ROS1 同版本，无跨解释器）
# 用法：source scripts/env_setup_ros38.sh
#
# 关键点：
#   1. onnxruntime-gpu 需要 nvidia/*/lib 在 LD_LIBRARY_PATH（pip 装的不自动加）
#   2. **不要 unset PYTHONPATH** —— ROS 的 rospy 在 /opt/ros/noetic 里，
#      清掉 PYTHONPATH 会导致 import rospy 失败。
#
# 若需 ROS：先 source /opt/ros/noetic/setup.bash，再 source 本脚本。

ROOT=/home/gaoduan/毕业设计/mgmd-linux
export PY38="$ROOT/.venv38/bin/python"

_NVLIBS=$(find "$ROOT/.venv38/lib/python3.8/site-packages/nvidia" -maxdepth 2 -type d -name lib 2>/dev/null | tr '\n' ':')
export LD_LIBRARY_PATH="${_NVLIBS}${LD_LIBRARY_PATH:-}"

# 让 mgmd / mgmd_onnx 可导入（不覆盖已有 PYTHONPATH）
case ":${PYTHONPATH:-}:" in
  *":$ROOT:"*) ;;
  *) export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" ;;
esac

echo "[env38] $($PY38 -V 2>&1)  nvidia 库目录 $(echo "$_NVLIBS" | tr ':' '\n' | grep -c .) 个"
