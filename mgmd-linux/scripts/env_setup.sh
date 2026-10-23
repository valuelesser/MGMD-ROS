#!/usr/bin/env bash
# MGMD 运行环境（SPEC §2.2）。所有脚本前 source 本文件。
export PROJECT=/home/gaoduan/毕业设计/mgmd-linux
export PY=/home/gaoduan/毕业设计/mgmd-linux/.venv/bin/python
# yolov5 v6.0 需要用大写 V5 的环境变量名
export YOLOV5_CONFIG_DIR=/home/gaoduan/毕业设计/.cache/ultralytics
export MPLCONFIGDIR=/home/gaoduan/毕业设计/.cache/matplotlib
export XDG_CACHE_HOME=/home/gaoduan/毕业设计/.cache
mkdir -p "$YOLOV5_CONFIG_DIR" "$MPLCONFIGDIR"
[ -f "$YOLOV5_CONFIG_DIR/Arial.ttf" ] || cp /usr/share/fonts/truetype/dejavu/DejaVuSans.ttf "$YOLOV5_CONFIG_DIR/Arial.ttf" 2>/dev/null
export PYTHONPATH="$PROJECT"

# ---- GPU 相关（SPEC §2.3）----
# NVIDIA 容器环境变量：便于在容器/沙箱内获得设备可见性
export NVIDIA_VISIBLE_DEVICES="${NVIDIA_VISIBLE_DEVICES:-all}"
export NVIDIA_DRIVER_CAPABILITIES="${NVIDIA_DRIVER_CAPABILITIES:-compute,utility}"
# 让 torch 不去探测 NVML 失败时刷屏
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"
# 允许 cuDNN 自动调优（固定 shape 场景收益明显）
export CUDNN_BENCHMARK="${CUDNN_BENCHMARK:-1}"
