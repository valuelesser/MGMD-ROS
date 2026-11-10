#!/usr/bin/env bash
# MGMD 运行环境（SPEC §2.2 / §8.28）。所有脚本前 source 本文件。
export PROJECT=/home/gaoduan/毕业设计/mgmd-linux
export PY=/home/gaoduan/毕业设计/mgmd-linux/.venv/bin/python
# yolov5 v6.0 需用大写 V5 的环境变量名
export YOLOV5_CONFIG_DIR=/home/gaoduan/毕业设计/.cache/ultralytics
export MPLCONFIGDIR=/home/gaoduan/毕业设计/.cache/matplotlib
export XDG_CACHE_HOME=/home/gaoduan/毕业设计/.cache
mkdir -p "$YOLOV5_CONFIG_DIR" "$MPLCONFIGDIR"
[ -f "$YOLOV5_CONFIG_DIR/Arial.ttf" ] || cp /usr/share/fonts/truetype/dejavu/DejaVuSans.ttf "$YOLOV5_CONFIG_DIR/Arial.ttf" 2>/dev/null
export PYTHONPATH="$PROJECT"

# ---- CUDA 库路径（ORT / TensorRT 需要）----
SP="$PROJECT/.venv/lib/python3.11/site-packages"
NV="$SP/nvidia"
_tl="$SP/tensorrt_cu12_libs"
_path=""
[ -d "$_tl" ] && _path="$_tl"
if [ -d "$NV" ]; then
  _nvpaths=$(ls -d "$NV"/*/lib 2>/dev/null | tr '\n' ':')
  _path="${_path:+$_path:}${_nvpaths%:}"
fi
[ -n "$_path" ] && export LD_LIBRARY_PATH="$_path${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# ---- GPU 相关 ----
export NVIDIA_VISIBLE_DEVICES="${NVIDIA_VISIBLE_DEVICES:-all}"
export NVIDIA_DRIVER_CAPABILITIES="${NVIDIA_DRIVER_CAPABILITIES:-compute,utility}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"
export CUDNN_BENCHMARK="${CUDNN_BENCHMARK:-1}"

# ---- 默认 TRT engine（若存在则自动启用，可用 MGMD_NO_TRT=1 关闭）----
if [ -z "$MGMD_NO_TRT" ] && [ -f "$PROJECT/outputs/lad_trt_fp16.engine" ]; then
  export MGMD_TRT_ENGINE="${MGMD_TRT_ENGINE:-$PROJECT/outputs/lad_trt_fp16.engine}"
fi
