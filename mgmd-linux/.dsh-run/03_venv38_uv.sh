#!/usr/bin/env bash
# �?uv �?.venv38（不依赖 apt �?python3.8-venv，可�?ROS 安装并行�?set -eo pipefail
ROOT=/home/gaodu/毕业设计/mgmd-linux
cd "$ROOT"
MIRROR=https://pypi.tuna.tsinghua.edu.cn/simple
PY38=$ROOT/.venv38/bin/python

echo "== 1) uv python 3.8 =="
/home/gaodu/.local/bin/uv python install 3.8

echo "== 2) venv =="
if [ ! -x "$PY38" ]; then
  /home/gaodu/.local/bin/uv venv .venv38 --python 3.8 --seed
fi
"$PY38" -V

echo "== 3) 基础依赖（清华镜像） =="
/home/gaodu/.local/bin/uv pip install --python "$PY38" --index-url "$MIRROR" \
  "numpy<2" opencv-python-headless "onnxruntime-gpu==1.19.2" \
  pandas pyyaml requests tqdm matplotlib seaborn scipy onnx onnx-simplifier

echo "== 4) torch CPU（只�?yolov5 NMS�?=="
/home/gaodu/.local/bin/uv pip install --python "$PY38" --index-url https://download.pytorch.org/whl/cpu \
  "torch==2.4.0+cpu" "torchvision==0.19.0+cpu"

echo "== 5) ONNX Runtime �?CUDA 12 运行库（cuDNN 9�?=="
/home/gaodu/.local/bin/uv pip install --python "$PY38" --index-url "$MIRROR" \
  "nvidia-cudnn-cu12>=9,<10" nvidia-cublas-cu12 nvidia-cuda-runtime-cu12 nvidia-cufft-cu12

echo "== 6) 验证 =="
"$PY38" - <<'PY'
import sys
print("Python", sys.version.split()[0])
import numpy, cv2, onnxruntime as ort
print("numpy", numpy.__version__, "cv2", cv2.__version__)
print("onnxruntime", ort.__version__)
print("providers", ort.get_available_providers())
import torch, torchvision
print("torch", torch.__version__, "torchvision", torchvision.__version__)
PY
chown -R gaodu:gaodu "$ROOT/.venv38"
echo "VENV38_OK"
