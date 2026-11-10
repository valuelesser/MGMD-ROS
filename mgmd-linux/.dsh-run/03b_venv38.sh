#!/usr/bin/env bash
# 用系统 python3.8 建 .venv38（与 ROS1 Noetic 同解释器版本）
set -eo pipefail
ROOT=/home/gaodu/毕业设计/mgmd-linux
cd "$ROOT"
MIRROR=https://pypi.tuna.tsinghua.edu.cn/simple
UV=/home/gaodu/.local/bin/uv
PY38=$ROOT/.venv38/bin/python

echo "== 1) venv =="
if [ ! -x "$PY38" ]; then
  python3.8 -m venv .venv38
fi
"$PY38" -V

echo "== 2) 基础依赖（清华镜像） =="
# 不装 onnx-simplifier：它要 CMake>=3.22（focal 只有 3.16），且 ROS 推理用不到
"$UV" pip install --python "$PY38" --index-url "$MIRROR" \
  "numpy<2" \
  opencv-python-headless \
  "onnxruntime-gpu==1.19.2" \
  pandas \
  pyyaml \
  requests \
  tqdm \
  scipy

echo "== 3) torch CPU（只给 yolov5 NMS） =="
if ! "$UV" pip install --python "$PY38" --index-url https://download.pytorch.org/whl/cpu \
      "torch==2.4.0+cpu" "torchvision==0.19.0+cpu"; then
  echo "[warn] cpu 索引失败，退回 PyPI 的 torch 2.4.0"
  "$UV" pip install --python "$PY38" --index-url "$MIRROR" \
      "torch==2.4.0" "torchvision==0.19.0"
fi

echo "== 4) ONNX Runtime 的 CUDA 12 运行库（cuDNN 9） =="
"$UV" pip install --python "$PY38" --index-url "$MIRROR" \
  "nvidia-cudnn-cu12>=9,<10" \
  nvidia-cublas-cu12 \
  nvidia-cuda-runtime-cu12 \
  nvidia-cufft-cu12

echo "== 5) 验证 =="
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
