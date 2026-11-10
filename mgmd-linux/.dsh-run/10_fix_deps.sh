#!/usr/bin/env bash
# 补齐 ORT CUDA EP 与 yolov5 utils 缺的依赖
set -eo pipefail
ROOT=/home/gaodu/毕业设计/mgmd-linux
MIRROR=https://pypi.tuna.tsinghua.edu.cn/simple
UV=/home/gaodu/.local/bin/uv
PY38=$ROOT/.venv38/bin/python

echo "== 1) 补 CUDA 运行库（curand/cusolver/cusparse） =="
"$UV" pip install --python "$PY38" --index-url "$MIRROR" \
  nvidia-curand-cu12 nvidia-cusolver-cu12 nvidia-cusparse-cu12

echo "== 2) 补 matplotlib / seaborn（yolov5 utils.metrics / plots 依赖） =="
"$UV" pip install --python "$PY38" --index-url "$MIRROR" matplotlib seaborn

echo "== 3) nvidia 库目录 =="
find "$ROOT/.venv38/lib/python3.8/site-packages/nvidia" -maxdepth 3 -type d -name lib | sort
echo "DEPS_OK"
