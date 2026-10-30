#!/usr/bin/env bash
# scripts/setup_env38.sh — 从零重建 Python 3.8 + ONNX 环境（SPEC-ROS §15.4）
#
# 为什么用 3.8：ROS1 Noetic 绑死 Python 3.8，单进程方案要求算法侧同版本。
# 用法：bash scripts/setup_env38.sh [venv 目录，默认 .venv38]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${1:-$ROOT/.venv38}"
MIRROR="${PIP_MIRROR:-https://pypi.tuna.tsinghua.edu.cn/simple}"

echo "=== 1/4 检查系统 python3.8 与 venv ==="
if ! command -v /usr/bin/python3 >/dev/null; then echo "缺少 /usr/bin/python3"; exit 1; fi
/usr/bin/python3 -V
if ! /usr/bin/python3 -m venv --help >/dev/null 2>&1; then
  echo "缺少 python3.8-venv，请先: sudo apt-get install -y python3.8-venv python3-pip"
  exit 1
fi

echo "=== 2/4 建 venv: $VENV ==="
if [ ! -x "$VENV/bin/python" ]; then
  /usr/bin/python3 -m venv "$VENV" --without-pip
fi
if ! "$VENV/bin/python" -m pip --version >/dev/null 2>&1; then
  TMP=$(mktemp -d)
  curl -sS https://bootstrap.pypa.io/pip/3.8/get-pip.py -o "$TMP/get-pip.py"
  "$VENV/bin/python" "$TMP/get-pip.py" -i "$MIRROR" >/dev/null
  rm -rf "$TMP"
fi
"$VENV/bin/python" -m pip --version

echo "=== 3/4 安装依赖（版本固定，见 SPEC-ROS §13.3） ==="
"$VENV/bin/python" -m pip install --no-cache-dir -i "$MIRROR" \
  "numpy<2" opencv-python-headless "onnxruntime-gpu==1.19.2" \
  torch==2.4.0 torchvision==0.19.0 \
  pandas pyyaml requests tqdm matplotlib seaborn scipy onnx onnx-simplifier

echo "=== 4/4 验证 ==="
"$VENV/bin/python" - <<'PY'
import sys, torch, cv2, numpy, onnxruntime as ort
print(f"  Python {sys.version.split()[0]}")
print(f"  torch {torch.__version__}  cuda={torch.cuda.is_available()}")
print(f"  cv2 {cv2.__version__}  numpy {numpy.__version__}")
print(f"  onnxruntime {ort.__version__}")
print(f"  providers {ort.get_available_providers()}")
PY
echo
echo "完成。使用前必须 source: source scripts/env_setup_ros38.sh"
