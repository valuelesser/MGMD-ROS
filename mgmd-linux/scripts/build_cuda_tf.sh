#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cuda=${CUDA_HOME:-/usr/local/cuda-12.1}
"$cuda/bin/nvcc" -O3 -arch="${MGMD_CUDA_ARCH:-sm_86}" --fmad=false -shared -Xcompiler=-fPIC \
    "$root/mgmd_onnx/cuda_tf.cu" -o "${1:?output library path required}"
