#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
prefix=${MGMD_LIBAV_PREFIX:-/opt/mgmd-g1-libav}
export PKG_CONFIG_PATH="$prefix/lib/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
cc -Wall -Wextra -Werror -shared -fPIC -Wl,--exclude-libs,ALL \
    "$root/mgmd_onnx/cuda_decoder.c" -o "${1:?output library path required}" \
    $(pkg-config --cflags --libs --static libavformat libavcodec libavutil)
