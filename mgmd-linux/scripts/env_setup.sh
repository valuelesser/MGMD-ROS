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
