#!/usr/bin/env bash
# 启动 ROS1 三节点（背景运行；GUI 由 06_gui.sh 单独开）
set -eo pipefail
ROOT=/home/gaodu/毕业设计/mgmd-linux
WS=$ROOT/ros1_ws
OUT=$ROOT/outputs/ros_viz
mkdir -p "$OUT" /tmp/ros_run

set +u
source /opt/ros/noetic/setup.bash
source "$WS/devel/setup.bash"
set -u

# 让 roslaunch 的 system python3.8 也能 import venv38 里的 onnxruntime/cv2/torch
VENV_SP=$ROOT/.venv38/lib/python3.8/site-packages
export PYTHONPATH="$VENV_SP:$ROOT:${PYTHONPATH:-}"
NVLIBS=$(find "$VENV_SP/nvidia" -maxdepth 3 -type d -name lib 2>/dev/null | tr '\n' ':')
export LD_LIBRARY_PATH="${NVLIBS}${LD_LIBRARY_PATH:-}"
export DISPLAY=:0
export PYTHONUNBUFFERED=1
export MGMD_ARD_ROOT="/mnt/c/Users/gaodu/Desktop/毕业设计/mgmd/data/videos/视频/ARD-MAV"
export MGMD_YOLOV5_DIR="$ROOT/third_party/yolov5-v6.0"

VIDEO=${1:-phantom30}
RATE=${2:-20}
LIMIT=${3:-900}
W=${4:-960}
H=${5:-720}
DEV=${6:-cuda}
OUTVID="$OUT/${VIDEO}_${W}x${H}.mp4"

echo "[run] video=$VIDEO rate=$RATE limit=$LIMIT ${W}x${H} device=$DEV -> $OUTVID"
rm -f "$OUTVID"
exec roslaunch mav_tracker_nodes track.launch \
  video:="$VIDEO" rate:="$RATE" limit:="$LIMIT" \
  width:="$W" height:="$H" loop:=false \
  device:="$DEV" d_min:=0.02 buffer:=6 \
  out_video:="$OUTVID" encoder:=nvenc visualize:=true
