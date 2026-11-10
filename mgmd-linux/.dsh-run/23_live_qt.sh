#!/usr/bin/env bash
# 常驻实时演示（Qt 版 v2）：roscore + 21_qt_viewer + 三节点（loop=true）
# 停止：bash .dsh-run/13_kill_all.sh
set +u
ROOT=/home/gaodu/毕业设计/mgmd-linux
WS=$ROOT/ros1_ws
LOG=$ROOT/outputs/logs
OUT=$ROOT/outputs/ros_viz
mkdir -p "$LOG" "$OUT"

source /opt/ros/noetic/setup.bash
source "$WS/devel/setup.bash"
ROSPATH=$PYTHONPATH

SP=$ROOT/.venv38/lib/python3.8/site-packages
NV=$(find "$SP/nvidia" -maxdepth 3 -type d -name lib 2>/dev/null | tr '\n' ':')
export PYTHONPATH="$SP:$ROOT:$ROSPATH"
export LD_LIBRARY_PATH="${NV}${LD_LIBRARY_PATH:-}"
export DISPLAY=:0
export PYTHONUNBUFFERED=1
export QT_X11_NO_MITSHM=1
export MGMD_ARD_ROOT="/mnt/c/Users/gaodu/Desktop/毕业设计/mgmd/data/videos/视频/ARD-MAV"
export MGMD_YOLOV5_DIR="$ROOT/third_party/yolov5-v6.0"

VIDEO=${1:-phantom30}
RATE=${2:-20}
W=${3:-960}
H=${4:-720}

echo "[0] 清理"
bash "$ROOT/.dsh-run/13_kill_all.sh" >/dev/null 2>&1 || true
for pat in 17_viewer.py 21_qt_viewer.py; do
  ps -eo pid,comm,args --no-headers | awk -v n="$pat" '$2 != "bash" && index($0, n) {print $1}' \
    | while read -r pid; do kill -TERM "$pid" 2>/dev/null; done
done
sleep 1

echo "[1] roscore"
setsid nohup roscore > "$LOG/live.core.log" 2>&1 < /dev/null &
for i in $(seq 1 25); do timeout 3 rostopic list >/dev/null 2>&1 && break; sleep 1; done
pgrep -x rosmaster >/dev/null && echo "    OK" || { echo "    FAIL"; tail -5 "$LOG/live.core.log"; exit 1; }

echo "[2] Qt 窗口"
PYTHONPATH="$ROOT:$ROSPATH" setsid nohup python3 "$ROOT/.dsh-run/21_qt_viewer.py" \
  > "$LOG/live.qt.log" 2>&1 < /dev/null &
sleep 4
tail -5 "$LOG/live.qt.log"

echo "[3] 三节点"
setsid nohup roslaunch mav_tracker_nodes track.launch \
  video:="$VIDEO" rate:="$RATE" limit:=0 \
  width:="$W" height:="$H" loop:=true \
  device:=cuda d_min:=0.02 buffer:=6 \
  out_video:="$OUT/live_${VIDEO}_${W}x${H}.mp4" encoder:=cpu visualize:=true \
  > "$LOG/live.launch.log" 2>&1 < /dev/null &

sleep 10
echo "[4] 状态"
timeout 5 rosnode list 2>&1
grep -E '已处理' "$LOG/live.launch.log" 2>/dev/null | tail -1
echo "--- viewer log ---"
tail -5 "$LOG/live.qt.log"
echo "QT_LIVE_V2_STARTED"
