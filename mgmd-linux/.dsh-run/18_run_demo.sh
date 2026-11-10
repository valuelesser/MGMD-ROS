#!/usr/bin/env bash
# 一键演示：roscore + 自绘窗口查看器 + 三节点流水线；结束时优雅收尾（保证 mp4 可读）
set +u
ROOT=/home/gaodu/毕业设计/mgmd-linux
WS=$ROOT/ros1_ws
RUN=/tmp/ros_run
OUT=$ROOT/outputs/ros_viz
mkdir -p "$RUN" "$OUT"
source /opt/ros/noetic/setup.bash
source "$WS/devel/setup.bash"
SP=$ROOT/.venv38/lib/python3.8/site-packages
NV=$(find "$SP/nvidia" -maxdepth 3 -type d -name lib 2>/dev/null | tr '\n' ':')
export PYTHONPATH="$SP:$ROOT:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${NV}${LD_LIBRARY_PATH:-}"
export DISPLAY=:0
export PYTHONUNBUFFERED=1
export MGMD_ARD_ROOT="/mnt/c/Users/gaodu/Desktop/毕业设计/mgmd/data/videos/视频/ARD-MAV"
export MGMD_YOLOV5_DIR="$ROOT/third_party/yolov5-v6.0"

VIDEO=${1:-phantom30}
RATE=${2:-20}
LIMIT=${3:-600}
W=${4:-960}
H=${5:-720}

echo "[0] 清理旧进程"
bash "$ROOT/.dsh-run/13_kill_all.sh" >/dev/null 2>&1 || true
sleep 1

echo "[1] roscore"
setsid nohup roscore > "$RUN/core.log" 2>&1 < /dev/null &
echo $! > "$RUN/core.pid"
for i in $(seq 1 25); do
  timeout 3 rostopic list >/dev/null 2>&1 && break
  sleep 1
done
pgrep -x rosmaster >/dev/null && echo "    roscore OK" || { echo "    roscore FAIL"; tail -20 "$RUN/core.log"; exit 1; }

echo "[2] 实时窗口 (17_viewer.py)"
setsid nohup "$ROOT/.venv38/bin/python" "$ROOT/.dsh-run/17_viewer.py" \
  > "$RUN/viewer.log" 2>&1 < /dev/null &
echo $! > "$RUN/viewer.pid"
sleep 3
ps -o pid,stat,args -p "$(cat "$RUN/viewer.pid")" | tail -1
tail -3 "$RUN/viewer.log" 2>/dev/null

echo "[3] 三节点流水线"
OUTVID="$OUT/${VIDEO}_${W}x${H}.mp4"
rm -f "$OUTVID"
setsid nohup roslaunch mav_tracker_nodes track.launch \
  video:="$VIDEO" rate:="$RATE" limit:="$LIMIT" \
  width:="$W" height:="$H" loop:=false \
  device:=cuda d_min:=0.02 buffer:=6 \
  out_video:="$OUTVID" encoder:=cpu visualize:=true \
  > "$RUN/launch.log" 2>&1 < /dev/null &
echo $! > "$RUN/launch.pid"

echo "[4] 等待回放结束（最多 120s）"
for i in $(seq 1 120); do
  if grep -qE "回放结束|到达 limit" "$RUN/launch.log" 2>/dev/null; then break; fi
  sleep 1
done

echo "[5] 优雅停止 visualize_node（写完整 mp4）"
VP=$(ps -eo pid,args --no-headers | awk '$2 !~ /bash|sh/ && /visualize_node\.py/ {print $1}' | head -1)
if [ -n "$VP" ]; then
  kill -TERM "$VP" 2>/dev/null
  for i in $(seq 1 12); do kill -0 "$VP" 2>/dev/null || break; sleep 1; done
  echo "    visualize pid=$VP 已停止"
else
  echo "    没找到 visualize_node"
fi
sleep 2

echo "[6] 收尾"
bash "$ROOT/.dsh-run/13_kill_all.sh" >/dev/null 2>&1 || true

echo "[7] 产物"
ls -l "$OUTVID" 2>/dev/null
ffprobe -v error -select_streams v:0 -show_entries stream=codec_name,width,height,nb_frames \
  -show_entries format=duration,size -of default=nw=1 "$OUTVID" 2>&1 | head -8
echo "DEMO_DONE"
