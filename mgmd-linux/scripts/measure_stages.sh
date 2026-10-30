#!/usr/bin/env bash
# 720p 各阶段耗时测量（SPEC-ROS §20）。
# 只用 PID 文件管理进程 —— 绝不用 pkill -f / pgrep -f，
# 否则会匹配到执行本脚本的 shell 自己并把它杀掉（已踩两次）。
set +u   # ROS setup.bash 引用未定义变量，不能开 -u

WS=/home/gaoduan/毕业设计/mgmd-linux/ros1_ws
ROOT=/home/gaoduan/毕业设计/mgmd-linux
PY=$ROOT/.venv38/bin/python
DUR=${1:-40}
TAG=${2:-720p}

source /opt/ros/noetic/setup.bash
source "$WS/devel/setup.bash"
source "$ROOT/scripts/env_setup_ros38.sh" >/dev/null

VID=${VIDEO:-/home/gaoduan/tracking/data_raw/mvdt/D1/cam2/cam2.mp4}
[ -z "$VID" ] && { echo "找不到视频"; exit 1; }
echo "视频: $VID"

cleanup() {
  for f in /tmp/st_pid_*; do
    [ -e "$f" ] || continue
    p=$(cat "$f" 2>/dev/null)
    [ -n "$p" ] && kill "$p" 2>/dev/null
  done
  sleep 1
  for f in /tmp/st_pid_*; do
    [ -e "$f" ] || continue
    p=$(cat "$f" 2>/dev/null)
    [ -n "$p" ] && kill -9 "$p" 2>/dev/null
    rm -f "$f"
  done
}
trap cleanup EXIT

rm -f /tmp/st_pid_*
# 清掉上次 run 残留的参数（rosmaster 可能还在别的 shell 里）
rosnode list >/dev/null 2>&1 || {
  nohup roscore > /tmp/st_core.log 2>&1 & echo $! > /tmp/st_pid_core
  sleep 4
}

nohup "$PY" "$WS/src/mav_tracker/mav_tracker_nodes/scripts/video_source.py" \
  _source:=file _path:="$VID" _width:=960 _height:=720 _rate:=${RATE:-0} _loop:=false \
  > /tmp/st_src.log 2>&1 & echo $! > /tmp/st_pid_src
sleep 3
nohup "$PY" "$WS/src/mav_tracker/mav_tracker_nodes/scripts/detector_node.py" \
  > /tmp/st_det.log 2>&1 & echo $! > /tmp/st_pid_det
sleep 6

"$PY" "$ROOT/scripts/stage_timing.py" --duration "$DUR" --csv "/tmp/stage_$TAG.csv"
RC=$?

echo
echo "=== detector 分段明细（前几帧）==="
grep -E "分段明细" /tmp/st_det.log | head -6
echo "=== detector 吞吐 ==="
grep -E "已处理" /tmp/st_det.log | tail -3
exit $RC
