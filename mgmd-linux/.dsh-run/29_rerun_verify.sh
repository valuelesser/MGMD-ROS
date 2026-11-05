#!/usr/bin/env bash
# Re-run the installed WSL environment without overwriting previous recordings.
set -eo pipefail
ROOT=/home/gaodu/毕业设计/mgmd-linux
RUN="$ROOT/outputs/rerun_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN"
source /opt/ros/noetic/setup.bash
source "$ROOT/ros1_ws/devel/setup.bash"
ROSPATH=$PYTHONPATH
SP="$ROOT/.venv38/lib/python3.8/site-packages"
export PYTHONPATH="$SP:$ROOT:$ROSPATH"
NV=$(find "$SP/nvidia" -maxdepth 3 -type d -name lib | tr '\n' ':')
export LD_LIBRARY_PATH="${NV}${LD_LIBRARY_PATH:-}"
export DISPLAY=:0 QT_X11_NO_MITSHM=1 PYTHONUNBUFFERED=1
export MGMD_ARD_ROOT=/mnt/c/Users/gaodu/Desktop/毕业设计/mgmd/data/videos/视频/ARD-MAV
export MGMD_YOLOV5_DIR="$ROOT/third_party/yolov5-v6.0"
CORE= VIEWER= LAUNCH=
cleanup() {
  if [ -n "$LAUNCH" ] && kill -0 "$LAUNCH" 2>/dev/null; then
    kill -INT "$LAUNCH" 2>/dev/null || true
    for i in $(seq 1 20); do kill -0 "$LAUNCH" 2>/dev/null || break; sleep 1; done
    if kill -0 "$LAUNCH" 2>/dev/null; then kill -TERM "$LAUNCH" 2>/dev/null || true; fi
  fi
  if [ -n "$VIEWER" ] && kill -0 "$VIEWER" 2>/dev/null; then
    kill -TERM "$VIEWER" 2>/dev/null || true
    for i in $(seq 1 5); do kill -0 "$VIEWER" 2>/dev/null || break; sleep 1; done
    # rospy's TERM handler can leave the Qt event loop running.
    if kill -0 "$VIEWER" 2>/dev/null; then kill -KILL "$VIEWER" 2>/dev/null || true; fi
  fi
  [ -z "$CORE" ] || kill -INT "$CORE" 2>/dev/null || true
  wait 2>/dev/null || true
}
trap cleanup EXIT
if timeout 3 rostopic list >/dev/null 2>&1; then
  echo 'An existing ROS master is active; refusing to interrupt it.'; exit 1
fi
echo "RUN=$RUN"
setsid roscore > "$RUN/core.log" 2>&1 < /dev/null & CORE=$!
for i in $(seq 1 25); do
  if timeout 2 rostopic list >/dev/null 2>&1; then break; fi
  sleep 1
done
timeout 3 rostopic list
PYTHONPATH="$ROOT:$ROSPATH" setsid python3 "$ROOT/.dsh-run/21_qt_viewer.py" > "$RUN/qt.log" 2>&1 < /dev/null & VIEWER=$!
sleep 3
setsid roslaunch mav_tracker_nodes track.launch video:=phantom30 rate:=20 limit:=1200 loop:=false width:=960 height:=720 device:=cuda d_min:=0.02 buffer:=6 encoder:=cpu visualize:=true out_video:="$RUN/phantom30.mp4" > "$RUN/launch.log" 2>&1 < /dev/null & LAUNCH=$!
sleep 18
timeout 5 rosnode list | tee "$RUN/nodes.txt"
timeout 8 rostopic hz /mav/detections > "$RUN/detection_hz.txt" 2>&1 || true
cat "$RUN/detection_hz.txt"
for i in 1 2; do
  xdotool search --name 'MGMD-ROS Qt' getwindowname 2>/dev/null | tee -a "$RUN/window.txt" || true
  sleep 3
done
tail -8 "$RUN/launch.log"
for i in $(seq 1 120); do
  grep -qE '回放结束|到达 limit' "$RUN/launch.log" && break
  kill -0 "$LAUNCH" 2>/dev/null || { echo 'roslaunch exited early'; exit 1; }
  sleep 1
done
grep -qE '回放结束|到达 limit' "$RUN/launch.log"
sleep 3
cleanup
trap - EXIT
ffprobe -v error -select_streams v:0 -show_entries stream=codec_name,width,height,nb_frames -show_entries format=duration,size -of default=nw=1 "$RUN/phantom30.mp4" | tee "$RUN/video_info.txt"
tail -20 "$RUN/launch.log"
echo "RERUN_DONE=$RUN"
