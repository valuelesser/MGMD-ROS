#!/usr/bin/env bash
set +u
source /opt/ros/noetic/setup.bash
source /home/gaodu/毕业设计/mgmd-linux/ros1_ws/devel/setup.bash
export DISPLAY=:0

echo "== procs =="
ps -eo pid,stat,etime,comm,args --no-headers | grep -E 'image_view|rosmaster|visualize_node|detector_node|video_source' | grep -v grep | head -10
echo "== nodes =="
timeout 5 rosnode list 2>&1
echo "== view log =="
tail -15 /tmp/ros_run/view.log 2>/dev/null
echo "== windows =="
for w in $(xdotool search --onlyvisible --name '.' 2>/dev/null); do
  echo "$w  $(xdotool getwindowname "$w" 2>/dev/null)  $(xdotool getwindowgeometry "$w" 2>/dev/null | tr '\n' ' ')"
done
echo "== mp4 =="
ls -l /home/gaodu/毕业设计/mgmd-linux/outputs/ros_viz/
echo "WINDOWS_DONE"
