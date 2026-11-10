#!/usr/bin/env bash
set +u
source /opt/ros/noetic/setup.bash
source /home/gaodu/毕业设计/mgmd-linux/ros1_ws/devel/setup.bash
export DISPLAY=:0
echo "== nodes =="
timeout 5 rosnode list 2>&1
echo "== det hz =="
timeout 8 rostopic hz /mav/detections 2>&1 | head -3
echo "== progress =="
grep '已处理' /home/gaodu/毕业设计/mgmd-linux/outputs/logs/live.launch.log | tail -1
echo "== mp4 =="
ls -lh /home/gaodu/毕业设计/mgmd-linux/outputs/ros_viz/live_phantom30_960x720.mp4 2>/dev/null
echo "== window =="
W=$(xdotool search --name "MGMD-ROS Qt" | head -1)
echo "id=$W title=[$(xdotool getwindowname "$W" 2>/dev/null)]"
echo "== git-ish files =="
ls /home/gaodu/毕业设计/mgmd-linux/outputs/logs/ 2>/dev/null
echo "STATUS_DONE"
