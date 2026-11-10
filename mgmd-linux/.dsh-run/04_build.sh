#!/usr/bin/env bash
# catkin 编译 mav_tracker_msgs + mav_tracker_nodes（以 root 运行）
set -eo pipefail
ROOT=/home/gaodu/毕业设计/mgmd-linux
WS=$ROOT/ros1_ws

set +u
source /opt/ros/noetic/setup.bash
cd "$WS"
catkin_make -j"$(nproc)"
source "$WS/devel/setup.bash"

echo "== rosmsg =="
rosmsg show mav_tracker_msgs/DetectionArray | head -20
echo "== devel 产物 =="
ls "$WS/devel/lib/mav_tracker_nodes" 2>/dev/null | head
ls "$WS/devel/lib/python3/dist-packages/mav_tracker_msgs/msg" 2>/dev/null | head
chown -R gaodu:gaodu "$WS"
echo "BUILD_OK"
