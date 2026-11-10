#!/usr/bin/env bash
set +u
source /opt/ros/noetic/setup.bash
echo "ROS_DISTRO=$ROS_DISTRO"
rosversion -d
command -v roscore
command -v roslaunch
command -v rostopic
python3 - <<'PY'
import sys
print("python", sys.version.split()[0])
import rospy, genpy
print("rospy ok", rospy.__file__)
import cv2
print("cv2", cv2.__version__)
PY
dpkg -l python3.8-venv | tail -1
ls -l /opt/ros/noetic/setup.bash
echo "VERIFY_DONE"
