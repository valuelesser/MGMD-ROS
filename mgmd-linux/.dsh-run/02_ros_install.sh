#!/usr/bin/env bash
# 安装 ROS1 Noetic（focal）。ROS1 官方签名 key 已于 2025-06-01 过期、Noetic �?EOL�?# 这里�?packages.ros.org 使用 trusted=yes（本机研究环境，可接受）�?set -eo pipefail
export DEBIAN_FRONTEND=noninteractive

echo "== 1) 先写源（trusted=yes），�?update =="
echo "deb [trusted=yes] https://mirrors.tuna.tsinghua.edu.cn/ros/ubuntu focal main" \
  > /etc/apt/sources.list.d/ros1.list
rm -f /usr/share/keyrings/ros-archive-keyring.asc /usr/share/keyrings/ros-archive-keyring.gpg
cat /etc/apt/sources.list.d/ros1.list

apt-get update 2>&1 | tail -6 || true

echo "== 2) 确认索引里有 noetic =="
apt-cache policy ros-noetic-ros-base | head -4

echo "== 3) base tools =="
apt-get install -y -qq curl gnupg >/dev/null

echo "== 4) install =="
apt-get install -y \
  ros-noetic-ros-base \
  ros-noetic-roslaunch \
  ros-noetic-message-generation \
  ros-noetic-message-runtime \
  ros-noetic-rqt-image-view \
  ros-noetic-image-view \
  ros-noetic-rosbash \
  build-essential cmake pkg-config \
  python3.8-venv python3-pip python3-catkin-pkg \
  xdotool ffmpeg

echo "== 5) verify =="
ls /opt/ros/noetic/setup.bash
set +u
. /opt/ros/noetic/setup.bash
rosversion -d
python3 -V
python3 -c "import rospy, genpy; print('rospy ok')"
echo "ROS_INSTALL_OK"
