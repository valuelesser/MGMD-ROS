#!/usr/bin/env bash
# 只启动 roscore（setsid 脱离会话，跨 wsl.exe 调用存活）
set +u
ROOT=/home/gaodu/毕业设计/mgmd-linux
source /opt/ros/noetic/setup.bash
source "$ROOT/ros1_ws/devel/setup.bash"
mkdir -p /tmp/ros_run
pkill -x rosmaster 2>/dev/null || true
pkill -x rosout 2>/dev/null || true
sleep 1
setsid nohup roscore > /tmp/ros_run/core.log 2>&1 < /dev/null &
for i in $(seq 1 20); do
  timeout 3 rostopic list >/dev/null 2>&1 && break
  sleep 1
done
timeout 5 rostopic list | head -3
pgrep -x rosmaster >/dev/null && echo "CORE_OK" || { echo "CORE_FAIL"; tail -20 /tmp/ros_run/core.log; }
