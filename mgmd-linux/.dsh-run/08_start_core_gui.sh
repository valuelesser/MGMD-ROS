#!/usr/bin/env bash
# 启动 roscore + image_view（真实弹窗，订阅 /mav/debug_image）
set +u
ROOT=/home/gaodu/毕业设计/mgmd-linux
source /opt/ros/noetic/setup.bash
source "$ROOT/ros1_ws/devel/setup.bash"
export DISPLAY=:0
export PYTHONUNBUFFERED=1
mkdir -p /tmp/ros_run

# 清掉上一轮（精确进程名）
pkill -x roscore 2>/dev/null || true
pkill -x image_view 2>/dev/null || true
if [ -f /tmp/ros_run/view.pid ]; then kill -TERM "$(cat /tmp/ros_run/view.pid)" 2>/dev/null; fi
if [ -f /tmp/ros_run/core.pid ]; then kill -TERM "$(cat /tmp/ros_run/core.pid)" 2>/dev/null; fi
sleep 1

echo "[1] roscore"
setsid nohup roscore > /tmp/ros_run/core.log 2>&1 &
echo $! > /tmp/ros_run/core.pid
for i in $(seq 1 20); do
  timeout 3 rostopic list >/dev/null 2>&1 && break
  sleep 1
done
timeout 5 rostopic list | head -3

echo "[2] image_view (/mav/debug_image)"
setsid nohup rosrun image_view image_view image:=/mav/debug_image \
  > /tmp/ros_run/view.log 2>&1 &
echo $! > /tmp/ros_run/view.pid
sleep 5
ps -o pid,stat,etime,cmd -p "$(cat /tmp/ros_run/view.pid)" || echo "image_view 退出了"
echo "CORE_GUI_OK"
