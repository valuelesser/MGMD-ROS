#!/usr/bin/env bash
# ROS + WSLg 弹窗连通性测试（先不起检测节点）
set +u
source /opt/ros/noetic/setup.bash
source /home/gaodu/毕业设计/mgmd-linux/ros1_ws/devel/setup.bash
export DISPLAY=:0
export PYTHONUNBUFFERED=1

mkdir -p /tmp/gui_test
pkill -x roscore 2>/dev/null || true
sleep 1

echo "[1] roscore"
nohup roscore > /tmp/gui_test/core.log 2>&1 &
echo $! > /tmp/gui_test/core.pid
for i in $(seq 1 15); do
  timeout 3 rostopic list >/dev/null 2>&1 && break
  sleep 1
done
timeout 5 rostopic list | head -3 || echo "roscore 未就绪"

echo "[2] image_view"
nohup rosrun image_view image_view image:=/mav/debug_image \
  > /tmp/gui_test/iv.log 2>&1 &
echo $! > /tmp/gui_test/iv.pid
sleep 8

echo "[3] 进程"
ps -o pid,stat,etime,cmd -p "$(cat /tmp/gui_test/iv.pid)" 2>/dev/null || echo "image_view 已退出"
echo "[4] 窗口"
xdotool search --name "image" 2>/dev/null | while read -r w; do
  echo "win=$w name=$(xdotool getwindowname "$w" 2>/dev/null)"
done
echo "[5] 日志"
tail -20 /tmp/gui_test/iv.log
echo GUI_TEST_DONE
