#!/usr/bin/env bash
# 停掉本轮 ROS 进程（按 PID 文件，不用 pkill -f）
set +u
for f in /tmp/ros_run/*.pid; do
  [ -e "$f" ] || continue
  p=$(cat "$f" 2>/dev/null)
  [ -n "$p" ] || continue
  # 先 TERM（让 visualize_node 写完整 mp4），再 KILL
  kill -TERM "$p" 2>/dev/null
done
sleep 4
for f in /tmp/ros_run/*.pid; do
  [ -e "$f" ] || continue
  p=$(cat "$f" 2>/dev/null)
  [ -n "$p" ] || continue
  kill -KILL "$p" 2>/dev/null
  rm -f "$f"
done
# roslaunch 的子进程（按 comm 精确匹配，避免误杀 shell）
for name in rosmaster rosout video_source.py detector_node.py visualize_node.py; do
  case "$name" in
    *.py) ps -eo pid,comm,args --no-headers | awk -v n="$name" '$2 != "bash" && index($0, n) {print $1}' ;;
    *) pgrep -x "$name" 2>/dev/null ;;
  esac | while read -r pid; do
    [ -n "$pid" ] && kill -KILL "$pid" 2>/dev/null
  done
done
echo "STOPPED"
