#!/usr/bin/env bash
# 彻底清理 ROS 进程/节点（按进程名精确匹配，不用 pkill -f 杀 shell）
set +u

echo "[1] roslaunch / rosout / master"
for p in $(pgrep -x roslaunch 2>/dev/null) $(pgrep -x rosmaster 2>/dev/null) \
         $(pgrep -x rosout 2>/dev/null) $(pgrep -x image_view 2>/dev/null); do
  kill -KILL "$p" 2>/dev/null
done

echo "[2] 节点脚本"
for pat in video_source.py detector_node.py visualize_node.py target_mover.py; do
  ps -eo pid,comm,args --no-headers 2>/dev/null \
    | awk -v n="$pat" '$2 != "bash" && $2 != "sh" && index($0, n) {print $1}' \
    | while read -r pid; do kill -KILL "$pid" 2>/dev/null; done
done

sleep 2
echo "[3] 残留"
ps -eo pid,comm,args --no-headers | grep -E 'rosmaster|image_view|detector_node|visualize_node|video_source' | grep -v grep || echo "无残留"
echo "KILLED"
