#!/usr/bin/env bash
# 优雅停止实时演示：先让 visualize_node 写完 mp4，再清其余进程
set +u
ROOT=/home/gaodu/毕业设计/mgmd-linux
LOG=$ROOT/outputs/logs

echo "[1] SIGTERM visualize_node（让它 close 写完整 mp4）"
VP=$(ps -eo pid,args --no-headers | awk '$2 !~ /bash|sh/ && /visualize_node\.py/ {print $1}' | head -1)
if [ -n "$VP" ]; then
  kill -TERM "$VP" 2>/dev/null
  for i in $(seq 1 15); do kill -0 "$VP" 2>/dev/null || break; sleep 1; done
  kill -0 "$VP" 2>/dev/null && echo "    仍在运行，稍后强杀" || echo "    已退出 pid=$VP"
else
  echo "    没有 visualize_node"
fi

echo "[2] 关闭 Qt 窗口"
QP=$(ps -eo pid,args --no-headers | awk '$2 !~ /bash|sh/ && /21_qt_viewer\.py/ {print $1}' | head -1)
if [ -n "$QP" ]; then
  kill -TERM "$QP" 2>/dev/null
  for i in $(seq 1 8); do kill -0 "$QP" 2>/dev/null || break; sleep 1; done
  echo "    Qt viewer pid=$QP 已停止"
else
  echo "    没有 Qt viewer"
fi

echo "[3] 清理其余 ROS 进程"
bash "$ROOT/.dsh-run/13_kill_all.sh" >/dev/null 2>&1 || true

echo "[4] 残留检查"
ps -eo pid,stat,args --no-headers | grep -E 'rosmaster|roslaunch|detector_node|visualize_node|video_source|21_qt_viewer' | grep -v grep || echo "    无残留"

echo "[5] 录制文件"
OUT=$ROOT/outputs/ros_viz/live_phantom30_960x720.mp4
ls -lh "$OUT" 2>/dev/null
ffprobe -v error -select_streams v:0 -show_entries stream=codec_name,width,height,nb_frames \
  -show_entries format=duration,size -of default=nw=1 "$OUT" 2>&1 | head -8
echo "STOP_DONE"
