#!/usr/bin/env bash
# 实时演示：roscore + 跟踪流水线（+ 可选 rqt_image_view 窗口）
# SPEC-ROS §23.3.4
#
# **用户决定：以后展示统一用这个办法。**
#
# 用法：
#   bash scripts/demo_live.sh                        # 默认 phantom30，不开窗
#   bash scripts/demo_live.sh phantom08              # 换视频
#   bash scripts/demo_live.sh phantom30 25 2000      # 视频 帧率 帧数上限
#   bash scripts/demo_live.sh phantom30 25 2000 rqt  # 同时开实时窗口
#   bash scripts/demo_live.sh stop                   # 停止
#
# 注意：
#   - limit:=0 + loop:=true 会无限跑（23 分钟即 710 MB），演示建议设 limit
#   - 关闭必须 SIGTERM，SIGKILL 会让 mp4 损坏（§23.3.6）
#   - 窗口里要选 /mav/debug_image 才有画面
set +u
source "$(dirname "${BASH_SOURCE[0]}")/ros_env.sh"
ros_env_init

RUNDIR=/tmp/demo_live
mkdir -p "$RUNDIR"

if [ "${1:-}" = "stop" ]; then
  stop_viz_gracefully
  kill_pidfile "$RUNDIR"
  kill_ros_procs
  echo "已停止全部服务"
  exit 0
fi

VIDEO=${1:-phantom30}
RATE=${2:-25}
LIMIT=${3:-0}
WITH_RQT=${4:-}
OUT=$ROOT/outputs/ros_viz/live_demo.mp4
mkdir -p "$ROOT/outputs/ros_viz"

stop_viz_gracefully
kill_pidfile "$RUNDIR"
kill_ros_procs >/dev/null 2>&1

echo "[1/3] 启动 roscore ..."
nohup roscore > "$RUNDIR/core.log" 2>&1 &
echo $! > "$RUNDIR/core.pid"
sleep 7
if ! timeout 8 rostopic list >/dev/null 2>&1; then
  echo "      ❌ roscore 启动失败，见 $RUNDIR/core.log"; exit 1
fi
echo "      ✅ roscore 就绪"

if [ "$WITH_RQT" = "rqt" ]; then
  echo "[2/3] 打开 rqt_image_view 窗口 ..."
  nohup rosrun rqt_image_view rqt_image_view > "$RUNDIR/rqt.log" 2>&1 &
  echo $! > "$RUNDIR/rqt.pid"
  sleep 8
  win=$(xdotool search --name "rqt_image_view" 2>/dev/null | tail -1)
  if [ -n "$win" ]; then
    xdotool windowactivate "$win" 2>/dev/null
    xdotool windowraise  "$win" 2>/dev/null
    xdotool windowmove   "$win" 60 60 2>/dev/null
    xdotool windowsize   "$win" 1000 780 2>/dev/null
    echo "      ✅ 窗口就绪（下拉框选 /mav/debug_image）"
  else
    echo "      ⚠️ 未找到窗口"
  fi
else
  echo "[2/3] 跳过实时窗口（末尾加 rqt 参数可开启）"
fi

echo "[3/3] 启动跟踪流水线 video=$VIDEO rate=$RATE limit=$LIMIT ..."
rm -f "$OUT"
nohup roslaunch mav_tracker_nodes track.launch \
    video:=$VIDEO width:=960 height:=720 rate:=$RATE limit:=$LIMIT loop:=true \
    out_video:=$OUT encoder:=nvenc > "$RUNDIR/launch.log" 2>&1 &
echo $! > "$RUNDIR/launch.pid"
sleep 12

echo
echo "=================== 就绪 ==================="
echo "录制：$OUT"
echo "日志：$RUNDIR/launch.log"
echo "停止：bash scripts/demo_live.sh stop"
echo "==========================================="
grep -E "已处理|已写" "$RUNDIR/launch.log" 2>/dev/null | tail -2
