#!/usr/bin/env bash
# 实时演示：roscore + rqt_image_view 窗口 + 跟踪流水线（SPEC-ROS §23.3.4）
#
# **用户决定：以后展示统一用这个办法。**
#
# 用法：
#   bash scripts/demo_live.sh                    # 默认 phantom30
#   bash scripts/demo_live.sh phantom08          # 换视频
#   bash scripts/demo_live.sh phantom30 25 2000  # 视频 帧率 帧数上限
#
# 停止：
#   bash scripts/demo_live.sh stop
#
# 注意：
#   - 全程用 PID 文件管理，**绝不用 pkill -f / pgrep -f**（会杀掉自身 shell）
#   - 关闭必须先 SIGTERM 让 visualize_node 收尾，SIGKILL 会让 mp4 损坏
#   - PYTHONUNBUFFERED=1 必须设，否则 roslaunch 下日志不可见（§23.3.2）
set +u

ROOT=/home/gaoduan/毕业设计/mgmd-linux
WS=$ROOT/ros1_ws
RUNDIR=/tmp/demo_live
mkdir -p "$RUNDIR"

stop_all() {
  # 先 SIGTERM visualize，让它写完 mp4 的 moov atom
  local viz
  # pgrep -f 会匹配到本脚本自身；用 python 进程 + args 精确筛选，且排除自己是 bash
  viz=$(pgrep -f "visualize_node.py" --ns $$ 2>/dev/null | while read p; do
          [ "$(ps -o comm= -p "$p" 2>/dev/null)" != "bash" ] && echo "$p"
        done | head -1)
  [ -n "$viz" ] && kill -TERM "$viz" 2>/dev/null && sleep 5
  for f in "$RUNDIR"/*.pid; do
    [ -e "$f" ] || continue
    p=$(cat "$f" 2>/dev/null)
    [ -n "$p" ] && kill -9 "$p" 2>/dev/null
    rm -f "$f"
  done
  for pat in rosmaster rosout; do
    pgrep -x "$pat" 2>/dev/null | xargs -r kill -9 2>/dev/null
  done
  for pat in detector_node.py video_source.py; do
    pgrep -f "$pat" 2>/dev/null | while read p; do
      [ "$p" != "$$" ] && [ "$(ps -o comm= -p "$p" 2>/dev/null)" != "bash" ] && kill -9 "$p" 2>/dev/null
    done
  done
  pgrep -f "rqt_image_view" 2>/dev/null | while read p; do
    [ "$(ps -o comm= -p "$p" 2>/dev/null)" = "python3" ] && kill -9 "$p" 2>/dev/null
  done
  echo "已停止全部服务"
}

if [ "${1:-}" = "stop" ]; then
  stop_all; exit 0
fi

VIDEO=${1:-phantom30}
RATE=${2:-25}
LIMIT=${3:-0}

source /opt/ros/noetic/setup.bash
source "$WS/devel/setup.bash"
source "$ROOT/scripts/env_setup_ros38.sh" >/dev/null 2>&1
export PYTHONUNBUFFERED=1 DISPLAY=:0

stop_all >/dev/null 2>&1

echo "[1/3] 启动 roscore ..."
nohup roscore > "$RUNDIR/core.log" 2>&1 &
echo $! > "$RUNDIR/core.pid"
sleep 7
timeout 8 rostopic list >/dev/null 2>&1 && echo "      ✅ roscore 就绪" || { echo "      ❌ 失败"; exit 1; }

echo "[2/3] 打开 rqt_image_view 窗口 ..."
nohup rosrun rqt_image_view rqt_image_view > "$RUNDIR/rqt.log" 2>&1 &
echo $! > "$RUNDIR/rqt.pid"
sleep 8
win=$(xdotool search --name "rqt_image_view" 2>/dev/null | tail -1)
if [ -n "$win" ]; then
  xdotool windowactivate "$win" 2>/dev/null
  xdotool windowraise "$win" 2>/dev/null
  xdotool windowmove "$win" 60 60 2>/dev/null
  xdotool windowsize "$win" 1000 780 2>/dev/null
  echo "      ✅ 窗口就绪（请在下拉框选 /mav/debug_image）"
else
  echo "      ⚠️ 未找到窗口，可能需要手动查看"
fi

echo "[3/3] 启动跟踪流水线 video=$VIDEO rate=$RATE limit=$LIMIT ..."
OUT=$ROOT/outputs/ros_viz/live_demo.mp4
mkdir -p "$ROOT/outputs/ros_viz"
rm -f "$OUT"
nohup roslaunch mav_tracker_nodes track.launch \
    video:=$VIDEO width:=960 height:=720 rate:=$RATE limit:=$LIMIT loop:=true \
    out_video:=$OUT encoder:=nvenc > "$RUNDIR/launch.log" 2>&1 &
echo $! > "$RUNDIR/launch.pid"
sleep 12

echo
echo "=================== 就绪 ==================="
echo "窗口：在 rqt_image_view 下拉框选 /mav/debug_image"
echo "录制：$OUT"
echo "日志：$RUNDIR/launch.log"
echo "停止：bash scripts/demo_live.sh stop"
echo "==========================================="
grep -E "已处理|已写" "$RUNDIR/launch.log" 2>/dev/null | tail -2
