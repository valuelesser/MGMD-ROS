#!/usr/bin/env bash
# 三节点单独测试（SPEC-ROS §23.3）
#
# 用法：
#   bash scripts/test_nodes.sh start     启动 roscore
#   bash scripts/test_nodes.sh stop      停掉全部
#   bash scripts/test_nodes.sh status    看进程与话题
#   bash scripts/test_nodes.sh round     跑一轮完整测试（三节点）
set +u
source "$(dirname "${BASH_SOURCE[0]}")/ros_env.sh"
ros_env_init

PY=$ROOT/.venv38/bin/python
SCRIPTS=$WS/src/mav_tracker/mav_tracker_nodes/scripts
RUNDIR=/tmp/node_test
mkdir -p "$RUNDIR"

_launch() {   # $1=名字 $2=日志 $3..=命令
  local name=$1 log=$2; shift 2
  nohup "$@" > "$log" 2>&1 &
  echo $! > "$RUNDIR/$name.pid"
  echo "  启动 $name  PID=$(cat "$RUNDIR/$name.pid")"
}

cmd_start() {
  _launch core "$RUNDIR/core.log" roscore
  sleep 6
  timeout 8 rostopic list >/dev/null 2>&1 \
    && echo "  roscore 就绪" || echo "  ❌ roscore 未就绪"
}

cmd_status() {
  echo "--- 进程 ---"
  local f n p
  for f in "$RUNDIR"/*.pid; do
    [ -e "$f" ] || continue
    n=$(basename "$f" .pid); p=$(cat "$f" 2>/dev/null)
    if [ -n "$p" ] && kill -0 "$p" 2>/dev/null; then
      echo "  $n ($p) 运行中"
    else
      echo "  $n 已退出"
    fi
  done
  echo "--- 节点 ---"
  timeout 8 rosnode list 2>&1 | grep -v rosout
}

cmd_round() {
  echo "=== 1/3 video_source ==="
  _launch src "$RUNDIR/src.log" "$PY" "$SCRIPTS/video_source.py" \
      _source:=file _path:=phantom30 _width:=960 _height:=720 \
      _rate:=10 _limit:=60 _loop:=false
  sleep 12
  echo "  日志:"; sed 's/^/    /' "$RUNDIR/src.log" 2>/dev/null | head -5
  echo "  话题频率:"; timeout 6 rostopic hz /camera/image_raw 2>&1 \
    | head -2 | sed 's/^/    /'

  echo "=== 2/3 detector_node ==="
  _launch det "$RUNDIR/det.log" "$PY" "$SCRIPTS/detector_node.py" _device:=cuda
  sleep 14
  echo "  日志:"; sed 's/^/    /' "$RUNDIR/det.log" 2>/dev/null | head -8

  echo "=== 3/3 visualize_node ==="
  _launch viz "$RUNDIR/viz.log" "$PY" "$SCRIPTS/visualize_node.py" \
      _out:=$RUNDIR/out.mp4 _encoder:=nvenc
  sleep 20
  echo "  日志:"; sed 's/^/    /' "$RUNDIR/viz.log" 2>/dev/null | head -5

  echo "=== 结果 ==="
  ls -la "$RUNDIR"/out.mp4 2>/dev/null | sed 's/^/  /' || echo "  ❌ 无输出视频"
}

case "${1:-round}" in
  start)  cmd_start ;;
  stop)   stop_viz_gracefully; kill_pidfile "$RUNDIR"; kill_ros_procs
          echo "已停止全部" ;;
  status) cmd_status ;;
  round)  cmd_start; cmd_round; cmd_status ;;
  *) echo "用法: $0 {start|stop|status|round}"; exit 1 ;;
esac
