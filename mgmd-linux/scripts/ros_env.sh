#!/usr/bin/env bash
# ROS 仿真/演示脚本的公共环境与进程管理（被 check_gazebo.sh / demo_live.sh
# / test_nodes.sh source）。
#
# **为什么要抽出来**：三个脚本各自重复了「设环境变量 + 用 PID 文件杀进程」，
# 而这部分恰恰是最容易出错的（pkill -f 杀掉自身、PYTHONUNBUFFERED 漏设、
# Gazebo 环境变量漏设导致静默失败）。集中一处，改一次全生效。
#
# 用法：
#   source "$(dirname "$0")/ros_env.sh"
#
# 提供：
#   ros_env_init          设 ROS + Gazebo 环境变量（含 5 个静默失败陷阱的规避）
#   kill_pidfile <dir>    按 PID 文件杀掉某目录下记录的全部进程
#   kill_ros_procs        杀掉常见 ROS/Gazebo 进程（**不含 gzclient GUI**）
#   stop_viz_gracefully   先 SIGTERM visualize_node 让它写完 mp4

[ -n "${_ROS_ENV_LOADED:-}" ] && return 0
_ROS_ENV_LOADED=1

ROOT=/home/gaoduan/毕业设计/mgmd-linux
WS=$ROOT/ros1_ws

# --------------------------------------------------------------------------- #
# 环境变量
# --------------------------------------------------------------------------- #
ros_env_init() {
  # set -u 会让 ROS 的 setup.bash 报「未绑定变量」
  set +u
  source /opt/ros/noetic/setup.bash
  source "$WS/devel/setup.bash"

  # ---- SPEC-ROS §23.3.2：不设则 roslaunch 日志全被块缓冲吞掉 ----
  export PYTHONUNBUFFERED=1
  export DISPLAY=${DISPLAY:-:0}

  # ---- SPEC-ROS §11.3.2：Gazebo 三个必须设的变量 ----
  # 不设 GAZEBO_MODEL_PATH → model://ground_plane 找不到，world 静默不加载
  export GAZEBO_MODEL_PATH=${GAZEBO_MODEL_PATH:-/usr/share/gazebo-11/models:$HOME/.gazebo/models}
  # 不设则 libCameraPlugin.so 找不到（该目录不在 ldconfig 缓存里）
  export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/gazebo-11/plugins:${LD_LIBRARY_PATH:-}
  # Gazebo 默认 11345，与 ROS 的 11311 不同，别搞混
  export GAZEBO_MASTER_URI=${GAZEBO_MASTER_URI:-http://localhost:11345}

  # 项目内的 py3.8 环境（onnxruntime / rospy 都在这里）
  if [ -f "$ROOT/scripts/env_setup_ros38.sh" ]; then
    source "$ROOT/scripts/env_setup_ros38.sh" >/dev/null 2>&1
  fi
}

# --------------------------------------------------------------------------- #
# 进程管理（**绝不用 pkill -f / pgrep -f 匹配命令行，会杀掉调用它的 shell**）
# --------------------------------------------------------------------------- #
_kill_pid() {
  local p=$1
  [ -n "$p" ] || return 0
  # 只杀真是目标程序名的进程，绝不杀 bash（否则就是杀自己）
  case "$(ps -o comm= -p "$p" 2>/dev/null)" in
    bash|sh) return 0 ;;
  esac
  { kill -9 "$p"; } 2>/dev/null
  disown "$p" 2>/dev/null
}

kill_pidfile() {
  local dir=$1 f p
  for f in "$dir"/*.pid; do
    [ -e "$f" ] || continue
    p=$(cat "$f" 2>/dev/null)
    _kill_pid "$p"
    rm -f "$f"
  done
}

# 先让 visualize_node 收到 SIGTERM，把 mp4 的 moov atom 写完
# （SIGKILL 会让文件损坏，SPEC-ROS §23.3.6）
stop_viz_gracefully() {
  local viz
  viz=$(ps -eo pid,comm,args --no-headers 2>/dev/null \
        | awk '$2!="bash" && $0 ~ /visualize_node\.py/ {print $1}' | head -1)
  if [ -n "$viz" ]; then
    { kill -TERM "$viz"; } 2>/dev/null
    disown "$viz" 2>/dev/null
    sleep 5
  fi
}

kill_ros_procs() {
  local pat p
  for pat in rosmaster rosout gzserver; do
    for p in $(pgrep -x "$pat" 2>/dev/null); do _kill_pid "$p"; done
  done
  for pat in detector_node.py video_source.py visualize_node.py target_mover.py; do
    for p in $(pgrep -f "$pat" 2>/dev/null); do _kill_pid "$p"; done
  done
}

# 一键全停：先优雅收尾视频，再清进程
stop_all_ros() {
  stop_viz_gracefully
  kill_ros_procs
}
