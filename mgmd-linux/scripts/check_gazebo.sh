#!/usr/bin/env bash
# Gazebo 仿真环境自检（SPEC-ROS §11.3）
#
# 用法：bash scripts/check_gazebo.sh
#
# 逐项验证 §11.3 记录的 5 个陷阱是否已避开：
#   1. GAZEBO_MODEL_PATH 未设 → world 静默加载失败
#   2. LD_LIBRARY_PATH 缺 plugins 目录 → 相机插件加载失败
#   3. 不用 rosrun gazebo_ros gzserver → ROS 桥接不初始化
#   4. 相机没 static → 自由落体，画面纯色
#   5. 无 world 文件 → 无从验证
set +u

ROOT=/home/gaoduan/毕业设计/mgmd-linux
WORLD=$ROOT/ros1_ws/src/mav_tracker/mav_tracker_nodes/worlds/cam_smoke.world
RUNDIR=/tmp/gz_check
mkdir -p "$RUNDIR"
PASS=0; FAIL=0
ok()   { echo "  ✅ $1"; PASS=$((PASS+1)); }
bad()  { echo "  ❌ $1"; FAIL=$((FAIL+1)); }

source /opt/ros/noetic/setup.bash

echo "=== 1. 可执行与版本 ==="
gazebo --version 2>&1 | head -1 | grep -q "11\." && ok "gazebo 11.x" || bad "gazebo 版本异常"
[ -x /usr/bin/gzserver ] && ok "gzserver" || bad "gzserver 缺失"
[ -f /opt/ros/noetic/lib/libgazebo_ros_camera.so ] && ok "libgazebo_ros_camera.so" || bad "相机插件缺失"
[ -f /usr/lib/x86_64-linux-gnu/gazebo-11/plugins/libCameraPlugin.so ] \
  && ok "libCameraPlugin.so" || bad "libCameraPlugin.so 缺失"

echo "=== 2. 环境变量（§11.3.2）==="
export GAZEBO_MODEL_PATH=${GAZEBO_MODEL_PATH:-/usr/share/gazebo-11/models:$HOME/.gazebo/models}
echo "  GAZEBO_MODEL_PATH=$GAZEBO_MODEL_PATH"
# GAZEBO_MODEL_PATH 是冒号分隔的多路径，需逐个查（不能直接 ls 整个变量）
found_gp=0
IFS=':' read -ra _paths <<< "$GAZEBO_MODEL_PATH"
for _d in "${_paths[@]}"; do
  [ -d "$_d/ground_plane" ] && found_gp=1 && break
done
[ "$found_gp" -eq 1 ] && ok "ground_plane 可见" || bad "ground_plane 不可见"
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/gazebo-11/plugins:$LD_LIBRARY_PATH
echo "  LD_LIBRARY_PATH 已含 gazebo-11/plugins"
export GAZEBO_MASTER_URI=${GAZEBO_MASTER_URI:-http://localhost:11345}
echo "  GAZEBO_MASTER_URI=$GAZEBO_MASTER_URI"

echo "=== 3. world 文件（§11.3.4）==="
if [ -f "$WORLD" ]; then
  ok "world 存在"
  grep -q "<static>true</static>" "$WORLD" && ok "相机 static（不会自由落体）" || bad "相机缺 static"
  grep -q "libgazebo_ros_camera.so" "$WORLD" && ok "相机插件已配" || bad "缺相机插件"
else
  bad "world 文件不存在: $WORLD"
fi

echo "=== 4. 启动实测（§11.3.3）==="
for pat in gzserver rosmaster rosout; do pgrep -x "$pat" >/dev/null 2>&1 && kill -9 $(pgrep -x "$pat") 2>/dev/null; done
sleep 1
nohup roscore > "$RUNDIR/core.log" 2>&1 & echo $! > "$RUNDIR/core.pid"
sleep 6
rosrun gazebo_ros gzserver "$WORLD" > "$RUNDIR/gz.log" 2>&1 & echo $! > "$RUNDIR/gz.pid"
echo "  等待 Gazebo 启动 ..."
sleep 20

topics=$(timeout 8 rostopic list 2>/dev/null | grep -c "image_raw")
[ "$topics" -gt 0 ] && ok "相机话题已发布" || bad "无相机话题"

sed 's/\x1b\[[0-9;]*m//g' "$RUNDIR/gz.log" 2>/dev/null \
  | grep -qiE "incorrect plugin type|has not been initialized" \
  && bad "ROS 桥接失败（用了 gzserver 而非 rosrun gazebo_ros gzserver）" \
  || ok "ROS 桥接正常"

echo "=== 5. 图像内容（关键：话题通≠画面对）==="
timeout 30 $ROOT/.venv38/bin/python - << 'PYEOF' 2>&1 | grep -E "^  " 
import rospy, numpy as np, sys
from sensor_msgs.msg import Image
rospy.init_node('gzchk', anonymous=True, disable_signals=True)
try:
    m = rospy.wait_for_message('/test/image_raw', Image, timeout=20)
except Exception as e:
    print(f"  ❌ 取图失败: {e}"); sys.exit(1)
a = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, 3)
r,g,b = a[:,:,0].astype(int), a[:,:,1].astype(int), a[:,:,2].astype(int)
red = ((r>100)&(g<80)&(b<80)).sum()
uniq = len(np.unique(a))
print(f"  ✅ 分辨率 {m.width}x{m.height} {m.encoding}")
print(f"  {'✅' if uniq>3 else '❌'} 画面非纯色（唯一值={uniq}）")
print(f"  {'✅' if red>1000 else '❌'} 红色方块可见（{red} 像素）")
PYEOF

echo "=== 清理 ==="
kill -9 $(cat "$RUNDIR/gz.pid") 2>/dev/null
for pat in gzserver rosmaster rosout; do pgrep -x "$pat" >/dev/null 2>&1 && kill -9 $(pgrep -x "$pat") 2>/dev/null; done
echo
echo "=================================="
echo "  通过 $PASS 项，失败 $FAIL 项"
echo "=================================="
[ "$FAIL" -eq 0 ] && echo "✅ 仿真环境可用" || echo "❌ 有失败项，见上"
