#!/usr/bin/env bash
# 真实弹窗：image_view 订阅 /mav/debug_image（WSLg DISPLAY=:0）
set -eo pipefail
set +u
source /opt/ros/noetic/setup.bash
source /home/gaodu/毕业设计/mgmd-linux/ros1_ws/devel/setup.bash
set -u
export DISPLAY=:0
export QT_X11_NO_MITSHM=1
exec rosrun image_view image_view image:=/mav/debug_image _autosize:=true
