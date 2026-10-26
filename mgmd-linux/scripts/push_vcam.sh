#!/usr/bin/env bash
# push_vcam.sh — 把视频推流到虚拟摄像头（SPEC §8.31 路径3）
#
# 用法：
#   bash scripts/push_vcam.sh                              # 默认 phantom70 循环推流
#   bash scripts/push_vcam.sh phantom63.mp4                # 指定视频
#   bash scripts/push_vcam.sh /path/to/v.mp4 --once        # 播一遍就退出
#   VIDEO_NR=11 bash scripts/push_vcam.sh v.mp4            # 指定设备号
#
# 说明：
#   -re  按原速推流（关键：不加则瞬间推完，摄像头读到的是最后一帧）
#   循环推流让 live_track.py 可以随时连接
set -e

VIDEO_NR="${VIDEO_NR:-10}"
ARD="/media/gaoduan/0C1CFCAF1CFC94C4/Users/gaodu/Desktop/毕业设计/mgmd/data/videos/视频/ARD-MAV/videos"
SRC="${1:-$ARD/phantom70.mp4}"
MODE="${2:-loop}"

DEV="/dev/video$VIDEO_NR"
if [ ! -e "$DEV" ]; then
  echo "[push] 虚拟摄像头 $DEV 不存在" >&2
  echo "[push] 先运行：sudo bash scripts/setup_vcam.sh" >&2
  exit 1
fi
if [ ! -f "$SRC" ]; then
  echo "[push] 视频不存在: $SRC" >&2
  exit 1
fi

echo "[push] 源: $SRC"
echo "[push] 目标: $DEV  ($(ffprobe -v error -select_streams v:0 \
  -show_entries stream=width,height,r_frame_rate -of csv=p=0 "$SRC" 2>/dev/null))"

ARGS=(-re -i "$SRC" -f v4l2 -pix_fmt yuv420p "$DEV")
if [ "$MODE" != "--once" ]; then
  ARGS=(-stream_loop -1 "${ARGS[@]}")
  echo "[push] 循环推流，Ctrl-C 停止"
fi
exec ffmpeg -loglevel warning "${ARGS[@]}"
