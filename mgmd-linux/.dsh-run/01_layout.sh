#!/usr/bin/env bash
# 一次性准备 WSL 侧目录布局（以 root 运行）
set -euo pipefail

SRC=/mnt/c/Users/gaodu/Desktop/opencode/mgmd_project_20261104/毕业设计
DST=/home/gaodu/毕业设计/mgmd-linux

echo "== 0) /home/gaoduan -> /home/gaodu =="
[ -e /home/gaoduan ] || ln -s /home/gaodu /home/gaoduan
ls -ld /home/gaoduan

echo "== 1) 备份旧工程（ros2_ws 版） =="
if [ -d "$DST" ] && [ ! -e "${DST}.ros2-old" ]; then
  mv "$DST" "${DST}.ros2-old"
  echo "moved -> ${DST}.ros2-old"
fi

echo "== 2) 复制当前工作区 mgmd-linux =="
mkdir -p "$DST"
rsync -a --exclude 'outputs' --exclude '__pycache__' --exclude '.git' \
  --exclude '*.engine' "$SRC/mgmd-linux/" "$DST/"
mkdir -p "$DST/outputs/ros_viz" "$DST/outputs/logs"

echo "== 3) third_party/yolov5-v6.0（ONNX 版 LAD 的 NMS 依赖） =="
mkdir -p "$DST/third_party"
rsync -a "$SRC/tracking/third_party/yolov5-v6.0/" "$DST/third_party/yolov5-v6.0/"
ls "$DST/third_party/yolov5-v6.0/models/yolo.py" >/dev/null && echo "yolov5 ok"

echo "== 4) ONNX 权重 -> WEIGHTS_DIR =="
mkdir -p /home/gaodu/tracking/data_raw/mgmd_weights
cp -f "$SRC/tracking/data_raw/mgmd_weights/lac.onnx" \
      "$SRC/tracking/data_raw/mgmd_weights/yolov5s_GLAD-crop.onnx" \
      /home/gaodu/tracking/data_raw/mgmd_weights/
chown -R gaodu:gaodu /home/gaodu/毕业设计/mgmd-linux /home/gaodu/tracking
ls -l /home/gaodu/tracking/data_raw/mgmd_weights

echo "== 5) 数据自检 =="
ARD="/mnt/c/Users/gaodu/Desktop/毕业设计/mgmd/data/videos/视频/ARD-MAV"
ls -ld "$ARD" "$ARD/videos" "$ARD/Annotations"
ls -l "$ARD/videos/phantom30.mp4"

echo "LAYOUT_OK"
