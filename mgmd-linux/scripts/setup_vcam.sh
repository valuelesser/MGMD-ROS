#!/usr/bin/env bash
# setup_vcam.sh — 加载 v4l2loopback 虚拟摄像头（SPEC §8.31 路径3）
#
# 需要 root。用法：
#   sudo bash scripts/setup_vcam.sh              # 创建 /dev/video10
#   sudo bash scripts/setup_vcam.sh --remove     # 卸载
#   sudo bash scripts/setup_vcam.sh --status     # 查看状态
set -e

VIDEO_NR="${VIDEO_NR:-10}"
LABEL="${LABEL:-MAV Test Camera}"

case "${1:-}" in
  --remove)
    echo "[vcam] 卸载 v4l2loopback"
    modprobe -r v4l2loopback 2>/dev/null || echo "[vcam] 未加载，无需卸载"
    exit 0
    ;;
  --status)
    echo "[vcam] 模块: $(lsmod | grep -c v4l2loopback || echo 0) 个"
    lsmod | grep v4l2loopback || true
    echo "[vcam] 设备:"
    ls -la /dev/video* 2>/dev/null || echo "  无"
    exit 0
    ;;
esac

if [ "$(id -u)" -ne 0 ]; then
  echo "[vcam] 需要 root：sudo bash $0" >&2
  exit 1
fi

# 已加载则先卸载（改参数需重新加载）
if lsmod | grep -q "^v4l2loopback"; then
  echo "[vcam] 已加载，先卸载"
  modprobe -r v4l2loopback || true
fi

echo "[vcam] 加载 v4l2loopback：video_nr=$VIDEO_NR label='$LABEL'"
modprobe v4l2loopback \
  video_nr="$VIDEO_NR" \
  card_label="$LABEL" \
  exclusive_caps=1 \
  max_buffers=8

sleep 1
if [ -e "/dev/video$VIDEO_NR" ]; then
  echo "[vcam] ✅ 创建成功: /dev/video$VIDEO_NR"
  ls -la "/dev/video$VIDEO_NR"
  echo
  echo "下一步：用 ffmpeg 把视频推流进去（见 scripts/push_vcam.sh）"
else
  echo "[vcam] ❌ 创建失败，检查 dmesg | tail -20" >&2
  exit 1
fi
