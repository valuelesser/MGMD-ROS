# ROS 跟踪输出视频

这些 mp4 由 `visualize_node` 生成，**不入库**（体积大，可随时重新生成）。

## 现有文件

| 文件 | 分辨率 | 帧数 | 说明 |
|---|---|---:|---|
| `track.mp4` | 960×720 | 369 | `track.launch` 默认输出（NVENC 编码） |
| `track_720p_nvenc.mp4` | 960×720 | 600 | 720p + NVENC，检测 1–2 个/帧 |
| `track_1080p_cpu.mp4` | 1920×1080 | 277 | 1080p + CPU 编码（检测全灭，仅作对照） |

## 重新生成

```bash
cd ros1_ws
source /opt/ros/noetic/setup.bash && source devel/setup.bash
source ../scripts/env_setup_ros38.sh

# 720p（推荐）
roslaunch mav_tracker_nodes track.launch video:=phantom30 \
    width:=960 height:=720 rate:=30 limit:=600

# 1080p 对照
roslaunch mav_tracker_nodes track.launch video:=phantom30 \
    width:=1920 height:=1080 rate:=30 limit:=300
```

输出路径由 `track.launch` 的 `out_video` 参数控制，
默认 `outputs/ros_viz/track.mp4`（本目录）。

## 视频内容

每帧叠加：
- 检测框（颜色按 `track_id` 区分）
- 标签：`#track_id conf`
- 轨迹历史连线（最近 30 个中心点）
- HUD（左上角）：`frame <视频帧号> det <检测数> cand <候选数> MFE/LAC/LAD <各阶段耗时>`
