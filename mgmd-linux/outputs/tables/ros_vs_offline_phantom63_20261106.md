# phantom63 全视频：ROS 实时展示与离线复现对比（2026-11-06）

固定随机种子 `20261106` 从本机 60 个 ARD-MAV 视频中抽中 `phantom63`。视频容器标称 1829 帧，实际解码 1799 帧，标注覆盖 1–1799 帧。两侧使用原始 1920×1080、推荐 MFE、`mfe,tf,lac,lad`、`d_min=0.02`、IoU≥0.25 的逐帧匈牙利一对一评测。离线版使用 CUDA；ROS 启动时请求 CUDA，但本次遗漏 `source scripts/env_setup_ros38.sh`，实际 LAC 回退 CPU、LAD 使用 CUDA。ROS 以 15 Hz 播放，启用 `visualize_node` 和 `rqt_image_view`，保存完整叠加视频。ROS 跟踪使用按帧间隔预测与 2.0 倍框对角线的补充关联门；离线版保持冻结的论文复现配置。

| 路径 | 评测帧 | TP | FP | FN | P | R | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 离线脚本原输出 | 1797 | 472 | 315 | 1325 | 0.5997 | 0.2627 | 0.3653 |
| 离线全帧统一口径 | 1799 | 472 | 315 | 1327 | 0.5997 | 0.2624 | 0.3650 |
| ROS 实时展示 | 1799 | 79 | 74 | 1720 | 0.5163 | 0.0439 | 0.0809 |

ROS 相对全帧离线：P −0.0834、R −0.2185、F1 −0.2841。离线脚本的三帧窗口未评测首尾两帧，统一口径将它们各计为 FN。ROS 收到全部 1799 张图，发布 303 条检测消息，全部通过 `header.stamp` 对齐；1496 帧没有检测消息。303 个已处理帧间平均跨度 5.83 源帧，最大 55 帧。每帧恰有一个 GT，故至少 1496 个 FN 来自未处理帧；其余 224 个 FN 发生在已处理帧。此历史结果混有 LAC CPU 回退，不能用来单独衡量 ROS 相对离线或新关联参数的影响。宽关联门可能引入误关联，需在相同输入帧序列下另做 ID switch 对照。

录制视频：`outputs/ros_viz/phantom63_20261106.mp4`（119.3 秒，不入 git）。ROS 原始检测和日志：`outputs/logs/ros_compare_phantom63_20261106/`；离线结果：`outputs/logs/testset/ros_compare_phantom63_20261106/phantom63.json`。摘要 JSON 与本表入库。

运行参数：

```bash
roslaunch mav_tracker_nodes track.launch video:=phantom63 rate:=15 \
  limit:=1799 loop:=false width:=1920 height:=1080 device:=cuda \
  visualize:=true out_video:=outputs/ros_viz/phantom63_20261106.mp4
rqt_image_view /mav/debug_image

.venv/bin/python scripts/run_testset.py --video phantom63 \
  --modules mfe,tf,lac,lad --mfe-preset recommended --d-min 0.02 \
  --device cuda --out-suffix ros_compare_phantom63_20261106 --force
```
