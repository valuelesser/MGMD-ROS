# phantom70 再测：处理帧率下降诊断（2026-11-06）

视频 `phantom70`，1920×1080，共 1349 帧；ROS 源 6 Hz，推荐 MFE、全模块、`d_min=0.02`。全部指标按原标注、IoU≥0.25、全 1349 帧匈牙利一对一匹配；离线列沿用先前同视频完整评测数据。

| 条件 | LAC / LAD | 已处理帧 | TP | FP | FN | P | R | F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 离线复现，既有结果 | PyTorch CUDA | 1347 | 908 | 29 | 441 | 0.9691 | 0.6731 | 0.7944 |
| 先前 ROS：旧关联、无展示 | CUDA / CUDA | 1272 | 831 | 23 | 518 | 0.9731 | 0.6160 | 0.7544 |
| 本轮 ROS：新关联、无展示 | CUDA / CUDA | 1155 | 799 | 37 | 550 | 0.9557 | 0.5923 | 0.7314 |
| 本轮 ROS：新关联、实时展示 | CUDA / CUDA | 963 | 642 | 30 | 707 | 0.9554 | 0.4759 | 0.6353 |
| 诊断失误：新关联、实时展示 | CPU / CUDA | 818 | 540 | 26 | 809 | 0.9541 | 0.4003 | 0.5640 |

第一轮展示测试遗漏 `source scripts/env_setup_ros38.sh`。日志明确报告 `libcublasLt.so.12` 不可见，LAC 静默回退到 CPU，LAD 仍在 CUDA。补上环境脚本后，LAC/LAD 均为 CUDA，处理帧数从 818 增至 963，F1 从 0.5640 增至 0.6353。CPU 回退一行仅供排障，不能当作 CUDA 结果。

在新关联且两模型均使用 CUDA 的条件下，启用 `visualize_node`、NVENC 写盘和 `rqt_image_view`，全视频处理帧从 1155 降到 963。节点日志的每帧 `step` 耗时中位数约从 159.45 ms 升到 186.77 ms；队列等待中位数从 33.54 ms 升到 48.36 ms。两个全程运行处理的帧集合并不完全相同，因此这些耗时差值包含场景采样和运行波动，不能精确拆成单个组件的耗时。

同 CUDA、无展示的前 300 帧短回放中，新关联处理 246 帧、旧关联处理 237 帧；`step` 中位数分别为 144.82 和 152.98 ms。另在视频中段 123 个连续帧上复用相同 MFE 候选做跟踪器单独计时，两种配置的 `update` 中位数均约 14.6 ms。未观察到新关联在该样本上造成大幅降速；先前旧关联无展示与本轮新关联无展示的差距仍可能包含完整视频后段的候选密度与运行波动，不能只归因于卡尔曼参数。

本轮正确 CUDA 展示视频：`outputs/ros_viz/phantom70_display6_cuda_20261106.mp4`，224.8 秒，写入 1349 帧。原始日志和逐帧 ROS 输出分别在 `outputs/logs/ros_compare_phantom70_display6_cuda_20261106/`、`outputs/logs/ros_compare_phantom70_noviz6_cuda_20261106/`；本表同名 JSON 保存具体数值。视频不入 git，由 `scripts/pack_deliverables.sh` 打包。

正确环境启动：

```bash
source /opt/ros/noetic/setup.bash
source ros1_ws/devel/setup.bash
source scripts/env_setup_ros38.sh
roslaunch mav_tracker_nodes track.launch video:=phantom70 rate:=6 \
  limit:=1349 loop:=false width:=1920 height:=1080 device:=cuda \
  visualize:=true out_video:=outputs/ros_viz/phantom70_display6_cuda_20261106.mp4
rqt_image_view /mav/debug_image
```
