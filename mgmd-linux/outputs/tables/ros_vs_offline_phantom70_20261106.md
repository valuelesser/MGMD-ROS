# phantom70 全视频：ROS 与离线复现对比（2026-11-06）

视频 `phantom70`，1920×1080，1349 帧；推荐 MFE，`mfe,tf,lac,lad`，`d_min=0.02`，CUDA。两侧均使用 ARD-MAV 原标注、IoU≥0.25、匈牙利一对一匹配。ROS 文件源以 6 Hz 发布，`video_source → detector_node`；本次关闭仅负责展示的 `visualize_node`。

| 路径 | 评测帧 | TP | FP | FN | P | R | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 离线脚本原输出 | 1347 | 908 | 29 | 439 | 0.9691 | 0.6741 | 0.7951 |
| 离线按全 1349 帧统一口径 | 1349 | 908 | 29 | 441 | 0.9691 | 0.6731 | 0.7944 |
| ROS 实时回放 | 1349 | 831 | 23 | 518 | 0.9731 | 0.6160 | 0.7544 |

ROS 相对全帧离线：P +0.0040，R −0.0571，F1 −0.0400。离线脚本因三帧窗口只评测 1347 帧；统一口径一行将首尾两帧各计为 FN。ROS 收到 1349 张原图，发布 1272 条检测消息，77 帧未产生检测消息；全部 1272 条通过 `header.stamp` 对齐到原图。ROS 的启动等待和运行时丢帧会影响召回，当前数据不能把 77 个 TP 差额全部归因于丢帧。

发现：`rospy.Publisher` 发布时重写了 `header.seq`；本次检测消息的 `seq` 为连续的消息计数，不能作为视频帧号评测。用图像和检测消息的相同时间戳完成对齐。这是后续 E3 逐帧对拍要修正的话题契约问题。

复现命令：

```bash
.venv/bin/python scripts/run_testset.py --video phantom70 \
  --modules mfe,tf,lac,lad --mfe-preset recommended --d-min 0.02 \
  --device cuda --out-suffix ros_compare_20261106 --force

# ROS 启动参数（在 ROS Noetic + .venv38 环境）
roslaunch mav_tracker_nodes track.launch video:=phantom70 rate:=6 \
  limit:=1349 loop:=false width:=1920 height:=1080 \
  device:=cuda visualize:=false
```

原始 ROS 检测、时间戳映射与日志在 `outputs/logs/ros_compare_20261106/`；离线单视频结果在 `outputs/logs/testset/ros_compare_20261106/phantom70.json`。这些大体量运行记录不入 git；本表与同名 JSON 摘要入库。
