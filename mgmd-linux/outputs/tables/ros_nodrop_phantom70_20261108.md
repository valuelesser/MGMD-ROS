# phantom70：降速无丢帧与同配置离线对照（2026-11-08）

## 结论

本视频支持“实时输入负载导致检测链路漏处理及时间间隔变化，是低召回的主要原因”。在 2 Hz 下，1349 张原图全部收到，帧 2–1348 的 1347 个有效中心帧全部产生检测消息；与独立同配置离线遍历相比，每一帧的 TP/FP/FN 及全部检测数量一致，P/R/F1 完全相同。未达到逐位数值相同，差异如下文所列；结论不外推为所有视频均无算法问题。

## 严格对照条件

- 两次 ROS 都使用 1920×1080、推荐 MFE、全模块、`d_min=0.02`、CUDA LAC/LAD、当前 FastSortTracker（真实帧间隔预测、补关联半径 2.0）、实时窗口和 NVENC 写盘。
- 两次都先初始化检测节点，运行相同的 180 帧、12 Hz 预热段并确认 LAD 被调用，再重启源回绕帧号，通过既有逻辑重置 tracker 和窗口。预热不参加正式计分或录制。
- 正式源均读到 EOF，不使用限帧处提前退出；最后一次正常限速等待有助于排空发布队列。两次正式源均收到全部 1349 张图。主要对照变量是输入频率 2 Hz / 30 Hz。
- 独立离线参考采用 `run_testset_onnx.run_video` 的顺序遍历，使用与 ROS 相同的模型和跟踪参数；不经过 ROS 消息传输、最新帧槽或在线窗口选择。
- 全部指标用完整 1349 帧 GT、IoU≥0.25、匈牙利一对一匹配。首尾两帧缺少 MFE 邻帧，仍作为漏检计入；不能将其称为运行时丢帧。

## 结果

| 条件 | 接收原图 | 已处理中心帧 / 1347 | TP | FP | FN | P | R | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ROS 30 Hz，同预热控制段 | 1349 | 276 | 180 | 4 | 1169 | 0.9783 | 0.1334 | 0.2348 |
| ROS 2 Hz，无运行时丢帧 | 1349 | 1347 | 945 | 42 | 404 | 0.9574 | 0.7005 | 0.8091 |
| 离线 ONNX，同模型和参数 | — | 1347 | 945 | 42 | 404 | 0.9574 | 0.7005 | 0.8091 |
| 历史离线，全帧口径（配置不同） | — | 1347 | 905 | 30 | 444 | 0.9679 | 0.6709 | 0.7925 |

同预热 30 Hz 段缺少 1071 个中心帧，中心帧覆盖率仅 20.49%；2 Hz 覆盖率为 100%。30 Hz 对照并未与离线参考并行争用 GPU，而是复用低速段已完成的离线结果。2 Hz 的 R 从 30 Hz 的 0.1334 恢复至 0.7005，F1 从 0.2348 恢复至 0.8091。

先前全测试集中的 phantom70 为并发启动、未按本协议预热的 30 Hz 运行（235 中心帧，F1=0.1888），属于另一启动条件，不用它单独归因输入速度。历史离线来自 `final_d002/phantom70.json`，跟踪和后端与当前同配置参考不同，因此分开列示。

## 数值差异核查

- 1347 个中心帧的检测数量均相同；包括首尾在内，每一帧的 TP/FP/FN 都相同。
- 按框与置信度 `rtol=1e-5, atol=1e-3` 的严格比较，有 6 帧未通过：366、571、634、796、963、1004。
- 前 5 帧存在最多 1 像素的框边缘差异；匹配框的最小 IoU 为 0.8853。全部输出中最大置信度绝对差为 0.0014085，出现在第 1004 帧。
- 这些差异没有改变任何帧的评测计数。本次没有确定其来自哪一项数值计算，不能将其直接断言为 GPU 随机性，也不能宣称 ROS 与离线逐位相等。
- 输入过快还会改变 MFE 邻帧间隔和跟踪轨迹，并非只有“未发检测消息的帧增加 FN”这一种影响。

## 证据与复现

低速目录：`outputs/logs/ros_nodrop_phantom70_2hz_20261108/`。包含正式 `capture.json`、`coverage.json`（三项缺失/额外帧检查均为空）、`numerical_audit.json`、结果、独立离线框及分节点日志。30 Hz 控制目录：`outputs/logs/ros_control_phantom70_30hz_warm_20261108/`，归档时同时复制到低速目录的 `30hz_control/`。两个正式录制文件均经 ffprobe 核对为 1080p、H.264、1349 帧。

低速运行提交：`7f357ae532b058970a8359455b07b0c045a922ba`；控制段运行提交：`934e3ca2a38227012f812e6eb1fb1a28de00937c`。两提交间仅评测工具、打包工具与 SPEC 有变化，生产节点和算法文件无差异。模型 SHA256 及具体计数见同名 JSON。

```bash
cd /home/gaoduan/毕业设计/mgmd-linux
source /opt/ros/noetic/setup.bash
source ros1_ws/devel/setup.bash
source scripts/env_setup_ros38.sh
# 另开终端运行 roscore 与 rqt_image_view /mav/debug_image
"$PY38" scripts/compare_ros_nodrop.py --rate 2 --out outputs/logs/nodrop_new
"$PY38" scripts/compare_ros_nodrop.py --rate 30 --allow-drops \
  --offline-reference outputs/logs/nodrop_new --out outputs/logs/control_new
bash scripts/pack_deliverables.sh --ros-run outputs/logs/ros_nodrop_phantom70_2hz_20261108
```
