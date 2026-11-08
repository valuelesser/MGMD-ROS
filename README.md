# MGMD 小型无人机检测与 ROS 实时跟踪

本项目复现论文 *Motion-Guided Small MAV Detection in Complex and Non-Planar Scenes*（arXiv:2410.10527v1），并将检测链路接入 ROS1 Noetic，支持本地视频回放和电脑摄像头实时输入。

当前开发工程位于 **`mgmd-linux/`**。项目包含离线论文复现和 ROS 实时系统两个阶段；现阶段聚焦图像上的无人机检测与跟踪，不包含飞行控制。

## 处理流程

```text
视频文件 / 摄像头
        ↓
video_source → /camera/image_raw
        ↓
detector_node：MFE → SORT/TF → LAC → LAD
        ↓
/mav/detections
        ↓
visualize_node → /mav/debug_image → rqt_image_view
        └─────→ 带框视频，可选 NVENC 编码
```

- **MFE**：通过帧间运动、图像配准和差分生成候选区域。
- **TF**：使用 SORT 和轨迹运动判据过滤候选。
- **LAC**：分类候选是否可能为无人机。
- **LAD**：局部检测并精修目标框。

ROS 检测使用 Python 3.8 和 ONNX Runtime，直接在检测节点内调用算法。离线版使用独立的 Python 3.11 环境。

## 目录

| 路径 | 内容 |
|---|---|
| `mgmd-linux/mgmd/` | 阶段一算法、标注加载和评测实现 |
| `mgmd-linux/mgmd_onnx/` | ROS 使用的 ONNX 模型包装、加速跟踪器、帧窗口 |
| `mgmd-linux/ros1_ws/` | ROS 节点、消息和 launch 文件 |
| `mgmd-linux/scripts/` | 环境、启动、评测、导出和打包脚本 |
| `mgmd-linux/tests/` | 算法与帧窗口测试 |
| `mgmd-linux/outputs/` | 运行日志、结果表格和演示视频 |
| `mgmd-linux/docs/SPEC.md` | 阶段一规格、结果与论文偏差记录 |
| `mgmd-linux/docs/SPEC-ROS.md` | 当前 ROS 阶段的事实源和维护记录 |
| `paper/` | PRD 和论文资料 |
| `tracking/` | 历史工程与第三方代码，非当前主开发入口 |
| `deliverables/` | 单独分发的演示视频交付包 |

## 环境与外部资产

已验证环境为 Ubuntu 20.04、ROS1 Noetic、NVIDIA RTX 3050 Laptop GPU。ROS 使用 `.venv38`（Python 3.8、PyTorch 2.4.0、ONNX Runtime GPU 1.19.2）；离线使用 `.venv`（Python 3.11、PyTorch 2.5.1）。FFmpeg 用于视频编码，`rqt_image_view` 用于实时显示。

**下文快速启动适用于当前已配置的电脑。** 仓库尚不是可直接在任意机器安装的发布包：数据集、模型权重、虚拟环境和编译产物需要另行准备，部分脚本仍含本机绝对路径。

| 资产 | 配置方式 / 当前位置 |
|---|---|
| ARD-MAV 视频与 XML 标注 | `MGMD_ARD_ROOT` 指向包含 `videos/`、`Annotations/` 的目录；未设置时自动探测本机数据盘 |
| 模型权重 | `mgmd-linux/mgmd/config.py` 中 `WEIGHTS_DIR`，当前为 `/home/gaoduan/tracking/data_raw/mgmd_weights` |
| ROS ONNX 模型 | 权重目录下的 `lac.onnx`、`yolov5s_GLAD-crop.onnx` |
| YOLOv5 v6.0 源码 | `MGMD_YOLOV5_DIR` 可覆盖，默认 `/home/gaoduan/tracking/third_party/yolov5-v6.0` |

在已安装 Noetic 的 Ubuntu 20.04 上，可参考以下命令重建 ROS Python 环境与工作区。新机器应先核对上述路径，以及 `scripts/env_setup_ros38.sh`、`scripts/ros_env.sh` 中的工程路径。

```bash
cd /home/gaoduan/毕业设计/mgmd-linux
bash scripts/setup_env38.sh
source /opt/ros/noetic/setup.bash
cd ros1_ws
catkin_make -DPYTHON_EXECUTABLE=/home/gaoduan/毕业设计/mgmd-linux/.venv38/bin/python
```

需要重新导出模型时，使用准备好依赖和原始权重的离线环境执行 `scripts/export_onnx.py`；详细要求见 [SPEC-ROS §13–15](mgmd-linux/docs/SPEC-ROS.md)。

## 快速启动：视频演示

```bash
cd /home/gaoduan/毕业设计/mgmd-linux
bash scripts/demo_live.sh phantom70 6 1349 rqt
```

脚本加载 ROS 与 CUDA 环境，启动三个节点、NVENC 录制和展示窗口。窗口下拉框选择 **`/mav/debug_image`**。

参数依次为视频名、源发布频率、最多发布帧数、是否打开窗口。该脚本固定输出 **960×720**，录制文件为 `outputs/ros_viz/live_demo.mp4`，再次启动会覆盖；脚本也会停止此前的项目 ROS 演示进程。

```bash
# 查看日志
tail -f /tmp/demo_live/launch.log

# 停止演示并完成录制收尾
bash scripts/demo_live.sh stop
```

### 手动启动与分辨率设置

```bash
cd /home/gaoduan/毕业设计/mgmd-linux
source /opt/ros/noetic/setup.bash
source ros1_ws/devel/setup.bash
source scripts/env_setup_ros38.sh

roslaunch mav_tracker_nodes track.launch \
  video:=phantom70 rate:=6 limit:=1349 loop:=false \
  width:=1920 height:=1080 device:=cuda \
  out_video:=/home/gaoduan/毕业设计/mgmd-linux/outputs/ros_viz/phantom70_manual.mp4
```

`roslaunch` 可在本机没有 ROS master 时自动启动它。另开终端显示结果：

```bash
source /opt/ros/noetic/setup.bash
rosrun rqt_image_view rqt_image_view /mav/debug_image
```

在 `roslaunch` 终端按 **Ctrl+C** 停止，等待视频保存完成。视频源到达帧数上限后停止发送，其他节点可能继续等待，仍需结束 launch。

| 参数 | 含义 | `track.launch` 默认值 |
|---|---|---|
| `video` | 数据集视频短名或视频完整路径 | `phantom30` |
| `rate` | 视频源发布频率，不是检测 FPS 保证值 | `30` |
| `limit` | 最多发送帧数，0 表示不限 | `0` |
| `loop` | 文件结束后循环播放 | `true` |
| `width` / `height` | 输入图像缩放尺寸，两者设为 0 保持原图 | `960` / `720` |
| `device` | 推理设备 | `cuda` |
| `visualize` | 启用画框、调试图发布与可选录制节点 | `true` |
| `out_video` | 带框视频保存路径 | 项目下 `outputs/ros_viz/track.mp4` |
| `encoder` | 编码方式 | `nvenc` |
| `frame_delta_predict` | 跟踪按实际视频帧间隔预测 | `true` |
| `association_radius_factor` | 未匹配框补关联的距离门限系数，0 关闭 | `2.0` |

**CUDA 环境脚本不能省略。** 即使设置 `device:=cuda`，动态库加载失败时仍可能退回 CPU。日志中应确认 `LAC provider=CUDAExecutionProvider` 和 `LAD provider=CUDAExecutionProvider`。

## 电脑摄像头

当前 `track.launch` 固定使用文件源，摄像头通过分别启动节点接入。每个新终端先执行：

```bash
cd /home/gaoduan/毕业设计/mgmd-linux
source /opt/ros/noetic/setup.bash
source ros1_ws/devel/setup.bash
source scripts/env_setup_ros38.sh
export PYTHONUNBUFFERED=1
```

依次在不同终端运行，检测节点就绪后再启动摄像头：

```bash
# 终端 1
roscore

# 终端 2
rosrun mav_tracker_nodes detector_node.py _device:=cuda

# 终端 3：空输出路径表示只预览、不录制
rosparam set /visualize_node/out "''"
rosrun mav_tracker_nodes visualize_node.py _fps:=15

# 终端 4：摄像头编号 0
rosrun mav_tracker_nodes video_source.py \
  _source:=camera _device:=0 _rate:=15 _width:=640 _height:=480

# 终端 5
rosrun rqt_image_view rqt_image_view /mav/debug_image
```

摄像头的 `_device:=0` 是设备编号，检测节点的 `_device:=cuda` 是推理设备。`width/height` 是输出缩放尺寸，并不设置摄像头硬件采集模式。停止时在各终端按 Ctrl+C；后续需要录制时，重新设置 `/visualize_node/out` 为具体文件路径后再启动展示节点。

2026-11-08 已验证本机 `/dev/video0` 能读取 640×480 图像，窗口正常，LAC/LAD 均使用 CUDA，检测消息输出约 15 Hz；当时没有合适的无人机目标，**未验证真实摄像头下的识别准确率**。

已有 ROS 摄像头驱动可以直接发布到 `/camera/image_raw`，此时无需再运行 `video_source`。当前检测入口按紧密排列的三通道 **BGR8** 读取图像；其他编码、带行填充的图像或网络视频流需要适配。

## 离线评测

以下命令使用已有 `.venv`，按推荐配置评测单个视频：

```bash
cd /home/gaoduan/毕业设计/mgmd-linux
source scripts/env_setup.sh
"$PY" scripts/run_testset.py --video phantom70 \
  --modules mfe,tf,lac,lad --mfe-preset recommended \
  --d-min 0.02 --device cuda --out-suffix manual --force
```

去掉 `--video phantom70` 会运行配置中的 15 个测试视频。输出位于 `outputs/logs/testset/manual/`。评测使用 **IoU≥0.25、匈牙利一对一匹配**。

离线脚本默认只评测实际处理的中心帧。与 ROS 比较时应统一标注帧范围，把未处理帧计入漏检；ROS 图像与检测结果用 `header.stamp` 对齐，不能直接用检测消息的 `header.seq` 作为原视频帧号。

## 已有结果与限制

最近一次完整 ROS 测试覆盖固定 15 个测试视频、28,337 个标注帧，采用 **1080p、30 Hz 源输入、CUDA 检测、实时展示与 NVENC 录制**。全帧结果如下：

| 对照 | P | R | F1 |
|---|---:|---:|---:|
| 论文 Table 1 报告值 | 0.8400 | 0.5900 | 0.6900 |
| 离线历史基线，统一全帧范围 | 0.8970 | 0.5407 | 0.6747 |
| ROS 完整实时链路 | 0.8349 | 0.1083 | 0.1917 |

ROS 实际处理 5,983 帧，覆盖率 21.11%，大量未处理帧被计为漏检；当前负载下不能逐帧处理 30 Hz 输入。离线使用已有冻结结果，不是本次重跑；论文与工程的实验条件存在偏差。完整逐视频、分场景统计见 [全测试集报告](mgmd-linux/outputs/tables/ros_testset_30hz_20261108.md)。

补充的 [phantom70 降速对照](mgmd-linux/outputs/tables/ros_nodrop_phantom70_20261108.md) 中，相同预热协议下，2 Hz 完整处理 1347/1347 个中心帧，P=0.9574、R=0.7005、F1=0.8091；与独立同模型同参数离线的逐帧 TP/FP/FN 完全一致，仍保留少量 1 像素级框与置信度差异。30 Hz 对照仅处理 276 个中心帧，F1=0.2348。这支持本视频的主要问题来自实时负载与漏处理，结论不外推到整个测试集。

```bash
# 加载 ROS/CUDA 环境，启动 roscore 和实时窗口后执行；新目录用于新一轮实验
"$PY38" scripts/compare_ros_nodrop.py --rate 2 --out outputs/logs/nodrop_new
"$PY38" scripts/compare_ros_nodrop.py --rate 30 --allow-drops \
  --offline-reference outputs/logs/nodrop_new --out outputs/logs/control_new
```

全量 ROS 评测入口如下。先在各终端加载上述 ROS/CUDA 环境，再分别运行；使用新的输出目录，并确保没有其他检测/视频源/展示节点占用同名话题。

```bash
# 终端 1
roscore
# 终端 2
rosrun rqt_image_view rqt_image_view /mav/debug_image
# 终端 3，工作目录为 mgmd-linux
"$PY38" scripts/run_ros_testset.py --rate 30 --out outputs/logs/ros_testset_30hz_new
```

脚本支持相同配置和源码版本的已完成视频断点续跑；每视频保留真实链路丢帧，按全部 GT 计分，并核查实际 CUDA、编码日志与检测时间戳对齐。

阶段一主结果为 P=0.897、R=0.541、F1=0.675、AP=0.524。它是 [SPEC §10.12](mgmd-linux/docs/SPEC.md) 记录的历史冻结口径，具体偏差和后续对齐修复见 §10.13，并非当前 ROS 版本的准确率承诺。

`phantom70` 的全 1349 帧对照如下，ROS 源均为 6 Hz、1920×1080，模型实际使用 CUDA：

| 配置 | 产生检测消息的帧数 | P | R | F1 |
|---|---:|---:|---:|---:|
| 离线历史结果，统一全帧口径 | 1347 个处理帧 | 0.9691 | 0.6731 | 0.7944 |
| ROS 新关联，无展示 | 1155 | 0.9557 | 0.5923 | 0.7314 |
| ROS 新关联，实时展示与写盘 | 963 | 0.9554 | 0.4759 | 0.6353 |

以上为特定视频与运行负载下的实测，不代表整个测试集。详细条件见 [帧率诊断表](mgmd-linux/outputs/tables/ros_phantom70_display_fps_20261106.md)。

- 运动差分方法对相对背景静止的目标较弱；模型面向无人机，普通物体未必会被识别。
- 高速运动、输入覆盖丢帧和候选中断会影响召回及 ID 保持。扩大关联范围仍属工程扩展，可能增加误关联。
- HUD 中 FPS 表示最近检测结果发布频率，不是源视频帧率或单帧推理耗时的倒数。
- 展示会把最近一次检测叠加到当前图像，跳帧时框可能滞后；画面不等于严格逐帧对齐结果。
- 展示、复制、编码和 CPU 上的 MFE 均会消耗资源；NVENC 已启用不代表整条链路都在 GPU 上执行。
- 性能优化候选已登记于 [SPEC-ROS §29](mgmd-linux/docs/SPEC-ROS.md)，当前决定为暂不实施。

## 检查与交付

```bash
cd /home/gaoduan/毕业设计/mgmd-linux
# 定向逻辑测试，不代替摄像头或 GPU 全链路验收
.venv/bin/python -m pytest tests/test_tf_fast_gap.py tests/test_ros_frame_window.py tests/test_tf.py -q

# 加载 ROS 环境后查看话题
rostopic list
rostopic hz /mav/detections

# 打包演示视频
bash scripts/pack_deliverables.sh
# 全量 ROS 测试的独立交付包（视频、检测、时间戳映射、日志）
bash scripts/pack_deliverables.sh --ros-testset outputs/logs/ros_testset_30hz_20261108_v2
```

演示视频不进入 Git，交付包位于仓库根目录的 `deliverables/`；结果表格与摘要 JSON 入库。代码或 SPEC 更新需要提交，ROS 代码与对应 SPEC 更新放在同一提交中。后续修改完成相应检查和提交后，还必须推送到 `origin` 的对应分支（当前主线 `master`），并核对远程与本地提交一致；远程仓库为 `git@github.com:valuelesser/MGMD-ROS.git`。

需求背景见 [PRD](paper/prd_mgmd_reproduction.md)，阶段一详见 [SPEC](mgmd-linux/docs/SPEC.md)，ROS 的实际实现、偏差和当前状态以 [SPEC-ROS](mgmd-linux/docs/SPEC-ROS.md) 为准。
