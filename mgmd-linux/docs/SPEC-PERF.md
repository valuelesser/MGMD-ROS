# SPEC-PERF · Docker 环境下的 ROS 性能优化

`spec_version: 0.1.5` · `branch: perf-optimization` · `base: docker@ed8cb69` · `prd_version: 4.2`

## 0. 版本与维护约定

本规格管理 Docker 下的 ROS 性能优化。继承 [阶段一 SPEC](SPEC.md) §0、[SPEC-ROS](SPEC-ROS.md) §0 与仓库 [AGENTS.md](../../AGENTS.md)。话题、帧序号和跟踪语义沿用 SPEC-ROS，阶段一算法与指标基线保持冻结。

| 规则 | 执行要求 |
|---|---|
| V1 | 每轮改动在 §8 追加记录，写明改了什么、为什么、影响哪些验收项及验证结果。失败或撤回的实验也记录实测结论。 |
| V2 | 性能相关代码、配置、测量脚本、规格及验证摘要放在同一次提交。提交信息使用 `perf: <改动> (SPEC-PERF §<节号>)`。涉及原 SPEC 契约时同次更新对应文档。 |
| V3 | 独立维护 `spec_version`，每轮功能性改动递增并在 §0.1 追加版本记录。纯文档修订注明范围。Git 提交标识对应实现与规格快照。 |
| V4 | 与 PRD、论文或原 ROS 契约的偏差写入 §7，注明适用范围、原因和受影响的验收项。 |
| V5 | 提交前完成与改动相称的测试、`git diff --check` 和暂存清单核对。提交后执行 `git show --stat HEAD`，报告提交哈希、验证结果和剩余改动。 |
| V6 | 提交性能表格和运行摘要。演示视频、rosbag、权重、依赖和构建产物保留在本地，登记生成方式与证据位置。 |

### 0.1 版本记录

| 版本 | 日期 | 对应提交 | 变更 |
|---|---|---|---|
| 0.1.5 | 2026-11-11 | 本次 G1.1 提交 | 实测两份模型的 CUDA I/O Binding 与同会话数值一致性 |
| 0.1.4 | 2026-11-11 | 本次 G0 提交 | 增加 Docker 能力探针、1080p 评测配置与实测记录 |
| 0.1.3 | 2026-11-11 | 本次文档提交 | 固定主分支现有 1080p 基线，细化同 ROS、实时展示和录像负载下的验证 |
| 0.1.2 | 2026-11-11 | 本次文档提交 | 细化工作包、接口、基准、对拍和上线条件 |
| 0.1.1 | 2026-11-11 | 本次文档提交 | 定义全 GPU 数据路径、能力探针和逐阶段验收 |
| 0.1.0 | 2026-11-11 | 初始规格提交 | 建立 Docker 性能优化基线、测量口径与提交规则 |

## 1. 范围

覆盖视频读取与缩放、ROS 图像传输、MFE、TF、LAC、LAD 和可视化。先用实测定位瓶颈，再逐项比较候选改动。摄像头和 TT 实机验收遵循 SPEC-ROS。

## 2. 环境与基线

代码起点为本地 `docker` 的 `ed8cb69`。运行环境使用 Docker Desktop 现有 `mgmd-ros` 容器，工程目录为 `/workspace/MGMD-ROS/mgmd-linux`，运行用户为 `valuelesser`。最近一次验证环境为 Ubuntu 20.04、ROS Noetic、Python 3.8.10、RTX 3050 Laptop GPU、ONNX Runtime GPU 1.19.2。启动入口、依赖、数据和模型路径见 [SPEC-ROS §2.4](SPEC-ROS.md#24-docker-desktop-回放环境)。

容器的迁移源码与 Git 分支曾存在差异。候选运行前核对 ROS 节点、`mgmd_onnx` 和评测脚本的实际文件，记录候选提交或文件哈希、配置、输入和模型哈希，确认编译产物对应本轮源码。

### 2.1 冻结主分支现有结果

用户所指主分支对应远程默认分支 `origin/master`。固定引用 `905a39563a7fe07587071e62d12f37a98cd1db19` 中的正式评测表及同名 JSON。原评测运行源码为 `570b801cc214199c232884d54484be1f7eecac14`，正式目录为 `outputs/logs/ros_testset_30hz_20261108_v2/`。直接引用已发布结果，跳过基线重跑和新片段筛选。 [主分支 1080p 评测报告](https://github.com/valuelesser/MGMD-ROS/blob/905a39563a7fe07587071e62d12f37a98cd1db19/mgmd-linux/outputs/tables/ros_testset_30hz_20261108.md)

| 基线项目 | 固定口径或既有结果 |
|---|---|
| 视频 | 沿用现有 ARD-MAV 测试视频及 GT。固定顺序为 phantom05、08、09、10、19、30、41、43、46、47、58、63、65、70、86。使用正式 JSON 的各视频标注帧范围，合计 28,337 帧。 |
| 输入及算法 | 原始 1920×1080，30 Hz；全模块，推荐 MFE，`d_min=0.02`，真实帧间隔预测，补关联半径系数 2.0。LAC/LAD 使用 CUDA，模型哈希取正式 JSON。 |
| ROS 及展示 | ROS Noetic，经实际话题采集。保持 `rqt_image_view` 可见并持续订阅 `/mav/debug_image`，同时 NVENC 录制 1080p H.264。每视频重启链路，冷启动与收尾计入墙钟。 |
| 处理与录像 | 源发布 28,337 帧，采集收到 28,335 帧，检测消息 5,983 条，覆盖率 21.11%。共录制 28,334 帧。含启动与收尾总运行约 19.38 分钟。 |
| 全帧检测 | TP 3,049、FP 603、FN 25,111，P 0.8349、R 0.1083、F1 0.1917。 |

主分支 §33 的 phantom70 低速和同预热控制用于解释帧窗口及数值差异。正式候选与 §32.1 的全量 30 Hz 结果对比。旧结果和 Docker 候选的执行环境按实记录，差值反映整套部署的变化。硬件、运行时或显示配置有差异时，在对照表登记，无法核实的项目标为未知。

## 3. 测量口径

- 正式候选按 §2.1 固定 15 个视频、完整原标注帧范围和顺序运行。保持原视频尺寸、播放速率、模型、阈值、跟踪参数和每视频冷启动。全程保留 ROS 采集、可见窗口和 NVENC 写盘。开发阶段用同一批视频的短段，例如 phantom30 和 phantom70。
- 复用主分支时间戳对齐和全帧计分。IoU ≥ 0.25 时逐帧匈牙利一对一匹配。漏处理帧保留 GT，并计入 FN。各视频 TP/FP/FN 先求和，再算全局 P/R/F1。检测消息数、已处理帧覆盖率和展示帧数分别统计。
- 正式结果报告逐视频与全量的检测消息数、覆盖率、TP/FP/FN、P/R/F1、墙钟和录像帧数。候选额外采集 p50/p95/p99、分段耗时、拷贝和显存。历史未提供同口径分位数的项目只列候选绝对值。候选复跑用于报告自身波动，历史基线保持冻结。

## 4. 验收

| 项目 | 判定 |
|---|---|
| P1 可复测 | 主分支基线引用锁定到提交及 JSON。候选有可定位的源码、容器配置、输入、参数和运行摘要。 |
| P2 性能收益 | 主指标为固定 30 Hz 负载下的处理覆盖率与全帧 F1。报告已有基线字段的差值、候选波动及画质和延迟护栏。 |
| P3 功能回归 | 检查三节点、ROS 话题、帧序号、实际 CUDA provider 及跟踪逻辑。用相同帧范围对拍检测框、置信度和跟踪 ID，说明差异及其依据。 |
| P4 可视化回归 | 涉及可视化时，验证 `/mav/debug_image`、桌面画面更新或录像完整性，按本轮涉及的入口记录。 |
| P5 GPU 驻留 | 严格 GPU 模式下，用时间线与各阶段输入输出设备检查，确认解码后的图像和中间张量在计算主链路无 H2D/D2H 往返，逐算子核对推理设备。检测消息、编码码流和兼容预览的输出边界单独统计。 |

## 5. 实施顺序

1. 锁定 §2.1 的主分支报告、视频清单和评测脚本。核对 Docker 运行副本，执行 G0 能力探针。
2. 在同一批现有视频的短段上依次打通 GPU 模型、MFE、TF 和输出，并用小样实测决定内核优化顺序。
3. 候选完成后先用单个现有视频走完整 ROS、桌面显示与录像协议，再跑全 15 个视频，与主分支正式结果对照。每轮按 §0 提交实现、版本和证据。

## 6. 全 GPU 流水线路线

目标是让解码后的像素、MFE 特征和候选、TF 跟踪状态、LAC/LAD 张量及叠框后的编码输入留在同一 GPU 上。CPU 负责文件读取和解封装、启动调度、时间戳、ROS 消息和输出封装。首选在现有容器中增加一个可选的同进程 GPU 后端，以 CUDA 缓冲区衔接 FFmpeg、CUDA 算子和 ONNX Runtime，保留现有后端作为逐帧参考。

ROS1 的 `sensor_msgs/Image` 以字节数组承载图像，GPU 计算主链在同一进程传递设备缓冲区。正式评测保留 `/camera/image_raw` 给原评测器对齐时间戳，保留 `/mav/debug_image` 给 rqt，并发布 `/mav/detections`。两个图像话题由独立输出支路下载，纳入正式负载和拷贝统计。NVENC 直接使用设备帧。 [ROS Noetic Image 消息定义](https://github.com/ros/common_msgs/blob/noetic-devel/sensor_msgs/msg/Image.msg)

| 阶段 | 实施与通过条件 |
|---|---|
| G0 · 能力与评测协议 | 核对运行源码和主分支评测配置。在现有视频短样上验证 CUDA 推理、NVDEC、NVENC、设备帧直通以及 1080p ROS/rqt 连接。交付能力摘要和参数清单，沿用 §2.1 的已有基线。 |
| G1 · 输入、设备缓冲与模型 | 首选 C++ FFmpeg/libav 的 CUDA 硬件帧接口，向当前 Python 3.8 进程提供设备帧适配。使用有界缓冲保留前、当前和未来帧，并用 CUDA stream/event 管理解码帧生命周期。先用合成设备帧验证 LAC/LAD I/O Binding、GPU 裁剪、归一化、softmax、NMS 与坐标回映，再接真实解码。检查 ONNX 实际算子落点及所有输出，排查 CPU fallback；使用支持的严格设置或运行 trace 明确判定。动态候选采用有界批次和掩码，先核对模型的实际 batch/shape，再评估 LAD 真批处理。 |
| G2 · MFE | 在 GPU 上实现缩放、灰度、稀疏金字塔 LK、单应性估计、warp、差分/阈值、形态学和连通域统计。可先验证 CUDA 版 OpenCV 的 LK 与可用算子；RANSAC、边界和连通域统计缺少等价设备路径时用 CUDA 实现。按中间结果对拍关键点及状态、单应矩阵、有效区掩膜、连通域面积和候选框，覆盖对齐失败、低纹理、边缘目标和镜头运动。保留原帧窗口和帧号语义。 |
| G3 · TF | 将 Kalman 状态、帧间隔预测、关联成本、门控、精确线性分配和轨迹更新迁至设备。先做小矩阵 GPU 分配原型并实测，与当前 SciPy 解对照等成本解、矩形矩阵、空集、漏帧和回放重置。按视频逐帧检查 ID 与轨迹生命周期。GPU 分配速度由当前候选分布和端到端同步成本决定。 |
| G4 · 输出和正式对照 | GPU 叠框和 NVENC 写盘与 ROS/rqt 同时运行。先以 phantom30 按正式 1080p/30 Hz 条件走通采集和计分，再用同一入口跑完 15 个视频。汇报逐视频和全局结果、帧序、窗口更新、录像完整性与显存峰值。用单独 trace 核对计算主链与输出支路。 |

G0 后可并行验证 G2 的单应性与 G3 的精确分配，提前发现等价性问题。G1–G3 先用原精度和阈值对拍。TensorRT、FP16 和更大的批次作为后续独立实验，分别量化延迟、吞吐和结果变化。各阶段应区分过渡模式的 CPU 路径与严格模式的验收结果。

FFmpeg 的 CUDA 硬件帧、NVDEC 和 NVENC 有官方支持。当前 WSL 容器的库映射、编解码格式和驱动组合由 G0 的实际运行确认。当前 PyNvVideoCodec 官方 Linux 轮子要求 Python 3.10 或以上，因此保留 ROS Python 3.8 时首选 libav/C++ 适配。[NVIDIA FFmpeg 指南](https://docs.nvidia.com/video-technologies/video-codec-sdk/13.0/ffmpeg-with-nvidia-gpu/index.html) · [PyNvVideoCodec 环境要求](https://docs.nvidia.com/video-technologies/pynvvideocodec/read-me/system-requirements-common.html)

ONNX Runtime I/O Binding 支持设备输入和输出，CUDA provider 可与调用方计算流协作。GPU 路径需验证张量布局、缓冲生命周期、流同步和实际算子设备。CUDA LK 和 GPU 精确分配已有参考实现，现有算法的边界条件和结果一致性仍由对拍验证。[ORT I/O Binding](https://onnxruntime.ai/docs/performance/tune-performance/iobinding.html) · [ORT CUDA Provider](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html) · [OpenCV CUDA LK](https://docs.opencv.org/doc/doxygen/html/d7/d05/classcv_1_1cuda_1_1SparsePyrLKOpticalFlow.html) · [NVIDIA GPU 线性分配](https://docs.nvidia.com/cugraph/latest/graph_support/algorithms/cpp_algorithms/linear_cpp/) · [Nsight Systems CUDA Trace](https://docs.nvidia.com/nsight-systems/UserGuide/#cuda-trace)

### 6.1 运行入口和缓冲区契约

- 保留 `track.launch` 作为参考入口。新增独立 GPU launch，将读取、检测和叠框放进同一个工作进程，仍由 ROS 管理启停。第一版覆盖现有 Docker 的文件回放。后续摄像头按输入格式重新验收，已解码的主机帧需要一次入口上传。
- 每个输入帧记录源帧号、循环轮次、视频 PTS、ROS 时间戳、到达时刻、尺寸、格式和逐平面 stride。设备帧由持有 FFmpeg 引用的对象管理。计算、预览和编码消费者完成后才能回收缓冲槽。FFmpeg CUDA 上下文与 Torch/ORT 的互操作先用单帧验证。统一 GPU 编号，显式使用同一上下文或确认可安全共享缓冲区。
- 首版用一条计算流串行衔接 MFE、TF、LAC 和 LAD，阶段间靠流内顺序。解码及输出接口各自使用事件交接。先复用有界的帧、裁剪、候选和跟踪状态缓冲。首次分配、预热、阻塞等待、缓冲峰值和溢出单独计数。有结果后再测多流并发。 [FFmpeg 硬件帧示例](https://www.ffmpeg.org/doxygen/trunk/hw_decode_8c-example.html) · [CUDA 异步执行](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html)
- 定义逐帧对拍与实时回放两种负载。对拍完整处理同一组可用帧，使用同一份解码输入隔离算法差异。实时按当前帧窗口和队列策略处理，记录等待未来帧、入口覆盖、缓冲淘汰和关闭时的尾帧。循环使用源帧号与轮次识别重置，先核对现有运行副本的契约。

### 6.2 G0 能力探针与正式评测准备

1. 容器恢复后记录镜像 ID、实际源码路径和指纹、Docker GPU 配置、驱动、FFmpeg 版本及编译参数。用项目用户分别跑 Torch CUDA 张量、两份模型实际推理、NVDEC 和 NVENC。记录真实算子设备、输入格式、输出格式和错误日志。
2. 用 H.264 样片验证解码到编码。以下是待在当前 FFmpeg 上验证的容器内探针，`VIDEO` 为样片路径，`RUN_DIR` 为本轮输出目录。再用最小 libav 适配器确认返回 CUDA 硬件帧及其 NV12 等实际底层格式，检查 trace 中的拷贝。 [NVIDIA FFmpeg 指南](https://docs.nvidia.com/video-technologies/video-codec-sdk/13.0/ffmpeg-with-nvidia-gpu/index.html)

   ```bash
   ffmpeg -hide_banner -loglevel verbose -hwaccel cuda -hwaccel_output_format cuda \
     -i "$VIDEO" -map 0:v:0 -an -vsync 0 -frames:v 120 -c:v h264_nvenc \
     "$RUN_DIR/gpu-codec-probe.mp4"
   ffprobe -v error -select_streams v:0 -count_frames \
     -show_entries stream=codec_name,width,height,nb_read_frames \
     "$RUN_DIR/gpu-codec-probe.mp4"
   ```

3. 读取固定版本的 `run_ros_testset.py` 与基线 JSON，核对现有视频和 GT、模型哈希、话题、编码、尺寸、帧范围与计分字段。候选评测入口复用采集、时间戳对齐、帧范围和计分，仅适配 GPU launch、退出判定和后端校验日志。首轮只验证接口及小样，正式评测器的历史离线对照数据可引用已有摘要，保持 ROS 计分逻辑一致。
4. 在候选上记录源帧号、到达、入队、窗口就绪、处理开始、检测发布和可选输出完成。GPU 阶段用 CUDA events，等待与墙钟分开统计。端到端延迟含等待未来帧，报告 p50/p95/p99。trace 另开诊断轮，正式候选保持与历史相同的 ROS 采集、显示和录制负载。

G0 交付探针脚本、正式评测配置清单和候选环境摘要。若编解码失败，分别核查库映射、编译能力、驱动 API、输入格式和实际设备帧。允许先用主机解码后单次上传验证计算链。该模式单独标记为过渡模式。正式对照要求 NVENC 与可见 rqt 同时正常运行。

#### 6.2.1 G0 实测记录（2026-11-11）

在现有 `mgmd-ros` 容器以 `valuelesser` 运行。镜像为 Ubuntu 20.04（ID `8feb4d8ca535`），RTX 3050 Laptop 4 GiB，驱动 621.14，Torch 2.4.1+cu121、cuDNN 9.1、ORT 1.19.2。实际运行源码的逐文件 SHA-256 在探针的 `environment.log`，容器迁移副本按该清单标识。两份模型哈希与主分支正式结果一致。15 个视频均存在，逐视频标注帧数与正式清单相符，合计 28,337 帧。固定视频、标注帧数、模型哈希及正式参数见 [ros1080p_eval.json](../configs/ros1080p_eval.json)。

| 项目 | 实测结果 |
|---|---|
| CUDA | Torch CUDA 张量运算通过。ORT profile 中 LAC 10 个、LAD 188 个执行事件标记为 CUDA，本次输入没有 CPU provider 执行事件。 |
| NVDEC / libav | 现有 `phantom30` 的前 120 个可播放帧返回 CUDA `AVFrame`，底层 NV12，1920×1080，stride 2048。PTS 单调递增，首帧 0、末帧 119119。 |
| NVENC / 显存帧直通 | 独立 NVENC 输出和 NVDEC→NVENC 输出均通过 H.264、1920×1080、120 帧读回校验。直通图的编码输入为 CUDA 帧。 |
| ROS/rqt/NVENC | 1080p、30 Hz、`phantom30` 前 600 帧。两个图像话题均为 1920×1080 bgr8，WSLg 上有可见 rqt 窗口且订阅 `/mav/debug_image`。该话题采样约 26.4 Hz。源发布 600 帧，检测处理 49 帧，H.264 录像可解码 526 帧。此次使用现有 ROS 计算后端，结果只用于环境验收。 |

现有容器创建时没有映射 `libnvcuvid.so.1` 和 `libnvidia-encode.so.1`。本轮从同一 Docker 驱动环境、启用 `compute,utility,video` 的临时容器中读取真实文件，临时放到 `/tmp/mgmd-g0-video-libs`。系统 FFmpeg 4.2.7 能加载编码库，但旧预设 API 返回 `unsupported param (12)`。本轮在 `/opt/mgmd-g0-ffmpeg` 独立构建 FFmpeg 6.1.3 和 `nv-codec-headers n12.1.14.0`。构建需启用 `file,pipe,fd` 协议，因为现有录像器以 `-i -` 传 BGR 帧。显式 `h264_cuvid` 路径曾产生 120 个编码包却只在默认编辑列表下读回 95 帧。通过的探针使用原生 H.264 解码器加 CUDA 硬件加速并检查设备像素格式，按正常播放范围验证 120 帧。

在容器工程目录下以项目用户复测。视频路径选择现有 ARD-MAV 原文件。第二条脚本需已有 WSLg 转接，使用独立 ROS master 端口，结束后关闭本轮节点和窗口。

```bash
export PATH=/opt/mgmd-g0-ffmpeg/bin:$PATH
export LD_LIBRARY_PATH=/tmp/mgmd-g0-video-libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
videos=(/data/videos/*/ARD-MAV/videos/phantom30.mp4)
bash scripts/probe_gpu_g0.sh "${videos[0]}" "outputs/g0_$(date +%Y%m%d_%H%M%S)"
bash scripts/probe_gpu_g0_ros.sh "outputs/g0_ros_$(date +%Y%m%d_%H%M%S)"
cc -Wall -Wextra -Werror scripts/probe_cuda_frames.c -o /tmp/mgmd-probe-cuda-frames \
  $(pkg-config --cflags --libs libavformat libavcodec libavutil)
/tmp/mgmd-probe-cuda-frames "${videos[0]}"
```

成功证据在容器 `outputs/g0_modern_20261111_065343/` 和 `outputs/g0_ros_20261111_065554/`。显存直通判定来自 libav 帧和 FFmpeg 图日志，完整候选计算链的拷贝 trace 留给 P5。基线继续使用主分支原报告。

### 6.3 G1–G4 实施任务与逐阶段对拍

**G1 模型与设备帧**

1. 先用固定设备张量验证 ORT CUDA I/O Binding。保留 CUDA 输入输出的引用，固定当前精度。确认每个模型实际输入名、输出名、形状和算子执行设备。分别比较参考裁剪张量、LAC logits/softmax、LAD 原始预测和最终精修框。
2. 实现按候选裁剪、边界处理、letterbox、归一化和 GPU 后处理。复用当前置信度门限、NMS 门限、排序与最近中心规则。用零候选、单候选、多候选、贴边框和等分数样例验证。先以单裁剪模型打通。需要扩大 batch 时先检查模型支持与输出等价性。
3. 接入 libav 解码适配。逐帧对照 PTS、帧数和色彩转换，覆盖 MP4 编辑列表、码流尾帧、退出和循环重置。像素对拍区分解码、色彩范围、缩放插值和算法误差。短样先隔离 GPU 模型，长样验证缓冲生命周期与显存稳定。

**G1.1 实测（2026-11-11）**

在现有 `mgmd-ros` 使用 Torch 2.4.1+cu121 与 ORT 1.19.2，以全零和固定种子输入对拍。对照组与候选组在同一 ORT 会话分别调用 `session.run()` 和 `run_with_iobinding()`，另设现有 `make_session()` 创建的独立会话观测跨会话差异。I/O Binding 使用 ORT 默认流。每种形状重复使用预分配的 CUDA 张量，绑定前等待 Torch 输入拷贝完成，运行后同步输出，再在对拍边界下载。输出全部核对形状、CUDA 设备和实际显存地址。

| 模型 | 输入与输出 | 同会话最大绝对差 | I/O Binding profile |
|---|---|---:|---|
| LAC | batch 1 和 4，`3×32×32`；1 个 logits 输出 | 0 | 40 个 CUDA 内核事件 |
| LAD | 固定 batch 1，`3×320×320`；主预测及 3 个辅助输出 | 0 | 376 个 CUDA 内核事件 |

上述绑定调用的 ORT profile 没有 CPU 算子或显式 Memcpy 节点。测试准备的上传与比较时的下载在绑定调用之外。完整 CUDA 拷贝时间线属于后续 P5 验收。独立会话对比中 LAD 主输出最大绝对差约 0.07365，超过 `rtol=1e-5`、`atol=1e-3` 的筛查条件。同会话逐元素一致，因此该差异也出现在两次独立的主机 I/O 推理间。跨会话差异的原因仍待定位，G1.2 须在真实候选上逐框核对阈值和选框影响。

在容器工程目录以项目用户加载 G0 记录的 CUDA 和 Python 环境后运行以下命令。探针使用原模型文件，输出目录须为新目录。

```bash
python3 scripts/probe_ort_iobinding.py /home/gaoduan/tracking/data_raw/mgmd_weights \
  "outputs/g1_1_$(date +%Y%m%d_%H%M%S)"
```

验证目录为容器 `outputs/g1_1_20261111_071119/`。 [验证摘要](../outputs/tables/g1_1_ort_iobinding.json)

**G2 MFE**

1. 先实现灰度、预处理、差分、阈值、形态学及连通域统计。用固定单应矩阵将图像算子与对齐算法分开验证。记录插值、边界填充、有效区、连通性、候选面积和排序。
2. 实测 CUDA 稀疏 LK 的关键点和状态，再接 GPU 单应性求解和透视变换。分别保存内点数、重投影误差、有效区、候选数量和位置。覆盖低纹理、无有效变换、镜头抖动和高候选帧。需要新增 CUDA 内核时放在 ROS GPU 后端，复用阶段一配置和参考输出。
3. 隔离对拍通过后接入帧窗口。缓存可复用的灰度帧等中间结果。逐轮测量每项缓存或融合对延迟、显存与数值结果的影响。

**G3 TF**

1. 在现有视频的开发短样上采样真实关联矩阵，记录行列分布、门控稀疏度和代价并列频率。用实际规模比较 CPU 参考与设备端精确分配原型，时间包含同步和必要拷贝。此对拍服务于实现校验。
2. 将 Kalman 预测、更新、关联和轨迹生命周期放到 GPU。验证空集、矩形矩阵、相同代价、同帧重复、跨帧跳跃、丢失重现和循环重置。对拍关联对、目标函数值和轨迹 ID。等价最优解导致 ID 变化时，先确定稳定的匹配规则并逐段检查其影响。
3. 在 MFE 和 TF 间传递有界设备候选数组及设备端计数。超过容量时显式报告、按规则扩容或终止严格模式。禁止无记录地截断候选或切换求解器。

**G4 输出与稳定性**

1. 实现 GPU 叠框，支持坐标、颜色、ID 和置信度。提交小体积检测结果给 ROS。预览从设备图像向主机输出支路独立传输，慢订阅者的丢帧单独计数。兼容预览保留时间戳、帧号和 BGR 编码语义。
2. 将叠框结果交给 NVENC，核对支持的输入格式、PTS、录制帧数与 EOF flush。整段解码输出录像，检查逐帧顺序和封装可读性。
3. 先跑 `phantom30` 全段 1,498 个标注帧，使用 1080p、30 Hz、可见 rqt、NVENC、ROS 采集同时开启。验证完整源帧范围、检测时间戳、评测输出与录像。通过后按原顺序跑固定 15 个视频，每段冷启动、退出、排空及记账。窗口持续可见，确认画面实际更新。其他模式只用于定位性能成本。

### 6.4 验收与提交顺序

| 验收 | 判定与证据 |
|---|---|
| 正确性 | 以循环轮次和源帧号对齐。记录未匹配框、每帧检测框数量、中心偏差、置信度偏差和 ID 切换。沿用 SPEC-ROS 的 E3 门槛作为最低框中心与检测帧数回归参考（分别小于 2 px 和 2%），逐帧记录阈值边缘差异和异常片段。跟踪 ID 连续性单独验收。 |
| 显存驻留 | 严格模式无计算主链整帧或中间结果 H2D/D2H，无未登记的 CPU 算子。对 ROS 检测元数据、码流和可选预览分别计量。阶段完成以设备指针、trace 和功能结果共同确认。 |
| 性能 | 同口径报告覆盖率相对 21.11% 的百分点变化、检测消息相对 5,983 的变化、全帧 F1 相对 0.1917 的变化，以及总墙钟和录像帧数。报告候选 p95/p99、丢帧和显存。GPU 驻留、性能与识别质量分别判定。 |
| 可重复 | 每轮交付运行摘要和结果表。记录 Git 提交、运行副本指纹、容器和依赖版本、配置、片段哈希、命令及 trace 位置。 |

按 `G0 探针与评测配置 → G1 I/O Binding 和设备帧 → G2 MFE → G3 TF → G4 ROS/rqt/NVENC 正式评测` 提交。每个可验证的小步同次提交代码、配置、SPEC 版本、§8 修正记录和结果。第一笔实现提交交付当前 Docker 能力探针及与主分支评测配置的核对结果。最终提交包含逐视频与全量对照表、候选摘要、录像和 trace 的本地路径。

## 7. 偏差表

当前尚无优化实现引入的偏差。后续按版本、偏差、原因和验收影响登记。

## 8. 修正记录

| 日期 | 版本 | 改动与原因 | 验收与证据 |
|---|---|---|---|
| 2026-11-11 | 0.1.5 | 增加 G1.1 双模型 CUDA I/O Binding 探针，确认全输出可写入预分配显存 | 同会话全部输出逐元素一致。独立 LAD 会话差异保留在摘要，G1.2 待对拍真实候选。 |
| 2026-11-11 | 0.1.4 | 实施 G0，添加 CUDA/编解码/libav 和 ROS/rqt 探针及固定评测配置 | CUDA、120 帧设备直通和 600 帧 ROS/NVENC 联合短测通过，见 §6.2.1。P5 与全量候选对照待后续。 |
| 2026-11-11 | 0.1.3 | 按用户指示沿用主分支既有 1080p ROS 结果，固定 15 个现有视频和相同展示、录制负载，取消重建基线 | 核对远程默认分支、评测报告、配置 JSON 和运行脚本。候选运行待实现。 |
| 2026-11-11 | 0.1.2 | 把全 GPU 路线拆为可执行工作包，定义缓冲区契约、基准采集和分阶段验收 | 文档规划。现有源码、历史验收口径和官方接口已核对，性能和硬件探针待实施。 |
| 2026-11-11 | 0.1.1 | 增加全 GPU 驻留路线和多场景新基线，覆盖编解码、MFE、TF、LAC/LAD、可视化和 ROS 边界 | 源码与官方接口已核对。Docker 引擎未连通，G0 和 P5 待实测。 |
| 2026-11-11 | 0.1.0 | 从本地 `docker` 创建优化分支并建立独立规格，约定每轮代码与规格同步提交 | 记录最近一次 Docker 回放证据与待复测项。P1–P4 等待首轮性能基线。 |
