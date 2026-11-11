# SPEC-PERF · Docker 环境下的 ROS 性能优化

`spec_version: 0.1.19` · `branch: perf-optimization` · `base: docker@ed8cb69` · `prd_version: 4.2`

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
| 0.1.19 | 2026-11-11 | 本次两流实验提交 | 验证固定窗口上的跨帧 MFE 与 TF/LAC/LAD 重叠 |
| 0.1.18 | 2026-11-11 | 本次并行方案记录 | 暂缓 LAD 多目标批处理，记录跨帧 CUDA 流水线的依赖和验收方式 |
| 0.1.17 | 2026-11-11 | 本次 warp 对照提交 | 加入 GPU 共享坐标图的 OpenCV remap，记录全片吞吐和候选差异 |
| 0.1.16 | 2026-11-11 | 本次 MFE 分段计时提交 | 记录 ROS 全片 MFE 热点和 WSLg 显示缩放实测 |
| 0.1.15 | 2026-11-11 | 本次推送约定提交 | 记录每次核验提交后推送远程对应分支的约定 |
| 0.1.14 | 2026-11-11 | 本次 HUD 与实时回放提交 | 恢复按源时钟跳过过期窗口，增加分阶段统计及全屏 rqt |
| 0.1.13 | 2026-11-11 | 本次 GPU ROS 提交 | 接通 MFE、TF、LAC、LAD 的 CUDA 数据链，验证 1080p 全片和 Windows 原生预览 |
| 0.1.12 | 2026-11-11 | 本次全片验证提交 | 完成 phantom30 1080p CUDA TF 对拍及现有 ROS 后端的桌面回放 |
| 0.1.11 | 2026-11-11 | 本次 G3 对照提交 | 同步容器 TF 参考，比较 CUDA 分配库和并行 TF 原型，登记独立卡尔曼备选 |
| 0.1.10 | 2026-11-11 | 本次 G2 对照提交 | 实测新增 OpenCV 4.12 CUDA 与 Torch 路径的数值、耗时及混合方案 |
| 0.1.9 | 2026-11-11 | 本次 G2 提交 | 开始迁移 MFE 图像算子，记录固定单应矩阵下的像素和候选差异 |
| 0.1.8 | 2026-11-11 | 本次 G1.4 提交 | 完成真实 NVDEC 帧到 LAC/LAD 的串联对拍和独立计时 |
| 0.1.7 | 2026-11-11 | 本次 G1.3 提交 | 接入 libav CUDA 解码帧并验证生命周期、PTS 和色彩 |
| 0.1.6 | 2026-11-11 | 本次 G1.2 提交 | 实现 CUDA 裁剪、预处理和模型后处理，使用两段 1080p 视频对拍 |
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

**G1.2 实测（2026-11-11）**

`mgmd_onnx/cuda_models.py` 增加 CUDA LAC/LAD 模型阶段。用原裁剪坐标规则对设备图像切片，按 OpenCV 的 8 位固定点规则缩放，并使用与原 NumPy 归一化一致的查表值。LAC 计算批量 softmax。LAD 每次推理一个裁剪，在设备上复用 YOLOv5 的 NMS、坐标回映及最近中心规则。ORT 输出直接绑定 CUDA 张量。 [OpenCV resize 实现](https://github.com/opencv/opencv/blob/4.x/modules/imgproc/src/resize.cpp)

从原 `phantom30`、`phantom70` 各读取前 120 帧，逐段使用容器实际的推荐 MFE 与 FastSortTracker 生成候选，两条路径逐帧共享候选及顺序。覆盖空候选、无效裁剪、贴边裁剪、等置信度的非重叠候选以及空 NMS 结果。G1.2 用主机解码后单次上传输入帧；参考和对拍日志在测试边界读取设备结果。设备输入接入、稳定缓冲与完整拷贝追踪由 G1.3 和 P5 验收。

首次试验中浮点插值导致 LAC 输入相差 1 个 8 位灰阶，出现 4 次阈值翻转；LAD 浮点归一化路径也出现 1 次有无输出差异和 1 次坐标差异。最终采用整数插值与精确归一化，复测结果如下。

| 视频 | 对拍中心帧 | LAC 候选 | LAC 阈值翻转 | LAD 候选 | 两路径均输出框 | LAD 框或置信度最大绝对差 |
|---|---:|---:|---:|---:|---:|---:|
| phantom30 | 118 | 13,635 | 0 | 291 | 134 | 0 |
| phantom70 | 118 | 1,113 | 0 | 135 | 116 | 0 |

两段的 LAC 输入和 logits、LAD 输入和主预测在同会话对拍中均逐元素相同。LAC softmax 最大绝对差约 `1.11e-16`。独立 LAD 会话的原始预测最大绝对差约 `0.62647`，本次 426 个候选中未出现出框变化或坐标变化。原始数值差异的成因仍待定位。当前实现保留 ORT 流同步，YOLOv5 NMS 的动态索引同步成本留待设备时间线验证。

容器工程目录以项目用户加载 G0 CUDA 和 Python 环境后复测。

```bash
a=(/data/videos/*/ARD-MAV/videos/phantom30.mp4)
b=(/data/videos/*/ARD-MAV/videos/phantom70.mp4)
python3 scripts/probe_cuda_models.py "outputs/g1_2_$(date +%Y%m%d_%H%M%S)" \
  "${a[0]}" "${b[0]}"
```

完整逐帧候选、ID 和框对照见容器 `outputs/g1_2_20261111_071848/frames.jsonl`。 [验证摘要](../outputs/tables/g1_2_cuda_models.json)

**G1.3 实测（2026-11-11）**

新增 libav CUDA 解码适配器。FFmpeg 6.1.3 使用 `current_ctx=1` 借用 Torch 当前 CUDA 上下文，兼容 ORT 已初始化的情况。NV12 两平面按实际指针、尺寸和步幅导入 Torch，每帧检查地址一致。BGR 在 GPU 生成独立张量。转换完成后等待计算流，再允许解码器回收输入面。设备帧转换首版使用单线程顺序读取与同步交接。 [FFmpeg CUDA 上下文接口](https://ffmpeg.org/doxygen/6.1/hwcontext__cuda_8h.html)

| 视频 | 解码至 EOF | PTS 范围（时基 1/30000） | Y/UV 步幅 | 前 120 帧 BGR 与原 OpenCV 解码绝对差（最大 / 均值） |
|---|---:|---|---|---|
| phantom30 | 1,498 帧 | 0–1,498,497 | 2048 / 2048 | 3 / 1.118 |
| phantom70 | 1,349 帧 | 0–1,349,348 | 2048 / 2048 | 3 / 1.077 |

两段均确认 PTS 递增、EOF 重复读取稳定、重置后源帧号和首帧一致，并验证早期返回的 BGR 张量在后续解码和重置后保持内容。GPU 色彩转换与原 OpenCV 4.2 解码存在差异。G1.4 分别测同像素模型等价性和原解码路径的输出差异。

在现有容器中，`/opt/mgmd-g1-libav` 由 G0 使用的 FFmpeg 6.1.3 与同一版本 nv-codec-headers 构建，增加 `--enable-pic` 供适配器链接。项目用户加载 G0 CUDA/Python 环境和视频驱动库后复测。

```bash
export LD_LIBRARY_PATH=/tmp/mgmd-g0-video-libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
bash scripts/build_cuda_decoder.sh /tmp/mgmd-cuda-decoder.so
a=(/data/videos/*/ARD-MAV/videos/phantom30.mp4)
b=(/data/videos/*/ARD-MAV/videos/phantom70.mp4)
python3 scripts/probe_cuda_decoder.py /tmp/mgmd-cuda-decoder.so \
  "outputs/g1_3_$(date +%Y%m%d_%H%M%S)" "${a[0]}" "${b[0]}"
```

证据目录为容器 `outputs/g1_3_20261111_073306/`。 [验证摘要](../outputs/tables/g1_3_cuda_decoder.json)

**G1.4 实测（2026-11-11）**

从现有视频各取前 120 帧，以原 MFE/TF 在 OpenCV 解码帧上生成固定候选。GPU 模型链直接使用 NVDEC 产生的设备帧。诊断时，另将设备 BGR 下载给参考模型，以相同像素和候选比较模型链；再用原 OpenCV 像素比较颜色转换的影响。两段 236 个中心帧中，同像素的 LAC 输入、logits、LAD 输入和主预测逐元素相同。14,748 个 LAC 候选无阈值翻转，LAD 框和置信度一致。单独 ORT 会话在同像素上另有 4 个 LAD 框坐标变化，保留为跨会话差异。

| 视频 | 原 OpenCV 像素对照 GPU 解码的 LAC 阈值变化 | 最终框有无变化 | 双方均出框时坐标变化 | GPU 链单帧 p50 / p95 | LAC 预处理 p50 / p95 |
|---|---:|---:|---:|---:|---:|
| phantom30 | 14 | 5 | 13 | 186.4 / 254.4 ms | 140.4 / 202.4 ms |
| phantom70 | 1 | 0 | 5 | 41.6 / 60.8 ms | 15.1 / 30.0 ms |

共同框的单坐标最大绝对差为 1 px。实际门控后原 OpenCV 路径输出 134/116 个框，GPU 解码路径输出 131/116 个框。逐框差异见 `frames.jsonl`。当前色彩转换与原路径的差异影响检测，须在 G2 对拍及正式验收中显式检查。正式主分支基线继续冻结。

计时单独回放固定候选，排除参考推理、画面下载、MFE/TF、ROS 展示与录像。以同步墙钟报告解码和各模型子阶段，样本含冷启动。代码路径没有显式整帧上传或下载。为单裁剪 LAD 调度，LAC 阈值在 GPU 计算后读回紧凑索引，共 3,384 字节。PyTorch 分配峰值为 87.4/85.3 MiB，另有 ORT 和解码器分配。完整 CUDA 拷贝时间线属于 P5。高候选片段中逐候选 LAC 裁剪与缩放成为本实现的主要开销，结果仍属于开发验收。

复用 G1.3 的库、环境和视频路径。

```bash
run="outputs/g1_4_$(date +%Y%m%d_%H%M%S)"
python3 scripts/probe_cuda_models.py --decoder /tmp/mgmd-cuda-decoder.so \
  "$run" "${a[0]}" "${b[0]}"
python3 scripts/measure_cuda_model_chain.py /tmp/mgmd-cuda-decoder.so \
  "$run/frames.jsonl" "$run/timed" "${a[0]}" "${b[0]}"
```

证据目录为容器 `outputs/g1_4_20261111_073306/`。 [逐视频摘要](../outputs/tables/g1_4_cuda_chain.json) · [分阶段计时](../outputs/tables/g1_4_cuda_timing.json)

**G2 MFE**

**首轮实测（2026-11-11）**。OpenCV CUDA 提供 `cvtColor`、`absdiff`、稀疏 PyrLK、滤波和 8 连通域标记。现有容器中的 OpenCV 5.0.0 返回 CUDA 设备数 0，Python 绑定没有暴露这些算子。现有 Torch 2.4.1+cu121 可用，首轮复用其设备运算、`grid_sample` 和池化实现灰度、反向 warp、差分、自适应阈值、二值中值、三帧组合及形态学。CUDA OpenCV 的 LK 和连通域适合后续独立构建验证，CUDA 连通域返回标签，区域统计仍需实现。 [OpenCV CUDA 图像处理](https://docs.opencv.org/doc/doxygen/html/d0/d05/group__cudaimgproc.html) · [绝对差](https://docs.opencv.org/4.13.0/d4/d03/cudaarithm_8hpp.html) · [稀疏 LK](https://docs.opencv.org/doc/doxygen/html/d7/d05/classcv_1_1cuda_1_1SparsePyrLKOpticalFlow.html) · [CUDA 滤波](https://docs.opencv.org/4.13.0/da/dc5/cudafilters_8hpp.html)

在现有 Docker 中用 `phantom30`、`phantom70` 各前 50 张 1080p 帧。两条路径使用相同解码像素、参考单应矩阵和有效区。两段各有 48 个中心帧。灰度、阈值全部一致。将相同对齐像素送入两条路径时，差分掩膜和最终形态学结果逐像素一致。

完整设备 warp 对拍中，试用 1/32 定点插值时实测最大像素差为 7，改用连续双线性后降为 1。两段分别有 21/48 和 1/48 个中心帧的有序候选列表变化。候选列表由 CPU 连通域读取设备生成的闭运算结果得到，作为差异定位。warp 舍入差异仍待收敛，GPU 有效区生成、LK、RANSAC 和连通域仍待迁移。上述是 MFE 分阶段对拍结果，正式 ROS 1080p 基线沿用主分支记录。

以项目用户加载 G0 的 CUDA/Python 环境后，从容器工程目录运行。

```bash
a=(/data/videos/*/ARD-MAV/videos/phantom30.mp4)
b=(/data/videos/*/ARD-MAV/videos/phantom70.mp4)
python3 scripts/probe_cuda_mfe.py "outputs/g2_fixed_$(date +%Y%m%d_%H%M%S)" \
  "${a[0]}" "${b[0]}"
```

探针退出状态用于同像素的灰度、掩膜和形态学对拍。warp 与候选差异另列。逐帧结果在容器 `outputs/g2_fixed_20261111_075104/frames.jsonl`。 [验证摘要](../outputs/tables/g2_fixed_cuda_mfe.json)

**OpenCV CUDA 实测（2026-11-11）**。新装的 OpenCV 4.12.0（CUDA 12.1）位于 `/opt/opencv-cuda-4.12.0`，实测 CUDA 设备数 1。默认 Python 仍加载 OpenCV 5.0.0 CPU。分别运行两个进程，先以 5.0.0 冻结 `phantom30` 与 `phantom70` 各前 50 张 1080p 帧、参考 H、有效区和结果，再在 4.12.0 环境中比较 CPU OpenCV、CUDA OpenCV、CUDA Torch，以及 CUDA OpenCV warp 加 CUDA Torch 掩膜和形态学的混合路径。每段 48 个中心帧参与对拍，前 5 帧排除出计时。算子计时为同步墙钟，输入上传和诊断下载在计时外。

| 算子 | OpenCV 4.12 CPU p50 范围（ms） | OpenCV 4.12 CUDA p50 范围（ms） | Torch CUDA p50 范围（ms） |
|---|---:|---:|---:|
| 灰度 | 0.29–0.43 | 0.69–0.73 | 1.45 |
| warp（每方向） | 3.03–3.07 | 1.34–1.40 | 7.81–7.82 |
| 差分、阈值、掩膜、中值（每方向） | 4.38–4.40 | 35.76–36.50 | 1.51 |
| 形态学 | 1.56–1.64 | 4.53–4.57 | 4.25–4.26 |
| 稀疏 LK（每方向） | 2.44–3.15 | 2.77–3.17 | — |
| 8 连通域标记 | 0.88–1.05 | 1.45–1.50 | — |

三条路径的灰度与 5.0.0 参考完全一致。CUDA OpenCV 与 Torch 的 warp 最大差均为 1 灰阶，4.12 CPU 对 5.0 CPU 最大差为 7。与 5.0 参考比较候选框集合，纯 CUDA OpenCV、纯 Torch CUDA 和混合路径分别有 **54/96、22/96、22/96 帧变化**。固定相同的对齐像素和有效区后，CUDA OpenCV 的方向掩膜仍有 269,816 个像素变化，最终形态学掩膜有 19,465 个像素变化。混合路径的设备指针共享通过，算子联调优先采用 CUDA OpenCV warp 和 Torch 掩膜。另以 Torch 显存构造 `GpuMat` 再执行恒等 warp，确认双向可共享设备缓冲。

稀疏 LK 在共同有效点上的位移差 p95 为 0.0034–0.0036 px，状态差为 `phantom30` 313/28,800、`phantom70` 5,733/28,800。候选接入前须用这些状态差验证 RANSAC 内点、回退与候选影响。CUDA 连通域在相同输入掩膜下 96 帧区域分区一致。本轮只计标签生成，区域统计和候选顺序另验。

复测时先在默认环境运行 `compare_mfe_opencv_cuda.py capture <fixtures> <phantom30> <phantom70>`，再将 `/opt/opencv-cuda-4.12.0/python` 加入 `PYTHONPATH` 首位、将该前缀的 `lib` 加入 `LD_LIBRARY_PATH`，运行 `compare_mfe_opencv_cuda.py compare <fixtures> <summary.json>`。本次目录 `outputs/g2_opencv_cuda_20261111_085437/`，原始冻结帧保留在容器。 [对照摘要](../outputs/tables/g2_opencv_cuda_comparison.json)

1. 先实现灰度、预处理、差分、阈值、形态学及连通域统计。用固定单应矩阵将图像算子与对齐算法分开验证。记录插值、边界填充、有效区、连通性、候选面积和排序。
2. 实测 CUDA 稀疏 LK 的关键点和状态，再接 GPU 单应性求解和透视变换。分别保存内点数、重投影误差、有效区、候选数量和位置。覆盖低纹理、无有效变换、镜头抖动和高候选帧。需要新增 CUDA 内核时放在 ROS GPU 后端，复用阶段一配置和参考输出。
3. 隔离对拍通过后接入帧窗口。缓存可复用的灰度帧等中间结果。逐轮测量每项缓存或融合对延迟、显存与数值结果的影响。

**G3 TF**

1. 在现有视频的开发短样上采样真实关联矩阵，记录行列分布、门控稀疏度和代价并列频率。用实际规模比较 CPU 参考与设备端精确分配原型，时间包含同步和必要拷贝。此对拍服务于实现校验。
2. 将 Kalman 预测、更新、关联和轨迹生命周期放到 GPU。验证空集、矩形矩阵、相同代价、同帧重复、跨帧跳跃、丢失重现和循环重置。对拍关联对、目标函数值和轨迹 ID。等价最优解导致 ID 变化时，先确定稳定的匹配规则并逐段检查其影响。
3. 在 MFE 和 TF 间传递有界设备候选数组及设备端计数。超过容量时显式报告、按规则扩容或终止严格模式。禁止无记录地截断候选或切换求解器。

#### G3 CUDA 分配库与 TF 原型对照

CPU 参考同步自当前容器的 `tf_fast.py`，SHA-256 为 `a9eb989a2c8c2d14185a563581015a280ec23d560e03e7a2ba0bef3e163d7240`。分支既有默认参数仍采用单轮 IoU。G3 回放显式启用按视频帧间隔预测和第二轮距离补配，参数为 `iou_thresh=.3, association_radius_factor=2, span=3, d_min=.02, max_age=3, min_hits=1`。两视频各取 118 个连续中心帧及其中每三个取一帧的 40 帧，另做 10 个合成帧。两组使用同一 CPU MFE 候选。

`torch-linear-assignment==0.0.6` 已在 Python 3.8.10、Torch 2.4.1+cu121、CUDA 12.1 上从 PyPI 源码包编译成功，构建时临时提供 `importlib-metadata==8.5.0`。源码包 SHA-256 为 `6aad76d3c976b40cf1572e81bed8aedbeafdacb817f290e137c7a76ad58fcd23`。公开接口将 CUDA 双精度代价降为单精度，631 次求解中有 62 次门控后的配对和目标值差异。底层双精度入口在这批矩阵上无门控配对或目标值差异。内核按矩阵并行，单路视频每轮仅一张矩阵。`phantom30` 连续段单次求解 p50/p95 为 SciPy 0.43/0.77 ms、库公开接口 16.28/25.79 ms、库双精度入口 28.39/44.93 ms。计时同步 CUDA，排除测试上传和下载。库未通过速度验收，未接入 TF。 [库文档](https://pypi.org/project/torch-linear-assignment/0.0.6/) · [求解对照](../outputs/tables/g3_cuda_assignment.json)

现有原型使用线程块内并行扫描求解列。326 帧的逐候选轨迹 ID、选中 ID、框、轨迹详情与生命周期均与参考一致。容量溢出、重置和设备有效计数通过。`phantom30` 连续段整段 TF 每帧 p50/p95 为 CPU 17.04/26.44 ms、CUDA 4.57/6.12 ms。稀疏 `phantom70` 连续段为 CPU 1.04/1.83 ms、CUDA 0.49/0.66 ms。保留原型用于 MFE→TF 设备串联，ROS 接入、CUDA trace 与正式 1080p 展示对比待验证。 [TF 回放摘要](../outputs/tables/g3_cuda_tf.json)

**卡尔曼备选**使用 `torch-kf==0.3.1` 的批量 CUDA 滤波与独立 GPU 分配求解器拼接。该版支持 Python 3.8。沿用现有 7 维状态、4 维测量、噪声矩阵、跳帧预测和面积保护，先用双精度对拍逐轨迹预测框与协方差。如果分段计时显示卡尔曼有优化空间，再验证批量收益及更新过程中的主机同步。本轮保留现有 CUDA 卡尔曼。 [torch-kf 文档](https://pypi.org/project/torch-kf/0.3.1/)

从容器的 `mgmd-linux` 运行 `PYTHONPATH=. python3 scripts/probe_cuda_assignment.py <summary.json> /data/videos/phantom30.mp4 /data/videos/phantom70.mp4`。执行 `bash scripts/build_cuda_tf.sh /tmp/mgmd-tf-g3.so` 后运行 `PYTHONPATH=. python3 scripts/probe_cuda_tf.py /tmp/mgmd-tf-g3.so <新输出目录> /data/videos/phantom30.mp4 /data/videos/phantom70.mp4`。逐帧日志在容器 `outputs/g3_tf_parallel_20261111/`。

#### phantom30 1080p 全片验证

输入按正常播放范围可解码 1,498 帧。对 `9fef1b9` 的 CUDA TF 原型使用 `probe_cuda_tf.py` 和 `--frames 1498`，先读取整个视频，再对 1,496 个有效三帧窗口进行连续对拍，并从中抽取 499 帧做跳帧对拍。两组逐候选轨迹 ID、筛选及生命周期差异均为零。连续段 170,615 个候选保留 112,485 个，TF p50/p95 为 CPU 6.02/11.46 ms、CUDA 2.47/4.60 ms。时间排除探针的上传和比对下载。原始证据在容器 `outputs/g3_tf_full_1080p_exact_20261111/`。 [全片 TF 摘要](../outputs/tables/g3_cuda_tf_full_phantom30.json)

ROS 回放沿用 G0 的 rqt、NVENC 和隔离 master 配置，将源帧限制改为 `limit:=0`，沿用 1920×1080、30 Hz、`loop:=false`、`frame_delta_predict:=true` 和 `association_radius_factor:=2.0`。ROS 使用现有 CPU FastSort 和 CUDA 模型推理。源在 50.1 秒内发布 1,498 帧，检测线程处理 227 帧，覆盖率 15.15%。主分支既有正式评测的同视频为 329 条检测消息、22.0% 覆盖率。本轮只读检测线程计数，未用正式采集器对齐消息和评分。可见 rqt 订阅叠框话题，话题抽样 30.49 Hz，输出 H.264 1920×1080 全段解码 1,491 帧，比发布少 7 帧。首轮缺失 NVENC 库，重跑前按 G0 恢复 FFmpeg 6.1.3 和驱动库路径。成功运行在容器 `outputs/g3_ros_full_1080p_nvenc_20261111/`，Windows 工作区同级的 `mgmd-1080p-phantom30-20261111/` 保存录像和日志。CUDA TF 接入 ROS 后仍需用正式采集器重做同口径对比。

#### GPU ROS 文件回放与 1080p 预览

`track_gpu.launch` 使用单进程 NVDEC → CUDA MFE → CUDA TF → CUDA LAC/LAD。MFE 用 OpenCV CUDA LK、Torch 批量单应性 RANSAC、warp 和形态学，以及 OpenCV CUDA 连通域和 Torch 区域统计。TF 沿用 G3 并行原型，ONNX 会话禁用 CPU 算子回退和运行时 provider 回退。LAC 裁剪使用 GPU 双线性采样，LAD 在 GPU 裁剪、推理及 NMS。ROS 发布边界读取检测元数据，预览支路下载 GPU 叠框后的 BGR 帧。动态索引和模型运行仍有主机同步。

当前版本的 MFE 采样、单应性有效区、连通域顺序与 CPU 实现有差异，LAC 批量插值与旧路径也可能有舍入差异。本轮验证完整执行和吞吐，待正式采集器评估 F1、ID 连续性和逐阶段输出。预览画绿色框，正式评测所需的标注样式、NVENC 写盘和 30 Hz 丢帧策略待接入；本入口顺序处理全部有效窗口，超出 30 Hz 时间预算时放慢回放。

在 `mgmd-ros` 中以 `valuelesser` 运行。先沿用现有 ROS demo 的环境，再显式选用 CUDA OpenCV 和 NVDEC 驱动目录。新入口默认使用 G1 和 G3 已生成的临时库，可通过 launch 参数覆盖路径。

```bash
source <(sed -n '1,/^cd "\$ROOT"$/p' /usr/local/bin/mgmd-ros-demo)
export PYTHONPATH=/opt/opencv-cuda-4.12.0/python:$PYTHONPATH
export LD_LIBRARY_PATH=/opt/opencv-cuda-4.12.0/lib:/tmp/mgmd-g0-video-libs:$LD_LIBRARY_PATH
export DISPLAY=127.0.0.1:47 QT_X11_NO_MITSHM=1
export ROS_MASTER_URI=http://127.0.0.1:11331 ROS_IP=127.0.0.1
roslaunch mav_tracker_nodes track_gpu.launch video:=/data/videos/phantom30.mp4
```

Windows 原生预览通过 PowerShell 7 执行 `scripts/preview_ros_native.ps1`。它订阅同一个 ROS master 的 `/mav/debug_image`，核验 1920×1080 BGR 布局，再逐帧传给 Windows FFplay，不走 WSLg 窗口缩放。只用原生预览时，launch 可传入 `rqt:=false`。300 帧短跑验证窗口实际打开，Windows 200% DPI 下的物理外框为 1924×1140。关闭 FFplay 即结束该预览订阅。

1080p 全片在 rqt 打开时读取 1,498 帧，连续处理 1,496 个中心帧，其中 1,169 帧有检测。墙钟 122.792 秒，**12.18 fps**，平均计算耗时 67.71 ms。Windows 原生预览与 rqt 同时开启的 300 帧短测处理 298 帧，9.59 fps，两次运行的片段长度和预览负载不同。Nsight 短段报告有 CUDA runtime 事件，未采到 GPU memcpy 或 kernel activity 表，因此严格拷贝验收仍未完成。容器日志和 trace 位于 `outputs/g4_gpu_ros_phantom30_20261111/`。 [运行摘要](../outputs/tables/g4_gpu_ros_phantom30.json)

#### 30 Hz 实时 GPU 回放、rqt HUD 与分段耗时

v0.1.13 在同一循环内解码、检测和发布，处理速度限制取帧速度。v0.1.14 的 `track_gpu.launch` 默认 `realtime:=true`，按单调时钟和 `rate:=30` 为每帧定位播放时刻，过期窗口只解码并维护最近三帧，跳过检测。处理窗口的相邻帧保持连续，TF 继续按源视频帧号预测。此入口的预览随检测结果更新。使用 `realtime:=false` 可离线处理所有有效窗口。

rqt ImageView 默认全屏，Escape 返回窗口模式，F11 全屏，`fullscreen:=false` 可用原窗口。实测 X11 客户区从 955×567 提高到当前 WSLg 输出上限 1440×810，ROS 图像维持原始 1920×1080。当前 WSLg 未提供 1920×1080 显示模式。HUD 在 ROS 预览支路下载图像后绘制，显示最近约 2 秒检测吞吐、源频率、处理与跳过计数、四阶段均值和最慢阶段。检测主链保持 CUDA。

全片 49.968 秒，读 1,498 帧，处理 545 个有效窗口、跳过 951 个，平均 10.91 fps。阶段 CUDA 事件均值/p95（ms）为 MFE 40.49/44.14、TF 1.98/2.79、LAC 3.38/8.65、LAD 16.96/30.03。MFE 占四阶段平均时间约 64%。四阶段时间未包含源解码、预览绘制、ROS 发布及节流。运行日志和 1080p HUD 样图在容器 `outputs/g4_gpu_ros_hud_20261111/`。 [运行摘要](../outputs/tables/g4_gpu_ros_hud_phantom30.json)

#### 1080p ROS 的 MFE 分段计时

容器内按前述环境运行 `roslaunch mav_tracker_nodes track_gpu.launch profile_mfe:=true`。该开关默认关闭。CUDA 事件按已处理中心帧统计，前、后两个方向同名阶段先在帧内相加。计时包含阶段内同步和调度间隔，可用于定位端到端瓶颈。

`phantom30` 全片读取 1,498 帧，按 30 Hz 源时钟在 49.974 秒内处理 511 个窗口，10.23 fps，MFE 均值 39.37 ms。最高项为 Torch 双精度坐标网格和透视采样，含 warp 边界同步，双向均值 16.222 ms、p95 16.659 ms，占 MFE 均值约 41%。第二项为 RANSAC，双向均值 6.247 ms、p95 7.579 ms。下一次 warp 优化可比较融合坐标生成、透视采样和有效区的 CUDA 实现，避免将 GPU 上求出的单应矩阵下载给 CPU。容器日志在 `outputs/g4_mfe_profile_20261111/launch.log`。 [完整分段数据](../outputs/tables/g4_mfe_profile_phantom30.json)

Windows 主屏实测物理尺寸 2880×1620，当前逻辑尺寸 1440×810，WSLg XWayland 和全屏 rqt 均为 1440×810。源图像仍为 1920×1080。若 WSLg 跟随 Windows 150% 显示缩放，逻辑尺寸可为 1920×1080。该配置未实测，本轮保留系统设置。 [WSLg 显示配置](https://github.com/microsoft/wslg/wiki/WSLg-Configuration-Options-for-Debugging)

#### OpenCV CUDA remap 设备路径

`roslaunch mav_tracker_nodes track_gpu.launch warp_backend:=opencv profile_mfe:=true` 启用实验路径。Torch 将 GPU 单应矩阵转换为 FP32 后生成两张源坐标图及有效掩膜，OpenCV CUDA remap 在默认 CUDA 流借用灰度图和坐标图地址，输出写入 Torch 预分配的 GPU 图像并校验指针。图像、矩阵、坐标图和掩膜没有显式主机往返。计时含坐标生成、采样、有效区和原有边界同步。

同一 `phantom30` 1080p 全片按 30 Hz 源时钟运行，rqt 和 HUD 开启。Torch 与 OpenCV 分别处理 511 和 672 个窗口，吞吐 10.23 → 13.45 fps。warp 均值/p95 由 16.222/16.659 降至 3.775/3.956 ms，MFE 由 39.37/41.50 降至 26.42/29.08 ms。两轮实时选帧不同。OpenCV 轮读取 1,498 帧，跳过 824 个窗口，488 帧有检测，耗时 49.951 秒。日志在容器 `outputs/g4_mfe_remap_20261111/launch.log`。

`compare_cuda_mfe_warp.py /tmp/mgmd-cuda-decoder.so <summary.json> /data/videos/phantom30.mp4 /data/videos/phantom70.mp4` 用相同解码帧对拍，每段 118 个中心帧。逐方向同矩阵对照最大差 1 灰阶，前后两方向的有效区差分别按视频累计为 40 和 317 像素。两段分别有 53 和 19 帧候选框集合变化，净候选数变化为 −3 和 0。保留 Torch 为默认，正式检测和轨迹质量通过后再考虑切换。 [候选与像素对照](../outputs/tables/g4_mfe_remap_comparison.json)

#### LAD 多目标批处理（暂缓）

当前 LAD ONNX 固定输入为 `[1, 3, 320, 320]`，每个 LAC 保留目标调用一次。以后验证动态批维或少量固定批规格，在显存中裁剪和拼批，按裁剪块独立执行 NMS、最近中心框选择和轨迹 ID 回传。先用固定 LAC 输出对拍框、置信度和 ID，再按候选数分组实测 batch 1/2/4/8 的 p50/p95、吞吐和峰值显存。超出选定批容量时分批处理。本轮仅记录。

#### 跨帧 CUDA 流水线（可行性记录）

按源帧时间顺序，在同一时刻可让较早窗口 `n` 执行 LAC/LAD，`n+1` 执行 TF，较晚窗口 `n+2` 执行 MFE。这里的下标表示已接纳中心窗口的顺序，实际源帧号可有跳跃。MFE 需要中心帧及前后各一帧。TF 按源帧号递增更新同一份轨迹状态，同帧 LAC/LAD 消费该帧的 TF 输出。模型推理仍可用当前 batch=1。

先试两条流重叠 MFE 与后处理，将耗时较小、带跨帧状态的 TF 保持在后处理流。用 CUDA event 交接候选缓冲并保护图像、轨迹输出和临时缓冲的生命周期。MFE 当前固定在默认流，须先验证 OpenCV LK、warp、CCL 的流兼容性并移除会阻塞整段的同步。模型侧验证当前 ORT 的 `user_compute_stream`、I/O Binding 和输出完成事件，避免由 CPU 等待阻塞 MFE 调度。两流验证后，可实测是否值得给 TF 单独开流。 [ONNX Runtime CUDA 流文档](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#user_compute_stream)

使用有界的在途窗口数，保持实时模式的过期跳帧，禁止把队列积压当作吞吐收益。先用固定源帧序列对拍候选、ID、框和分数，再以相同 warp 后端、1080p、30 Hz 源时钟和 rqt/HUD 实测 GPU 时间线重叠、检测帧率、端到端延迟、跳帧和峰值显存。以 OpenCV remap 回放数据为量级参考，MFE 26.42 ms、TF 1.86 ms、LAC/LAD 合计 19.42 ms。两侧会竞争同一张 RTX 3050 的计算和带宽，实际收益须实测。

#### 跨帧两流固定窗口实验

在当前 Docker 使用 `scripts/probe_cuda_frame_overlap.py /tmp/mgmd-cuda-decoder.so /tmp/mgmd-tf-g3.so <video> <summary.json>`。先用 12 帧预热，再按最多 12 个中心窗口一组复用同一批 NVDEC 显存输入，每组两种模式分别重置 TF，交替运行顺序。两组都使用 OpenCV remap、相同 CUDA 流配置和绑定到后处理流的 ORT，只有候选帧提交次序不同。MFE 工作线程提前处理一个后续窗口，后处理按源帧号消费候选。OpenCV 4.12 的 CCL Python 接口无流参数，仍经默认流，前后保留同步。

| 1080p 视频 | 窗口数 | 串行计算 fps | 两流计算 fps | 最终框、置信度、ID 或候选数量有差异的窗口 |
|---|---:|---:|---:|---:|
| phantom30 | 138 | 24.07 | 26.80 | 0 |
| phantom70 | 138 | 29.54 | 32.79 | 0 |

置信度和坐标容差为 1e-6。相邻窗口阶段的 CUDA event 区间累计交叠 3.34 秒和 2.64 秒，包含阶段内空隙，不能据此计算内核同时执行的时间。两模式混合运行的 PyTorch 峰值分配约 224–227 MiB，范围不含 ORT、OpenCV 和 NVDEC。计时不含解码、ROS 发布、预览或 30 Hz 源节拍。默认 ROS 路径短测读取 12 帧并处理全部 10 个有效窗口。两流还需验证实时调度、跨组持续跟踪及进程总显存。容器日志在 `outputs/g4_cuda_overlap_20261111/`。 [phantom30 摘要](../outputs/tables/g4_cuda_overlap_phantom30.json) · [phantom70 摘要](../outputs/tables/g4_cuda_overlap_phantom70.json)

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

每次提交完成并核验后，推送到 `origin/perf-optimization` 并核对远程分支提交。首次推送建立上游跟踪。推送失败时记录原因和本地提交状态。用户明确要求暂不推送时暂停推送。

## 7. 偏差表

当前尚无优化实现引入的偏差。后续按版本、偏差、原因和验收影响登记。

## 8. 修正记录

| 日期 | 版本 | 改动与原因 | 验收与证据 |
|---|---|---|---|
| 2026-11-11 | 0.1.19 | 验证两流固定窗口原型 | 两段共 276 个窗口通过对拍，计算吞吐提升约 11%。 |
| 2026-11-11 | 0.1.18 | 登记 LAD 批处理待办和跨帧并行设计 | 纯文档修订，记录阶段依赖、CUDA 流和后续验收。 |
| 2026-11-11 | 0.1.17 | 对拍全显存 OpenCV remap 和 Torch warp | 1080p 全片 13.45 fps。236 个固定窗口中 72 帧候选集合变化，继续按实验参数启用。 |
| 2026-11-11 | 0.1.16 | 增加可选 MFE 子阶段计时，核对 Windows 主屏和 WSLg 尺寸 | 511 个 1080p 中心帧中，warp 平均 16.222 ms，为最慢环节。 |
| 2026-11-11 | 0.1.15 | 同步 SPEC 与仓库维护约定，每次核验提交后推送对应远程分支 | 纯文档修订。远程分支与本地 HEAD 一致作为推送验收。 |
| 2026-11-11 | 0.1.14 | 增加 HUD 和全屏 rqt，按 30 Hz 源时钟处理过期窗口 | 1080p 全片处理 545 个窗口，10.91 fps。MFE 平均 40.49 ms，为最慢阶段。 |
| 2026-11-11 | 0.1.13 | 增加四阶段 CUDA ROS launch 和 Windows 原生 1080p 预览 | 全片处理 1,496 帧，12.18 fps。300 帧原生预览通过，正式评分、拷贝验收和 NVENC 待接入。 |
| 2026-11-11 | 0.1.12 | 记录 phantom30 1080p 全片验证 | CUDA TF 的 1,496 个连续窗口逐帧一致。现有 ROS 后端发布 1,498 帧，NVENC 录像解码 1,491 帧。 |
| 2026-11-11 | 0.1.11 | 同步容器 TF 参考，验证 CUDA 分配库及并行 TF，记录独立卡尔曼备选 | 631 矩阵对照与 326 帧回放完成。CUDA 分配库未达速度验收，并行 TF 原型继续用于设备串联。 |
| 2026-11-11 | 0.1.10 | 验证新装 OpenCV CUDA 的 MFE 算子，并对照已有 Torch 路径 | 96 帧数值对照与分段计时完成。混合 warp 路径、CUDA LK 和连通域的后续接入边界见 §6.3。 |
| 2026-11-11 | 0.1.9 | 核对可用 CUDA 库并实现固定矩阵 MFE 的 GPU 图像阶段 | 两视频 96 个中心帧对拍。同像素灰度、掩膜和形态学一致。warp 最大差 1 灰阶，22 帧候选列表变化。 |
| 2026-11-11 | 0.1.8 | 将 NVDEC 设备帧与 CUDA 模型前后处理串联，并独立测量各阶段 | 236 帧同像素对拍通过。登记颜色转换导致的 15 次 LAC 阈值变化及 5 次最终框有无变化。 |
| 2026-11-11 | 0.1.7 | 增加 NVDEC CUDA 帧适配与设备端 NV12→BGR | 两段视频解到 EOF，指针共享、PTS、重置和帧生命周期通过。记录色彩差异。 |
| 2026-11-11 | 0.1.6 | 迁移 CUDA 裁剪、整数缩放、归一化和 LAC/LAD 后处理 | 两段现有视频 236 个中心帧对拍，LAC 无阈值翻转，LAD 框和置信度同会话一致。 |
| 2026-11-11 | 0.1.5 | 增加 G1.1 双模型 CUDA I/O Binding 探针，确认全输出可写入预分配显存 | 同会话全部输出逐元素一致。独立 LAD 会话差异保留在摘要，G1.2 待对拍真实候选。 |
| 2026-11-11 | 0.1.4 | 实施 G0，添加 CUDA/编解码/libav 和 ROS/rqt 探针及固定评测配置 | CUDA、120 帧设备直通和 600 帧 ROS/NVENC 联合短测通过，见 §6.2.1。P5 与全量候选对照待后续。 |
| 2026-11-11 | 0.1.3 | 按用户指示沿用主分支既有 1080p ROS 结果，固定 15 个现有视频和相同展示、录制负载，取消重建基线 | 核对远程默认分支、评测报告、配置 JSON 和运行脚本。候选运行待实现。 |
| 2026-11-11 | 0.1.2 | 把全 GPU 路线拆为可执行工作包，定义缓冲区契约、基准采集和分阶段验收 | 文档规划。现有源码、历史验收口径和官方接口已核对，性能和硬件探针待实施。 |
| 2026-11-11 | 0.1.1 | 增加全 GPU 驻留路线和多场景新基线，覆盖编解码、MFE、TF、LAC/LAD、可视化和 ROS 边界 | 源码与官方接口已核对。Docker 引擎未连通，G0 和 P5 待实测。 |
| 2026-11-11 | 0.1.0 | 从本地 `docker` 创建优化分支并建立独立规格，约定每轮代码与规格同步提交 | 记录最近一次 Docker 回放证据与待复测项。P1–P4 等待首轮性能基线。 |
