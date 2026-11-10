# SPEC-PERF · Docker 环境下的 ROS 性能优化

`spec_version: 0.1.1` · `branch: perf-optimization` · `base: docker@ed8cb69` · `prd_version: 4.2`

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
| 0.1.1 | 2026-11-11 | 本次文档提交 | 定义全 GPU 数据路径、能力探针和逐阶段验收 |
| 0.1.0 | 2026-11-11 | 初始规格提交 | 建立 Docker 性能优化基线、测量口径与提交规则 |

## 1. 范围

覆盖视频读取与缩放、ROS 图像传输、MFE、TF、LAC、LAD 和可视化。先用实测定位瓶颈，再逐项比较候选改动。摄像头和 TT 实机验收遵循 SPEC-ROS。

## 2. 环境与基线

代码起点为本地 `docker` 的 `ed8cb69`。运行环境使用 Docker Desktop 现有 `mgmd-ros` 容器，工程目录为 `/workspace/MGMD-ROS/mgmd-linux`，运行用户为 `valuelesser`。最近一次验证环境为 Ubuntu 20.04、ROS Noetic、Python 3.8.10、RTX 3050 Laptop GPU、ONNX Runtime GPU 1.19.2。启动入口、依赖、数据和模型路径见 [SPEC-ROS §2.4](SPEC-ROS.md#24-docker-desktop-回放环境)。

容器的迁移源码与 Git 分支曾存在差异。首次性能测量前，核对实际执行的 ROS、`mgmd_onnx` 和脚本文件，记录其提交或文件哈希、工作区差异、运行参数、输入视频和模型哈希。需要同步源码时保留容器原有差异，并确认编译产物对应本轮源码。

2026-11-10 在容器迁移版本上，以 20 Hz、960×720 回放 `phantom30`，LAC 和 LAD 均使用 CUDA。视频源发布 1,498 帧，检测处理 1,405 帧，检测话题采样约 19.7 Hz。该记录用于核对运行环境。分支可比较的性能基线待核对源码后重测。2026-11-11 建档时 Docker 引擎未连通，本轮未重跑性能测试。

## 3. 测量口径

- 首轮包括 `phantom30` 的 960×720、20 Hz，以及重新选取的低候选、高候选、快速运动和无目标片段。覆盖部署所需的分辨率和帧率，含非限速吞吐档。固定帧范围、模型、阈值、缓冲和预热方式。同一组基线与候选使用相同配置，桌面预览和录像单独测量。
- 记录输入帧数、检测处理帧数、输出频率、丢帧、端到端延迟 p50/p95，以及 MFE、TF、LAC、LAD 和可视化的分段耗时。分别统计检测消息和可视化帧，注明 GPU 计时的同步边界。
- 在同一容器中分别重复基线和候选至少三轮，记录各轮结果与中位数。测量饱和吞吐时另列输入速率。20 Hz 回放结果用于观察该负载下的延迟与丢帧。

## 4. 验收

| 项目 | 判定 |
|---|---|
| P1 可复测 | 基线和候选有可定位的源码、容器配置、输入、参数和运行摘要。 |
| P2 性能收益 | 报告目标指标的基线值、候选值、变化和轮间波动，同时报告延迟、吞吐和丢帧的变化。 |
| P3 功能回归 | 检查三节点、ROS 话题、帧序号、实际 CUDA provider 及跟踪逻辑。用相同帧范围对拍检测框、置信度和跟踪 ID，说明差异及其依据。 |
| P4 可视化回归 | 涉及可视化时，验证 `/mav/debug_image`、桌面画面更新或录像完整性，按本轮涉及的入口记录。 |
| P5 GPU 驻留 | 严格 GPU 模式下，用时间线与各阶段输入输出设备检查，确认解码后的图像和中间张量在计算主链路无 H2D/D2H 往返，逐算子核对推理设备。检测消息、编码码流和兼容预览的输出边界单独统计。 |

## 5. 实施顺序

1. 核对 Docker 运行副本，建立固定负载的可重复基线。
2. 按 §6 做能力探针。根据新测量安排内核和集成工作，优先验证存在兼容性或算法等价风险的环节。SPEC-ROS §20–21 的历史结果仅用于提出待复测的问题。
3. 每轮测量单项改动，完成受影响的回归检查，再按 §0 提交实现、版本和结果。

## 6. 全 GPU 流水线路线

目标是让解码后的像素、MFE 特征和候选、TF 跟踪状态、LAC/LAD 张量及叠框后的编码输入留在同一 GPU 上。CPU 负责文件读取和解封装、启动调度、时间戳、ROS 消息和输出封装。首选在现有容器中增加一个可选的同进程 GPU 后端，以 CUDA 缓冲区衔接 FFmpeg、CUDA 算子和 ONNX Runtime，保留现有后端作为逐帧参考。

ROS1 的 `sensor_msgs/Image` 以字节数组承载图像，现有检测节点从消息读入 NumPy。GPU 模式将节点间的图像计算移到同一进程的设备缓冲区。`/mav/detections` 发布小体积元数据，`/mav/debug_image` 由兼容预览分支按需下载。需要继续提供 `/camera/image_raw` 时，也由单独的兼容分支发布，并在测量中记录其成本。[ROS Noetic Image 消息定义](https://github.com/ros/common_msgs/blob/noetic-devel/sensor_msgs/msg/Image.msg)

| 阶段 | 实施与通过条件 |
|---|---|
| G0 · 能力与新基线 | 核对运行源码，记录 Docker 镜像、驱动、CUDA/cuDNN/ORT、FFmpeg 编译项和可用显存。在同一容器分别实际运行 CUDA 推理、H.264 NVDEC 解码、NVENC 编码及设备帧直通小样，验证设备帧格式、帧数、时间戳和编码输出。设备枚举仅作诊断。用 Nsight Systems/CUPTI 核对拷贝和 GPU 执行，另用无分析器运行比较性能。任一硬件能力失败时记录故障层级，允许继续验证计算链，严格 GPU 模式暂不通过。 |
| G1 · 输入、设备缓冲与模型 | 首选 C++ FFmpeg/libav 的 CUDA 硬件帧接口，向当前 Python 3.8 进程提供设备帧适配。使用有界缓冲保留前、当前和未来帧，并用 CUDA stream/event 管理解码帧生命周期。先用合成设备帧验证 LAC/LAD I/O Binding、GPU 裁剪、归一化、softmax、NMS 与坐标回映，再接真实解码。检查 ONNX 实际算子落点及所有输出，排查 CPU fallback；使用支持的严格设置或运行 trace 明确判定。动态候选采用有界批次和掩码，先核对模型的实际 batch/shape，再评估 LAD 真批处理。 |
| G2 · MFE | 在 GPU 上实现缩放、灰度、稀疏金字塔 LK、单应性估计、warp、差分/阈值、形态学和连通域统计。可先验证 CUDA 版 OpenCV 的 LK 与可用算子；RANSAC、边界和连通域统计缺少等价设备路径时用 CUDA 实现。按中间结果对拍关键点及状态、单应矩阵、有效区掩膜、连通域面积和候选框，覆盖对齐失败、低纹理、边缘目标和镜头运动。保留原帧窗口和帧号语义。 |
| G3 · TF | 将 Kalman 状态、帧间隔预测、关联成本、门控、精确线性分配和轨迹更新迁至设备。先做小矩阵 GPU 分配原型并实测，与当前 SciPy 解对照等成本解、矩形矩阵、空集、漏帧和回放重置。按视频逐帧检查 ID 与轨迹生命周期。GPU 分配速度由当前候选分布和端到端同步成本决定。 |
| G4 · 输出和集成 | 在 GPU 叠框，向 NVENC 传设备帧。CPU 封装压缩码流并发布检测结果。分别运行无预览、有预览、录像三档，检查帧序、窗口更新、录像完整性和显存峰值。用 trace 检查无预览计算主链的设备驻留，并按 P1–P5 汇报。 |

G0 后可并行验证 G2 的单应性与 G3 的精确分配，提前发现等价性问题。G1–G3 先用原精度和阈值对拍。TensorRT、FP16 和更大的批次作为后续独立实验，分别量化延迟、吞吐和结果变化。各阶段应区分过渡模式的 CPU 路径与严格模式的验收结果。

FFmpeg 的 CUDA 硬件帧、NVDEC 和 NVENC 有官方支持。当前 WSL 容器的库映射、编解码格式和驱动组合由 G0 的实际运行确认。当前 PyNvVideoCodec 官方 Linux 轮子要求 Python 3.10 或以上，因此保留 ROS Python 3.8 时首选 libav/C++ 适配。[NVIDIA FFmpeg 指南](https://docs.nvidia.com/video-technologies/video-codec-sdk/13.0/ffmpeg-with-nvidia-gpu/index.html) · [PyNvVideoCodec 环境要求](https://docs.nvidia.com/video-technologies/pynvvideocodec/read-me/system-requirements-common.html)

ONNX Runtime I/O Binding 支持设备输入和输出，CUDA provider 可与调用方计算流协作。GPU 路径需验证张量布局、缓冲生命周期、流同步和实际算子设备。CUDA LK 和 GPU 精确分配已有参考实现，现有算法的边界条件和结果一致性仍由对拍验证。[ORT I/O Binding](https://onnxruntime.ai/docs/performance/tune-performance/iobinding.html) · [ORT CUDA Provider](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html) · [OpenCV CUDA LK](https://docs.opencv.org/doc/doxygen/html/d7/d05/classcv_1_1cuda_1_1SparsePyrLKOpticalFlow.html) · [NVIDIA GPU 线性分配](https://docs.nvidia.com/cugraph/latest/graph_support/algorithms/cpp_algorithms/linear_cpp/) · [Nsight Systems CUDA Trace](https://docs.nvidia.com/nsight-systems/UserGuide/#cuda-trace)

## 7. 偏差表

当前尚无优化实现引入的偏差。后续按版本、偏差、原因和验收影响登记。

## 8. 修正记录

| 日期 | 版本 | 改动与原因 | 验收与证据 |
|---|---|---|---|
| 2026-11-11 | 0.1.1 | 增加全 GPU 驻留路线和多场景新基线，覆盖编解码、MFE、TF、LAC/LAD、可视化和 ROS 边界 | 源码与官方接口已核对。Docker 引擎未连通，G0 和 P5 待实测。 |
| 2026-11-11 | 0.1.0 | 从本地 `docker` 创建优化分支并建立独立规格，约定每轮代码与规格同步提交 | 记录最近一次 Docker 回放证据与待复测项。P1–P4 等待首轮性能基线。 |
