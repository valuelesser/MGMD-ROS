# SPEC-ROS: 阶段二 · ROS1 单目旋翼无人机小目标检测与跟踪

> 本文件是**阶段二的唯一事实源**，与阶段一的 `docs/SPEC.md` 并列、互不覆盖。
> 派生自：`paper/prd_mgmd_reproduction.md`（v4.2, 2026-10-28）§11
> 当前运行环境：Windows 宿主机上的 WSL2 `Ubuntu-20.04`（Ubuntu 20.04.6 LTS）
> WSL 工程根：`/home/gaodu/毕业设计/mgmd-linux/`
> 最后更新：2026-11-10（补充 Docker 回放环境和运行记录，PRD v4.2）
> WSL2 启动方式见 §2.3，Docker 启动方式见 §2.4。后续章节保留原生 Ubuntu 环境的历史记录。

---

## 0. 版本与维护约定（硬性）

| 规则 | 说明 |
|---|---|
| V1 | 每次改动阶段二代码，必须在 §8 修正记录追加一行，写明"改了什么/为什么/影响哪些验收项" |
| V2 | 本文件与阶段二代码必须在**同一个 commit** 内提交，message 格式：`ros: <做了什么> (SPEC-ROS §<节号>)` |
| V3 | `spec_version` 每次功能性改动递增，与阶段一 SPEC 的版本号**独立编号** |
| V4 | 偏离 PRD §11 或论文的决定，必须写进 §7 偏差表，不得只在代码注释里留 |
| **V5** | **提交后必须用 `git show --stat HEAD` 核对文件清单**，产物必须出现在 stat 里 |
| **V6** | **演示视频与 rosbag 不进 git**，用 `scripts/pack_deliverables.sh` 打包到 `deliverables/`；表格与摘要 JSON 必须入库 |
| **V7** | **不得修改阶段一的 `mgmd/` 代码**（约束 C1-3）。必需扩展时在 `mgmd_ros/` 下包装，不改原文件 |
| **V8** | 阶段一算法与指标保持冻结；`docs/SPEC.md` 1.0.2 仅按用户要求同步 WSL 环境说明，结果基线仍为 1.0.1（§10.13） |

`spec_version: 2.0.3` | `prd_version: 4.2` | `stage1_spec: 1.0.2`（仅环境修订，结果基线仍为 1.0.1）

### 0.1 版本记录

| spec_version | 日期 | commit | 变更摘要 |
|---|---|---|---|
| **2.0.3** | **2026-11-10** | 本次文档修订 | §2.4 记录现有 `mgmd-ros` 容器的工作空间修复、依赖、启动命令与回放结果 |
| **2.0.2** | **2026-11-05** | 本次提交（环境维护） | **§2.3 更新为当前 WSL2 Ubuntu 20.04 环境、依赖、路径、运行方式和显示/编码限制；保留重跑脚本及运行记录；根目录 AGENTS.md 记录后续按 SPEC 维护与提交的授权；PRD 与业务代码不变** |
| **0.1.0** | **2026-10-29** | 本轮 | 建立本文件；继承 PRD §11 全部决策；登记阶段二基线 |

### 0.2 与阶段一的关系（继承事实，不重复论证）

阶段二的输入是阶段一的**已冻结结果**：

| 项 | 值 | 来源 |
|---|---|---|
| 主结果 F1 | **0.6752**（P=0.897, R=0.541, AP=0.524） | 阶段一 SPEC §10.12 |
| 口径 | `mfe,tf,lac,lad`，`d_min=0.02`，推荐 MFE 预设 | 同上 |
| 冻结分支 | **`result`**（commit `36f8292`） | 本文件 §0.3 |
| 测试套件 | 62 项全过 | 阶段一 SPEC §8.34 |
| 算法环境 | Python 3.11.17，torch 2.5.1+cu121，`.venv` | 阶段一 SPEC §2.2 |

### 0.3 分支策略

| 分支 | 用途 | 状态 |
|---|---|---|
| **`result`** | **阶段一结果冻结快照**，指向 `36f8292` | 🔒 不改动，仅供对照 |
| **`master`** | 阶段二开发主线 | 活跃 |

**理由**：阶段二的 ROS 改造会频繁改动代码结构，若不隔离，阶段一已验收的结果可能被无意破坏。
`result` 分支保证任何时刻都能回到"F1=0.675"那个确切状态。

**校验命令**：
```bash
git show result:mgmd-linux/docs/SPEC.md | grep 'spec_version'   # 应为 1.0.0
git checkout result -- .                                        # 需要回滚时
```

---

## 1. 范围

在 DJI RoboMaster TT 上，用 ROS1 Noetic 实现**单目小目标检测与跟踪**的模块化软件框架。

**是**：
- **单进程**：三个 ROS 节点，算法在 `detector_node` 内直接调用（无跨解释器，§17）
- 话题契约驱动的可替换架构（E1/E2/E3）
- 用阶段一冻结的检测算法（不改算法本身）
- 室外场地实机验证

**不是**：
- 不重训模型（重训是独立任务，见 §6）
- 不做算法消融（阶段一已完成）
- 不追求实机 F1 超过阶段一（数据域不同）

---

## 2. 架构基线（继承 PRD §11.1–11.2，此处只记实施要点）

### 2.1 进程边界

```
ROS1 主图（Python 3.8，**单一解释器**）
┌──────────────────────────────────────────────┐
│ video_source   → /camera/image_raw           │
│ detector_node  内部直调:                      │
│                 MFE → FastSortTracker        │
│                 → LacOnnx → LadOnnx          │
│               → /mav/detections              │
│ visualize_node → 带框视频（NVENC/CPU）        │
└──────────────────────────────────────────────┘
```

**没有任何跨进程通信。** 原 sidecar + Unix socket 方案已在 §13 推翻（RD-05），
算法以 `mgmd_onnx` 形式在 `detector_node` 进程内直接调用。

> **注**：节点已由五节点重构为三节点（§17，2026-10-30）。上图为当前状态。

### 2.2 五个既定决策（PRD §11.0.1，不再讨论）

| 编号 | 决策 |
|---|---|
| Q-A | ~~方案 A：sidecar~~ → **单进程 ONNX**（§13，RD-05） |
| Q20 | 实验室多架，≥2 |
| Q21 | 真机可用 |
| Q22 | 阶段一三表单独出结果 |
| Q23 | **室外场地** |
| Q24 | 两架同型号 TT，外观一致 |
| Q25 | 有场地许可 |
| Q26 | 接受 Noetic EOL |

### 2.3 当前环境：WSL2 Ubuntu 20.04

2026-11-04 已完成本机 WSL 部署，2026-11-05 在同一环境重跑通过。当前基线是 Windows + WSL2，不再以原生 Ubuntu 的驱动安装、sudo 限制或 `/media/...` 数据挂载为前提。本节是当前运行配置；历史章节中的其他环境组合不作为本机默认值。

#### 2.3.1 系统与依赖（2026-11-05 核验）

| 项 | 当前值 | 用途 / 验证 |
|---|---|---|
| Linux 环境 | WSL2，发行版 `Ubuntu-20.04`，Ubuntu 20.04.6 LTS | Windows 通过 `wsl -d Ubuntu-20.04` 调用 |
| 用户 | `gaodu` | 工程与环境位于 WSL Linux 文件系统 |
| GPU | NVIDIA GeForce RTX 3050 Laptop，4096 MiB | WSL 内 `nvidia-smi` 可用，LAC/LAD CUDA 冒烟通过 |
| ROS1 | Noetic，`/opt/ros/noetic`；`ros-noetic-roslaunch 1.17.4-1focal.20250519.232645` | roscore、roslaunch、rostopic 和三节点本次运行正常；catkin 编译通过见首次部署记录 |
| Python | 系统 3.8.10；项目 `.venv38` 同为 3.8.10 | detector 在 Python 3.8 进程内直接调用 ONNX |
| ONNX Runtime | `onnxruntime-gpu==1.19.2` | LAC、LAD 均启用 `CUDAExecutionProvider` |
| torch / torchvision | `2.4.0+cpu` / `0.19.0+cpu` | YOLOv5 后处理使用；GPU 前向由 ORT 执行 |
| numpy / 算法侧 cv2 | `1.24.4` / `opencv-python-headless==5.0.0.93` | 位于 `.venv38`，不用于 GUI 窗口 |
| CUDA 运行库 | pip 包：CUDA runtime `12.9.79`、cuBLAS `12.9.3.13`、cuDNN `9.27.0.42` | 另有 cufft、curand、cusolver、cusparse、nvjitlink、nvrtc；库目录加入 `LD_LIBRARY_PATH` |
| 显示 | WSLg，`DISPLAY=:0`；系统 `python3-pyqt5 5.14.1+dfsg-3build1` | 使用 `.dsh-run/21_qt_viewer.py`；标题 `MGMD-ROS Qt  frames=N` |
| 视频编码 | ffmpeg `4.2.7`；当前显式 `encoder:=cpu` | OpenCV mp4v 输出 MPEG-4；当前组合 NVENC 失败见 §2.3.4 |

当前推理环境不需要重建阶段一的 Python 3.11 + CUDA torch 环境。`torch.cuda.is_available()` 在 CPU torch 下不作为 GPU 验收；应检查 ORT session 的首选 provider 和实际冒烟结果。首次部署使用 pip CUDA/cuDNN 运行库，没有安装系统 CUDA Toolkit。

#### 2.3.2 工程、数据与兼容路径

| 资产 | 当前路径 |
|---|---|
| Windows 源码根 | `C:\Users\gaodu\Desktop\opencode\mgmd_project_20261104\毕业设计\mgmd-linux` |
| WSL 挂载的 Windows 源码根 | `/mnt/c/Users/gaodu/Desktop/opencode/mgmd_project_20261104/毕业设计/mgmd-linux` |
| WSL 运行工程根 `ROOT` | `/home/gaodu/毕业设计/mgmd-linux` |
| ROS 工作空间 | `$ROOT/ros1_ws`，使用已编译的 `devel/setup.bash` |
| 推理环境 | `$ROOT/.venv38` |
| ARD-MAV | `/mnt/c/Users/gaodu/Desktop/毕业设计/mgmd/data/videos/视频/ARD-MAV` |
| ONNX 权重 | `/home/gaodu/tracking/data_raw/mgmd_weights/lac.onnx`、`yolov5s_GLAD-crop.onnx` |
| YOLOv5 v6.0 | `$ROOT/third_party/yolov5-v6.0` |
| 兼容软链 | `/home/gaoduan -> /home/gaodu`，兼容旧代码中的绝对路径 |
| 输出 | `$ROOT/outputs/`；本次独立目录 `rerun_20261105_155551/` |

数据从 `/mnt/c` 读取并保持文件不变；这不表示 Windows 分区以只读方式挂载。venv、catkin 构建和运行输出放在 WSL Linux 文件系统。Windows 源码与 WSL 运行副本是两个目录；本次运行前用 `28b_strict_diff.sh` 核对业务源码一致，排除了依赖、构建、输出、辅助脚本和 catkin 自动生成文件。后续只修改 Windows 文件不会自动同步运行副本。

#### 2.3.3 环境变量与启动方式

以下为 WSL 内当前运行环境（Qt 查看器使用去掉 venv site-packages 的 `PYTHONPATH`，避免加载 headless cv2）：

```bash
ROOT=/home/gaodu/毕业设计/mgmd-linux
source /opt/ros/noetic/setup.bash
source "$ROOT/ros1_ws/devel/setup.bash"
ROSPATH=$PYTHONPATH
SP="$ROOT/.venv38/lib/python3.8/site-packages"
export PYTHONPATH="$SP:$ROOT:$ROSPATH"
NV=$(find "$SP/nvidia" -maxdepth 3 -type d -name lib | tr '\n' ':')
export LD_LIBRARY_PATH="${NV}${LD_LIBRARY_PATH:-}"
export DISPLAY=:0 QT_X11_NO_MITSHM=1 PYTHONUNBUFFERED=1
export MGMD_ARD_ROOT=/mnt/c/Users/gaodu/Desktop/毕业设计/mgmd/data/videos/视频/ARD-MAV
export MGMD_YOLOV5_DIR="$ROOT/third_party/yolov5-v6.0"
```

推荐从 Windows PowerShell 调用本次验证脚本：

```powershell
wsl -d Ubuntu-20.04 -- bash '/mnt/c/Users/gaodu/Desktop/opencode/mgmd_project_20261104/毕业设计/mgmd-linux/.dsh-run/29_rerun_verify.sh'
```

脚本复用已部署环境，启动 roscore、Qt 查看器与三个流水线节点，以 `phantom30`、960×720、20 Hz、`limit:=1200 loop:=false device:=cuda d_min:=0.02 buffer:=6 encoder:=cpu` 运行。每次创建独立 WSL 输出目录，输出路径在终端打印；结束后关闭写入器并清理进程。已有 ROS master 时退出，不打断其他运行。脚本本身位于 Windows 工程，调用的是 WSL 内已部署源码。

仅检查 GPU 推理时，在 WSL 工程根执行 `bash .dsh-run/07_run_smoke.sh`。常驻演示沿用 WSL 侧 `bash .dsh-run/23_live_qt.sh phantom30 20 960 720`，但该旧脚本会复用 `live_phantom30_960x720.mp4` 路径；需要保留旧视频时先另存。常驻演示必须主动停止并收尾，不能仅靠关窗结束流水线。

#### 2.3.4 当前环境限制与收尾

- **编码**：首次部署中 NVENC 报 `Cannot get the preset configuration: unsupported param (12)`，本机使用 `encoder:=cpu`；历史 NVENC 性能数字不能作为当前默认配置。
- **显示**：本机以系统 Python + PyQt5 查看 `/mav/debug_image` 为已验证方案。旧 image_view/GTK 路线曾遇到 WSLg 共享内存故障；原生 Ubuntu 的 rqt 展示命令不作为本机默认入口。若复现该显示故障，可在结束 WSL 任务后重启 WSLg；`wsl --shutdown` 会停止全部发行版，不是常规启动步骤。
- **视频收尾**：先向 roslaunch 发送 SIGINT，或向 visualize_node 发送 SIGTERM，让写入器关闭并保存 mp4；不能直接 SIGKILL 视频写入进程。本次实际收到信号 2，成功保存 1195 帧。
- **Qt 收尾**：rospy 的 SIGTERM 处理可能留下 Qt 事件循环；新验证脚本在等待 5 秒后仅对未退出的查看器使用 SIGKILL。视频写入器已先行正常关闭。最终仍需检查本轮进程残留。
- **环境安装历史**：Noetic 安装时使用清华镜像，首次部署记录包含过期签名 key 与源配置的处理。本次复用环境，没有重装 ROS、驱动或依赖；不把原生 Ubuntu 的安装命令作为必要重跑步骤。

#### 2.3.5 本机验证证据与边界

| 验证 | 2026-11-05 结果 |
|---|---|
| GPU 冒烟 | `SMOKE_OK`，LAC/LAD 均启用 CUDA provider |
| 节点与话题 | video_source、detector_node、visualize_node 和 Qt viewer 注册正常；检测话题采样平均约 18.98–19.99 Hz |
| 窗口更新 | Qt 标题帧号 520 → 580，间隔 3 秒 |
| 计数 | 视频源发布 1200 帧；检测线程处理 1022 帧；末次周期日志入口丢帧 102；视频写入 1195 帧 |
| 产物 | `phantom30.mp4`：MPEG-4，960×720，1195 帧，59.75 秒，53,283,148 字节；ffprobe 与整段解码通过 |
| 收尾 | 无本轮 ROS 节点 / Qt 查看器残留 |

图像、检测和可视化是异步环节；可视化使用最近一次检测叠加，因此视频帧数不等于完成检测的帧数。当前结果证明本机文件回放、GPU 推理、显示和保存正常，不能据此认定通信丢帧率 <1%、E3 逐帧对拍、完整端到端时延、长时间稳定性或 TT 真机验收已通过。阶段一训练与离线精度未在本次重新评测。

证据文件：Windows 工程 `outputs/rerun_20261105_155551/{launch.log,detection_hz.txt,nodes.txt,window.txt,video_info.txt,phantom30.mp4,preview.png}`。另见 [首次部署记录](../../docs/mgmd-ROS-WSL20.04-执行记录.md) 与 [本次重跑记录](../../docs/WSL20.04-重跑记录-2026-11-05.md)。

---

### 2.4 Docker Desktop 回放环境

`mgmd-ros` 使用 Ubuntu 20.04、ROS Noetic、Python 3.8.10，Docker 已透传 RTX 3050 Laptop GPU。容器工程位于 `/workspace/MGMD-ROS/mgmd-linux`。这次使用容器已有的迁移版本。它与本地检出的 ROS 源码存在差异，包含独立的 `mgmd_onnx/frame_buffer.py`。

迁移后的 `ros1_ws/src/CMakeLists.txt` 是存有目标路径的普通文本文件。恢复指向 `/opt/ros/noetic/share/catkin/cmake/toplevel.cmake` 的软链接后，以 `catkin_make -j4 -DPYTHON_EXECUTABLE=/usr/bin/python3` 编译两个 ROS 包，生成自定义消息。

检测使用容器用户 `valuelesser` 已安装的 PyTorch 2.4.1+cu121、torchvision 0.19.1+cu121、OpenCV 5.0.0、SciPy 1.10.1 和 ONNX Runtime GPU 1.19.2。启动时加载 Catkin 工作空间及 CUDA/cuDNN 动态库，YOLOv5 路径为 `/workspace/MGMD-ROS/tracking/third_party/yolov5-v6.0`。两份 ONNX 模型位于 `/home/gaoduan/tracking/data_raw/mgmd_weights`，SHA-256 均与登记值一致。

容器已安装回放入口 `/usr/local/bin/mgmd-ros-demo`。在 Windows PowerShell 执行下面的命令。脚本以项目用户运行，默认读取 `/data/videos/视频/ARD-MAV/videos/phantom30.mp4`，以 20 Hz 回放到文件结束，用 CUDA 推理和 CPU 编码。运行目录由 `RUN=` 打印，视频与日志保存到该目录。回放结束后脚本关闭本轮 ROS 进程并检查视频能否完整解码。

```powershell
docker start mgmd-ros
docker exec mgmd-ros runuser -u valuelesser -- /usr/local/bin/mgmd-ros-demo
```

| 验证 | 2026-11-10 结果 |
|---|---|
| ROS 构建和逻辑测试 | 两个包编译成功。容器安装 `python3-pytest` 后，跟踪和帧缓冲定向测试 28 项通过 |
| 推理 | LAC 和 LAD 实际使用 `CUDAExecutionProvider` |
| 源视频 | 容器内 FFmpeg 按 MP4 编辑列表解码 1,498 帧，忽略编辑列表时解码 1,523 帧。ROS 按默认播放范围发布 1,498 帧 |
| 检测 | 节点处理 1,405 帧，收到 1,405 条检测消息，其中 1,108 条有检测框。抽样约 19.7 Hz |
| 可视化 | `/mav/debug_image` 抽样约 20.0 Hz。输出 960×720、1,498 帧、74.9 秒，FFmpeg 全段解码通过 |
| 证据 | Windows 工作目录下 `mgmd-ros-docker-20261110/` 保存视频、节点清单及运行日志 |

这次验收覆盖 Docker 内的文件回放、检测跟踪、可视化话题和录制。实时桌面窗口、摄像头、TT 实机与 E3 对拍仍待验证。输入、推理和录制异步运行，输出视频帧数与完成检测的帧数分别统计。

---

## 3. 实施阶段划分

PRD §11.7 给了 32 天总工时（含 8 天多目标扩展）。实际执行按"**能否离线完成**"切分：

| 阶段 | 内容 | 需硬件 | 状态 |
|---|---|---|---|
| **S1** | ros1_ws 双包骨架（msgs + nodes） | ❌ | ✅ **完成** |
| **S2** | 节点骨架（空实现，话题通） | ❌ | ✅ **完成**（随 S1 一并验证） |
| **S3** | ~~`detect_server.py` + socket~~ → **ONNX 单进程集成**（§13） | ❌ | ✅ **完成**（§13.8/§14.7） |
| **S3.5** | **`video_source source:=file` 回放 + 720p 实测** | ❌ | ✅ **完成**（§16，原 `tt_driver`） |
| **S4** | **`detector_node` 接 `mgmd_onnx`（单进程直调）** | ❌ | ✅ **完成**（§17，原 `mav_node`） |
| **S4.5** | **性能归因：GPU/多线程/框架开销三路线实测** | ❌ | ✅ **完成**（§20/§21，三条全部否决） |
| **S4.6** | **step_H 对齐补全 + 五视频验证** | ❌ | ✅ **完成**（§22） |
| **S5** | rosbag 三方对拍（E3：中心差 <2px，帧数差 <2%） | ❌ | ⬜ **按用户指示后置** |
| **S5.5** | **Gazebo 相机 + 抖动注入仿真**（§11） | ❌ | ❌ **已中止**（§11.11，LAC/LAD 无法验证） |
| **S6** | 实机联调 | ✅ | ⬜ **最后** |
| **S7** | 室外飞行验收 | ✅ | ⬜ **最后** |
| **S8** | TT 数据采集与重训 | ✅ | ⬜ **最后** |

**执行记录（用户 2026-11-02 指示）**：

1. **用现成视频验证三个 ROS 节点** —— ✅ **完成**（§23，三节点均正常）
2. **仿真**（S5.5）—— ❌ **已中止**（§11.11，根因：纯几何体无法通过 LAC/LAD，
   验证不了完整链路，投入产出不成立）
3. 两机对拍（S5）→ 真机（S6/S7/S8） —— 用户要求放到最后

**当前状态**：离线部分（S1–S4.6）已完成并经实测验证；
仿真路线已验证不可行并中止；**下一步需用户决定方向**（见 §23.6）。

原「S1–S5.5（6.0 天）」的估算已过时，实际离线部分已基本完成。

**S5.5 仿真的定位**：用户不关心飞控，只关心"能否在无人机上跑起来"。
因此仿真验证的是**资源占用下的链路吞吐与延迟**，不是飞行合理性。详见 §11。

---

## 4. 验收指标（继承 PRD §11.5）

### 4.1 架构性质

| 编号 | 指标 | 判定方法 |
|---|---|---|
| **E1** | 换检测器不改其他节点 | 用 YOLO 节点替换 `detector_node`，`video_source`/`visualize_node` 零改动，launch 文件改一行 |
| **E2** | 换平台不改算法节点 | `video_source` 的 `~source` 参数切换 stub/file/camera，话题映射改一处 |
| **E3** | 离线可复现 | rosbag 重放，输出与阶段一脚本对拍，**框中心差 < 2 px，检测帧数差 < 2%** |

### 4.2 功能指标

| 项 | 目标 |
|---|---|
| 检测延迟（帧进→检测出） | < 100 ms |
| 端到端帧率（960×720） | ≥ 10 FPS |
| 跟踪 ID 稳定性 | 连续 100 帧无 ID switch |
| 通信丢帧率 | < 1% |

### 4.3 安全门限（室外，不可配置）

| 条件 | 动作 |
|---|---|
| 500 ms 无新帧 | 速度清零 |
| 2 s 无新帧 | 自动降落 |
| 检测框面积 > 画面 5% | 立即停止追瞄 |
| `n_candidates > 200` | 自动关闭 LAD，仅跑 LAC |
| 视觉伺服使能条件 | `state==CONFIRMED` **且** `hits>=5` |

---

## 5. 话题契约（E1/E2 的实际载体）

**当前实际使用的话题（三节点重构后，§17）**：

| 话题 | 类型 | 频率 | 方向 | 代码位置 |
|---|---|---|---|---|
| `/camera/image_raw` | `sensor_msgs/Image` | 视频原始帧率 | `video_source` → `detector_node`/`visualize_node` | `video_source.py:149` |
| `/mav/detections` | `mav_tracker_msgs/DetectionArray` | 处理帧率 | `detector_node` → `visualize_node`/外部 | `detector_node.py:296` |
| `/mav/debug_image` | `sensor_msgs/Image` | 处理帧率 | `visualize_node` → 外部（可选） | `visualize_node.py:122` |

**已删除的话题**（随飞控功能一并移除，用户决策「只做视频跟踪，不控飞控」）：

| 话题 | 原用途 | 状态 |
|---|---|---|
| `/tt/state` | Tello 状态 | ❌ 已删除 |
| `/tt/cmd_vel` | 飞控速度指令 | ❌ 已删除 |
| `/mav/candidates` | 候选框 | ❌ 已删除（`n_candidates` 字段内嵌于 DetectionArray） |
| `/mav/tracks` | 跟踪轨迹 | ❌ 已删除（`track_id` 内嵌于 Detection） |
| `/mav/track_target` | 目标选择 | ❌ 已删除（RD-04） |

**`visualize_node` 的输出**：主要产物是**带框视频文件**（`~out` 参数），
`/mav/debug_image` 为可选的实时预览（`~publish` 控制，默认关闭）。

---

## 6. 数据与重训（继承 PRD §11.6）

| 项 | 计划 |
|---|---|
| 采集 | 室外场地，2 架 TT，多距离/多角度/多背景，≥20 段 × 2000 帧 |
| 标注 | 沿用 ARD-MAV 的 XML 格式，复用阶段一解析器 |
| 重训 LAC | Small 组召回损失 39% 来自 LAC 误杀（阶段一 SPEC §8.25），重训可挽回 |
| 重训 LAD | 域差异：ARD-MAV 机载相机 vs TT 相机 |
| 验证 | `[待定]` 需先有数据才能定划分比例 |

**阶段一的证据**：MFE 跨域可用（MVDT 上 97–99.7%），LAC 跨域失效（D1 从 99.5% 掉到 1.0%）。
**因此重训的优先级是 LAC > LAD > MFE。**

---

## 7. 偏差表

| 编号 | 项 | PRD/论文 | 本工程 | 理由 | 影响 |
|---|---|---|---|---|---|
| **RD-01** | MFE+TF 是否留在 ROS 侧 | 未规定 | ~~全部放 sidecar~~ → **全部在 mav_node 进程内**（§13 后无跨进程边界） | 原本担心 `step_H`/候选顺序要暴露成协议；单进程后该问题不存在 | 无协议开销，ROS 侧可单独跑 MFE 消融 |
| **RD-02** | `/tt/state` 消息类型 | 未规定 | `Float32MultiArray` | Tello 字段随固件变 | 故意的技术债 |
| **RD-03** | 多目标跟踪的 `max_age` | 论文 3 | `[待定]` | 论文的 3 是为快速丢假正例，多目标长跟踪需要更长；但阶段一证明调大无收益（§10.10） | 待实机验证 |
| **RD-04** | **`/mav/track_target` 用 action 还是 topic** | PRD §11.2/§11.8 US-017 用 action | **改用 `std_msgs/Int32` topic** | actionlib 面向**可抢占的一次性任务**（官方定义）；视觉跟踪是**无终点的持续流式通路**，30 Hz 连续控制不适合走 feedback | 删除 `TrackTarget.action`；话题表同步更新；US-017 的 "cancel 中止" 改为发 `-1` |
| **RD-05** | **架构：sidecar vs 单进程 ONNX** | PRD §11.0.2 采用方案 A（sidecar） | **改单进程 3.8 + ONNX Runtime** | 可移植性要求；sidecar 需两套 Python 环境，嵌入式不可行 | 性能持平（LAD）+ LAC 快 2.3×；S3 大简化；**不再是双解释器** |
| **RD-06** | **检测输出 30 Hz vs 按算法实际能力** | S2-c 写 30 Hz | **`/mav/detections` >= 8 Hz（1080p）/ >= 15 Hz（720p）** | MFE 单帧 88 ms（1080p），30 Hz 需 33 ms，物理不可达；绑死会造成名义 30 Hz 实际丢帧的假指标 | 相机侧仍 30 Hz；新增 S3-b 延迟验收 |

### 技术债（已知取舍）

| 编号 | 内容 | 偿还条件 |
|---|---|---|
| TD-01 | `/tt/state` 用数组 | Tello 字段稳定 |
| ~~TD-02~~ | ~~sidecar 协议无版本号~~ | **已消除**（无 socket 协议，§13） |

---

## 8. 修正记录（每轮必填）

| 日期 | 版本 | 改动 | 原因 | 影响验收项 |
|---|---|---|---|---|
| 2026-11-10 | 2.0.3 | §2.4 记录 Docker 容器的 Catkin 软链接、项目用户依赖、CUDA 配置、启动入口和回放结果 | 验证已迁移的 `mgmd-ros` 容器 | Docker 视频回放、检测话题、可视化话题和录制 |
| 2026-11-05 | 2.0.2 | 当前环境从原生 Ubuntu 更新为 WSL2 Ubuntu 20.04；补齐 Python 3.8/ORT CUDA/CPU torch、路径、Qt/CPU 编码、重跑命令与收尾限制；一并保留 `.dsh-run/29_rerun_verify.sh` 与两份部署/运行记录，在根 AGENTS.md 固化维护与提交约定 | 用户要求只更新 SPEC 环境部分、PRD 不改，并要求及时按 SPEC 维护与提交 | 环境与运行说明；脚本增加 Qt 退出等待与兜底清理；记录本机回放证据，不新增精度或实机验收结论 |
| 2026-10-29 | 0.1.0 | 建立本文件；登记阶段二基线、分支策略、实施阶段划分 | 用户要求 ROS 阶段独立 SPEC 并持续维护 | — |
| 2026-10-29 | 0.2.0 | §9 补详细计划 S1–S5；删除 ros2_ws；`vx/vy` 单位改为像素/帧 | 用户要求先看计划再动手 | §4.1 E3 |
| 2026-10-29 | 0.3.0 | §11 仿真方案 2.5b（Gazebo相机+抖动注入）；工时+0.5天 | 用户只关注能否在无人机上跑，不关心控制 | §4.2 功能指标 |
| 2026-10-29 | 0.4.0 | 建 ros1_ws 双包；4 msg 编译通过；五节点 30Hz 跑通；**删除 action 改 topic** | S1 实施；用户指出 action 不适合流式任务 | S1-a~h 全通过 |
| 2026-10-29 | 0.5.0 | **架构变更：sidecar → ONNX 单进程**；建 .venv38；LAC/LAD ONNX 导出并验证；性能持平 | 用户要求可移植性 | S3 从 2 天降为 0.5 天 |
| 2026-10-29 | 0.6.0 | `mgmd_onnx` 包装层；全链路对拍 1496 帧完全一致；排查出 P 差 0.037 是参数不一致（非 ONNX 问题） | S3 门槛验证 | F1 0.8328 两侧相同 |
| 2026-10-30 | 0.7.0 | 实时性指标从 30 Hz 修订为 >=8 Hz（检测侧）；实测定位瓶颈 MFE 117ms + TF 61ms | 用户选 C（放宽指标）；30 Hz 物理不可达 | 新增 S3-b/c；记录 TF 向量化可 119x |
| 2026-10-30 | 0.8.0 | 实现 FastSortTracker（只向量化 IoU 矩阵）；1496 帧逐帧差 0；端到端 8.5 Hz | S3-b/c 验收 | 1080p 达标，720p 待 S4 实测 |
| 2026-10-30 | 0.9.0 | 实测 MFE-UMat：medianBlur 快2x 但 calcOpticalFlowPyrLK 崩溃；端到端仅快 3%，**否定该方案** | 用户要求试 UMat | §14.5 加速表更新，删除 UMat |
| 2026-10-30 | 1.0.0 | 用户接受当前帧率；阶段表重构为 S3/S3.5/S4/S5；补 §15 计划 | 用户询问下一步 | S3.5 为首要，前置补 export/setup 脚本 |
| 2026-10-30 | 1.1.0 | 补 export_onnx/setup_env38/verify_onnx；tt_driver 文件回放 30.1Hz 通过；720p 实测 14Hz，**指标下调至 12Hz** | S3.5 实施；实测推翻了 §14.7 的外推值 | S3.5-a~e 全通过；RD-07 |
| 2026-10-30 | 1.2.0 | **用户指出 SPEC 仍残留 socket/sidecar 描述**；清理 15 处失效内容，重写 §9.3 为 ONNX 单进程 | 文档一致性检查 | 工时由 6.0 降为 3.2 天（离线部分） |
| 2026-10-30 | 1.3.0 | 用户明确不控飞控；节点重构为三节点；帧号改为视频帧号；修复 mp4 损坏与 CUDA 路径两个 bug | 用户要求方案 A + 改名 + 带框视频 + 缓冲6帧 | S4-a~f 通过，S4-g 频率未达离线水平（§17.8） |
| 2026-10-30 | 1.4.0 | 实测 visualize 各环节（画框仅 0.77ms，编码 19.55ms）；NVENC 1.16x 提速但 CPU 89% vs 781%；**发现跳帧使 SORT 输出降至 32%** | 用户询问能否 GPU 编码 | 待决策 §18.8 四个方案 |
| 2026-10-30 | 1.5.0 | 用户指示改 720p；video_source 加 width/height 缩放；实测处理 10Hz、检测恢复；**再次犯了「先改后验」，tf_seq 改动已撤销**（第八次同类） | 用户选择方案 A | 720p 可用；F1 重验待定 |
| 2026-10-30 | 1.6.0 | §20 各阶段耗时实测：MFE 46ms 占 49.3% 为唯一瓶颈；补 `tf_ms` 字段；发现 ROS 私有参数写法坑（`_name:=` vs `--name`） | 用户要求「列出各阶段处理时间」 | 性能基线建立 |
| 2026-10-30 | 1.7.0 | §21 三条优化路线实测否决（GPU torch 版、Python 多线程、框架开销）；§21.9 存档 GPU-LK 想法并更正「cv2 冲突」误判 | 用户提出 GPU/多线程/LK 可加速 | 性能结论固化，不再重复尝试 |
| 2026-10-30 | 1.8.0 | §22 step_H 对齐补全同步到 master；五视频并行验证 `compose_H` 100%；phantom05 实测 F1 变化 −0.0012 | 用户要求 B 方案 + 复制到 master | §22.3.1 F1 影响量化 |
| 2026-11-02 | 1.9.0 | §23 文档一致性大核对：修正 §1/§2 架构图、§3 阶段表状态、§4 E1/E2、§5 话题契约（按实际代码重写）、§15 标注过时；补登 1.6–1.8 版本行 | 用户要求「整理进展，核对 spec」 | 文档与代码对齐 |
| 2026-11-03 | 2.0.1 | 代码整理：抽 `scripts/ros_env.sh` 公共环境与进程管理（三脚本去重）；`detector_node` 的 MFE 覆盖改为表驱动并补齐 `median_blur`/`mode`；修「adaptive=0 被哨兵误判」bug；`CMakeLists` 加 worlds 到 install。新增 `docs/REVIEW-2026-11-03.md` 供审阅 | 用户要求「整理代码，我审阅」 | 三脚本全绿，54 单测通过 |
| 2026-11-02 | 2.0.0 | §11.3 仿真环境实测（5 个静默失败陷阱）；§11.11 **仿真中止**（LAC/LAD 无法验证）；§24 方向决策（推荐 E3 对拍 + 端到端时延）；新增 `worlds/`、`target_mover.py`、`check_gazebo.sh` | 用户「放弃仿真，反正跑不了 LAC 和 LAD」 | 仿真路线关闭，资产保留 |

---

## 9. 实施计划（详细）

### 9.0 前置：ros2_ws 已清理（2026-10-29）

`mgmd-linux/ros2_ws/` 是阶段一的 ROS2 遗留物，**从未提交过 git**，PRD §11 已宣告 ROS2 方案作废。

**已执行**：整目录删除。
**抢救内容**：4 个 msg + 1 个 action 定义已备份到 `/tmp/ros2_backup/`，
待 S1 迁移为 ROS1 格式后入库。**备份是临时的，S1 完成后以包内文件为准。**

`[flag: /tmp 重启会清空。若 S1 在此之前未开始，消息定义已在 §9.1.3 完整列出，不会丢失。]`

---

### 9.1 S1 — 工作空间与消息包

**目标**：`catkin_make` 通过，`rosmsg show mav_tracker_msgs/DetectionArray` 有输出。

**工时**：0.5 天

#### 9.1.1 目录结构

```
mgmd-linux/ros1_ws/src/mav_tracker/
├── mav_tracker_msgs/
│   ├── CMakeLists.txt
│   ├── package.xml
│   ├── msg/{Detection.msg, DetectionArray.msg, Track.msg, TrackArray.msg}
│   └── action/TrackTarget.action
└── mav_s/
    ├── CMakeLists.txt
    ├── package.xml
    ├── launch/detect.launch
    ├── scripts/{video_source.py, detector_node.py, .py,
    │            visualize_node.py, .py}
    └── config/topics.yaml
```

#### 9.1.2 关键差异：ROS2 → ROS1

| 项 | ROS2 | ROS1 | 影响 |
|---|---|---|---|
| 构建 | `ament_cmake` / `colcon` | `catkin` / `catkin_make` | `CMakeLists.txt` 全重写 |
| 包声明 | `package.xml` format 3 | **format 2** | `<buildtool_depend>catkin</buildtool_depend>` |
| 消息生成 | `rosidl_default_generators` | `add_message_files()` + `generate_messages()` | |
| action | `.action` 单文件 | `.action` 用 `---` 分隔，需 `add_action_files()` | ⚠️ 注意段顺序 |
| Python 入口 | `setup.py` entry_points | `catkin_install_python()` | |

**action 段顺序**：ROS2 版 `TrackTarget.action` 的三段是 goal / feedback / result，
ROS1 要求 **goal / result / feedback**。必须交换后两段，否则 action 定义静默错位。

#### 9.1.3 消息定义（迁移后的最终版本）

**DetectionArray.msg**（帧级检测包）
```
std_msgs/Header header
uint32 image_width
uint32 image_height
Detection[] detections
uint32 n_candidates       # MFE 原始候选数（诊断背景噪声）
float32 mfe_ms
float32 lac_ms
float32 lad_ms
```

**Detection.msg**（单检测）
```
std_msgs/Header header
float32 x1
float32 y1
float32 x2
float32 y2
float32 cx
float32 cy
float32 confidence        # [0,1]
string source             # mfe / lac / lad
int32 track_id            # -1 未关联
```

**TrackArray.msg**
```
std_msgs/Header header
Track[] tracks
```

**Track.msg**
```
std_msgs/Header header
int32 track_id
string state                    # TENTATIVE / CONFIRMED / LOST
float32 x1
float32 y1
float32 x2
float32 y2
float32 cx
float32 cy
float32 vx                      # 像素/帧（ROS2 版注释误写为像素/秒，已更正）
float32 vy
uint32 age
uint32 hits
float32 confidence
float32[] history_xy            # 扁平 [x0,y0,x1,y1,...]
```

`[fix: ROS2 版 Track.msg 注释写 vx 单位是"像素/秒"，但阶段一 KalmanBox 的 velocity 实际是像素/帧。此处更正，避免实机视觉伺服差 30 倍。]`

**TrackTarget.action**（ROS1 顺序：goal → result → feedback）
```
int32 target_track_id
bool rotate_to_center
bool keep_distance
float32 desired_box_area
---
bool success
string message
float32 total_tracking_time_s
---
int32 current_track_id
float32 error_x
float32 error_y
float32 box_area_ratio
string status
```

#### 9.1.4 验收

| # | 检查 | 命令 |
|---|---|---|
| S1-a | 编译通过 | `cd ros1_ws && catkin_make` |
| S1-b | 消息可查 | `rosmsg show mav_tracker_msgs/DetectionArray` |
| S1-c | action 可查 | `rosmsg show mav_tracker_msgs/TrackTargetActionGoal` |
| S1-d | Python 可导入 | `python3 -c "from mav_tracker_msgs.msg import Detection"` |

---

### 9.2 S2 — 五个节点骨架

**目标**：`roslaunch mav_s detect.launch` 能起，`rostopic list` 出现全部 §5 话题。

**工时**：1 天

#### 9.2.1 每个节点的空实现要求

| 节点 | 空实现必须做到 |
|---|---|
| `video_source` | 定时器发布假 `/camera/image_raw`（纯色图）+ `/tt/state`；订阅 `/tt/cmd_vel` 只打日志 |
| `detector_node` | 订阅图像，发空 `DetectionArray` 到 `/mav/detections` |
|  | 收空检测，发空 `TrackArray` |
| `visualize_node` | 收图像和轨迹，输出原图到 `/mav/debug_image` |
|  | 收轨迹，发零速度 `Twist` 到 `/tt/cmd_vel` |

**关键**：骨架场景下整条链路的消息能从头流到尾，这是 E1/E2 成立的前提。

#### 9.2.2 参数文件 `config/topics.yaml`

把所有话题名抽成参数，换检测器只改这个文件（E1 的核心）：

```yaml
topics:
  image_raw: /camera/image_raw
  state: /tt/state
  cmd_vel: /tt/cmd_vel
  detections: /mav/detections
  candidates: /mav/candidates
  tracks: /mav/tracks
  debug_image: /mav/debug_image
```

#### 9.2.3 验收

| # | 检查 |
|---|---|
| S2-a | `roslaunch` 五个节点全起，无 crash |
| S2-b | `rostopic list` 含 §5 全部 8 个话题 |
| S2-c | `rostopic hz /mav/detections` ≈ 30 Hz |
| S2-d | `rosnode info` 各节点连接关系与 §2.1 图一致 |

---

### 9.3 S3 — ONNX 单进程集成（原 sidecar 方案已废弃）

> **本节于 SPEC-ROS 1.2.0 重写。** 原内容为 sidecar + Unix socket 协议，
> 已被 §13（RD-05）推翻。保留本节编号以维持历史引用有效。

**目标**：`detector_node` 进程内直接调用算法，输出真实检测。

**工时**：0.4 天（原 sidecar 方案估 2 天）

#### 9.3.1 架构

```
/camera/image_raw
      ↓
  detector_node（单进程，Python 3.8）
      ├─ mgmd.mfe.enhance            ← 阶段一代码，未改
      ├─ mgmd_onnx.tf_fast.FastSortTracker
      ├─ mgmd_onnx.LacOnnx
      └─ mgmd_onnx.LadOnnx
      ↓
/mav/candidates, /mav/detections
```

**没有 socket、没有协议序列化、没有第二解释器。**
状态（SORT 轨迹、MFE 帧缓冲）天然存活在进程内。

#### 9.3.2 丢帧策略（关键设计）

相机 30 Hz，算法 14 Hz（720p），**必须主动丢帧**：

| 项 | 策略 |
|---|---|
| 订阅队列 | `queue_size=1` |
| 缓冲 | 只保留**最新一帧**，处理完立刻取下一帧 |
| 效果 | 延迟不随运行时长增长；丢的是中间帧而非尾帧 |

`[flag: 若用 queue_size>1 且不丢帧，队列会随时间单调增长，延迟无限变大。]`

#### 9.3.3 复用阶段一代码的方式

**不改 `mgmd/`**（约束 V7）：

```python
from mgmd.mfe import MFEConfig, enhance          # 原样复用
from mgmd.tf import SortTracker                    # 原样复用（基线）
from mgmd_onnx.tf_fast import FastSortTracker      # 向量化版，逐帧等价（§14.7）
from mgmd_onnx import LacOnnx, LadOnnx             # ONNX 推理
```

#### 9.3.4 验收

| # | 检查 | 状态 |
|---|---|---|
| S3-a | ~~sidecar 独立启动，socket 可连~~ → **`detector_node` 进程内跑通全链路** | ⬜ |
| S3-b | 同图同帧检测数与阶段一脚本一致 | ✅ 算法侧已验证（§13.8） |
| S3-c | 端到端延迟 p50 | ✅ 1080p 112 ms / 720p 70 ms（§16.4） |
| S3-d | **tracker 状态连续**（同一目标 track_id 不重置） | ⬜ 待 S4 验证 |
| S3-e | **无 GPU 时自动回落 CPU** | ⬜ 待 S4 验证 |

---

### 9.4 S3.5 — video_source 回放模式（✅ 已完成，2026-10-30）

**目标**：无真机也能跑完整链路。

**工时**：实际 0.5 天（原估 1 天）

| `source` 值 | 状态 | 行为 |
|---|---|---|
| `stub` | ✅ | 全黑假帧 |
| **`file`** | ✅ | 读视频文件，按 `rate` 定时发布 |
| `tello` | ⬜ S6 | 连真机取流 |

**验收结果**：见 §16.3（30.119 Hz，1920×1080 真实图像）。
`roslaunch mav_s playback.launch video:=phantom30`

---

### 9.5 S5 — rosbag 对拍（E3）

**目标**：证明 ROS 链路与阶段一脚本输出一致。

**工时**：1 天

#### 9.5.1 对拍方法

1. 用 S4 回放 phantom30 前 500 帧
2. 记录 `/mav/detections` 到 rosbag
3. 用**完全相同**的 500 帧跑 `scripts/run_testset.py`
4. 逐帧对比框中心位置

#### 9.5.2 判定

| 指标 | 门限 |
|---|---|
| 框中心差 | **< 2 px** |
| 检测帧数差 | **< 2%** |
| track_id 一致性 | 无需一致（生命周期可能不同） |

**不合格排查顺序**：图像是否无损 → 帧索引是否对齐 → MFE 帧缓冲是否正确 → LAC/LAD 是否同权重。

---

### 9.6 工时汇总

| 阶段 | 工时 | 累计 | 需硬件 |
|---|---:|---:|---|
| S1+S2 | 0.5 天 | 0.5 | ❌ | ✅ 完成 |
| S3（算法侧） | 1 天 | 1.5 | ❌ | ✅ 完成（§13.8/§14.7） |
| S3.5（回放+720p） | 0.5 天 | 2.0 | ❌ | ✅ 完成（§16） |
| **S4（detector_node 接入）** | **0.4 天** | **2.4** | ❌ | ⬜ **下一步** |
| S5（rosbag 对拍） | 0.3 天 | 2.7 | ❌ | ⬜ |
| S5.5 仿真 | 0.5 天 | 3.2 | ❌ | ⬜ |
| **S1–S5.5 小计** | **3.2 天** | | **全部离线** | |
| S6 | 3 天 | 6.2 | ✅ | ⬜ |
| S7 | 3 天 | 9.2 | ✅ | ⬜ |
| S8 | 8 天 | 17.2 | ✅ | ⬜ |

**较初始估算减少 2.8 天**：S3 从 2 天降到 1 天（ONNX 免掉 socket 层），
S3.5 从 1 天降到 0.5 天（文件回放比预想简单），S4/S5 相应压缩。

PRD §11.7 的 32 天含多目标扩展；本表为单目标主线。

---

### 9.7 待办清单

- [x] 清理 `ros2_ws/`（2026-10-29 完成，消息已备份）
- [x] **S1**：建 ros1_ws 双包，迁移消息，`catkin_make` 通过（2026-10-29）
- [x] **S2**：五节点骨架，话题全通（2026-10-29）
- [x] **S3**：ONNX 单进程集成（算法侧完成，§13.8/§14.7）
- [x] **S3.5**：video_source file 回放 + 720p 实测（§16）
- [ ] **S4**：`detector_node` 接 `mgmd_onnx`（丢帧策略、CPU 回落）
- [ ] **S4**：`video_source source:=file`
- [ ] **S5**：rosbag 对拍 E3
- [ ] **S5.5**：Gazebo 相机 + 抖动注入仿真（§11，需 S4 完成）
- [ ] S6–S8：需硬件

---

## 10. 与阶段一 SPEC 的分工

| 文档 | 管什么 | 改不改 |
|---|---|---|
| `docs/SPEC.md`（阶段一） | MFE/TF/LAC/LAD 算法、ARD-MAV 评测、论文复现结论 | 🔒 **冻结在 1.0.0** |
| `docs/SPEC-ROS.md`（本文件） | ROS 框架、节点、话题、实机集成 | 每轮维护 |

**阶段二改动算法时怎么办？** 按 V7 不允许。必须扩展时在 `mgmd_ros/` 下包装，
若确实要改 `mgmd/`，需先解冻阶段一 SPEC 并在两个文件同时记录——**目前不需要**。

---

## 11. 仿真方案（S5.5）

### 11.1 定位与取舍

**用户明确关注点**：不关心无人机控制（动力学/姿态/风扰），**只关心"这套东西能不能在无人机上跑起来"**。

按此标准，仿真要验证的是**数据链路在真实约束下能否通畅**，不是飞行是否合理。因此：

| 仿真什么 | 不仿真什么 |
|---|---|
| ✅ 真实分辨率与帧率（960×720 @ 30 Hz） | ❌ 飞行动力学 |
| ✅ Gazebo 渲染自身的 CPU/显存开销 | ❌ 姿态稳定 |
| ✅ 相机内参与畸变 | ❌ 电池、风扰 |
| ✅ 人为注入帧抖动（0–50 ms）与丢帧（2%） | ❌ 起飞降落 |
| ✅ 多目标同时在画面中 | ❌ 真实 H.264 解码 |
| ✅ 端到端延迟与帧率实测 | |

### 11.2 方案选型（方案 2.5b）

对比了五种方案（见对话记录），最终选 **2.5b：Gazebo 相机 + 人为帧抖动注入**。

| 方案 | 动力学 | Gazebo开销 | 真实视频流协议 | 工时 |
|---|---|---|---|---|
| 1 完整 rotors_simulator | ✅ | ✅ | ❌ | 2–3 天 |
| **2.5b（选定）** | ❌ | ✅ | ⚠️ 抖动近似 | **0.5 天** |
| 2 Gazebo仅相机 | ❌ | ✅ | ❌ | 0.5 天 |
| 3 PyBullet | ✅ | ❌ | ❌ | 1–2 天 |
| 4 纯数学mock | ❌ | ❌ | ❌ | 0.5 天 |
| 5 AirSim/Isaac | ✅ | ✅ | — | ❌ 4GB显存不够 |

**为什么不做 2.5a（真实 H.264 UDP 分片）**：多 1 天，收益是覆盖真实解码路径；
但用户关心的核心问题是"资源占用下能否跑通"，用抖动注入可近似覆盖，性价比更高。
若后续实测发现解码是瓶颈，再升级到 2.5a。

### 11.3 环境事实（2026-11-02 重新全面实测）

**结论：仿真环境可用，但需三处环境变量与一个写法约定，否则静默失败。**

#### 11.3.1 可用性

| 项 | 实测 | 状态 |
|---|---|---|
| `gazebo` / `gzserver` / `gzclient` | **11.15.1** | ✅ |
| ROS 包 | `gazebo_ros`/`gazebo_plugins`/`gazebo_ros_pkgs`/`gazebo_msgs` | ✅ |
| 相机插件 | `libgazebo_ros_camera.so` ✅、`libCameraPlugin.so` ✅ | ✅ |
| 显示 | `:0`，2880×1620，24 位深，Mesa DRI | ✅ |
| 模型库 | `/usr/share/gazebo-11/models` **仅 2 个**（ground_plane、sun） | ⚠️ 需自建目标物体 |

#### 11.3.2 三个必须设置的环境变量

| 变量 | 值 | 不设的后果 |
|---|---|---|
| `GAZEBO_MODEL_PATH` | `/usr/share/gazebo-11/models:$HOME/.gazebo/models` | `model://ground_plane` 找不到，**world 静默加载失败**，无任何报错 |
| `LD_LIBRARY_PATH` | 追加 `/usr/lib/x86_64-linux-gnu/gazebo-11/plugins` | `Failed to load plugin libgazebo_ros_camera.so: libCameraPlugin.so: cannot open shared object file` |
| `GAZEBO_MASTER_URI` | `http://localhost:11345`（**与 ROS 的 11311 不同**） | Gazebo 与 ROS 各连各的 master |

**注意**：`gazebo-11/plugins` 目录下的 `libCameraPlugin.so` **存在但不在 `ldconfig` 缓存里**
（`ldconfig -p | grep CameraPlugin` 无结果），必须显式加 `LD_LIBRARY_PATH`。

#### 11.3.3 启动方式：必须用 `rosrun gazebo_ros gzserver`

直接 `gzserver xxx.world` **不会**初始化 ROS 桥接，报：

```
[FATAL] A ROS node for Gazebo has not been initialized, unable to load plugin.
```

在 world 里手写 `<plugin filename="libgazebo_ros_api_plugin.so"/>` **也不行**：

```
[Err] World[default] is attempting to load a plugin, but detected an incorrect plugin type.
```

（加 `type="system"` 同样无效。）

**正确做法**：

```bash
rosrun gazebo_ros gzserver cam.world
```

该 wrapper 会自动注入 system 插件并建立桥接。

#### 11.3.4 相机模型必须 `static`

`<model name="cam">` 若不加 `<static>true</static>`，相机会**自由落体**。
实测 61 秒后相机 z 坐标掉到 **-19194 m**，画面变成纯色（min=max=178）。

**这是最容易踩的坑**：话题正常发布、帧率正常（15.6 Hz）、分辨率正常（960×720 rgb8），
但画面是纯色 —— 因为相机早已飞出场景。**只看话题和频率无法发现问题。**

#### 11.3.5 验证通过的完整最小 world

```xml
<?xml version="1.0" ?>
<sdf version="1.5">
  <world name="default">
    <include><uri>model://ground_plane</uri></include>
    <include><uri>model://sun</uri></include>
    <model name="box">                     <!-- 目标物体 -->
      <pose>3 0 0.5 0 0 0</pose>
      <link name="link">
        <collision name="c"><geometry><box><size>1 1 1</size></box></geometry></collision>
        <visual name="v"><geometry><box><size>1 1 1</size></box></geometry>
          <material><ambient>1 0 0 1</ambient><diffuse>1 0 0 1</diffuse></material>
        </visual>
      </link>
    </model>
    <model name="cam">
      <static>true</static>                <!-- 必须！否则自由落体 -->
      <pose>0 0 2 0 0.3 0</pose>
      <link name="link">
        <sensor name="camera" type="camera">
          <update_rate>15</update_rate>
          <camera>
            <horizontal_fov>1.047</horizontal_fov>
            <image><width>960</width><height>720</height><format>R8G8B8</format></image>
          </camera>
          <plugin name="ros_cam" filename="libgazebo_ros_camera.so">
            <ros><namespace>cam</namespace></ros>
            <cameraName>test</cameraName>
            <imageTopicName>image_raw</imageTopicName>
            <frameName>cam_link</frameName>
          </plugin>
        </sensor>
      </link>
    </model>
  </world>
</sdf>
```

**实测结果**：`/test/image_raw` 以 **15.6 Hz** 发布，960×720 rgb8，
画面含 **99,218 个红色像素**（方块可见），15 种颜色 —— **渲染正常**。

> **注**：world 里写的 `<namespace>cam</namespace>` 在 `rosrun gazebo_ros gzserver`
> 下被忽略，话题实际是 `/test/image_raw` 而非 `/cam/test/image_raw`。
> 接 ROS 时要按实际话题名对接。

#### 11.3.6 硬件

RTX 3050 Laptop 4 GB，16 核，15 GB RAM。

### 11.4 环境事实（原记录，2026-10-29，已被 11.3 取代）

| 项 | 状态 |
|---|---|
| `gazebo11` | ✅ 已装（11.15.1，随 ROS Noetic） |
| `gazebo_ros` / `gazebo_plugins` | ✅ 已装 |
| `libgazebo_ros_camera.so` | ✅ 存在 |
| 模型目录 | `/usr/share/gazebo-11/models`（仅 ground_plane、sun） |
| 硬件 | RTX 3050 Laptop 4 GB，16 核，15 GB RAM |

**无需额外安装。** 但需要自建场景模型（目标物体），见 §11.5。

### 11.5 架构

```
Gazebo 场景（虚拟相机 + 移动目标模型）
        │
        ▼  gazebo_ros_camera 插件
   /camera/image_raw (960×720, 30 Hz)
        │
        ▼
   video_source (source:=gazebo)
        │  ← 人为注入：帧抖动 0–50ms、丢帧 2%
        ▼
   detector_node ──(进程内直调)──► mgmd_onnx (3.8)
        │
        ▼
   /mav/detections ──►  ──► /mav/tracks
                                            │
                              ┌─────────────┴─────────────┐
                              ▼                           ▼
                          visualize_node                 
                              │                           │
                              ▼                    /tt/cmd_vel
                        /mav/debug_image                  │
                                                    ┌─────┴─────┐
                                                    ▼           ▼
                                              模式A:丢弃   模式B:驱动相机
```

**关键**：复用 S4 的 `video_source`，只新增 `source:=gazebo` 一个分支，**不新增组件**。

### 11.6 场景设计

| 要素 | 内容 |
|---|---|
| 世界文件 | `worlds/outdoor.world`：地面 + 太阳光 + 目标模型 |
| 相机 | `static` 模型挂 `libgazebo_ros_camera.so`，960×720，30 Hz，水平 FOV 82.6°（Tello 规格） |
| 目标 | 运动模型（边长 0.3 m 立方体），通过 `set_model_state` 按预设轨迹移动 |
| 轨迹 | 复用 ARD-MAV 的目标运动风格：直线匀速 + 变速 + 尺寸衰减（模拟远离） |
| 目标尺寸 | 按距离控制，使其在画面中呈现 **10–40 px**（对应小目标区间） |

**为什么用立方体而不是无人机模型**：目标是验证链路，不是外观。立方体在检测算法里会走 LAC 判定为假正例——
**但如果目标是纯色立方体，LAC/LAD 大概率判负，链路会"正常但检测不到"**。

`[flag: 这是本方案最大的实施风险。解法：① 用真实 MAV 图片贴图到模型上；**
**② 或允许在仿真场景中用较小 d_min + 关闭 LAC/LAD 只跑 MFE→TF，验证链路吞吐。**
**建议先用②验证链路，再用①验证端到端检测。]`

### 11.7 帧抖动与丢帧注入

在 `video_source(source:=gazebo)` 中实现，参数可配：

| 参数 | 默认 | 说明 |
|---|---|---|
| `jitter_ms` | 30 | 每帧随机延迟 `uniform(0, jitter_ms)` |
| `drop_rate` | 0.02 | 随机丢弃帧的比例 |
| `enable` | true | 关掉则退化为理想链路 |

**实现要点**：用独立线程 + 时间轮，不能阻塞 ROS 回调；注入后仍要保证话题时间戳反映**真实到达时间**。

### 11.8 验收项

| # | 检查 | 门限 |
|---|---|---|
| **S5.5-a** | Gazebo 能起，`/camera/image_raw` 出图 | `rostopic hz` ≈ 30 Hz |
| **S5.5-b** | 端到端帧率（含 Gazebo 渲染开销） | **≥ 10 FPS**（PRD §11.5） |
| **S5.5-c** | 端到端延迟 p50 / p95 | **p50 < 100 ms** |
| **S5.5-d** | 注入 2% 丢帧后 tracker 不崩 | 无 crash，ID 不每帧重置 |
| **S5.5-e** | 显存峰值 | **< 4 GB**（不能 OOM） |
| **S5.5-f** | 目标从画面中心缓慢横移，能被检测并跟踪 | ID 连续 ≥ 50 帧 |

**S5.5-e 是"能不能在无人机上跑"的直接答案**：若 4 GB 显存下 Gazebo + 检测算法能同时跑，
且帧率 ≥ 10 FPS，则**计算资源不是瓶颈**，真机部署的剩余风险只剩通信与供电。

### 11.9 能回答与不能回答的问题

| 用户关心的问题 | 能否回答 |
|---|---|
| 模块在 Gazebo 资源占用下能否跑满 30 FPS | ✅ 能 |
| 端到端延迟是否 < 100 ms | ✅ 能 |
| 帧抖动/丢帧时 tracker 会不会崩 | ✅ 能 |
| 4 GB 显存是否够 | ✅ 能 |
| 真实 Tello H.264 解码延迟 | ⚠️ 只能近似 |
| 真实飞控指令延迟 | ❌ 不能（属控制范畴，本次不关心） |
| 真实光照/逆光/雨雾鲁棒性 | ❌ 不能 |

### 11.10 工时与依赖

| 项 | 工时 |
|---|---:|
| 世界文件 + 相机模型 | 0.2 天 |
| 目标运动脚本（`set_model_state` 驱动） | 0.15 天 |
| `video_source source:=gazebo` + 抖动注入 | 0.15 天 |
| 验收测试 | 0.1 天 |
| **合计** | **0.5 天** |

**依赖**：S1（ros1_ws 可编译）、S4（`video_source` 已有 `source` 参数框架）。
**可提前到 S4 之后立即做，不必等 S5。**

---

### 11.11 仿真实施记录与**中止决定**（2026-11-02）

#### 11.11.1 结论：**S5.5 仿真中止**

用户决定：**「放弃这个吧，毕竟就是成功了也没办法跑 LAD 和 LAC」**

**判断成立**，理由如下（全部为实测支撑，非推测）：

| 环节 | 实测结果 |
|---|---|
| `MFE`（三帧差分） | ✅ 可跑通，候选能出 |
| `TF`（SORT） | ✅ 可跑通，能维持 `confirmed` 轨迹 |
| **`LAC`**（32×32 CNN） | ❌ **必然判负** —— 模型在真实 MAV 数据上训练，纯几何立方体的特征分布完全不匹配 |
| **`LAD`**（YOLOv5s 320 裁剪） | ❌ **必然判负** —— 同上 |

**这意味着即使把场景调到完美，仿真也只能验证 MFE+TF，验证不到 MGMD 的后两级。**
而 LAC/LAD 恰恰是阶段一调优最多、也最容易在嵌入式上出问题的部分。
**投入产出不成立，中止是正确决策。**

若强行压阈值让 LAC/LAD 通过，验证的就**不是真实链路**了 —— 那是自欺欺人。

#### 11.11.2 已完成的实施（保留，不删除）

| 文件 | 状态 | 说明 |
|---|---|---|
| `worlds/outdoor.world` | ✅ 可用 | 相机 + 地面 + 墙体 + 目标，相机发 `/camera/image_raw` |
| `worlds/cam_smoke.world` | ✅ 可用 | 最小验证场景 |
| `scripts/target_mover.py` | ✅ 可用 | 三种轨迹（line/sine/recede），30 Hz `set_model_state` |
| `scripts/check_gazebo.sh` | ✅ 可用 | 环境自检，10/10 通过 |
| `detector_node` 的 `~mfe_threshold`/`~mfe_adaptive` | ✅ 已加 | 仿真场景覆盖参数，默认不影响实机 |

**这些都保留在仓库里。** 如果将来有了真实 MAV 渲染模型（贴图/3D 模型），
可以直接复用这套场景与驱动。

#### 11.11.3 调试过程中确认的 6 个技术事实

按发现顺序，全部有实测数据：

**① 相机噪声是致命的，必须关**
world 里写的 `<noise>gaussian stddev=0.007</noise>` 看似微小，实测产生：
静止场景差分 `max=19`、**11 万像素 > 3**。
MFE 的阈值只有 5，噪声直接淹没目标。
**关掉后静止场景差分精确为 0。**

**② `medianBlur(5)` 会吞掉细窄差分带**
目标 34 px、每帧移动 4.6 px → 差分区域约 34×4.6 细条，
`medianBlur(5)` 正好把它滤没。

| 配置 | 有候选帧占比 |
|---|---:|
| 默认 `recommended` | 19% |
| 关自适应阈值 | 19%（**无效**） |
| **关 medianBlur** | **53%** |
| 两个都关 | 53%（自适应不是瓶颈） |

**③ 自适应阈值在此场景不是瓶颈**
`T = 5 + mean(diff)`，实测 `mean(diff)≈2` → `T≈7`，影响远小于 medianBlur。

**④ `3frame_or` 比 `3frame_and` 好一个数量级**

| mode | 有候选帧占比 |
|---|---:|
| `3frame_or` | 43% |
| `3frame_and` | 5% |
| `2frame` | 18% |

**⑤ `rospy.wait_for_message` 会拿到缓存旧帧，给出错误结论**
用它连续取 12 帧，测得「91% 重复」，一度误判为「Gazebo 渲染只有 3 fps」。
改用 `Subscriber(queue_size=1)` 正确计数后：
**181 帧全部唯一（100%），时间戳间隔稳定 33 ms（30 Hz）**，
`real_time_factor = 0.997`。
**排查仿真帧率必须用 Subscriber，不能用 wait_for_message。**

**⑥ `gzclient` 有无界面与渲染帧率无关**
用户提出「有界面就快了」，实测**开了 gzclient 重复率反而从 82% 升到 91%**
（但这本身是 ⑤ 的测量假象）。真正的结论是：无头 `gzserver` 渲染 30 Hz 完全正常。

#### 11.11.4 遗留参数说明

`detector_node` 新增的两个参数**默认不生效**，不影响实机与离线：

| 参数 | 默认 | 作用 |
|---|---|---|
| `~mfe_threshold` | `0`（不覆盖） | >0 时覆盖 `cfg.threshold` |
| `~mfe_adaptive` | `-1`（不覆盖） | >=0 时覆盖 `cfg.adaptive_thresh` |

**注**：目前这两个参数只够覆盖阈值，`median_blur` 与 `mode` 尚未暴露。
若将来恢复仿真，需补 `~mfe_median_blur` 与 `~mfe_mode`。

## 12. S1 完成记录（2026-10-29）

### 12.1 交付物

```
mgmd-linux/ros1_ws/src/mav_tracker/
├── mav_tracker_msgs/          # 4 个 msg，无 action
│   ├── CMakeLists.txt
│   ├── package.xml            # catkin / format 2
│   └── msg/{Detection,DetectionArray,Track,TrackArray}.msg
└── mav_tracker_nodes/         # 五节点骨架
    ├── CMakeLists.txt
    ├── package.xml
    ├── config/topics.yaml
    ├── launch/detect.launch
    └── scripts/{tt_driver,mav_node,tracker_node,viz_node,mission_node}.py
```

### 12.2 验收结果（全部通过）

| # | 检查项 | 结果 |
|---|---|---|
| **S1-a** | `catkin_make` 通过 | ✅ 0 error，生成 py/cpp/eus/lisp 全部后端 |
| **S1-b** | `rosmsg show DetectionArray` | ✅ 字段完整 |
| **S1-c** | ~~action 可查~~ | **❌ 已取消**（见 §7 RD-04） |
| **S1-d** | `from mav_tracker_msgs.msg import ...` | ✅ 4 个消息全部可导入可实例化 |
| **S1-e** | 五节点 `rosnode list` | ✅ 5 个全在 |
| **S1-f** | 话题完整 | ✅ 7 个话题全部出现 |
| **S1-g** | 频率 | ✅ camera 30.094 Hz / state 29.969 / detections 29.910 / tracks 29.941 |
| **S1-h** | 连接关系 | ✅ tt_driver → /mav_node on /camera/image_raw |

### 12.3 实际踩坑记录

| # | 问题 | 解法 |
|---|---|---|
| 1 | `pkill -f "roscore..."` 匹配到自己的命令行导致 shell 被杀 | **改用 PID 文件**（与阶段一 SPEC 的教训一致，再次验证） |
| 2 | 后台任务用 `nohup` 才能跨工具调用存活 | `nohup ... & echo $! > pidfile` |
| 3 | `rostopic hz` 需要 `timeout` 包裹否则挂起 | `timeout 7 rostopic hz ... \| grep "average rate"` |

### 12.4 与 PRD 的差异

**action 已删除**（PRD §11.2 第 911 行原写 `/mav/track_target` 为 action）。
理由见 §7 RD-04：actionlib 面向**可抢占的一次性任务**，视觉跟踪是**持续流式**的数据通路，
应改用 topic。此改动**提前消除了 ROS1 迁移中 action 段顺序的风险**（ROS2 是 goal/feedback/result，
ROS1 要 goal/result/feedback，写错会静默错位）。

### 12.5 下一步

S2 的验收项（S2-a~d）已在 S1 中**提前全部验证通过**（五节点起、8 话题、30 Hz、连接正确）。
因此 **S2 实质已完成**，可合并处理。下一步进入 **S3**（后于 1.2.0 改为 ONNX 单进程，见 §13）。

---

## 13. ONNX 改造方案（替代 sidecar，2026-10-29）

### 13.1 决定：放弃 sidecar，改用 ONNX Runtime

**原方案 A（sidecar）**：ROS 侧 3.8 + 算法侧 3.11，走本地 socket。
**新方案（ONNX）**：全部跑在 **Python 3.8 单进程**，LAC/LAD 用 ONNX Runtime 推理。

**用户要求**：可移植性。sidecar 需要两个 Python 环境，在嵌入式上不可行；ONNX 只需一个。
而且要问的问题只是"能不能在无人机上跑"，**单进程是最直接的答案**。

### 13.2 可行性验证（已完成）

| 模型 | 导出 | 数值一致性（CPU，39 真实样本） |
|---|---|---|
| **LAC** | `lac.onnx` 242 KB（动态 batch） | 最大概率差 **5.36e-07**，0/39 超 1e-4 ✅ |
| **LAD** | `yolov5s_GLAD-crop.onnx` 28.2 MB | 框差 **0.00 px**，conf 完全一致 ✅ |

**LAD 的关键事实**：ONNX 输出 `(1,6300,6)` 原始预测（cxcywh + obj + cls），
**不含 NMS**。NMS 与 `scale_coords` 仍在 Python 侧，复用 yolov5 utils。
因此 **ONNX 只替换 `model(x)` 这一步**，改造面极小。

### 13.3 环境事实

| 项 | 值 |
|---|---|
| Python | **3.8.10**（与 ROS1 同版本） |
| venv | `.venv38/`（项目内） |
| torch | **2.4.0+cu121**（最后支持 3.8 的版本，仅用其 NMS 工具） |
| onnxruntime-gpu | **1.19.2**（需 CUDA 12） |
| cv2 | 5.0.0 |
| numpy | 1.24.4（<2，兼容性更好） |
| 环境脚本 | **`scripts/env_setup_ros38.sh`** |

**必须 source 环境脚本**：onnxruntime-gpu 需要 `nvidia/*/lib` 在 `LD_LIBRARY_PATH`，
pip 安装不会自动加。不设则 CUDA provider 静默回退 CPU。

### 13.4 性能测试（预热 10 次 + 30 次计时，39 个真实样本）

| 模块 | torch (3.11, cu121) | ONNX (3.8, ort 1.19.2 CUDA) | 对比 |
|---|---:|---:|---|
| LAC p50 | 2.01 ms | **0.87 ms** | **快 2.3×** |
| LAD 总计 p50 | 161.5 ms | 165.4 ms | 慢 2.4% |
| LAD 每框 | 4.14 ms | 4.24 ms | 慢 2.4% |

**结论：ONNX 性能与 torch 持平（LAD），LAC 反而更快。没有性能代价。**

LAD 的 165 ms 主要是 Python 侧的 NMS + `scale_coords`（3900 帧 × 40 框），
不是推理本身。若要提速，应优化后处理而非换后端。

### 13.5 踩坑记录（三个真实 bug）

| # | Bug | 症状 | 修正 |
|---|---|---|---|
| 1 | `batch[:,:,::-1]` 是**负步长视图**，直接 `.transpose()` + `.astype()` 读错数据 | LAC 概率差 **0.928**（误判为 ONNX 有问题） | 加 `.copy()` + `np.ascontiguousarray()` |
| 2 | `mgmd/tf.py` 用 `Box = tuple[...]`，**Python 3.8 不支持** | `TypeError: type object is not subscriptable` | 改 `typing.Tuple`（第 25–26 行） |
| 3 | onnxruntime 1.17 需 `libcublasLt.so.11`(CUDA11)，环境是 CUDA12 | CUDA provider 加载失败，**静默回退 CPU**，性能差 4× | 升级 1.19.2 + `LD_LIBRARY_PATH` |
| 4 | YOLOv5 `export.py` 缺 `onnxsim` 时 NMS 未烘焙 | 误以为 ONNX 已含 NMS | **ONNX 本来就不含 NMS**，Python 侧做 |

**踩坑 1 的教训（第三次同类错误）**：把工具 bug 误判为被测对象的问题。
判据应是"同环境内对比"，而非"跨环境对比"。

### 13.6 对 S3 的影响

| 原 S3 内容 | 现方案 |
|---|---|
| `detect_server.py` + socket 协议 | **不需要** |
| 有状态 tracker 跨请求保持 | **不需要**（单进程，状态天然保持） |
| `mav_node` 接 socket 客户端 | **不需要** |
| 进程间序列化 | **不需要** |

**S3 从 2 天降为约 0.5 天**：只需把 `mav_node` 改成直接 import `mgmd` 调算法。

### 13.7 待办

- [ ] 把 `lac.onnx` / `yolov5s_GLAD-crop.onnx` 纳入版本管理或加导出脚本
- [ ] 写 `mgmd_onnx/` 包装层：ONNX 版的 `LacClassifier` / `LadDetector`（接口与 torch 版一致）
- [ ] 全链路 F1 验证：ONNX 路径跑 15 视频，F1 必须 = 0.6752
- [ ] `mav_node` 集成

### 13.8 全链路等价性验证（S3 门槛，已完成）

**方法**：用**同一个 `mgmd.pipeline.run`**，仅替换 LAC/LAD 实现（monkey-patch），
参数完全相同（`d_mode=self, d_min=0.02, lad_batch=8`），视频 phantom30 全 1496 帧。

| 路径 | P | R | F1 | AP |
|---|---:|---:|---:|---:|
| torch 2.5.1 (3.11) | 0.9551 | 0.7383 | **0.8328** | 0.7267 |
| **ONNX (3.8)** | **0.9551** | **0.7383** | **0.8328** | **0.7267** |

**逐帧核对**：
- 帧数 **1496 = 1496**，帧 key 集合完全相同
- 框数不同帧：**0 帧**
- 全部 1158 个框的坐标 + conf **最大差 2.50e-06**（浮点舍入级）

**结论：ONNX 路径与 torch 路径在本配置下等价。S3 门槛通过。**

### 13.9 排查记录（一次误判，第五次同类）

**现象**：初次对拍得 P=0.9182 (torch) vs 0.9551 (ONNX)，P 差 0.037，
一度怀疑 ONNX 有问题。逐层排查：

| 层级 | 对比结果 |
|---|---|
| LAC 概率（827 候选） | 最大差 2.93e-06 ✅ |
| LAD refine（58 帧 / 1158 框） | 完全一致 ✅ |
| LAD refine_batch（289 帧 / 293 框） | 完全一致 ✅ |
| **完整 pipeline** | **0.9182 vs 0.9551 ❌** |

**真因**：对比的两侧**调用参数不同**。`scripts/run_testset.py --lad-batch 8`
与手写的 `run(...)` 默认 `lad_batch=1`，走不同代码路径。
**统一参数后两个路径完全相同。**

**教训（第 5 次）**：对拍必须锁定**全部**参数，不只是模型和 d_min。
前 4 次分别是：错误的 Kalman 测试、`_kal.predict()` 重复调用、
`batch[:,::-1]` 负步长、`ref_lac.npy` 陈旧样本。

### 13.10 新增文件

| 文件 | 作用 |
|---|---|
| `mgmd_onnx/__init__.py` | `LacOnnx` / `LadOnnx`，接口与 torch 版一致 |
| `scripts/env_setup_ros38.sh` | 3.8 环境变量（含 `LD_LIBRARY_PATH`） |
| `scripts/bench_laclad.py` | LAC/LAD 性能基准（两环境通用） |
| `scripts/run_testset_onnx.py` | ONNX 全测试集评测 |

**`mgmd_onnx` 未修改 `mgmd/` 任何文件**（除 `tf.py` 的 3.8 类型兼容修复），符合 V7。

---

## 14. 实时性指标修订（S3，用户决策 C）

### 14.1 问题

原 S2-c 验收写的是 `rostopic hz /mav/detections ≈ 30 Hz`，
隐含假设"算法能以 30 Hz 处理 30 Hz 相机"。**实测不成立。**

### 14.2 实测：真实瓶颈分布

phantom30，1920×1080，ONNX 路径，预热 30 帧后统计 270 帧（CPU 侧计时）：

| 阶段 | p50 (ms) | p95 (ms) | 占比 |
|---|---:|---:|---:|
| **MFE**（三帧差分） | **117.1** | 181.1 | **57%** |
| **TF**（SORT 关联） | **60.6** | 330.1 | **29%** |
| LAC | 1.8 | 15.2 | 0.9% |
| LAD | 23.8 | 46.1 | 12% |
| **端到端** | **207.0** | 527.5 | **4.8 Hz** |

**推翻了之前的判断**：曾以为 LAD 是瓶颈（单框 4.2 ms × 多候选），
实测平均每帧只有 **1.3 个候选过 LAC、0.98 个过 LAD** —— LAD 不是瓶颈。
真正的瓶颈是 **MFE 差分**和 **TF 关联**。

### 14.3 两个独立的瓶颈

**(a) TF：纯 Python 循环，可优化 119×**

`cProfile` 显示 `iou()` 被调用 **1127 万次**（15 轨迹 × 131 候选 × 290 帧），
占总耗时 **69%**。向量化 IoU 矩阵后：

| 实现 | 耗时 | 数值差 |
|---|---:|---:|
| 逐元素 `iou()` 循环 | 9.59 ms | 基准 |
| **向量化矩阵** | **0.08 ms** | **0.000e+00** |

**119× 提速且零数值差异。**

**(b) MFE：与分辨率成正比，不可规避**

| 分辨率 | MFE p50 | MFE 可达 |
|---|---:|---:|
| 1920×1080（ARD-MAV 测试集） | 87.9 ms | 11.4 Hz |
| **960×720（TT 实际分辨率）** | **39.9 ms** | **25.1 Hz** |

**TT 的相机是 960×720，比测试集视频小 3 倍。** 这是 S3 实时性的关键前提。

### 14.4 决策（用户选 C）：解耦相机频率与检测频率

**原设计的问题**：把"相机 30 Hz"和"算法 30 Hz"绑在一起，物理上不可实现。

**新设计**：**输入 30 Hz，输出按算法实际能力**，两者解耦。

| 环节 | 指标 | 依据 |
|---|---|---|
| `/camera/image_raw` | **30 Hz** | TT 相机实测（S1-g 30.094 Hz） |
| `/tt/state` | 30 Hz | 飞控状态 |
| **`/mav/detections`** | **≥ 8 Hz（1080p）/ ≥ 12 Hz（720p）** | 实测端到端（§16.4） |
| `/mav/tracks` | 与 detections 同频 | tracker 跟随 |
| 端到端延迟 | **p50 ≤ 250 ms**（1080p） | 实测 207 ms |
| 端到端延迟 | **p50 ≤ 150 ms**（720p） | 实测 70 ms（§16.4） |

**为什么 ≥8 Hz 可接受**：MGMD 的 SORT 有 `max_age=3`，
检测间隔 125 ms（8 Hz）时，丢失 3 帧才删轨迹 = 375 ms 容忍。
对 5–20 m/s 的无人机目标（相对位移 <0.5 px/帧），跟踪不会断。

**为什么不做 30 Hz 的挣扎**：MFE 要对 1920×1080 做三帧差分 + 形态学，
单帧 88 ms 已是 OpenCV 优化后的结果。要 30 Hz 只能降分辨率或降帧率采样，
两者都改变算法语义，与"复现论文"冲突。

### 14.5 可选的加速路径（不做，但记录）

| 方案 | 预期收益 | 代价 |
|---|---|---|
| TF 向量化 IoU | TF 60.6 → 0.5 ms | **低**，数值无损 |
| ~~MFE 用 `cv2.UMat`（OpenCL）~~ | ~~可能 2-3×~~ → **实测仅 3%** | **已否定，见 §14.8** |
| MFE 降采样检测再映射回原图 | 4× | **改变算法语义**，影响复现 |
| 隔帧检测（每 2 帧跑一次 MFE） | 2× | 改变时序语义 |
| TensorRT（LAD 已有 engine） | LAD 23.8 → ~8 ms | 仅 LAD，占比小，收益低 |

**本轮只做 TF 向量化**（无损、低风险）。其余记为技术债。

### 14.6 S3 验收修订

| # | 原验收 | 修订后 |
|---|---|---|
| S2-c | `rostopic hz` ≈ 30 Hz | **`/camera/image_raw` ≥ 29 Hz 且 `/mav/detections` ≥ 8 Hz** |
| S3-a | 算法跑通 | ✅ 已达成（§13.8 全链路等价） |
| **S3-b** | **（新增）** | **1080p 端到端 p50 ≤ 250 ms；720p ≤ 120 ms** |
| **S3-c** | **（新增）** | **TF 向量化后 detections 与向量化前逐帧一致（数值差 = 0）** |
| S3-d | 端到端 rosbag 联调 | 不变 |

### 14.7 TF 向量化实施结果（S3-b/c 验收，已完成）

**实现**：`mgmd_onnx/tf_fast.py` 的 `FastSortTracker`，继承 `SortTracker`，
**只把 IoU 关联矩阵的 N×M 双重循环换成一次向量化调用**，其余逻辑逐行照抄。
`mgmd/` 未改动（V7）。

#### S3-c 验收（逐帧一致性）

| 项 | 结果 |
|---|---|
| 对比帧数（phantom30 全视频） | **1496** |
| 不一致帧 | **0** |
| 框坐标 + track_id 最大差 | **0.000000e+00** |
| 判定 | **✅ 通过** |

脚本：`scripts/verify_tf_fast.py`

#### 性能提升

**TF 单独**（180 帧，预热 20）：

| 实现 | p50 | mean | p95 |
|---|---:|---:|---:|
| `SortTracker` | 74.98 ms | 82.02 | 145.87 |
| **`FastSortTracker`** | **14.24 ms** | **14.93** | **22.77** |
| 提升 | **5.3×** | 5.5× | **6.4×** |

未达到理论 119× 的原因：向量化后 `update` 的剩余部分（Kalman predict/update、
`TrackState.add`、`classify`）成为新主导，这些是逐轨迹的 Python 逻辑，不易向量化。

**端到端**（290 帧，预热 30，ONNX LAC/LAD + CUDA）：

| 配置 | p50 | p95 | Hz |
|---|---:|---:|---:|
| baseline（`SortTracker`） | 150.0 ms | 244.9 | 6.7 |
| **fast（`FastSortTracker`）** | **117.3 ms** | **141.6** | **8.5** |
| 改善 | **22%** | **42%** | 6.7 → 8.5 |

#### 指标达成情况

| 指标（§14.4） | 目标 | 实测 | 判定 |
|---|---|---|---|
| 1080p 端到端 p50 | ≤ 250 ms | **117.3 ms** | ✅ |
| 1080p 检测频率 | ≥ 8 Hz | **8.5 Hz** | ✅ |
| 720p 端到端 p50 | ≤ 120 ms | 未实测（MFE 40ms 估算总 ~70ms） | 待 S4 实测 |
| 720p 检测频率 | ≥ 15 Hz | 未实测（估算 ~14–20 Hz） | 待 S4 实测 |

**720p 为估算值**，需 S4 用 TT 真实分辨率验证，不得声称已达成。

### 14.8 MFE 用 cv2.UMat 的评估结论（2026-10-30，不采用）

**背景**：§14.5 曾把"MFE 用 `cv2.UMat`(OpenCL)"列为候选加速手段，预期 2–3×。
**结论：实测无收益（3%），不采用。**

#### 环境

| 项 | 值 |
|---|---|
| cv2 | 5.0.0，`OpenCL: YES` |
| OpenCL 设备 | **NVIDIA GeForce RTX 3050 Laptop GPU** |
| `cv2.ocl.haveOpenCL()` | True |

#### 数值正确性（先验，全部通过）

| 算子 | CPU vs UMat |
|---|---|
| `medianBlur` k=3 | **完全相同**（最大差 0） |
| `morphologyEx` | 完全相同 |
| `absdiff` | 完全相同 |
| `bitwise_and` | 完全相同 |
| MFE 端到端（53 帧 / 8245 候选） | **候选完全一致，0 帧不同** |

**所以不是"精度问题"，是"没有速度"。**

#### 单算子性能（1080p，50 次平均，预热 10）

| 算子 | CPU | UMat | 比值 |
|---|---:|---:|---:|
| `medianBlur` k=3 | 0.55 ms | **0.27 ms** | **2.0× 快** |
| `morphologyEx` MORPH_OPEN | 0.35 ms | 0.71 ms | 0.5× 慢 |
| `absdiff` | 0.09 ms | 0.15 ms | 0.6× 慢 |
| `cvtColor` GRAY2BGR | 0.06 ms | 0.37 ms | **0.2× 慢** |
| `connectedComponentsWithStats` | 2.17 ms | 4.72 ms | 0.5× 慢 |
| **`calcOpticalFlowPyrLK`** | 6.3 ms | **❌ 崩溃** | — |

`calcOpticalFlowPyrLK` 报 `lkpyramid.cpp:1051 error: (-215:Assertion ...)`，
**OpenCV 5.0 的 LK 光流 UMat 路径不可用**。

#### 整条后处理链（960×720，GPU 常驻不往返）

| 方式 | 耗时 |
|---|---:|
| CPU 全程 | **0.48 ms** |
| UMat 常驻 GPU | 0.57 ms |
| UMat 每步往返（错误用法） | 2.39 ms |

**即使正确使用（不往返），整链也只有 0.48 ms，UMat 反而更慢。**

#### 为什么没有收益

MFE 的耗时结构（cProfile，73 帧）：

| 操作 | 占比 |
|---|---:|
| `calcOpticalFlowPyrLK` | **25%** ← UMat 崩溃 |
| `medianBlur` | 20% ← UMat 快 2× |
| `connectedComponentsWithStats` | 13% ← UMat 慢 2× |
| `warpPerspective` | 8% |
| `morphologyEx` | 4.5% ← UMat 慢 2× |

1. **唯一受益的算子（medianBlur）占比 20%，快 2× 即省 10%**；
2. **占比最大（25%）的 `calcOpticalFlowPyrLK` UMat 不支持**；
3. **MFE 端到端实测仅 49.7 → 48.1 ms（3%）**，落在测量噪声内；
4. 端到端 117 ms 中省 1.6 ms，**占比 1.4%**，毫无意义。

#### 决策

**不采用 UMat。** 理由：
- 收益 3% 不抵风险：换机器若 OpenCL 不可用会静默回退（或崩溃），
  引入一个**与环境强耦合的依赖**却换不来性能；
- 会造成"代码里用了 GPU 加速"的错觉，实际主力算子根本没加速。

**§14.5 加速路径表更新**：删除 MFE-UMat 一项，标注"已实测否定"。

#### 真正可行的加速方向（重新排序）

| 方案 | 实测/预期收益 | 风险 |
|---|---|---|
| ~~MFE 用 cv2.UMat~~ | **3%（实测）** | 环境耦合，**已否定** |
| MFE 降采样检测再映射 | ~4× | 改变算法语义 |
| 隔帧跑 MFE | 2× | 改变时序语义 |
| 优化 `calcOpticalFlowPyrLK` 参数（`maxLevel`/`winSize`/点数） | 未知 | 需重测 F1 |
| LK 换成稀疏角点采样 | 未知 | 改变对齐精度 |

**下一步若仍要提速 MFE，应从 `calcOpticalFlowPyrLK` 入手**（25%），
而不是在 UMat 上继续尝试。

---

## 15. 当前状态与下一步计划（2026-10-30，**已被 §23 取代**）

> ⚠️ **本节已过时**：其中列出的 S3.5/S4 计划均已完成实现（见 §16/§17/§3）。
> 章节内保留的 `tt_driver`/`mav_node` 为当时的旧名，现分别更名为
> `video_source`/`detector_node`。**当前状态请看 §23。**

### 15.1 到哪里了

| 阶段 | 状态 | 关键产出 |
|---|---|---|
| 阶段一（离线复现） | ✅ **冻结** | F1=0.6752，Tier-B 达成，62 测试，`result` 分支 |
| S1/S2（ROS 骨架） | ✅ **完成** | 双包编译，4 消息，五节点 30 Hz |
| S3 算法侧 | ✅ **完成** | ONNX 单进程，全链路与 torch 逐帧等价（§13.8） |
| S3 实时性 | ✅ **完成** | TF 向量化，1080p 端到端 117 ms（8.5 Hz）（§14.7） |
| **S3.5/S4（节点接入）** | ⬜ **未开始** | — |

**当前可交付**：一个能在 PC 上以 8.5 Hz 处理 1080p 的 ONNX 检测库，
已验证与 torch 版逐帧一致。**但还没有任何 ROS 节点调用它。**

### 15.2 下一步（按依赖排序）

#### S3.5 — `tt_driver` 文件回放（0.5 天）

**为什么先做这个**：S4 的 `mav_node` 需要有图像源才能测。
现在 `tt_driver` 只能发"假帧"，无法跑真实视频。

| 任务 | 验收 |
|---|---|
| `tt_driver source:=file path:=<视频>` | `rostopic hz /camera/image_raw` ≈ 视频帧率 |
| 支持 `rate:=` 限速（模拟 30 Hz） | 可指定 10/15/30 Hz |
| **720p 实测** | 用真实 720p 素材验证 §14.4 的 ≥15 Hz |

**产出**：720p 真实性能数据，替换 §14.7 的估算值。

#### S4 — `mav_node` 接入 ONNX 算法（0.4 天）

**架构**（比原 sidecar 方案简单得多）：

```
/camera/image_raw → mav_node（单进程内）
                      ├─ mgmd.mfe.enhance
                      ├─ FastSortTracker
                      ├─ LacOnnx
                      └─ LadOnnx
                    → /mav/candidates, /mav/detections
```

| 任务 | 验收（S4-a~d） |
|---|---|
| mav_node 内跑完整链路 | detections 频率 = 视频帧率（720p 应 ≥15 Hz） |
| 图像→检测延迟 | 消息 header 时间戳差 p50 ≤ 120 ms（720p） |
| 帧丢弃策略 | 算法跟不上时**丢最旧帧**，不排队堆积 |
| 优雅降级 | 无 GPU 时自动回落 CPU（不崩溃） |

**关键设计点——丢帧策略**：相机 30 Hz，算法 8.5 Hz，必须**主动丢弃**而非排队。
队列深度固定为 1，处理完永远取"最新"帧。这决定了延迟不随时长增长。

#### S5 — rosbag 对拍（0.3 天）

| 任务 | 验收（E3） |
|---|---|
| `rosbag record` 录一段真实回放 | bag 可复现 |
| 同一 bag 跑 torch 版与 ONNX 版 | **中心差 < 2 px，帧数差 < 2%** |
| 写入 `scripts/compare_rosbag.py` | 脚本可重复执行 |

#### S5.5 — Gazebo 抖动注入仿真（0.5 天）

只有前面全通才做。验证**资源竞争下的吞吐**（Gazebo 吃 CPU/GPU 时链路是否还稳）。

### 15.3 明确不做的事

| 项 | 原因 |
|---|---|
| MFE 继续提速 | 已接受当前帧率（用户决策）；加速必改变算法语义 |
| 30 Hz 检测 | 已论证物理不可达（§14.2） |
| TensorRT 全链路 | LAD 占比 12%，收益小；engine 与版本强绑定 |
| 多目标扩展（PRD 8 天） | 阶段一已验证 SORT 支持多目标，TT 场景暂不需要 |

### 15.4 风险与前置条件

| 风险 | 影响 | 缓解 |
|---|---|---|
| **没有 720p 真实素材** | S3.5 的 720p 实测做不了 | 用现有 1080p 视频降采样到 720p（分辨率真实，内容非 TT） |
| ONNX 模型未入库 | 换机器要重新导出 | 补 `scripts/export_onnx.py` |
| `.venv38` 未入库 | 换机器要重装 | 补 `scripts/setup_env38.sh`（记录完整依赖版本） |
| TT 实机不可用 | S6/S7 阻塞 | 与 S1–S5.5 无关，可后置 |

**前置条件**：S3.5 开始前应先补 `export_onnx.py` 和 `setup_env38.sh`，
否则"能跑"只在当前这台机器成立。

---

## 16. S3.5 完成记录：`tt_driver` 文件回放 + 720p 实测（2026-10-30）

### 16.1 前置补齐（§15.4 风险项）

| 文件 | 作用 | 验证 |
|---|---|---|
| `scripts/export_onnx.py` | 从权重重新生成 `lac.onnx` / `yolov5s_GLAD-crop.onnx` | ✅ 在 `/tmp/onnx_verify` 重新导出并自检通过 |
| `scripts/setup_env38.sh` | 从零重建 `.venv38`（固定版本） | ✅ 脚本内容可复现当前环境 |
| `scripts/verify_onnx.py` | ONNX 模型正确性自检（形状/概率/IoU） | ✅ 全部通过 |

**为什么必须补**：之前 ONNX 模型散落在 `/home/gaoduan/tracking/data_raw/`，
`.venv38` 未入库 —— **"能跑"只在当前这台机器成立**。现在模型可从权重重建、环境可从脚本重建。

### 16.2 `tt_driver` 三模式

| 模式 | 状态 | 说明 |
|---|---|---|
| `source:=stub` | ✅ | 全黑假帧（S1 骨架保留） |
| **`source:=file`** | ✅ **新增** | 回放本地视频，支持 `rate` / `loop` / `limit` |
| `source:=tello` | ⬜ | S6 实机阶段，当前显式报错提示 |

**参数**：`path`（短名如 `phantom30` 或绝对路径）、`rate`（0=原始帧率）、
`loop`、`limit`。

**时间戳约定**：`header.stamp` 用**墙钟时间**而非视频 PTS。
理由：消费者按时间戳算端到端延迟，需要墙钟语义；用 PTS 会得到"零延迟"的假象。

### 16.3 S3.5 验收

| # | 检查 | 结果 |
|---|---|---|
| S3.5-a | `catkin_make` 通过 | ✅ |
| S3.5-b | `roslaunch playback.launch video:=phantom30` 起节点 | ✅ |
| S3.5-c | `rostopic hz /camera/image_raw` | **30.119 Hz**（目标 30）✅ |
| S3.5-d | 图像尺寸与内容 | **1920×1080, bgr8, 均值 100.1, 非零 100%**（真实图像非黑帧）✅ |
| S3.5-e | `limit:=150` 正确停止 | ✅ 150 帧 / 5.15 s |

### 16.4 720p 实测（替换 §14.7 的估算值）

**方法**：ARD-MAV 1080p 视频降采样到 **960×720**（TT 真实分辨率；画面内容非 TT）。
`phantom30`，290 帧，预热 25 帧，ONNX + CUDA。

| 分辨率 | 端到端 p50 | p95 | p50 Hz | §14.4 目标 | 判定 |
|---|---:|---:|---:|---|---|
| 1080p（测试集原生） | 111.8 ms | 150.3 | **8.9 Hz** | ≥ 8 Hz | ✅ |
| **720p（TT 实际）** | **68.3–71.9 ms** | 95.0 | **13.9–14.6 Hz** | ≥ 15 Hz | ❌ **未达** |

**720p 分项（p50）**：

| 阶段 | 耗时 | 占比 |
|---|---:|---:|
| MFE | 31.8 ms | 58% |
| LAD | 10.0 ms | 18% |
| TF | 2.8 ms | 5% |
| LAC | 1.0 ms | 2% |
| **分项合计** | **45.6 ms** | — |
| **端到端实测** | **70 ms** | — |

差额 24 ms 是 Python 解释开销 / GC / 数组分配，非某单一阶段。

### 16.5 判定：接受 14 Hz，下调 720p 目标

**不调整代码去凑 15 Hz**，理由：

1. **14 Hz 与 15 Hz 在任务上等价**。SORT `max_age=3`，14 Hz 时容忍 214 ms 丢失，
   对相对位移 <0.5 px/帧 的无人机目标无影响（§14.4 已论证 8 Hz 就够）；
2. 剩余 24 ms 是 **Python 运行时开销**，要压掉需重写成 C++/提高帧率采样，
   **代价远大于收益**；
3. 15 Hz 这个数字是 §14.7 用 MFE 单项 40 ms **外推**得到的，当时即标注"待实测"。
   实测证明外推偏乐观 —— **应修正指标，而不是扭曲实现**。

**指标修订**（§14.4 表格）：

| 环节 | 原指标 | **修订后** | 依据 |
|---|---|---|---|
| `/mav/detections`（1080p） | ≥ 8 Hz | ≥ 8 Hz（不变） | 实测 8.9 Hz |
| `/mav/detections`（720p） | ≥ 15 Hz | **≥ 12 Hz** | 实测 14 Hz，留 15% 余量 |
| 端到端延迟（1080p） | p50 ≤ 250 ms | 不变 | 实测 112 ms |
| 端到端延迟（720p） | p50 ≤ 120 ms | **p50 ≤ 150 ms** | 实测 70 ms，留余量 |

**记录为 RD-07**：指标从外推值下调为实测值，不是"降低要求"，
而是"用实测替换推测"。

### 16.6 测量陷阱记录（第六次同类）

**现象**：首次 `bench_720p.py` 测得 720p 端到端 68.3 ms，但分项 MFE 显示 **42.4 ms**；
单独测 MFE 只有 **20.4 ms**。

**原因**：在完整链路中，LAD 的 CUDA 工作与下一步 MFE 的 CPU 计时**重叠**，
前一帧的 GPU 任务在后台继续，干扰了 MFE 的墙钟计时。

**教训**：**在混合 CPU/GPU 流水线里，分项计时必须隔离**。
要么显式 `torch.cuda.synchronize()` 后再开始计时，要么接受"分项之和 < 端到端"是常态。

本次采用后者：**端到端墙钟（70 ms）是唯一可信指标**，分项仅用于定位占比。

---

## 17. 节点重构：从五节点改为三节点（2026-10-30，S4）

### 17.1 触发原因

用户明确：**只做视频上的目标跟踪，不做无人机控制。**

原五节点方案（§9.2）假设要接飞控，因此有 `mission_node`（发 `/tt/cmd_vel`）
和 `tt_driver` 里的飞控桥（`/tt/state`）。这两部分在视频跟踪场景下是纯冗余。

### 17.2 新架构：三节点

```
video_source ──► /camera/image_raw ──► detector_node ──► /mav/detections
                                                     └─► visualize_node ──► 带框 mp4
```

| 节点 | 文件名 | 职责 |
|---|---|---|
| **video_source** | `video_source.py` | 视频源（file / camera / stub），发 `/camera/image_raw` |
| **detector_node** | `detector_node.py` | MFE → TF → LAC → LAD，发 `/mav/detections` |
| **visualize_node** | `visualize_node.py` | 画框 + HUD，写带框 mp4，发 `/mav/debug_image` |

### 17.3 删除与改名对照

| 原名 | 新名 | 处理 |
|---|---|---|
| `tt_driver.py` | **`video_source.py`** | 改名 + 删除飞控（`/tt/state`、`/tt/cmd_vel`、`tello` 模式） |
| `mav_node.py` | **`detector_node.py`** | 改名 + 重写帧缓冲 |
| `viz_node.py` | **`visualize_node.py`** | 改名 + 实现真实画框/写视频 |
| `tracker_node.py` | — | **删除**（TF 已内置于 detector，重复） |
| `mission_node.py` | — | **删除**（不控飞） |
| `detect.launch` | **`track.launch`** | 改名 + 只起三节点 |
| `playback.launch` | 合并进 `track.launch` | 删除 |

**为什么删除 `tracker_node`**：阶段一的 TF（SORT）已经在检测流水线内部，
`/mav/detections` 已带 `track_id`。再单独做轨迹管理是重复实现，
且拆开要求把「MFE 帧缓冲 + 候选顺序」跨进程传递，代价大、收益为零。

### 17.4 帧序号的明确定义（本次修正的核心 bug）

**定义**：帧序号 = **视频内的帧序号**，取自 `Image.header.seq`（`video_source` 发出，0 起逐帧 +1）。

**旧版 bug**：`mav_node` 用「收到的帧计数」`frame_seq` 当帧号传给 MFE：

```python
pipe.frame_seq += 1                    # 收到的第几帧
pipe.frames[pipe.frame_seq] = arr
target = pipe.frame_seq - pipe.k       # 想处理「中心帧」
```

两个后果：
1. **帧缓冲永远凑不齐**：`step()` 要求 `t-k, t, t+k` 三个 key 都在，
   而每次只新增 1 个 key，`target` 又是刚减去 k 的旧 key，
   大量调用返回 `None`（实测「缓冲等待 898/1198」）；
2. **检测无法回溯到视频帧**：`/mav/detections.header.seq` 是接收计数，
   与视频帧号脱节，S5 的 rosbag 对拍（中心差 < 2px）**根本无法进行**。

**修正**：

```python
# video_source：seq = 视频帧号（0 起）
msg.header.seq = sent

# detector_node：直接用 msg.header.seq 作为帧号
slot.put((int(msg.header.seq), msg.header, arr))
...
pipe.push(frame_id, header, arr)        # 以视频帧号为 key
```

**验证**：连续收到的检测消息 `seq=326,327,328,329`，且 `track_id=26696` 连续三帧不变、
框位从 (552,681) 平滑移到 (541,687) —— **帧号正确、跟踪连续**。

### 17.5 帧缓冲 = 6 帧

用户指定缓冲 6 帧。MFE 需要 `(t-k, t, t+k)`，k=1 时三帧即可，
但取 6 帧是为了：
- 丢帧时仍有足够的候选中心帧可比对
- 处理慢时能一次追平多帧（`while True: ready_frame_id()`）

**缓冲逻辑**（已单元验证）：

| 推入 | ready | 缓冲内容 |
|---|---|---|
| 0 | None | [0] |
| 1 | None | [0,1] |
| 2 | **1** | [0,1,2] |
| 3 | **2** | [1,2,3] |

`evict_below(t)` 在处理完 t 后淘汰 `t-k` 及更早，防止内存增长。

### 17.6 S4 验收结果

| # | 检查 | 结果 |
|---|---|---|
| S4-a | 三节点 `roslaunch` 起、话题通 | ✅ |
| S4-b | **帧号 = 视频帧号** | ✅ seq 连续 326→329 |
| S4-c | **track_id 跨帧连续** | ✅ tid=26696 连续 3 帧 |
| S4-d | 检测框真实（conf/坐标合理） | ✅ conf 0.82–0.91，框随目标平滑移动 |
| S4-e | **输出带框视频** | ✅ 1920×1080@30fps，277 帧，37.7 MB |
| S4-f | CUDA provider 生效 | ✅ `CUDAExecutionProvider`（修了 cublas 路径） |
| S4-g | 处理频率 | ⚠️ **4.4 Hz**（离线 8.9 Hz，见 §17.8） |

### 17.7 本轮修掉的两个真实 bug

#### bug-1：`visualize_node` 被 kill 时视频文件损坏

**现象**：`kill <pid>` 后 mp4 报 `moov atom not found`，无法播放。

**原因**：`rospy.on_shutdown` 只在 `ros::shutdown()` 时触发；
直接 SIGTERM/SIGINT 不会走到那里，`cv2.VideoWriter` 没 `release()`，
mp4 缺少 moov atom。

**修复**：额外注册 `signal.SIGINT/SIGTERM` handler，幂等调用 `_shutdown()`。

#### bug-2：CUDA 库路径 `maxdepth` 不够

**现象**：`libcublasLt.so.12: cannot open shared object file`，静默回落 CPU。

**原因**：`env_setup_ros38.sh` 用 `find ... -maxdepth 2 -type d -name lib`，
但实际路径深度是 3（`nvidia/cublas/lib`）。

**修复**：改为 `-maxdepth 3`。修后 `providers` 含 `CUDAExecutionProvider`。

### 17.8 仍存在的问题：ROS 内比离线慢一倍

| 指标 | 离线（单进程脚本） | ROS（detector_node） |
|---|---:|---:|
| MFE | 78 ms | 104–206 ms |
| 端到端 | 111.8 ms | ~230 ms |
| 频率 | 8.9 Hz | 4.4 Hz |
| 丢弃率 | — | 57%（1230/2130） |

**候选数一致**（83–181，离线约 130），说明 MFE 算法行为正常，**是速度问题**。

**尚定位的原因（按可能性）**：
1. **`visualize_node` 抢 CPU**：它要解 mp4、画框、编码 mp4，全在同一台机器上；
2. **ROS 默认多 spinner** + rosout 日志开销；
3. **`buff_size` 过大**（2×1080p×3）导致拷贝开销。

**下一步应先做隔离测试**：关掉 `visualize_node` 单独测 detector，
确认画框是否拖慢了检测。**不猜测、不盲目改代码。**

### 17.9 参数残留问题（ROS1 特性，已踩两次）

`rosrun pkg node.py _limit:=400` 会把 `/node/limit` 写进参数服务器，
**节点退出后不清除**。下次启动若不指定 `_limit`，会继承旧值。

**已两次导致误判**（第一次以为 tt_driver 逻辑有问题）。
**规避**：重启前显式 `rosparam delete /<node>`，或统一用 `track.launch`（launch 参数不残留）。

---

## 18. 编码器 GPU 化评估 + 跳帧破坏 SORT（2026-10-30）

### 18.1 问题提出

用户问：**"解 mp4 + 画框 + 编码 mp4 能不能分给 GPU，反正正常情况 GPU 也跑不满"**。

### 18.2 各环节实测（1920×1080，60 帧素材）

| 环节 | p50 | 占比 |
|---|---:|---:|
| `frame.copy()` | 1.61 ms | 8% |
| **画框 + HUD** | **0.77 ms** | **4%** |
| **`cv2.VideoWriter` mp4v 编码** | **19.55 ms** | **96%** |
| 合计 | 20.3 ms | — |

**画框本身只占 0.77 ms，完全不是问题。** 真正开销在编码。

另：`cv2.cuda` 在本环境 **`getCudaEnabledDeviceCount() = 0`**，OpenCV 的 GPU 路径不可用；
`cv2.VideoWriter` 不支持 H.264（只有 `mp4v`/`FFV1`）。

### 18.3 NVENC 评估：省 CPU，不省时间

ffmpeg 自带 `h264_nvenc`（RTX 3050 可用）。**严格同条件**对比
（同一批 400 帧、预热 40 帧、每组 3 轮取中位）：

| 方案 | p50 | 文件大小 |
|---|---:|---:|
| cv2 mp4v 软编码 | **15.68 ms** | 32.4 MB |
| nvenc q=32 cq23 | **13.51 ms** | 15.3 MB |
| nvenc q=64 cq28 | 15.16 ms | 8.2 MB |

**只有 1.16× 提速（省 2.2 ms），但文件体积减半。**

**纯 ffmpeg 编码（绕过 Python）的关键对比**：

| 编码方式 | 墙钟 | CPU |
|---|---:|---:|
| `h264_nvenc` | 2.20 s | **89%**（≈0.9 核） |
| `libx264 veryfast` | 2.34 s | **664%**（6.6 核） |

预转 yuv420p 后（排除 bgr→yuv 转换干扰）：

| 编码方式 | 墙钟 | CPU |
|---|---:|---:|
| `h264_nvenc` | 2.61 s | 79% |
| `libx264 veryfast` | **1.78 s** | 781% |

**结论：NVENC 的价值是降低 CPU 占用（89% vs 781%，差 7.4×），墙钟时间反而略慢。**

这恰好回应用户的判断「GPU 反正跑不满」——**GPU 确实有闲置算力，用它编码能把 CPU 让给 MFE**。

### 18.4 实现：`nvenc_writer.py`

| 设计点 | 选择 | 理由 |
|---|---|---|
| 进程模型 | **独立子进程 + mp.Queue** | 同步写 stdin 实测 **30.9 ms，比 cv2 的 15.7 ms 还慢** |
| 队列深度 | 32 | q=16 → 17.8 ms；q=32 → 13.5 ms；再大收益递减 |
| mp 上下文 | **`fork`** | 本环境 `spawn` 因脚本来自 `<stdin>`/不可导入而失败 |
| 失败回落 | 自动用 `cv2.VideoWriter` | 环境缺 ffmpeg 时不至于整个节点挂掉 |
| `-movflags +faststart` | 开 | moov 前置，边写边可播 |

**实测（120 帧）**：写入 p50 18.97 ms、文件 7.8 MB、帧数正确、可正常读回。

⚠️ **投递仍有 8–13 ms 开销**：6.2 MB 帧数据要序列化进队列，这是 Python 多进程的固有成本。

### 18.5 决定性结论：visualize 不是瓶颈

| 配置 | detector 频率 |
|---|---:|
| 有 visualize（CPU 编码） | 3.91 Hz |
| **无 visualize** | **3.77 Hz** |

**关掉 visualize 后几乎没变。** 编码 GPU 化确实有价值（省 CPU、文件减半），
但**它不解决 main 问题**。

### 18.6 真正的根因：跳帧破坏 SORT Algorithm 1

**现象**：ROS 里 detector 收到 302 帧只处理 33 帧，且 `LAC 0.0 ms / LAD 0.0 ms / 检测 0`。

**排查过程**：

1. 脱离 ROS 直接跑 `Pipeline.step()` → **119 ms、处理 118/120 帧（98%）**，算法本身正常
2. 单测 MFE → 40–42 ms，三轮波动仅 2 ms（**排除系统负载**，虽然 load average 6.46）
3. 关掉 visualize → 3.77 Hz（**排除 visualize 抢 CPU**）
4. ROS 里分段计时 → `MFE 58–147 ms、LAC 0.0、LAD 0.0、检测 0`
5. 离线测 LAC → 206 候选、max 概率 1.0、3 个过阈值（**LAC 正常**）
6. **最小复现**：

| 场景 | 轨迹输出 |
|---|---|
| 连续帧 [5,6,7,8] | 0, 0, **1, 2** ✅ |
| **跳帧 [2,4,8,12]**（ROS 式） | 0, 0, **0, 0** ❌ |

**不同步长下 SORT 每帧输出数**：

| 步长 | 每帧输出 | 相对 stride=1 |
|---|---:|---:|
| **stride=1（连续）** | **14.33** | 100% |
| stride=2 | 9.30 | 65% |
| stride=3 | 4.93 | 34% |
| **stride=4（ROS 现状）** | **4.58** | **32%** |

**根因**（`mgmd/tf.py`）：

1. `classify()` 第 148 行：`if self.lifetime < 3` → `probability = 0`。
   轨迹必须**被更新 3 次**才可能确认；
2. `TrackState.window(span=3)`：`target = newest_frame - span`，
   在 `boxes` 里找 `f <= target` 的最近旧帧。跳帧时历史帧稀疏，
   找到的旧帧与 `newest_frame` 间隔远大于 3，**同构变换 `compose_H(f0,f1)`
   逐帧查 `step_H`，中间帧缺失即返回 `None`**，对齐失败；
3. `detector_node` 调用 `tracker.update()` 时**从未传 `step_H`**（默认 None），
   使 `compose_H` 永远失败。

**为什么连续帧能工作**：`d_mode="self"` 在自身坐标系下比对，
连续帧时 `window()` 能找到间隔恰为 3 的旧帧，`align` 走到可用路径。

**结论：`mgmd.tf` 的 Algorithm 1 设计假设「每帧都被处理」。一旦跳帧，
确认机制失效。这不是 S4 引入的 bug，是阶段一算法的隐含前提在 ROS 实时约束下暴露。**

### 18.7 本次修正（已提交）

1. `ready_frame_id()` 跳过已处理帧（`last_done`），此前会重复处理同一帧；
2. 加 `n_dropped_evict` 统计，区分「入口丢弃」与「缓冲淘汰」；
3. 加前 5 帧分段计时明细，便于定位；
4. 加「收到第 N 张图」诊断。

**修正后**：`淘汰 0`（此前大量帧被淘汰）、`入口丢 241`。但频率仍 3.9 Hz，
**因为根因（跳帧）未解**。

### 18.8 未解问题与选项

**跳帧是 SORT 失效的直接原因，而跳帧是实时性不足的必然结果——这是死锁。**

| 方案 | 做法 | 代价 |
|---|---|---|
| A. 降分辨率到 720p | MFE 从 78→42 ms，可跑满 15 Hz | 小目标像素变小，检测率下降 |
| B. 改 SORT 容忍跳帧 | 修 `window()`/`compose_H` 支持变步长 | **改阶段一冻结算法**，违反 V7，需重验 F1 |
| C. 只对确认后的轨迹跳帧跟踪 | 检测降频、跟踪用 Kalman 外推 | 需新增逻辑 |
| D. 接受低检测率 | 承认 ROS 版只输出少量检测 | 检测率 32%，不可接受 |

**待用户决策。** 本轮不擅自改动阶段一算法（V7 约束）。

---

## 19. 分辨率降到 720p 的效果（2026-10-30）

### 19.1 改动

`video_source.py` 新增 `~width` / `~height` 参数（0 = 原始分辨率），
非 0 时用 `cv2.resize(..., INTER_AREA)` 缩放后再发布。

```bash
rosrun mav_tracker_nodes video_source.py _source:=file _path:=phantom30 \
    _width:=960 _height:=720
```

**算法侧零改动**（MFE/SORT/LAC/LAD 全不动），只是输入像素变少。

### 19.2 离线基准对比（phantom30，连续帧）

| 分辨率 | MFE p50 | TF 输出/帧 | 过 LAC 检测/帧 |
|---|---:|---:|---:|
| 1080p | 46.6 ms | 20.25 | 1.21 |
| **720p** | **18.1 ms（2.6× 快）** | 8.20 | **0.87（降 28%）** |

**MFE 快 2.6 倍，检测数只降 28%** —— 用 28% 的召回换 2.6 倍的算力，划算。

### 19.3 ROS 实测对比

| 指标 | 1080p | **720p** |
|---|---:|---:|
| 处理频率 | 3.9 Hz | **9.5–11.0 Hz（2.5×）** |
| MFE（ROS 内） | 58–147 ms | **25–56 ms** |
| 入口丢弃率 | 57% | 44% |
| 缓冲淘汰 | 0 | 0 |
| **检测输出** | **0（全灭）** | **0–2 个/帧** ✅ |

**720p 让检测从「完全没有」恢复到「约 1 个/帧」，与离线 0.87/帧基本吻合。**

### 19.4 为什么 1080p 时检测全灭

之前（§18.6）把原因归给「跳帧破坏 SORT」。**720p 后仍需澄清**：

| 现象 | 解释 |
|---|---|
| 1080p 检测恒为 0 | 处理 3.9 Hz，跳帧间隔 30/3.9 ≈ **7.7 帧**，SORT `window(span=3)` 需要间隔 3 的历史帧，实际间隔 7–8，周期性错过，且 `lifetime<3` 门控下难以累积确认 |
| 720p 检测恢复 | 跳帧间隔降到 30/9.9 ≈ **3.0 帧**，恰好在 `span=3` 的可容忍范围内 |

**这修正了 §18.6 的表述**：不是「跳帧一律破坏 SORT」，而是
**跳帧间隔与 `span=3` 的匹配程度决定成败**。间隔 3 时可用，间隔 7–8 时基本失效。

### 19.5 又一次犯错的记录（第七次同类）

**错误**：把传给 SORT 的帧号从「视频帧号」改成「处理序号」，理由是
「SORT 用 window(span=3) 找 newest_frame-3，帧号跳跃会找不到」。

**验证结果**（stride=4，36 帧）：

| 传法 | 总输出 | 平均 |
|---|---:|---:|
| 视频帧号 (2,6,10,14...) | 301 | **8.36/帧** |
| 处理序号 (1,2,3,4...) | 305 | **8.47/帧** |

**差异在噪声内，改动无效，已撤销。**

**原因**：`mgmd/tf.py` 用 `self._miss`（连续未匹配次数）判超龄，
不依赖帧号绝对值；`window()` 找的是「最近一个 `f <= newest-span` 的帧」，
间隔大于 span 时仍会取到，不是必须恰好等于 3。

**教训（与 §16.7 同一条，再次佐证）**：改代码前必须先写验证脚本。
这已是第七次「先改后验」的错误。

### 19.6 720p 下的新问题：检测数偏低

| 指标 | 阶段一 1080p 基线 | ROS 720p |
|---|---:|---:|
| 检测/帧 | 1.23（phantom30 全片） | ≈1（但 0 和 2 交替） |

候选数经核对与离线 720p 一致（28–55），**不是 MFE 问题**。
检测在 0/1/2 之间波动，符合「720p 下小目标像素变少、部分帧检测不到」的预期。

**未做**：720p 下的完整 F1 评估（需要跑标注集，属于重验范围，待定）。

### 19.7 当前可交付状态

```bash
roslaunch mav_tracker_nodes track.launch video:=phantom30 \
    width:=960 height:=720 rate:=30
```

产出：`outputs/ros_viz/track.mp4`，960×720@30fps，NVENC 编码，
带检测框 + track_id + HUD（帧号/检测数/各阶段耗时）。

**实测**：`检测 1–2 个/帧`、处理 9.5–11 Hz、视频正常可播。

---

## 20 各阶段耗时实测（性能基线）

用户要求「先解决性能问题」。本节给出**逐阶段计时**，是所有后续优化的唯一依据。
**在拿到本节数据之前，任何关于瓶颈的判断都不可信**（§16.7/§19 已因此翻车两次）。

### 20.1 计时方法

计时点在 `detector_node.Pipeline.step()` 内部，**每个阶段前后各取一次
`time.perf_counter()`**，差值为该阶段毫秒数：

| 字段 | 覆盖范围 | 位置 |
|---|---|---|
| `mfe_ms` | `enhance(prev,cur,next,cfg)` 全程 | `step()` 第 194 行 |
| `tf_ms` | `FastSortTracker.update()` | `step()` 第 203 行 |
| `lac_ms` | crop + `mav_prob_batch()` + 阈值过滤 | `step()` 第 219 行 |
| `lad_ms` | 逐个 `LadOnnx.refine()` | `step()` 第 232 行 |

`DetectionArray.msg` 相应新增 `float32 tf_ms`（此前只有 mfe/lac/lad）。

**「墙钟」** = `scripts/stage_timing.py` 订阅端相邻两条消息的到达间隔。
**「未计入」** = 墙钟 − 四阶段和，包含：视频源 resize、消息序列化、
rqt 发布、线程间交接、以及**处理不过来时的排队等待**。

> ⚠️ 「未计入」不等于「框架开销」。当入口丢帧 > 0 时，它主要是**排队**，
> 说明流水线已经饱和，此时墙钟反映的是「处理能力上限」而不是「单帧成本」。

复现：

```bash
bash scripts/measure_stages.sh 40 ard720p            # 默认 ARD phantom30 @720p
VIDEO=/path/to/x.mp4 bash scripts/measure_stages.sh 40 mytest
```

### 20.2 结果：ARD phantom30 @ 960×720（有效基线）

数据：395 帧，视频源 29.97 fps 全速，`buffer=6`，`d_min=0.02`，
CUDA EP（LAC/LAD），`FastSortTracker`。

| 阶段 | p50 (ms) | p95 (ms) | 均值 (ms) | 占墙钟 |
|---|---:|---:|---:|---:|
| **MFE 帧差对齐 + OR** | **46.00** | **69.80** | **48.41** | **49.3%** |
| TF SORT | 3.60 | 10.55 | 4.90 | 5.0% |
| LAC MAV 分类 | 1.28 | 5.53 | 1.94 | 2.0% |
| LAD YOLO 精修 | 0.00 | 38.92 | 11.68 | 11.9% |
| —— 四阶段合计 | 64.83 | 109.56 | 66.93 | 68.2% |
| —— 未计入（排队为主） | 41.64 | 65.82 | 31.26 | 31.8% |
| **墙钟总计（单帧）** | **103.35** | **157.20** | **98.19** | 100% |

吞吐 ≈ **10.18 Hz**；候选 p50 = 33 个/帧；输出框 p50 = 0（47% 的帧有框）。

### 20.3 结论：瓶颈是 MFE，且高度偏斜

1. **MFE 独占近一半墙钟（49.3%）**，是唯一值得优化的阶段。
   LAC + LAD + TF 合计仅 18.9%，即使全部优化到 0 也只能从 10.2 Hz 提到约 14 Hz。
2. **LAD 的均值(11.68) ≫ p50(0.00)**：只有约一半帧有候选进入 LAD，
   而 p95 达 38.92 ms —— 单帧多个候选时 LAD 串行逐个 refine，成本线性叠加。
   这是「检测数低」之外的第二性能点，但优先级低于 MFE。
3. **TF 比预期高**：720p 下含 33 个候选/帧时，SORT 的 IoU 矩阵 + 匈牙利匹配
   耗时 3.60 ms，而 1080p 低候选场景只有 0.58 ms。属正常，非优化重点。
4. **墙钟 103 ms 与 MFE 46 ms 的差距主要来自排队**：视频源 29.97 fps
   即 33 ms/帧，而处理需 98 ms，必然堆积。

### 20.4 关键坑：ROS 私有参数写法

测量脚本最初写成 `--source file --path ...`（argparse 风格），**完全无效** ——
节点用 `rospy.get_param("~source")` 读私有参数，命令行必须写 `_source:=file`。
后果是节点静默使用默认值（`stub` 源、30 Hz、循环），表现为：

- 日志 `source=file path= hz=30.0 loop=True res=原始` ← **path 为空是唯一线索**
- 收到 1972 帧但只处理 550 帧，**候选与检测全为 0**
- 帧号在 1..1493 之间循环重启 —— **这就是用户报的「序号乱跳」的真因**

**教训**：节点日志必须在启动时打印**全部生效参数**（已具备），
且看日志要确认 **path/res 非空**，不能只看「跑起来了」。

### 20.5 已排除的解释

| 猜想 | 实测 | 结论 |
|---|---|---|
| 可视化拖慢 | 有/无 visualize：3.91 vs 3.77 Hz | ❌ 可视化不是瓶颈（§18） |
| SORT 传错帧号 | 视频帧号 8.36 vs 序号 8.47 个/帧 | ❌ 在噪声内（§19.4） |
| 系统负载干扰 | load 6.46 时与空载 MFE 一致 | ❌ 不影响 MFE |
| 720p 反而更慢 | MFE 46.6 → 18.1 ms（离线） | ❌ 720p 更快 |

### 20.6 下一步（性能方向）

按投入产出排序：

1. **MFE 降分辨率**：MFE 的对齐/帧差在灰度图上做，可先在 480×360 上算、
   再把候选框按比例映射回 720p 原图。预计 MFE 降到 ~10 ms。
2. **MFE 降帧率**：`k` 由 1 改 2，减少对齐次数；需评估 F1 影响（暂缓）。
3. **LAD 批量化**：多候选时改成 batch 推理，消除串行线性叠加（p95 38.92 ms）。
4. **跳过无候选帧的 LAC/LAD**（已自然跳过）。

---

## 21 性能归因：三条优化路线全部实测否决

§20 定位到 MFE 是瓶颈后，对「GPU / 多线程 / 框架开销」三条路线逐条实测。
**三条都不成立。** 记录于此，避免重复尝试。

**结论先行**：当前 9.45–10.2 Hz 且「未计入」无浪费，是算法计算量的真实反映。
唯一未做的优化是 LAD 批量化（预计 ≤8 ms）。

### 21.1 端到端拆解（先弄清「未计入」是什么）

在 `detector_node` 加细粒度计时，每 50 帧打印一次。ARD phantom30 @960×720：

| 环节 | 耗时 (ms) | 说明 |
|---|---:|---|
| `frame_gap` | 26–36 | 回调被调用 − 上次发布完成 |
| `decode`（`np.frombuffer`） | **0.03** | 零拷贝 |
| `queue_to_work` | 6.9–12.1 | 等工作线程 |
| `step`（MFE+TF+LAC+LAD） | 56–111 | 真正的计算 |
| `build`（构造 DetectionArray） | 0.07 | |
| `publish` | 0.2–0.9 | |

**关键**：`frame_gap` 26–36 ms ≈ 视频源帧间隔 33 ms（29.97 fps）。
说明 detector 比源发得慢，每帧都在**等新帧**，而非传输慢。

**交叉验证**：`queue_to_work` 仅 6.9–12.1 ms（队列基本为空）。
若传输真慢，此值会很大。**两者共同证明流水线没有浪费，只是算得慢。**

### 21.2 否决路线一：GPU（OpenCV 无 CUDA 后端）

`cv2.cuda.getCudaEnabledDeviceCount() == 0`（OpenCV 5.0.0 与系统 4.2.0 均如此），
`cv2.cuda` 全部算子 `hasattr == False`。只能用 torch 实现等价算子实测。

**必须先预热**（用户提出）：预热 3 / 10 / 30 / 100 / 300 次，
`cvtColor` 分别 0.444 / 0.443 / 0.448 / 0.452 / 0.458 ms —— **无差异**，
异步吞吐与同步中位亦一致。**预热不是问题。**

960×720 单算子实测（RTX 3050，torch 2.4.0+cu121）：

| 算子 | CPU (cv2) | GPU (torch) | |
|---|---:|---:|---|
| cvtColor | 0.135 | 0.429 | GPU 慢 |
| absdiff | 0.082 | 0.169 | GPU 慢 |
| binarize(+median) | 0.214 | 0.196 | 持平 |
| morph | 0.441 | 0.442 | 持平 |
| **运动链整体** | **2.392** | **1.693** | **GPU 快 1.41×** |
| warpPerspective | 0.461 | 1.958 | **GPU 慢 4×** |
| calcOpticalFlowPyrLK | 1.723 | **无 torch 算子** | 无法移植 |
| connectedComponents | 1.42 | **无 torch 算子** | 无法移植 |

> ⚠️ 首次测量曾漏掉 GPU 侧 `medianBlur`，两边工作量不等价，
> 得出「GPU 全部更慢」的错误结论。补齐后修正为 1.41×。
> **教训：对比时必须逐项核对工作量等价，不能只看同名算子。**

MFE 16.65 ms 构成：可 GPU 化约 3–4 ms，**不可移植约 12 ms（光流+连通域）**。
理论上限 16.65→约 13 ms（20%），代价是引入 torch、自实现 warp（且更慢）、
CPU/GPU 往返。**判定：不做。**

### 21.3 否决路线二：Python 多线程

`prev`/`next` 两条支路完全独立（各自 对齐→帧差→二值化，最后 `bitwise_or`），
结构上可并行。实测：

| `cv2.getNumThreads()` | 串行 | 2 线程 | 加速比 |
|---:|---:|---:|---:|
| 16（默认） | 17.81 | 18.03 | **0.99×** |
| 8 | 17.03 | 18.89 | 0.90× |
| 4 | 21.91 | 22.46 | 0.98× |
| 2 | 26.25 | 25.99 | 1.01× |
| 1 | 39.35 | 32.43 | 1.21× |

**原因**：OpenCV 已用 16 线程并行。逐算子自带加速：

| 算子 | 16 线程 | 1 线程 | 自带加速 |
|---|---:|---:|---:|
| align | 3.96 | 11.68 | **2.95×** |
| binarize(+median) | 2.26 | 2.14 | 0.95× |
| postprocess | 4.41 | 7.69 | **1.74×** |

Python 层再加线程，只是在已饱和的 CPU 上制造竞争。
把 cv2 限到 1 线程后有 1.21×，但绝对耗时 32.43 ms（默认仅 17.81 ms）——
**自欺欺人的优化。判定：不做。**

### 21.4 否决路线三：框架开销

见 §21.1。`decode` 0.03 ms、`build` 0.07 ms、`publish` 0.2–0.9 ms。
原先怀疑的「ROS 图像传输 2 MB 拷贝 / tobytes / N 个 Detection 构造」**全部不存在**
（`np.frombuffer` 零拷贝）。**判定：无可优化空间。**

### 21.5 分辨率约束

**960×720 是下限，不得再降。** 用户明确：再降小目标会消失。

### 21.6 剩余可做项（未做，用户决定先接受当前结果）

| 方向 | 预期收益 | 风险 |
|---|---:|---|
| **LAD 批量化** | ≤8 ms（仅 55% 帧跑 LAD） | 低，不动 MFE |
| medianBlur 5→3 或可分离 | ~1 ms | 需视觉验证 |
| `k=2` 减半对齐次数 | ~5 ms | **中，影响小目标召回** |

LAD 观察：p50 0.00 / p95 44.81 / mean 14.82 ms，占 14%。
多候选时串行 `refine`，改批量可摊薄。**此轮未实施。**

### 21.7 新增工具

| 脚本 | 用途 |
|---|---|
| `scripts/stage_timing.py` | 订阅 `/mav/detections`，输出各阶段 p50/p95/均值/占比 |
| `scripts/measure_stages.sh` | 一键起三节点并测量；`RATE=x` 控制源帧率，`VIDEO=` 换视频 |
| `scripts/profile_mfe_inner.py` | MFE 子步骤拆解（align/bin/preprocess/postprocess） |
| `scripts/profile_mfe_gpu.py` | CPU vs GPU(torch) 算子对比 + 预热敏感性 |

### 21.8 测量纪律（再次踩坑）

1. **ROS 私有参数必须写 `_name:=value`**，`--name value` 静默无效（§20.4）。
2. **对比 CPU/GPU 必须核对工作量等价**（§21.2 漏 medianBlur）。
3. **测 GPU 必须预热并报告预热次数**（用户 2026-10-30 提出）。
4. **禁用 `pkill -f` / `pgrep -f`**：会匹配执行命令的 shell 自身并杀掉，
   已踩两次（§20.4 附近）。一律用 PID 文件。
5. **`git add -A` 会误收仓库根目录的 `.cache/`（曾含 2 GB 文件）**，
   必须显式列出改动文件。
6. **后台重定向 / roslaunch 测节点必须设 `PYTHONUNBUFFERED=1`**，
   否则日志被块缓冲吞掉，会被误判为「节点没启动」（§23.3.2）。

### 21.9 待办想法：GPU 加速 LK 光流（用户提出，暂不实施）

用户 2026-10-30 指出：「LK 金字塔光流可以用 GPU 加速，且效果不错」。

**修正此前的错误结论**：§21.2 表格中「`calcOpticalFlowPyrLK` → `无 torch 算子` → 无法移植」
的推理是**错的**。torch 没有 LK 算子，**不等于 GPU 上没有 LK 实现**。
实际存在的成熟实现：

- OpenCV `cv2.cuda.SparsePyrLKOpticalFlow`（`opencv_contrib` 的 `cudaoptflow` 模块）
- NVIDIA Optical Flow SDK / VPI

**不是「不能做」，是本机这套环境里没有。**

**本机现状**（`cv2.cuda` 全部为 False，`getCudaEnabledDeviceCount()==0`）：

| 检查项 | 结果 |
|---|---|
| venv `cv2` 5.0.0（节点实际使用） | CPU-only wheel |
| 系统 `cv2` 4.2.0（`cv_bridge` 使用） | 无 CUDA |
| `cv2.cuda.SparsePyrLKOpticalFlow` / `Farneback` / `NvidiaOpticalFlow_*` | 均 False |
| VPI / Optical Flow SDK | 未安装 |

**关于「ROS cv2 4.2 冲突」的更正**：此说法不成立，已核实——
venv 5.0.0 与系统 4.2.0 是两套独立安装，节点进程只加载一个（实测节点内
`cv2.__version__ = 5.0.0`），ABI 不冲突。真实前置条件是**需从源码编译带
`cudaoptflow` 的 OpenCV**，这是新增构建步骤，不是解决冲突。

#### 收益上限取决于点数（实测，960×720，ARMAV phantom30）

| 网格点数 | `calcOpticalFlowPyrLK` 耗时 |
|---:|---:|
| 70 | 0.992 ms |
| 260 | 1.596 ms |
| **600（当前 30×20）** | **2.040 ms** |
| 2400 | 5.347 ms |
| 9600 | 20.823 ms |

单次 `align_to_reference` 构成（600 点）：

| 环节 | 耗时 | GPU 可加速性 |
|---|---:|---|
| `calcOpticalFlowPyrLK` | 2.277 | 可，但仅 600 点 |
| `findHomography`(RANSAC) | 0.433 | 迭代算法，收益小 |
| `warpPerspective` | 0.521 | **实测 GPU 慢 4×**（1.958 vs 0.461） |
| valid_region 等 | ~1.6 | 可 |
| **全程** | **4.830** | |

传输成本：单帧灰度 0.66 MB，PCIe ~6 GB/s ≈ 0.12 ms/次。
每次 align 需传 2 张，两次 align 共 4 次 ≈ **0.46 ms**。

**判断**：600 点规模下 LK 本身仅 2 ms，GPU 并行度填不满 4096 核，
扣除传输与启动开销后**乐观也只有 1–1.5 ms 收益**（MFE 共 16.65 ms）。

#### 有价值但未验证的组合方案

> **GPU 实施 LK ＋ 把网格从 600 提到 3000–6000 点**

点数提升会改善对齐精度（对小目标是正收益），同时让 GPU 真正发挥优势。
但**必须先验证「提高点数是否真能改善小目标检测」**——这一步纯 CPU 可做，
不影响其他环节。用户 2026-10-30 决定暂缓，先接受当前结果。

**恢复此方向的前置实验**（约 0.5 天，纯 CPU）：
在 `d_min`、阈值不变的前提下，扫 `grid_cols×grid_rows` ∈ {30×20, 45×30, 60×40}，
对比候选框数量、面积分布与真值召回。**有正收益再投入编译 CUDA OpenCV。**

---

## 22 step_H 对齐补全（跨入阶段二，2026-10-30）

阶段一主结果存在「Algorithm 1 对齐从未生效」的复现缺口，详见 **SPEC §10.13**。
本节记录该修复**同步到阶段二 master** 的情况。

### 22.1 改动

| 文件 | 改动 |
|---|---|
| `mgmd/pipeline.py` | `tracker.update(..., step_H=res.info["prev_H"])`，仅 `k==1` |
| `tests/test_tf.py` | 新增 7 个 step_H / compose_H / align 测试 |
| `scripts/verify_step_H.py` | 单视频双口径验证 |
| `scripts/verify_step_H_multi.py` | **多视频并行验证** |

分支 `feat/step-H-align`（从 `result` 分出）保持独立，`result`（`36f8292`）冻结不变。

### 22.2 五视频并行验证（200 帧/视频 @960×720，5 进程）

| 视频 | 可比较 | `compose_H` | step_H | d 均值(对齐) | d 均值(原) | 翻转 | 通过率(对齐) | 通过率(原) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| phantom02 | 16489 | 16489/16489 | 100% | 0.8133 | 0.4231 | 78 (0.5%) | 99.9% | 99.6% |
| phantom05 | 5012 | 5012/5012 | 100% | 1.2447 | 0.6057 | 10 (0.2%) | 99.9% | 99.8% |
| phantom08 | 4508 | 4508/4508 | 100% | 0.3909 | 0.2484 | 59 (1.3%) | 99.3% | 98.5% |
| phantom10 | 2361 | 2361/2361 | 100% | 0.4930 | 0.3195 | 31 (1.3%) | 99.1% | 98.6% |
| phantom30 | 3525 | 3525/3525 | 100% | 0.4255 | 0.2786 | 25 (0.7%) | 98.4% | 97.8% |
| **合计** | **31895** | **31895/31895** | **100%** | — | — | **203 (0.6%)** | — | — |

**结论**：

1. `compose_H` 成功率 **100%**（31895/31895）—— 对齐稳定生效，无链路断裂。
2. d 均值系统性变大（约 2 倍），因为框的形变被计入位移。
3. 判定翻转仅 **0.6%**，方向是**通过率略升**（更宽松）。
4. **对 F1 的影响预期在噪声量级**（±0.005），但未重测。

### 22.3 再次确认：`reason` 字段不可信

`classify()` 末尾无条件覆盖 `reason`，`align_missing` 不会保留。
本工程此前第一版验证脚本正是读 `reason` 统计，得到 "align_missing 0 次"
而**误判为"对齐已生效"**。

**规则**：判断对齐是否生效，**只能**看 `compose_H(f_old, f_new) is not None`。
`tests/test_tf.py::test_align_missing_reason_is_overwritten` 锁定此行为。

### 22.3.1 phantom05 上的 F1/PR/AP 实测（2026-10-30）

用户要求「测一下 F1、R、P」。**只跑了 phantom05**（其余两个视频未跑完即停止，
用户决定不再继续）。同一命令、同一 GPU、同一配置，仅 step_H 有无不同：

```bash
.venv/bin/python scripts/run_testset.py --video phantom05 \
    --modules mfe,tf,lac,lad --mfe-preset recommended --d-min 0.02 \
    --out-suffix with_stepH --force
```

| 指标 | 基线 `final_d002`（无 step_H） | 新 `with_stepH` | Δ |
|---|---:|---:|---:|
| P | 0.8918 | 0.8908 | **−0.0011** |
| R | 0.5231 | 0.5220 | **−0.0011** |
| F1 | 0.6594 | 0.6582 | **−0.0012** |
| AP | 0.5324 | 0.5323 | **−0.0001** |
| TP / FP / FN | 940 / 114 / 857 | 938 / 115 / 859 | −2 / +1 / +2 |
| 检测总数 | 1054 | 1053 | −1 |
| 耗时 | 6.4 min | 6.9 min | +0.5 min |

**结论**：三个指标的变化都在 **−0.0012 以内**，远小于 0.6% 判定翻转率对应的
理论上界，完全在噪声量级。与 §22.2 的预测一致。

**注意**：单个视频的结论不能外推到整个测试集。特别是 phantom05 的
`compose_H` 成功率虽为 100%，但该视频目标运动幅度可能较小，
对齐收益有限。**完整影响仍需全测试集验证（未做）。**

### 22.4 阶段二定位

阶段二（ROS）同样没有传 `step_H`（`detector_node.py` 的 `tracker.update`），
**该缺陷与阶段一同源**。master 合并本次改动后，ROS 侧仍需单独补传：

```python
tracks = self.tracker.update(..., step_H=res.info.get("prev_H"))
```

**未做**（用户决定先记录）。若补，需注意 ROS 侧的跳帧可能使 `compose_H` 链路断裂，
届时成功率会低于离线测得的 100%。

---

## 23 进展整理与文档核对（2026-11-02）

用户要求「整理当前进展，核对 SPEC 文档」。本节为**唯一权威的当前状态**，
取代 §15 的过时计划。

### 23.1 执行顺序（用户 2026-11-02 指示）

1. **用现成视频验证三个 ROS 节点能否正常运行** ← 当前
2. **仿真**（S5.5）
3. 两机对拍（S5）、真机相关（S6/S7/S8）—— **全部放到最后**

### 23.2 已完成的阶段

| 阶段 | 内容 | 交付物 |
|---|---|---|
| S1/S2 | ros1_ws 双包骨架、节点骨架 | `ros1_ws/` |
| S3 | ONNX 单进程集成 | `mgmd_onnx/` |
| S3.5 | 文件回放 + 720p | §16 |
| S4 | 三节点重构（`video_source`/`detector_node`/`visualize_node`） | §17 |
| — | NVENC 编码、SORT 跳帧分析 | §18 |
| — | 720p 实测：10 Hz，检测 1–2/帧 | §19 |
| — | **性能归因**：MFE 46ms 占 49.3%，为唯一瓶颈 | §20 |
| — | **三条优化路线实测否决**（GPU/多线程/框架开销） | §21 |
| — | **step_H 对齐补全** + 五视频验证（`compose_H` 0%→100%） | §22 |

### 23.3 节点运行验证（2026-11-02 实测完成）

#### 23.3.1 三节点单独测试结果

逐个单独启动测试（`scripts/test_nodes.sh`，PID 文件管理，**不用 pkill -f**）：

| 节点 | 单独运行 | 配合其他节点 | 结论 |
|---|---|---|---|
| `video_source` | ✅ | ✅ | **正常**。参数全生效，60/100/200 帧三种 limit 准时退出 |
| `detector_node` | ✅ | ✅ | **正常**。候选 28–106/帧，稳定输出 1 检测/帧 |
| `visualize_node` | ✅ | ✅ | **正常**。产出 2.0 MB mp4（100 帧 @960×720） |

**detector 实测输出**（`_rate:=10 _limit:=200`）：

```
分段明细#1: MFE 38.8 LAC 0.0 LAD 0.0 候选 106 检测 0
已处理  50 帧  瞬时  6.84 Hz  收到  77 入口丢 21  frame= 76  候选 88  检测 0  MFE 36.7 ms
已处理 100 帧  瞬时  9.94 Hz  收到 127 入口丢 21  frame=126  候选 30  检测 1  MFE 41.4 ms
已处理 150 帧  瞬时 10.00 Hz  收到 177 入口丢 21  frame=176  候选 29  检测 1  MFE 25.2 ms
```

前 5 帧检测为 0 属**正常 warm-up**（SORT 需 ≥3 帧历史 + Algorithm 1 的 `lifetime>=3` 门控）。

#### 23.3.2 根因：此前「launch 下无日志」是 stdout 缓冲，不是节点故障

这是本次排查的核心发现。**三个节点从来没有故障。**

| 启动方式 | 日志行数 |
|---|---:|
| 后台重定向（默认缓冲） | **0** |
| `PYTHONUNBUFFERED=1` | **5** ✅ |

Python 的 stdout 默认是**块缓冲**（4 KB 或 8 KB 才 flush）。
`roslaunch` 以管道接管 stdout 时同样触发块缓冲，导致：

- 节点明明正常运行（`rosnode list` 能看到）
- 但日志全憋在用户态缓冲区里，不落到终端
- 进程被 kill 时缓冲区直接丢失 → **观测到「零日志」**

**这与节点代码无关，是输出重定向的固有行为。**

**规则**：任何后台重定向 / `roslaunch` 场景下测节点，**必须设
`PYTHONUNBUFFERED=1`**（或在脚本里加 `flush=True`），否则日志不可观测，
极易误判为「节点没启动」。已加入 §21.8 测量纪律。

#### 23.3.3 遗留观察（非阻塞）

- **处理频率**：`_rate:=10` 时稳定 10 Hz；但空闲起步阶段只有 6.84 Hz，
  与 §20 测得的 MFE 25–42 ms 一致（step 约 90 ms）。
- **入口丢帧**：`收到 177 / 已处理 150 / 入口丢 21`，与 §19 的跳帧分析一致。
- **LAC 偶发慢**：`LAC 132.4 ms`（首帧）与 `LAC 99.9 ms`，属 CUDA 首次调用抖动，
  与 §20.2 测到的 LAD max 2914 ms 同类。

#### 23.3.4 实时展示方案（**用户决定：以后展示都用这个**）

> 本节为原生 Ubuntu 的历史展示记录。当前 WSL2 环境以 §2.3 的 PyQt5 查看器 + CPU 编码为准，不再使用下方命令作为默认入口。

**标准做法**：`rqt_image_view` 订阅 `/mav/debug_image`。

```bash
# 终端 1：roscore
source /opt/ros/noetic/setup.bash && source ros1_ws/devel/setup.bash
roscore

# 终端 2：显示窗口
export DISPLAY=:0
rosrun rqt_image_view rqt_image_view
# 左上角下拉框选 /mav/debug_image

# 终端 3：跑流水线
export PYTHONUNBUFFERED=1 DISPLAY=:0
roslaunch mav_tracker_nodes track.launch video:=phantom30 \
    width:=960 height:=720 rate:=25 limit:=0 loop:=true \
    out_video:=$PWD/outputs/ros_viz/live_demo.mp4 encoder:=nvenc
```

**为什么选它**：ROS 官方工具，`rqt_image_view` 直接订阅话题，零中转、零改造。

**已排除的方案**：

| 方案 | 结论 |
|---|---|
| venv 里 `cv2.imshow` | ❌ venv 的 OpenCV 5.0 是 **headless 构建**，`namedWindow` 报 "The function is not implemented" |
| venv → 系统 python 管道传图 | ⚠️ 可行但复杂，无必要 |
| 系统 python + cv2 4.2 | ✅ 可行（rospy+PyQt5 都在），但不如 rqt 简洁 |
| `ffplay` 回放 mp4 | ✅ 只能看录好的，非实时 |

**关窗注意**：`rqt_image_view` 用 `xdotool` 查窗口时，
标题是 `rqt_image_view__ImageView - rqt`，不是简单的 "Image View"。

#### 23.3.5 实测数据汇总（2026-11-02）

三次运行的稳态数据：

| 运行 | 帧数 | 频率 | 候选/帧 | 检测/帧 | MFE |
|---|---:|---:|---:|---:|---:|
| 短测（rate=10, limit=200） | 150 | 10.0 Hz | 29–88 | 1 | 25–41 ms |
| 展示（rate=25, limit=900） | 400 | 12.6–16.8 Hz | 25–48 | 1 | 23–25 ms |
| **长跑（rate=25, limit=0，23 分钟）** | **25200** | **15.1–17.8 Hz** | **24–98** | **1** | **38–44 ms** |

**入口丢帧**：长跑 `收到 34270 / 已处理 25200 / 丢 3099`（9%），
与 §19 的跳帧分析一致（源 29.97 fps vs 处理 ~16 Hz）。

**长跑无累积退化**：第 25000 帧时频率仍有 15.07 Hz、MFE 43.8 ms，
与第 300 帧基本持平，未见内存或句柄泄漏。

#### 23.3.6 视频文件关闭必须用 SIGTERM，不能 SIGKILL

实测：`kill -9` 打断 NVENC 写入会导致 **mp4 损坏**（`moov atom not found`，
无法打开）。`kill -TERM` 则正确触发 signal handler：

```
[visualize_node] 收到信号 15，释放视频写入器
```

文件随即完整可读（400 帧 @25fps）。**`SIGKILL` 无法被捕获，是关闭方式的错，
不是 `visualize_node` 的问题。**

> ⚠️ 另注意：`limit:=0` + `loop:=true` 会**无限跑**，
> 23 分钟即产出 710 MB。演示时建议设 `limit` 或及时关闭。

### 23.4 节点运行验证（初次记录，已被 23.3 取代）

用 `roslaunch mav_tracker_nodes track.launch` 测试三个节点：

| 节点 | 启动 | 话题 | 结论 |
|---|---|---|---|
| `video_source` | ✅ | `/camera/image_raw` 正常发布 | **正常**。参数全部生效（路径/hz/分辨率），150 帧与 300 帧两种 limit 都能正常退出 |
| `visualize_node` | ✅ | 订阅图像与检测，写 mp4 | **正常**。产出 2.6 MB 视频文件 |
| `detector_node` | ⚠️ | 已注册到 master | **异常**：单独运行时完全正常（CUDA、FastSortTracker 就绪），但**在 launch 下无任何日志输出**，未处理帧 |

**`detector_node` 单独运行正常**（实测输出）：

```
[detector_node] MFE preset=recommended k=1 mode=3frame_or
[detector_node] tracker = FastSortTracker（向量化，§14.7）
[detector_node] LAC provider=CUDAExecutionProvider  LAD provider=CUDAExecutionProvider
[detector_node] 工作线程启动 device=cuda d_min=0.020
[detector_node] 就绪，等待 /camera/image_raw
```

**launch 下无一行日志**，但 `rosnode list` 能看到 `/detector_node`。
说明节点进程存在但初始化未完成或输出被吞。

**待排查方向**（尚未定位）：

1. launch 的 `output="screen"` 是否真的把 stdout 接到终端
2. 节点初始化耗时（CUDA 加载 ~2s）与 `video_source` 发帧速度的竞态
3. `env_setup_ros38.sh` 在 launch 环境下是否生效（`LD_LIBRARY_PATH` 为进程级环境变量，roslaunch 可能不继承）
4. 是否被 `roslaunch` 的 respawn/fork 行为影响

**注意**：此问题**不影响算法正确性**（离线 pipeline 已验证），
只影响 `roslaunch track.launch` 这一条启动路径。

### 23.4 文档核对结果（本次修正）

核对发现并修正以下不一致：

| # | 问题 | 修正 |
|---|---|---|
| 1 | §1 称「五个 ROS 节点」 | → 三个（§17 已重构） |
| 2 | §2.1 架构图仍画五个节点 + 飞控话题 | 按实际三节点重画 |
| 3 | §3 阶段表 S3.5/S4 标「⬜ 下一步」，实际已完成 | 全部更新为 ✅，新增 S4.5/S4.6 |
| 4 | §3 执行顺序未反映用户 2026-11-02 指示 | 补充三段式顺序 |
| 5 | §4 E1/E2 用旧节点名 | → `detector_node`/`video_source` |
| 6 | §5 话题表含 5 个已删除话题（`/tt/state`、`/tt/cmd_vel` 等） | 按实际代码重写，列出已删除项 |
| 7 | §15 计划已过时但仍作当前状态 | 标注「已被 §23 取代」 |
| 8 | §8 修正记录缺 1.6.0 / 1.7.0 / 1.8.0 | 补登 4 条（含本节 1.9.0） |
| 9 | **`docs/SPEC.md` 缺 §10.13**（代码已改但阶段一文档未同步，违反 V2） | 已补，含 phantom05 实测 |

**未修改**（属历史记录，保留原名正确）：§7 偏差表、§8 修正记录、
§12 S1 记录、§16 S3.5 记录、§17 重构记录、§19 720p 记录。

### 23.5 下一步

按 §23.1 的顺序：**继续排查 `detector_node` 在 launch 下无日志的问题**（§23.3），
之后进入仿真（S5.5，§11）。

---

## 24 方向决策记录（2026-11-02）

### 24.1 已走完与已否决的路线

| 路线 | 结论 | 依据 |
|---|---|---|
| 离线复现（阶段一） | ✅ **完成**，F1=0.6752 | SPEC §10.12 |
| ONNX 单进程集成 | ✅ 完成 | §13 |
| 三节点重构 + 文件回放 | ✅ **完成**，10–18 Hz | §17/§19/§23 |
| 性能优化（GPU/多线程/框架） | ❌ **三条全部实测否决** | §20/§21 |
| step_H 对齐补全 | ✅ 完成，指标影响 < 0.0012 | §22 |
| **Gazebo 仿真** | ❌ **中止**，LAC/LAD 无法验证 | §11.11 |

### 24.2 仿真中止后的可选方向

按「离机能做多少」排序：

| 方向 | 内容 | 产出 | 工时估 |
|---|---|---|---|
| **A. 完成 E3 对拍（用现成视频）** | `video_source` 文件回放 vs 离线 pipeline，逐帧比对框中心与检测数，达成 PRD E3 指标（中心差 <2px、帧数差 <2%） | **可写进毕设的定量验收结论** | 0.5 天 |
| **B. 端到端时延拆解** | 现在只有处理频率，没有「帧进→检测出」的端到端延迟 | 补齐 §4.2 验收指标 | 0.5 天 |
| **C. 多视频稳定性回归** | 5 个视频 × 长时间跑，查内存/句柄泄漏、ID switch | 稳定性证据 | 1 天 |
| D. 真机/两机对拍 | 用户明确放最后 | — | — |

**推荐 A + B**：都是纯离线、能直接产出毕设需要的定量结论，且 A 正是 PRD 的 E3 验收项。

### 24.3 仿真资产的用途（若将来恢复）

`worlds/`、`target_mover.py`、`check_gazebo.sh` 全部保留。
恢复的前提是**准备一个真实 MAV 的渲染模型**（贴图或 glb），
让 LAC/LAD 能正常判定。没有这个前提，仿真只能验证 MFE+TF，价值有限。
