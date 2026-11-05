# MGMD-linux × ROS1 在 WSL Ubuntu 20.04 上的运行记录

- 日期：2026-11-04
- 目标：在 WSL `Ubuntu-20.04` 里跑通 `毕业设计/mgmd-linux` 的 ROS1（Noetic）三节点流水线，
  用本机桌面上的 ARD-MAV 数据，真实弹窗显示检测结果。
- 结论：**已跑通**。三节点在 RTX 3050 4GB 上稳定 **~20 Hz**，ONNX Runtime 走 CUDA，
  实时窗口可见，输出 mp4 可正常播放。

---

## 1. 最终状态

| 项 | 结果 |
|---|---|
| ROS | Noetic（`/opt/ros/noetic`），`roscore/roslaunch/rostopic` 正常 |
| 工作空间 | `ros1_ws` catkin 编译通过：4 个消息 + 4 个节点 |
| Python | 系统 3.8.10（ROS 侧）+ 项目内 `.venv38`（同 3.8.10） |
| 推理后端 | `onnxruntime-gpu 1.19.2`，provider = **CUDAExecutionProvider**（LAC/LAD 均 GPU） |
| 数据 | `C:\Users\gaodu\Desktop\毕业设计\mgmd\data\videos\视频\ARD-MAV`（60 视频 / 60 标注目录），经 `/mnt/c` 只读 |
| 话题 | `/camera/image_raw` → `/mav/detections`（**19.8–20.1 Hz**）→ `/mav/debug_image`（20 Hz） |
| 实时窗口 | **`MGMD-ROS Qt  frames=N`**（PyQt5，WSLg RAIL 窗口；xdotool 可见、帧号持续增长） |
| 输出视频 | `outputs/ros_viz/phantom30_960x720.mp4`：960×720、mpeg4、**300 帧 / 15.0 s / 14.9 MB**，可播放 |

实测耗时（phantom30，960×720，RTX 3050 4GB）：

| 阶段 | 耗时 |
|---|---|
| MFE | 18–27 ms/帧（偶发 40–53 ms） |
| LAC | 0.44 ms/次（benchmark，batch1） |
| LAD refine（含预处理+NMS） | **10.5 ms/次** |
| ORT 前向 | ~8 ms/次 |
| 端到端 step | 30–47 ms/帧 |
| 入口丢帧 | ~16%（源 20 Hz vs 处理 ~20 Hz，启动阶段堆积） |
| 候选/帧 | 24–55 |
| 检测/帧 | 0–2（多数为 1，暖启动前 5 帧为 0） |

---

## 2. 与原始环境（SPEC-ROS §2）的差异

| 项 | 原记录（另一台机） | 本机 |
|---|---|---|
| 用户 | `gaoduan` | `gaodu`（用 `/home/gaoduan -> /home/gaodu` 软链兼容脚本里硬编码的 ROOT） |
| CUDA | 12.x + cuDNN（系统或 torch 带） | **pip 安装的 CUDA 12.9 运行库 + cuDNN 9.27**，无系统 CUDA Toolkit |
| torch | 2.4.0（cu121，默认） | **2.4.0+cpu**（只给 yolov5 的 NMS 用，避免与 ORT 的 cuDNN 9 冲突） |
| NVENC | 可用（18 ms/帧） | **不可用**：ffmpeg 4.2 报 `Cannot get the preset configuration: unsupported param (12)`，改用 `encoder:=cpu` |
| 显示 | rqt_image_view | 自写 `17_viewer.py`（image_view 在 WSLg 下窗口不映射） |
| 数据路径 | `/media/gaoduan/...` | `MGMD_ARD_ROOT=/mnt/c/.../ARD-MAV`（代码已支持该环境变量） |

---

## 3. 可复现步骤

所有脚本在 `/home/gaodu/毕业设计/mgmd-linux/.dsh-run/`（辅助脚本，不属于工程源码）。

```bash
# 3.1 目录布局：/home/gaoduan 软链 + 工程落地 + third_party + ONNX 权重
bash .dsh-run/01_layout.sh

# 3.2 安装 ROS Noetic（走清华镜像；官方 key 2025-06-01 已过期 → trusted=yes）
bash .dsh-run/02_ros_install.sh

# 3.3 建 .venv38（系统 python3.8；onnxruntime-gpu + CPU torch + pip CUDA 库）
bash .dsh-run/03b_venv38.sh

# 3.4 编译
bash .dsh-run/04_build.sh

# 3.5 单独验证 ONNX GPU 推理（不经过 ROS）
bash .dsh-run/07_run_smoke.sh
bash .dsh-run/11_run_bench.sh

# 3.6 一次性演示（600 帧，结束自动收尾）
bash .dsh-run/19_run_demo_viewer.sh phantom30 20 600 960 720

# 3.7 常驻实时演示（loop=true，窗口一直更新）
bash .dsh-run/23_live_qt.sh phantom30 20 960 720

# 停止一切 ROS 进程
bash .dsh-run/13_kill_all.sh
```

`23_live_qt.sh` 的窗口标题是 **`MGMD-ROS Qt  frames=N`**（Qt/PyQt5），位于 Windows 桌面（WSLg RAIL）。
窗口是 Qt 画的，关掉窗口不会结束节点；要停用 `13_kill_all.sh`。
`20_live.sh` 是旧的 GTK 版（窗口标题 `MGMD-ROS debug`），仅在 WSLg 共享内存正常时可用。

---

## 4. 环境变量（运行期必设）

```bash
source /opt/ros/noetic/setup.bash
source ros1_ws/devel/setup.bash
SP=$ROOT/.venv38/lib/python3.8/site-packages
export PYTHONPATH="$SP:$ROOT:$PYTHONPATH"          # 让 roslaunch 的 system python3 找到 ORT/cv2
export LD_LIBRARY_PATH="$(find $SP/nvidia -maxdepth 3 -type d -name lib | tr '\n' ':')$LD_LIBRARY_PATH"
export DISPLAY=:0
export PYTHONUNBUFFERED=1                           # 不加则 roslaunch 日志被块缓冲吞掉
export MGMD_ARD_ROOT="/mnt/c/Users/gaodu/Desktop/毕业设计/mgmd/data/videos/视频/ARD-MAV"
export MGMD_YOLOV5_DIR="$ROOT/third_party/yolov5-v6.0"
```

---

## 5. 踩到的坑与修复（全部有实测证据）

| # | 现象 | 根因 | 修复 |
|---|---|---|---|
| 1 | `apt update` 报 `EXPKEYSIG F42ED6FBAB17C654` | ROS1 签名 key 已于 **2025-06-01 过期**，Noetic 已 EOL | 源改 `[trusted=yes]`（本机研究环境可接受） |
| 2 | `packages.ros.org` 下载 Packages 卡住 | 官方源在国内不稳定 | 换 `mirrors.tuna.tsinghua.edu.cn/ros/ubuntu`（200, 0.25 s） |
| 3 | `onnxsim` 编译失败 | 它要 **CMake ≥ 3.22**，focal 只有 3.16 | 不装 `onnx-simplifier`（推理不需要） |
| 4 | ORT 报 `libcurand.so.10: cannot open` | 只装了 cudnn/cublas/cufft/cudart | 补 `nvidia-curand-cu12`（+cusolver/cusparse） |
| 5 | LAD 调用 `ModuleNotFoundError: matplotlib` | yolov5 v6.0 `utils/metrics.py` 顶层 import matplotlib | 装 `matplotlib>=3.7`（+seaborn） |
| 6 | `[nvenc] ffmpeg 退出码 1: Cannot get the preset configuration` | WSL 的 ffmpeg 4.2 + 新驱动的 nvenc preset 不兼容 | `encoder:=cpu`（cv2 mp4v），27 MB/600 帧 |
| 7 | `rosrun image_view` 进程在、`/mav/debug_image` 20 Hz，但**没有窗口** | image_view 在回调线程里 `imshow`，WSLg 的 GTK/Wayland 下不映射 | 自写 viewer：回调只存帧、**主线程** 显示；GTK 版（`17_viewer.py`）可用，但见 #12 |
| 8 | `kill -9` 后 mp4 `moov atom not found` | 写入器未收尾 | 停止顺序：先 `SIGTERM visualize_node` 等它 `writer.close()`，再清理其它进程 |
| 9 | 新旧节点同名导致新节点被 shutdown（`new node registered with same name`） | 上一轮 detector/visualize 未退出就起新一轮 | 每轮先跑 `13_kill_all.sh`（按 PID/精确进程名，不用 `pkill -f`） |
| 10 | `roscore` 在 `wsl.exe` 会话结束后被杀 | 同一会话组收到 SIGHUP/SIGINT | `setsid nohup ... &` 脱离会话 |
| 11 | `/tmp/ros_run` 日志偶发消失 | WSL 的 systemd-tmpfiles 清理 `/tmp` | 常驻演示日志改写到 `outputs/logs/` |
| **12** | **用户实际看不到视频窗口，只看到一个报错** | ① WSLg 的 RDP 共享内存通道失败：`rdp_allocate_shared_memory: Failed to open "/mnt/shared_memory/{...}" with error: Input/output error`（weston.log）；② OpenCV 的 GTK 窗口在这个通道异常时会显示异常/空白 | **`wsl --shutdown` 重启 WSLg**，并把查看器换成 **PyQt5**（`21_qt_viewer.py`，Qt 走 xcb，不再依赖 OpenCV GTK）。恢复后窗口 `MGMD-ROS Qt frames=N` 持续更新（实测 1194→1254/3s） |

---

## 5.1 故障复盘：为什么第一次"只看得到报错、看不到画面"

1. 窗口确实创建了（weston 日志有 `appId: MGMD-ROS debug / WindowId: 0x4`），说明 ROS → 图像 → GUI 链路是通的。
2. 但同一时刻 weston 报：
   `rdp_allocate_shared_memory: Failed to open "/mnt/shared_memory/{8190e1e6-...}" with error: Input/output error`
   —— WSLg 向 Windows 传画面的 RDP 共享内存通道坏了，窗口有"壳"但没有像素。
3. 期间 `wlog.log` 还记录了 FreeRDP 重新连接（11:11），说明 RDP 会话抖动过；`stderr.log` 里 Xwayland
   `Failed to initialize glamor, falling back to sw` 也指向同一类渲染降级。
4. OpenCV 的 GTK 窗口在这种状态下表现为空白/异常窗口，用户侧看到的就是"一个报错"。

**修复**：`wsl --shutdown` 让 WSLg 整体重启（共享内存与 RDP 通道重建），并把查看器从
OpenCV GTK 换成 **PyQt5（xcb）**；重启后 `weston.log` 不再出现 `rdp_allocate_shared_memory` 错误，
窗口 `MGMD-ROS Qt frames=N` 每 3 秒稳定 +60 帧（20 fps）。

复现命令：

```bash
wsl --shutdown
# 重新进入发行版后
bash .dsh-run/23_live_qt.sh phantom30 20 960 720
bash .dsh-run/25_focus_window.sh          # 置顶并摆到 (40,40)
bash .dsh-run/26_status.sh                # 看节点/频率/窗口标题
bash .dsh-run/27_stop.sh                  # 优雅停止（写完整 mp4）
```

本次实时演示最终产物（2026-11-04 06:23 停）：`outputs/ros_viz/live_phantom30_960x720.mp4`，
mpeg4 / 960×720 / **12 218 帧 / 610.9 s / 523 MB**，`visualize_node` 收到 SIGTERM 后正常收尾，
`ffprobe` 无 `moov atom` 错误。停止后 `rosmaster`/节点/Qt 窗口/GPU 计算进程均无残留。

---

## 6. 数据与模型

| 资产 | 路径 | 说明 |
|---|---|---|
| 视频 | `/mnt/c/Users/gaodu/Desktop/毕业设计/mgmd/data/videos/视频/ARD-MAV/videos/*.mp4` | 60 个；phantom30 = 1920×1080 / 29.97fps / 1523 帧 |
| 标注 | 同目录 `Annotations/phantomNN/*.xml` | 60 个目录，Pascal VOC |
| ONNX 权重 | `/home/gaodu/tracking/data_raw/mgmd_weights/{lac.onnx, yolov5s_GLAD-crop.onnx}` | `WEIGHTS_DIR` 硬编码到 `/home/gaoduan/tracking/...`，靠软链命中 |
| YOLOv5 源码 | `mgmd-linux/third_party/yolov5-v6.0` | 从 `tracking/third_party` 复制（ONNX 版 LAD 的 NMS 用） |

---

## 7. 待确认 / 未做

1. **1200p/1080p 原始分辨率**：本次跑 960×720（SPEC 的推荐口径）。1080p 是否仍 ~20 Hz 未测。
2. **D4（4K）**：未跑；4 GB 显存需要 `downscale`。
3. **NVENC**：本机不可用，若需要更低编码开销，可装新版 ffmpeg（apt 只有 4.2）或改用 `-preset p1` 之外兼容参数。
4. **`image_view`/`rqt_image_view`**：`image_view` 在 WSLg 下不可用；已改为 PyQt5 查看器（实测可用，窗口持续刷新）。
5. **长跑稳定性**：本次实时跑已处理 8 500+ 帧（约 7 分钟）稳定 ~19.6 Hz；SPEC 的 23 分钟长跑未复测。
6. **实时 mp4 体积**：`loop=true` 时文件持续增长（约 390 MB / 7 分钟，≈ 55 MB/min），长期演示要设 `limit` 或定期清。
7. **辅助脚本** `.dsh-run/` 未纳入 git；如需保留，建议移到 `scripts/wsl/` 并登记到 SPEC。

---

## 8. 复现用脚本清单（`.dsh-run/`）

| 脚本 | 作用 |
|---|---|
| `01_layout.sh` | 软链、工程复制、third_party、ONNX 权重 |
| `02_ros_install.sh` / `02b_verify_ros.sh` | 装 ROS Noetic / 验证 |
| `03b_venv38.sh` | 建 `.venv38`（ORT GPU + CPU torch + pip CUDA 库） |
| `10_fix_deps.sh` | 补 curand/cusolver/matplotlib/seaborn |
| `04_build.sh` | `catkin_make` + `rosmsg show` |
| `07_smoke_onnx.py` / `07_run_smoke.sh` | 不经 ROS 的 ONNX GPU 冒烟 |
| `11_bench_lad.py` / `11_run_bench.sh` | LAD/LAC 延迟基准 |
| `17_viewer.py` | 主线程 cv2 窗口（GTK；WSLg 共享内存异常时会出问题） |
| `21_qt_viewer.py` | **PyQt5 窗口（当前推荐）**：订阅 `/mav/debug_image`，标题带帧号 |
| `23_live_qt.sh` | 常驻实时演示（Qt 窗口版，loop=true） |
| `24_gui_check.sh` | GTK/Qt 窗口连通性自检 + xdotool/weston 证据 |
| `25_focus_window.sh` | 把视频窗口置顶/移到 (40,40)/改 1000×760 |
| `26_status.sh` | 一键查看节点、话题频率、处理进度、mp4 大小、窗口标题 |
| `27_stop.sh` | **优雅停止**：先 TERM visualize 写完 mp4，再关窗口、清进程，最后 ffprobe 校验 |
| `19_run_demo_viewer.sh` | 一次性演示（含窗口、优雅收尾、ffprobe 校验） |
| `20_live.sh` | 常驻实时演示（GTK 窗口版） |
| `13_kill_all.sh` | 停掉全部 ROS 进程 |
| `12_list_windows.sh` | 进程/节点/窗口状态 |
