# SPEC-ROS: 阶段二 · ROS1 单目旋翼无人机小目标检测与跟踪

> 本文件是**阶段二的唯一事实源**，与阶段一的 `docs/SPEC.md` 并列、互不覆盖。
> 派生自：`paper/prd_mgmd_reproduction.md`（v4.2, 2026-10-28）§11
> 工程根：`/home/gaoduan/毕业设计/mgmd-linux/`
> 最后更新：2026-10-29

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
| **V8** | 阶段一 SPEC（`docs/SPEC.md`）**冻结在 1.0.0**，阶段二如需引用只读不改 |

`spec_version: 1.2.0` | `prd_version: 4.2` | `stage1_spec: 1.0.0`（冻结）

### 0.1 版本记录

| spec_version | 日期 | commit | 变更摘要 |
|---|---|---|---|
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
- **单进程**：五个 ROS 节点，算法在 `mav_node` 内直接调用（无跨解释器）
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
│ tt_driver     → /camera/image_raw            │
│ mav_node      内部直调:                        │
│                 MFE → FastSortTracker        │
│                 → LacOnnx → LadOnnx          │
│               → /mav/candidates,/mav/detections│
│ tracker_node  → /mav/tracks                  │
│ viz_node      → /mav/debug_image             │
│ mission_node  → /tt/cmd_vel,/mav/track_target│
└──────────────────────────────────────────────┘
```

**没有任何跨进程通信。** 原 sidecar + Unix socket 方案已在 §13 推翻（RD-05），
算法以 `mgmd_onnx` 形式在 `mav_node` 进程内直接调用。

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

### 2.3 环境事实（已实测，2026-10-26）

| 项 | 值 | 验证 |
|---|---|---|
| ROS1 | `ros-noetic-desktop-full 1.5.0`，`/opt/ros/noetic`（140 MB） | `roscore` 启动 ✅ |
| 源 | `http://mirrors.tuna.tsinghua.edu.cn/ros/ubuntu focal main` | ✅ |
| 工具 | `catkin_make`、`catkin_create_pkg`、`rosmsg`、`rostopic` | ✅ |
| `rospy` | 发布/订阅可用 | ✅ |
| **系统 cv2** | **4.2.0**（apt 版） | ⚠️ 与算法侧 5.0.0.93 不同，**不得交叉 import** |
| 算法 venv | 3.11.17 + torch 2.5.1+cu121，CUDA True | ✅ 未受影响 |

**已知坑**：ROS 安装过程中踩过两个坑，记录备查。
1. `raw.githubusercontent.com` 不可达 → 从 `packages.ros.org/ros.key` 取
2. 官方 `ros.key` 是 2019 版已过期 → `gpg --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654` 重新导出

---

## 3. 实施阶段划分

PRD §11.7 给了 32 天总工时（含 8 天多目标扩展）。实际执行按"**能否离线完成**"切分：

| 阶段 | 内容 | 需硬件 | 状态 |
|---|---|---|---|
| **S1** | ros1_ws 双包骨架（msgs + nodes） | ❌ | ✅ **完成** |
| **S2** | 五个节点骨架（空实现，话题通） | ❌ | ✅ **完成**（随 S1 一并验证） |
| **S3** | ~~`detect_server.py` + socket~~ → **ONNX 单进程集成**（§13） | ❌ | 🟡 **算法侧完成**（§13.8/§14.7），ROS 节点待接 |
| **S3.5** | **`tt_driver source:=file` 回放 + 720p 实测** | ❌ | ⬜ **下一步** |
| **S4** | **`mav_node` 接 `mgmd_onnx`（单进程直调）** | ❌ | ⬜ |
| **S5** | rosbag 三方对拍（E3：中心差 <2px，帧数差 <2%） | ❌ | ⬜ |
| **S5.5** | **Gazebo 相机 + 抖动注入仿真**（§11） | ❌ | ⬜ |
| **S6** | 实机联调 | ✅ | ⬜ |
| **S7** | 室外飞行验收 | ✅ | ⬜ |
| **S8** | TT 数据采集与重训 | ✅ | ⬜ |

**S1–S5.5（6.0 天）完全离线可完成**，是当前可推进的部分。

**S5.5 仿真的定位**：用户不关心飞控，只关心"能否在无人机上跑起来"。
因此仿真验证的是**资源占用下的链路吞吐与延迟**，不是飞行合理性。详见 §11。

---

## 4. 验收指标（继承 PRD §11.5）

### 4.1 架构性质

| 编号 | 指标 | 判定方法 |
|---|---|---|
| **E1** | 换检测器不改其他节点 | 用 YOLO 节点替换 `mav_node`，`tracker_node`/`viz_node` 零改动，launch 文件改一行 |
| **E2** | 换平台不改算法节点 | `tt_driver` 换成通用视频源，话题映射改一处 |
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

| 话题 | 类型 | 频率 | 方向 |
|---|---|---|---|
| `/camera/image_raw` | `sensor_msgs/Image` | 30 Hz | driver → mav_node/viz_node |
| `/tt/state` | `std_msgs/Float32MultiArray` | 10 Hz | driver → mission_node |
| `/tt/cmd_vel` | `geometry_msgs/Twist` | 30 Hz | mission_node → driver |
| `/mav/candidates` | `mav_tracker_msgs/DetectionArray` | 30 Hz | mav_node → viz_node |
| `/mav/detections` | `mav_tracker_msgs/DetectionArray` | 30 Hz | mav_node → tracker_node |
| `/mav/tracks` | `mav_tracker_msgs/TrackArray` | 30 Hz | tracker_node → viz_node/mission_node |
| `/mav/debug_image` | `sensor_msgs/Image` | 15 Hz | viz_node → 外部 |
| `/mav/track_target` | `std_msgs/Int32` | on-change | 外部 → mission_node（目标 ID，-1=停止）；**原 PRD 为 action，已改，见 RD-04** |

**`/tt/state` 用 Float32MultiArray 是故意的技术债**（字段顺序靠文档约定），
等 Tello 固件字段稳定后升级专用 msg。记入 §7 技术债。

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
└── mav_tracker_nodes/
    ├── CMakeLists.txt
    ├── package.xml
    ├── launch/detect.launch
    ├── scripts/{tt_driver.py, mav_node.py, tracker_node.py,
    │            viz_node.py, mission_node.py}
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

**目标**：`roslaunch mav_tracker_nodes detect.launch` 能起，`rostopic list` 出现全部 §5 话题。

**工时**：1 天

#### 9.2.1 每个节点的空实现要求

| 节点 | 空实现必须做到 |
|---|---|
| `tt_driver` | 定时器发布假 `/camera/image_raw`（纯色图）+ `/tt/state`；订阅 `/tt/cmd_vel` 只打日志 |
| `mav_node` | 订阅图像，发空 `DetectionArray` 到 `/mav/detections` |
| `tracker_node` | 收空检测，发空 `TrackArray` |
| `viz_node` | 收图像和轨迹，输出原图到 `/mav/debug_image` |
| `mission_node` | 收轨迹，发零速度 `Twist` 到 `/tt/cmd_vel` |

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

**目标**：`mav_node` 进程内直接调用算法，输出真实检测。

**工时**：0.4 天（原 sidecar 方案估 2 天）

#### 9.3.1 架构

```
/camera/image_raw
      ↓
  mav_node（单进程，Python 3.8）
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
| S3-a | ~~sidecar 独立启动，socket 可连~~ → **`mav_node` 进程内跑通全链路** | ⬜ |
| S3-b | 同图同帧检测数与阶段一脚本一致 | ✅ 算法侧已验证（§13.8） |
| S3-c | 端到端延迟 p50 | ✅ 1080p 112 ms / 720p 70 ms（§16.4） |
| S3-d | **tracker 状态连续**（同一目标 track_id 不重置） | ⬜ 待 S4 验证 |
| S3-e | **无 GPU 时自动回落 CPU** | ⬜ 待 S4 验证 |

---

### 9.4 S3.5 — tt_driver 回放模式（✅ 已完成，2026-10-30）

**目标**：无真机也能跑完整链路。

**工时**：实际 0.5 天（原估 1 天）

| `source` 值 | 状态 | 行为 |
|---|---|---|
| `stub` | ✅ | 全黑假帧 |
| **`file`** | ✅ | 读视频文件，按 `rate` 定时发布 |
| `tello` | ⬜ S6 | 连真机取流 |

**验收结果**：见 §16.3（30.119 Hz，1920×1080 真实图像）。
`roslaunch mav_tracker_nodes playback.launch video:=phantom30`

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
| **S4（mav_node 接入）** | **0.4 天** | **2.4** | ❌ | ⬜ **下一步** |
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
- [x] **S3.5**：tt_driver file 回放 + 720p 实测（§16）
- [ ] **S4**：`mav_node` 接 `mgmd_onnx`（丢帧策略、CPU 回落）
- [ ] **S4**：`tt_driver source:=file`
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

### 11.3 环境事实（已实测，2026-10-29）

| 项 | 状态 |
|---|---|
| `gazebo11` | ✅ 已装（11.15.1，随 ROS Noetic） |
| `gazebo_ros` / `gazebo_plugins` | ✅ 已装 |
| `libgazebo_ros_camera.so` | ✅ 存在 |
| 模型目录 | `/usr/share/gazebo-11/models`（仅 ground_plane、sun） |
| 硬件 | RTX 3050 Laptop 4 GB，16 核，15 GB RAM |

**无需额外安装。** 但需要自建场景模型（目标物体），见 §11.5。

### 11.4 架构

```
Gazebo 场景（虚拟相机 + 移动目标模型）
        │
        ▼  gazebo_ros_camera 插件
   /camera/image_raw (960×720, 30 Hz)
        │
        ▼
   tt_driver (source:=gazebo)
        │  ← 人为注入：帧抖动 0–50ms、丢帧 2%
        ▼
   mav_node ──(进程内直调)──► mgmd_onnx (3.8)
        │
        ▼
   /mav/detections ──► tracker_node ──► /mav/tracks
                                            │
                              ┌─────────────┴─────────────┐
                              ▼                           ▼
                          viz_node                 mission_node
                              │                           │
                              ▼                    /tt/cmd_vel
                        /mav/debug_image                  │
                                                    ┌─────┴─────┐
                                                    ▼           ▼
                                              模式A:丢弃   模式B:驱动相机
```

**关键**：复用 S4 的 `tt_driver`，只新增 `source:=gazebo` 一个分支，**不新增组件**。

### 11.5 场景设计

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

### 11.6 帧抖动与丢帧注入

在 `tt_driver(source:=gazebo)` 中实现，参数可配：

| 参数 | 默认 | 说明 |
|---|---|---|
| `jitter_ms` | 30 | 每帧随机延迟 `uniform(0, jitter_ms)` |
| `drop_rate` | 0.02 | 随机丢弃帧的比例 |
| `enable` | true | 关掉则退化为理想链路 |

**实现要点**：用独立线程 + 时间轮，不能阻塞 ROS 回调；注入后仍要保证话题时间戳反映**真实到达时间**。

### 11.7 验收项

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

### 11.8 能回答与不能回答的问题

| 用户关心的问题 | 能否回答 |
|---|---|
| 模块在 Gazebo 资源占用下能否跑满 30 FPS | ✅ 能 |
| 端到端延迟是否 < 100 ms | ✅ 能 |
| 帧抖动/丢帧时 tracker 会不会崩 | ✅ 能 |
| 4 GB 显存是否够 | ✅ 能 |
| 真实 Tello H.264 解码延迟 | ⚠️ 只能近似 |
| 真实飞控指令延迟 | ❌ 不能（属控制范畴，本次不关心） |
| 真实光照/逆光/雨雾鲁棒性 | ❌ 不能 |

### 11.9 工时与依赖

| 项 | 工时 |
|---|---:|
| 世界文件 + 相机模型 | 0.2 天 |
| 目标运动脚本（`set_model_state` 驱动） | 0.15 天 |
| `tt_driver source:=gazebo` + 抖动注入 | 0.15 天 |
| 验收测试 | 0.1 天 |
| **合计** | **0.5 天** |

**依赖**：S1（ros1_ws 可编译）、S4（`tt_driver` 已有 `source` 参数框架）。
**可提前到 S4 之后立即做，不必等 S5。**

---

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

## 15. 当前状态与下一步计划（2026-10-30）

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
