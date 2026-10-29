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

`spec_version: 0.1.0` | `prd_version: 4.2` | `stage1_spec: 1.0.0`（冻结）

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
- 五个 ROS 节点 + 一个 sidecar 进程
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
ROS1 主图 (Python 3.8)          sidecar (Python 3.11)
┌────────────────────┐          ┌────────────────────┐
│ tt_driver          │          │                    │
│ mav_node  ◄────socket──────►  │  detect_server.py  │
│ tracker_node       │          │  MFE→TF→LAC→LAD    │
│ viz_node           │          └────────────────────┘
│ mission_node       │
└────────────────────┘
```

**socket 协议只有两样**：帧进去，检测出来。其余全不暴露。

### 2.2 五个既定决策（PRD §11.0.1，不再讨论）

| 编号 | 决策 |
|---|---|
| Q-A | **方案 A：sidecar** |
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
| **S1** | ros1_ws 双包骨架（msgs + nodes） | ❌ | ⬜ 未开始 |
| **S2** | 五个节点骨架（空实现，话题通） | ❌ | ⬜ |
| **S3** | `detect_server.py` + socket 协议 | ❌ | ⬜ |
| **S4** | `tt_driver source:=file` 回放模式 | ❌ | ⬜ |
| **S5** | rosbag 三方对拍（P1/P2 一致性） | ❌ | ⬜ |
| **S6** | 实机联调 | ✅ | ⬜ |
| **S7** | 室外飞行验收 | ✅ | ⬜ |
| **S8** | TT 数据采集与重训 | ✅ | ⬜ |

**S1–S5 完全离线可完成**，是当前可推进的部分。

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
| `/mav/track_target` | action | on-demand | 外部 → mission_node |

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
| **RD-01** | MFE+TF 是否留在 ROS 侧 | 未规定 | **全部放 sidecar** | `step_H` 与候选顺序是内部依赖，跨解释器要暴露成协议 | ROS 侧不能单独跑 MFE 消融 |
| **RD-02** | `/tt/state` 消息类型 | 未规定 | `Float32MultiArray` | Tello 字段随固件变 | 故意的技术债 |
| **RD-03** | 多目标跟踪的 `max_age` | 论文 3 | `[待定]` | 论文的 3 是为快速丢假正例，多目标长跟踪需要更长；但阶段一证明调大无收益（§10.10） | 待实机验证 |

### 技术债（已知取舍）

| 编号 | 内容 | 偿还条件 |
|---|---|---|
| TD-01 | `/tt/state` 用数组 | Tello 字段稳定 |
| TD-02 | sidecar 协议无版本号 | 出现第二版协议时 |

---

## 8. 修正记录（每轮必填）

| 日期 | 版本 | 改动 | 原因 | 影响验收项 |
|---|---|---|---|---|
| 2026-10-29 | 0.1.0 | 建立本文件；登记阶段二基线、分支策略、实施阶段划分 | 用户要求 ROS 阶段独立 SPEC 并持续维护 | — |

---

## 9. 待办

- [ ] S1：建 `ros1_ws/src/mav_tracker/{mav_tracker_msgs, mav_tracker_nodes}`
- [ ] 把从未提交的 `ros2_ws` 消息定义迁移到 ROS1 包（4 msg + 1 action）
- [ ] S2：五个节点空实现，`roslaunch` 能起，`rostopic list` 有全部话题
- [ ] S3：`detect_server.py` socket 协议 + 与 `mgmd/pipeline.py` 对接
- [ ] S4：`tt_driver source:=file` 回放
- [ ] S5：rosbag 对拍 E3
- [ ] 清理 `ros2_ws/`（从未提交，确认删除）

---

## 10. 与阶段一 SPEC 的分工

| 文档 | 管什么 | 改不改 |
|---|---|---|
| `docs/SPEC.md`（阶段一） | MFE/TF/LAC/LAD 算法、ARD-MAV 评测、论文复现结论 | 🔒 **冻结在 1.0.0** |
| `docs/SPEC-ROS.md`（本文件） | ROS 框架、节点、话题、实机集成 | 每轮维护 |

**阶段二改动算法时怎么办？** 按 V7 不允许。必须扩展时在 `mgmd_ros/` 下包装，
若确实要改 `mgmd/`，需先解冻阶段一 SPEC 并在两个文件同时记录——**目前不需要**。
