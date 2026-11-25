# GPU 常驻计算实验

`spec_version: 0.1.3` · `branch: master` · 原实验分支 `exp/all-in-gpu` · 2026-11-25

## 目标和维护

在现有 mgmd-ros 容器内，减少解码至检测发布之间的主机传输和阻塞。沿用共享有序 TF、源帧率调度、三帧确认、HUD 和 ROS 消息接口。预览整帧下载不属于实验范围。

每次改动同时更新本规格、递增版本、提交代码和证据并推送本分支。记录命令、输入、环境、结果、失败和未验证项。合并主分支另行决定。以当前父提交作为 A/B 对照，使用 phantom30 的 1080p 原片，跳过开头 25 帧，预热 12 窗。固定窗口检验正确性，实时桌面检验吞吐，分别记录口径。

## 实验项目

检查 NVDEC/BGR 交接、MFE 动态连通域统计、TF 动态行数、确认缓存、LAC 裁剪常量和筛选、ORT 同流等待、LAD 图内复制及 NMS、ROS 检测元数据。区分设备传输、主机调度、GPU 跨流依赖、必须交给 ROS 的数据。

## 0.1.0 实现记录

| 环节 | 实验实现 | 保留的边界 |
|---|---|---|
| OpenCV NVDEC/BGR | 将正式读取提交到具名 CUDA 流，灰度准备在自己的流上等待解码事件；用事件保留 OpenCV 缓冲区所有者 | 启动跳过 25 帧、EOF、销毁和容器内解码器自己的工作单独核算 |
| MFE → TF | 按 BUF 的 2×2 tile 标签在显存累计面积、首像素和边界，按首像素稳定顺序返回最多 512 个框以及 GPU 上的实际计数 | 复用现有 TF 溢出报错约定。有效数可以超过 512，禁止静默截断 |
| TF → LAC | 合并同一布尔条件产生的索引，坐标、ID 和分数共享一次设备索引；每条流缓存图像边界 | 动态批次仍需获知有效数量。确认匹配另需区分新行和复用行 |
| 确认缓存 | 发布线程按中心帧顺序在 GPU 更新 ID、连续计数和置信度；计算线程持有不可变快照，并通过 CUDA 事件和 record_stream 保护读写顺序和生命周期 | 确认规则仍按有序发布推进，实时派发时哪个快照已可见受时序影响 |
| LAC / LAD 执行 | 校验 ORT 的用户 CUDA 流就是消费者流；绑定、输入、输出保留到完成事件，取消每次 ORT 调用后的显式流同步 | 计算流结束事件仍用于统计与空闲派发，单 ROI LAD 仍由主机提交 |
| LAD 后处理 | 单类别 CUDA greedy NMS，保留前 300 个框，再按原规则选择离裁剪中心最近的框，维持阈值、取整和裁边 | 每个 ROI 回固定长度一行，末尾一次压紧；模型固定为 320 像素、6300×6 输出 |
| LAD 输出头 | 可选内存内拆分现有 ONNX 的三个尺度，将其 xy、wh、obj/class 绑定 CUDA 输出，在当前流完成拼接 | 原权重文件不改写；拆头独立于其余实验开关 |
| ROS / HUD | 状态、候选数和框合并成一次检测消息回读；GPU 直接读取页锁定的 HUD 排版缓冲区 | HUD 字符仍来自 CPU，映射内存仍有主机总线流量。每次 draw 后原 download 的同步也保护下一次改写文字缓冲区。ROS1 检测消息序列化仍在 CPU |

启动时的模型加载、CUDA 常量、字形图集和预热不计入稳态零复制目标。固定批次 LAD 和填满 TF 最大容量的 LAC 仅作为独立实验。

### 复现

在 `mgmd-ros` 的 `~/mgmd-showcase/mgmd-linux` 使用主分支源码，配套权重、OpenCV、TF、BUF 和预览库沿用展示环境。

```bash
source scripts/showcase_env.sh
/usr/local/cuda/bin/nvcc --shared -Xcompiler -fPIC --fmad=false -O3 -arch=sm_86 \
  mgmd_onnx/resident_ops.cu -o /tmp/mgmd-all-in-gpu.so
roslaunch mav_tracker_nodes showcase.launch resident_library:=/tmp/mgmd-all-in-gpu.so resident_split_head:=true
```

Windows 原生预览使用本分支 `mgmd-linux/scripts/showcase.ps1`，传入同样两个 launch 参数。省略 `resident_split_head` 即保留原 LAD 输出头；省略两个参数即使用原计算路径。Linux/ROS 的 rqt 或 Windows 原生预览均使用原有 1920×1080 BGR 图像话题。

```bash
gcc -shared -fPIC scripts/resident_copy_audit.c -ldl -o /tmp/allgpu-copy-hook.so
LD_PRELOAD=/tmp/allgpu-copy-hook.so python3 scripts/verify_resident.py --transfers
LD_PRELOAD=/tmp/allgpu-copy-hook.so python3 scripts/verify_resident.py --windows 32 --split-head --controlled-conv --audit-stages
python3 scripts/verify_resident.py --windows 300
python3 scripts/verify_resident.py --windows 300 --split-head
python3 scripts/verify_resident.py --windows 300 --baseline-control
python3 scripts/verify_resident.py --windows 1496 --split-head --controlled-conv
python3 scripts/probe_resident_batches.py
LD_PRELOAD=/tmp/allgpu-copy-hook.so python3 scripts/probe_resident_slots.py --slots 4
LD_PRELOAD=/tmp/allgpu-copy-hook.so python3 scripts/probe_resident_slots.py --slots 8 --windows 1496
```

### 实时桌面实测

2026-11-24，RTX 3050 Laptop GPU，同容器、同 ROS 入口、同 Windows FFplay、透明 HUD、两条计算流、共享 TF、三帧确认。顺序为原路径、拆头、保留原头；各一次。每组读取 1498 帧。FPS 为发布处理窗口数除以正式播放墙钟时间。

| 运行配置 | 处理 / 跳过 | FPS | 源帧至发布 P50 / P95（ms） | MFE / TF / LAC / LAD 均值（ms） |
|---|---:|---:|---:|---:|
| 原路径 | 1075 / 421 | 21.49 | 139.37 / 201.71 | 46.70 / 3.47 / 10.06 / 21.31 |
| 实验，保留原输出头 | 1073 / 423 | 21.44 | 150.35 / 214.97 | 47.66 / 3.16 / 10.39 / 20.63 |
| 实验，拆头 | 1126 / 370 | 22.49 | 136.33 / 213.86 | 44.02 / 2.98 / 9.52 / 20.19 |

拆头组合本轮多处理 51 个窗口，P95 发布延迟增加 12.15 ms。各组实时选中的窗口不同，各一次测量不足以归因到某个单独改动，也不足以确定长期增益。

### 正确性与其他方案

默认 ORT 配置下，各取前 302 个解码帧，单计算流按中心帧顺序提交与确认，逐窗比较。原头和拆头两组的所有 BGR 像素、候选数、跟踪数、复用数、检测框坐标、ID 均一致，300 窗共 343 行检测，290 行复用。最终同输入的原始与映射 HUD 逐像素一致。置信度最大绝对误差为原头 0.000808、拆头 0.000804。重复原模型对照误差达到 0.001119，三组都未通过预设的 1e-5 置信度判据。保持判据，并单独使用固定卷积算法及关闭 TF32 的诊断对照。正式回放使用原 ORT 配置。

同一帧的四个真实图像裁剪、单流、预热 3 次、正式 12 次，时间含主机提交和末尾流完成。LAC 单次纯模型推理 112 行为 0.590 ms，填满 1024 行为 2.255 ms，前 112 行最大 logit 差 0.001736。LAD 原模型单目标 4.025 ms，四次串行含拼接 18.503 ms；将 Reshape 和 Resize 尺寸改为固定批 4 后为 8.105 ms。批 4 的未取整原始坐标最大差 0.23742 像素，obj 最大差 0.000075。此处只度量裁剪后的模型，未将批 4 接入默认 ROS。批次不足时无条件填满会引入额外计算。

拆头微测使用同一 CUDA 张量反复调用。原 LAD 每次出现 H2D 24、50400、24、24 字节；拆头的 ORT 和 Torch 拼接范围中未观察到 H2D/D2H。探针只覆盖被标记线程经动态链接的 CUDA runtime 调用，无法证明驱动、编解码库后台线程或硬件总线零传输。

在独立的固定卷积设置下，拆头与原路径逐窗处理完整 1498 帧 / 1496 窗；输出 1408 行，复用 1107 行。所有解码帧像素、逐窗候选数和跟踪数、复用数、最终框坐标、ID、置信度均完全一致，最大置信度差为 0。此实验串行配对消费两个实现的输出，用于数值验收。实时 FPS 使用上方默认 ORT 双流桌面回放。CUDA 非空 NMS 的重叠框、严格阈值和中心选择均通过；空掩膜计数为零，构造 1024 个有效连通域时保留首 512 个框并返回实际计数 1024。该用例验证了区域统计的溢出输入，沿用 TF 原有状态检查。

### 传输及剩余等待

同一组 32 个完整窗口、34 个解码帧、两套独立的单流状态，启用固定卷积、拆头和确认复用，末尾比较一次 HUD。两边得到 31 行检测，27 行复用，输出完全相同。按调用线程最内层的标记归因。下表为这些范围累计调用次数，不含测试脚本取值比较、ROS 发布和预览下载。

| 范围 | H2D 原 → 实验 | D2H 原 → 实验 | 显式或库内 runtime 流同步 原 → 实验 |
|---|---:|---:|---:|
| 解码 read | 0 → 0 | 0 → 0 | 34 → 0 |
| MFE 区域统计 | 32 → 0 | 64 → 0 | 96 → 0 |
| TF 输出压紧 | 0 → 0 | 64 → 32 | 64 → 32 |
| 确认分流及 detect 外层 | 54 → 0 | 135 → 60 | 189 → 60 |
| LAC 及筛选 | 30 → 0 | 60 → 30 | 120 → 30 |
| LAD 每目标推理、NMS 及末尾压紧 | 152 → 0 | 45 → 20 | 86 → 20 |
| 一次 HUD 绘制 | 1 → 0 | 0 → 0 | 0 → 0 |
| 合计 | 269 → 0 | 368 → 142 | 589 → 142 |

实验路径的 142 次 D2H 各为 4 字节，合计 568 字节，均伴随流同步。另有每边 32 次完成事件主机等待，用于返回已完成的计算流及其耗时。CPU/ROS 仍要获取最终行数据，每条检测用 6 个数值，额外一行承载状态和候选数；检测结果由一次合并回读交付。未将 ROS 订阅协议或预览下载计入上述合计。每帧预览原样下载 6,220,800 字节。CUDA 事件依赖、映射 HUD 缓冲区、主机调度与实测显式内存复制是不同开销。

### 固定槽位的整段计算实验

独立探针将 MFE 的设备计数直接交给 TF，对全部 1024 个 TF 槽位做安全裁剪和 LAC；在显存内按原顺序选择通过的轨迹，固定提交一批 LAD，GPU 完成 NMS、坐标还原和打包。只在末端读取结果以及真实有效数。此实验关闭确认复用以覆盖更重的 LAD 输入，使用单计算流；对拍复用同窗 TF 输出的动态 LAC/LAD 路径。溢出单独计数，该窗口不列入准确性通过项。预览和 ROS 耗时未列入此探针。

前 64 窗中每窗实际最多 5 个目标需要 LAD。4 槽溢出 5 窗，能容纳的 59 窗最终坐标、ID 和置信度全同。8 槽覆盖全部 64 窗，结果全同。剔除前两窗，两组从灰度到结果打包的 CUDA 事件均值为 39.19 / 47.99 ms。标记区间内仅观测到 D2D，4 槽与 8 槽分别为 1240 / 1488 次。长度回读由固定计算量取代，批满时有收益，目标稀少时会计算空槽。

完整 phantom30 的 1496 个窗口使用 8 槽复测，最多正好 8 个目标需要 LAD，溢出 0 窗，全部窗口的最终框、ID、置信度与动态参考一致。剔除前两窗，1494 窗的计算段 CUDA 事件均值 45.96 ms。在所拦截的调用中仅记录到 35856 笔 D2D；H2D、D2H、流同步和事件主机等待均为 0。CPU 仍负责发起算子，末端再把打包结果交还给主机。这是独立的单流、逐窗计算实验，未加入 ROS 默认路径。8 槽仅覆盖本片，通用版本需要在超过容量时续批或回退。

[桌面运行日志摘要、锁步比较、传输计数和槽位测试数据](../outputs/all_in_gpu_20261124/_summary.json)

## 0.1.1 ROS 单流桌面对照

2026-11-25 在同一 RTX 3050 Laptop GPU 和 `mgmd-ros` 容器顺序运行原单流、实验单流、实验双流，各完整回放一遍。三组均使用 `phantom30`、1920×1080、源速 30000/1001 Hz、12 窗预热、三帧轨迹确认、CUDA HUD 与 Windows 原生 FFplay，并发布原尺寸 BGR 预览。每组读取 1498 帧。GPU 每 200 ms 在 Windows 主机采样，取 ROS 暖机完毕至回放结束的区间，指标包含整张 GPU 的活动。实验两组均启用拆头及动态目标数。

| 配置 | 实际 FPS | 处理 / 跳过 | 发布延迟 P50 / P95（ms） | 平均 GPU 占用 | 显存峰值（MiB） | MFE / TF / LAC / LAD 均值（ms） |
|---|---:|---:|---:|---:|---:|---:|
| 原单流 | 17.40 | 870 / 626 | 104.05 / 159.14 | 50.31% | 1147 | 31.31 / 2.37 / 4.93 / 11.74 |
| 实验单流 | 20.38 | 1019 / 477 | 98.83 / 149.95 | 52.93% | 1148 | 27.83 / 2.10 / 4.12 / 9.56 |
| 实验双流 | 23.27 | 1164 / 332 | 126.97 / 201.53 | 59.38% | 1379 | 39.44 / 2.82 / 8.58 / 19.27 |

原单流到实验单流的 FPS 变化为 17.1%，发布 P95 变化为 -5.8%。实验双流对实验单流的 FPS 变化为 14.2%，发布 P95 变化为 34.4%。实时选帧和异步确认使各组处理窗口及复用次数不同，阶段均值不能隔离单项改动的贡献。

MFE 仍是三组中耗时最长的阶段。原单流至实验单流，区域统计均值由 4.375 ms 降至 1.480 ms。实验单流的 RANSAC 为 9.920 ms，实验双流为 16.023 ms。

从 Windows 的 `mgmd-linux` 目录运行。

```powershell
pwsh -File .\scripts\showcase.ps1 -MasterPort 11575 dual_stream:=false
pwsh -File .\scripts\showcase.ps1 -MasterPort 11576 dual_stream:=false resident_library:=/tmp/mgmd-all-in-gpu.so resident_split_head:=true
pwsh -File .\scripts\showcase.ps1 -MasterPort 11577 dual_stream:=true resident_library:=/tmp/mgmd-all-in-gpu.so resident_split_head:=true
```

[ROS 日志摘要及各阶段数据](../outputs/resident_single_20261125/_summary.json)

## 0.1.3 主分支展示验收

2026-11-25 从合并提交 `5261deb` 部署 `~/mgmd-showcase/mgmd-linux` 并构建 GPU 常驻库。用上一节的双流参数完整播放 `phantom30`，两条计算流共用 TF，开启 LAC、LAD、三帧确认、CUDA HUD 与 LAD 拆头。Windows 原生 1920×1080 窗口响应正常，回放结束后正常退出。

读入 1498 帧，处理 1052 窗，按实时策略跳过 444 窗。正式回放历时 50.483 秒，处理速率 20.84 FPS，发布延迟 P50 / P95 为 136.04 / 208.79 ms。MFE / TF / LAC / LAD 均值为 44.13 / 3.04 / 9.79 / 19.79 ms。本次为一次桌面验收，实验分支的 23.27 FPS 来自上一节的独立回放。[主分支验收摘要](../outputs/master_resident_showcase_20261125/_summary.json)
