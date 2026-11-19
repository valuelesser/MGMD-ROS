# MGMD GPU 展示版

默认回放 `phantom30.mp4`。ROS 发布带轨迹 ID 的检测框和原尺寸 1920×1080 BGR 图像，HUD 显示实际处理 FPS、跳过窗口数、各阶段均值及 MFE 分项。默认开启 LAC、LAD、双流和三帧确认后的模型跳过。

## Windows 一键展示

在本分支的仓库根目录用 PowerShell 7 执行。当前电脑已将展示代码部署到 `mgmd-ros`。启动前确保容器在运行，且 `ffplay.exe` 在 Windows PATH 中。

```powershell
pwsh -File .\mgmd-linux\scripts\showcase.ps1
```

脚本在容器内启动独立 ROS master（默认端口 11530）和 `showcase.launch`，同时打开 Windows 原生 1080p 窗口。视频播完自动退出，提前关闭窗口也会结束本次展示。端口冲突时可加 `-MasterPort 11532`。Windows 播放器接收固定尺寸的 1080p BGR 图像，其他尺寸请使用 rqt。

```powershell
# 单流，两级模型每次都运行
pwsh -File .\mgmd-linux\scripts\showcase.ps1 dual_stream:=false skip_confirmed:=false
# 只用 LAC 筛选 TF 框
pwsh -File .\mgmd-linux\scripts\showcase.ps1 enable_lad:=false
# 直接用 LAD 处理 TF 候选
pwsh -File .\mgmd-linux\scripts\showcase.ps1 enable_lac:=false
# 查看 MFE + TF 的输出
pwsh -File .\mgmd-linux\scripts\showcase.ps1 enable_lac:=false enable_lad:=false
```

## ROS 入口与参数

在已配置图形显示的 ROS 环境中，launch 会一并打开全屏 rqt，按 Esc 可切回普通窗口。当前 Windows Docker 采用上面的原生播放器。容器没有 X 显示时可加 `rqt:=false`，图像仍发布到 `/mav/debug_image`。

```bash
# 在 mgmd-ros 内，以 valuelesser 运行
source ~/mgmd-showcase/mgmd-linux/scripts/showcase_env.sh
roslaunch mav_tracker_nodes showcase.launch
# 例如，关闭复用并更换视频与源帧率
roslaunch mav_tracker_nodes showcase.launch video:=/data/videos/example.mp4 rate:=25 skip_confirmed:=false
```

| 参数 | 默认 | 对展示的影响 |
|---|---|---|
| `enable_lac` | true | 对 TF 候选运行 LAC 分类，关闭后跳过分类且不加载 LAC 模型 |
| `enable_lad` | true | 对保留的框运行 LAD 精修，关闭后不加载 LAD 模型 |
| `dual_stream` | true | 两条计算流共享一套按序更新的 TF；false 将灰度、MFE、TF 和模型放到一条计算流 |
| `skip_confirmed` | true | 同一 ID 在三个连续源帧都有 LAD 检出后，后续可用当前 TF 框和最近一次 LAD 置信度。仅当两级模型都开启时生效；false 让所有 TF 候选继续走启用的模型 |
| `video`、`rate` | `/data/videos/phantom30.mp4`、29.97002997 Hz | 填容器内视频路径及该视频源帧率；rate 控制播放节拍 |
| `decoder_leading_frames` | 默认视频 25，其他路径 0 | 对齐此前 phantom30 的测试区间；填 0 可从解码器第一帧开始 |
| `realtime`、`limit` | true、0 | 实时模式会跳过已错过播放期限的窗口；false 顺序处理全部窗口。limit 限制读入帧数，0 表示读到结尾 |

**怎样读检测结果。** `/mav/detections` 的 `source=lad` 表示 LAD 输出的精修框与置信度。只开 LAC 时，`source=lac` 使用 TF 框和 LAC 的分类概率。两级都关时，`source=tf` 使用 TF 框，置信度 0 表示没有模型评分。两级都开且复用已确认轨迹时，`source=tf` 使用最近 LAD 置信度。已确认的 ID 从有序输出中消失后要重新确认；已派发的窗口可能仍做一次推理。关闭模型的阶段耗时显示为 0。

## 做过哪些优化

| 环节 | 当前展示版采用的做法 |
|---|---|
| 解码与 MFE | OpenCV cudacodec 用 NVDEC 解码，BGR 图像在显存中直接交给 PyTorch。相邻三帧窗口共享按帧号缓存的灰度图。透视坐标在 GPU 计算，OpenCV CUDA remap 执行采样，移除了 warp 后的显式流同步。 |
| 连通域与 TF | 每条计算流独立运行 BUF 连通域，消除了该路径的共享锁。TF 的卡尔曼预测和匹配在 GPU 上运行，按窗口顺序更新同一份轨迹状态。 |
| 双流与模型 | 空闲的计算流可以接下一窗口，ROS 按序发布。LAC、LAD 使用绑定到对应 CUDA 流的 ONNX Runtime 会话。三帧确认后可省去该轨迹的两级推理。 |
| 启动与预览 | 正式回放前用 12 个窗口预热，随后重置跟踪器和灰度缓存并重放这些帧。GPU 合成框和透明 HUD，最后下载到主机并发布 ROS 图像。 |

同一台 RTX 3050 Laptop、同一 1080p 视频和 Windows FFplay 环境下，此前完整回放的双流 / 单流分别达到 **22.68 / 18.70 处理 FPS**，源帧到发布的 P95 为 **193.69 / 150.44 ms**。双流让更多窗口并行完成，也会因 GPU 竞争拉长单个窗口的 MFE 耗时。这两组均开启两级模型与三帧复用。关闭模型会改变检测语义，各模式的识别精度尚未统一评测。RANSAC 尺度常量缓存和 CUDA Graph 仅做过诊断实验。[完整对照与实验条件](docs/SPEC-PERF-CONFIRMED-SINGLE-STREAM.md)

## 在当前 Docker 环境重新部署此分支

源码位于 Windows 工作树，容器使用独立的运行副本。更新分支后，从仓库根目录执行以下 PowerShell 命令。容器使用现有 Noetic 消息工作区、CUDA 版 OpenCV、TF/BUF/HUD 动态库和 ONNX 权重。动态库的构建命令见 [GPU 环境规格](docs/SPEC-PERF-GPU.md)。

```powershell
$bundle = Join-Path $env:TEMP 'mgmd-showcase.tgz'
tar -czf $bundle -C mgmd-linux mgmd mgmd_onnx ros1_ws/src/mav_tracker scripts/showcase_env.sh scripts/build_cuda_preview.sh
docker cp $bundle mgmd-ros:/tmp/mgmd-showcase.tgz
docker exec -u valuelesser mgmd-ros bash -c 'set -e; d="$HOME/mgmd-showcase/mgmd-linux"; mkdir -p "$d"; cd "$d"; tar xzf /tmp/mgmd-showcase.tgz; sed -i "s/\r$//" scripts/*.sh ros1_ws/src/mav_tracker/mav_tracker_nodes/scripts/*.py; chmod +x ros1_ws/src/mav_tracker/mav_tracker_nodes/scripts/*.py'
```
