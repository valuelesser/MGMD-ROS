# SPEC-PERF-SINGLE · ROS GPU 单流优化

`spec_version: 0.1.0` · `branch: perf-single-stream` · `base: perf-optimization@bd0da6b`

## 1. 范围与基线

从两流 HUD 版本分支，先用现有 `track_gpu.launch` 的 `legacy` 模式建立单流基线。MFE、TF、LAC、LAD 在同一应用计算路径按帧串行推进。此定义不约束 NVDEC、OpenCV 或 ONNX Runtime 内部使用的流。HUD 继续显示处理 FPS、四阶段及 MFE 分项的近两秒均值。

基线用现有 `mgmd-ros` 容器及 `phantom30` 1080p 视频，维持 30 Hz、rqt/HUD。记录明确的解码器、warp、选帧及计时设置。固定帧序列用于对拍候选、跟踪 ID、检测框和置信度。实时全片记录读帧、处理、跳过、FPS、发布延迟 p50/p95，以及分项耗时均值和 p95。两流历史结果仅作参考，后续单流改动使用本分支同配置基线比较。

### 1.1 首轮实测

```bash
roslaunch --screen mav_tracker_nodes track_gpu.launch warp_backend:=opencv \
  decoder_backend:=opencv decoder_leading_frames:=25 profile_wall:=true
```

1080p、30 Hz、rqt/HUD 和 MFE 分项计时开启。沿用 OpenCV GPU 解码环境，25 帧偏移只适用于本次 `phantom30`。单流 `legacy` 路径未开启灰度复用。2026-11-13 全片读取 1,498 帧，处理 561 个窗口，跳过 935 个，398 个窗口有检测。运行 49.982 s，处理速率 11.22 FPS，源帧到 ROS 发布延迟 p50/p95 为 118.28/151.18 ms。

| 阶段 | 均值 / p95（ms） |
|---|---:|
| MFE | 33.78 / 40.68 |
| TF | 2.27 / 3.29 |
| LAC | 6.17 / 13.02 |
| LAD | 28.03 / 43.08 |

MFE 子阶段均值分别为灰度 4.47、LK 3.23、RANSAC 9.38、warp 4.49、差分与阈值及中值 2.91、形态学 4.20、连通域 0.46、区域统计 3.85 ms。原始日志在容器 `outputs/single_stream_20261113/launch.log`。

## 2. 维护约定

继承 [SPEC-PERF §0](SPEC-PERF.md) 和 [仓库维护约定](../../AGENTS.md)。每轮改动同步更新本文版本、修正记录、验证证据，并与代码同次提交。失败实验也记录条件和结果。提交前核查 `git diff --check` 和暂存清单，提交后核对 `git show --stat HEAD`，每次推送到 `origin/perf-single-stream` 并核对远程提交。视频、模型和构建产物留在本地。涉及原 ROS 契约的修改同步维护对应规格。

## 3. 修正记录

| 日期 | 版本 | 改动 | 验证 |
|---|---|---|---|
| 2026-11-13 | 0.1.0 | 建立分支与规格，ROS GPU 默认恢复单流 | ROS 日志确认默认 `legacy`，1080p 全片 11.22 FPS，见 §1.1。 |
