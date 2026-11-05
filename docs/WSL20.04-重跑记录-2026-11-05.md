# WSL Ubuntu 20.04 重跑记录（2026-11-05）

本次复用了 `/home/gaodu/毕业设计/mgmd-linux` 的现成环境，运行 MGMD ROS1 三节点流水线。未重装依赖、未修改业务源码。WSL 与 Windows 侧源码通过现有 `28b_strict_diff.sh` 比对一致（排除依赖、构建产物、输出、辅助脚本和 catkin 生成文件）。

## 实测结果

- 系统：WSL2 Ubuntu 20.04.6；GPU：RTX 3050 Laptop，4096 MiB。
- ONNX 冒烟：`07_run_smoke.sh` 返回 `SMOKE_OK`；LAC、LAD 均包含 `CUDAExecutionProvider`。
- 输入：ARD-MAV `phantom30`，960×720，20 Hz，limit=1200，不循环。
- ROS：video_source、detector_node、visualize_node 和 Qt viewer 均正常注册。
- `/mav/detections` 采样平均频率约 18.98–19.99 Hz。
- Qt 窗口标题帧号从 520 增至 580，间隔 3 秒；后续观察为 778。
- 视频源正常发布 1200 帧后退出；检测线程共处理 1022 帧，末次周期日志记录入口丢帧 102。源发布、检测处理和视频写入计数来自不同异步环节，不能视为每帧均完成检测。
- 输出：MPEG-4，960×720，1195 帧，59.75 秒，53,283,148 字节。视频写入器收到 SIGINT 后正常关闭；预览帧可见检测框与分数。
- 收尾时发现 rospy 的 SIGTERM 处理没有退出 Qt 事件循环，已单独终止查看器，并在新脚本中加入限时退出处理。最终检查无本轮 ROS 节点或 Qt 查看器残留。

本次是运行与显示验证，不是全数据集精度评测或长时间稳定性测试。

## 结果位置

- WSL：`/home/gaodu/毕业设计/mgmd-linux/outputs/rerun_20261105_155551/`
- Windows：`毕业设计/mgmd-linux/outputs/rerun_20261105_155551/`
- 视频 `phantom30.mp4`，预览 `preview.png`，运行日志 `launch.log`，频率记录 `detection_hz.txt`，节点记录 `nodes.txt`，窗口帧号 `window.txt`。

## 再次运行

在 Windows PowerShell 执行：

```powershell
wsl -d Ubuntu-20.04 -- bash '/mnt/c/Users/gaodu/Desktop/opencode/mgmd_project_20261104/毕业设计/mgmd-linux/.dsh-run/29_rerun_verify.sh'
```

新脚本每次创建独立 WSL 输出目录，弹出 Qt 窗口，运行 1200 帧后收尾；若已有 ROS master，脚本会退出以避免打断现有任务。输出默认留在 WSL，新目录路径由脚本打印。
