# MGMD ROS 视频复现交付版

版本标识：`ros-video-v2.4.3`（与 SPEC-ROS 2.4.3 对应）。代码以该 Git 标签为准；评测归档保留各次运行时的源码提交号和原始日志。

## 交付范围

本版提供阶段一 MGMD 离线复现、ROS1 Noetic 三节点视频/本机摄像头检测与展示、15 视频实时负载评测，以及单视频降速对照。项目目前是**已配置机器上的视频演示和研究复现版本**，没有完成 TT 实机、室外飞行、正式 rosbag E3 验收或通用安装包。生产算法与启动参数在本次整理中未改动。

| 内容 | 入口或结果 |
|---|---|
| 源码与启动说明 | [README](README.md)；版本标签 `ros-video-v2.4.3` |
| 阶段一结果与口径 | [SPEC](mgmd-linux/docs/SPEC.md) §10.12–10.13，F1=0.6752 |
| ROS 实现与维护约定 | [SPEC-ROS](mgmd-linux/docs/SPEC-ROS.md) §0、§32–§34 |
| 15 视频 30 Hz 完整 ROS 链路 | [报告](mgmd-linux/outputs/tables/ros_testset_30hz_20261108.md)；P=0.8349、R=0.1083、F1=0.1917，处理覆盖率 21.11% |
| phantom70 2 Hz/30 Hz 对照 | [报告](mgmd-linux/outputs/tables/ros_nodrop_phantom70_20261108.md)；2 Hz 中心帧覆盖 100%，F1=0.8091，与同配置独立离线逐帧评测计数一致 |

论文 Table 1 的 P/R/F1=0.84/0.59/0.69 与本工程的模型、数据划分和运行条件存在差异。30 Hz 结果代表实时负载下的全帧统计；低速单视频结果不能外推到全测试集。

## 大文件归档

以下文件在仓库根目录 `deliverables/`，按 SPEC-ROS V6 不进入 Git。两份 SHA256 已核对，归档目录及关键结果文件已检查。

| 文件 | 大小 | SHA256 | 包含内容 |
|---|---:|---|---|
| `mgmd_ros_testset_30hz_20261108_v2.tar.gz` | 1,479,309,462 字节 | `557a69be8958fe1792d6934a6129e2dcebd85f870fcee1d29dc7391a98c141bf` | 15 段录制、捕获/检测结果、时间戳映射、节点日志、汇总及清单；运行源码 `570b801` |
| `mgmd_ros_nodrop_phantom70_2hz_20261108.tar.gz` | 639,863,794 字节 | `a9c85ef2687954af5f7a45c635aec84da9b228958579c807feff4d1163d4af7e` | 2 Hz 与 30 Hz 录制和结果、同配置离线框、数值审计、日志及清单；2 Hz 运行源码 `7f357ae`，30 Hz 控制运行源码见包内结果 |

各归档旁有同名 `.sha256` 和 `_MANIFEST.md`。在 `deliverables/` 执行：

```bash
sha256sum -c mgmd_ros_testset_30hz_20261108_v2.tar.gz.sha256
sha256sum -c mgmd_ros_nodrop_phantom70_2hz_20261108.tar.gz.sha256
```

## 接收后运行

当前仓库**不包含** ARD-MAV 原始视频与标注、模型权重/ONNX 文件、Python 虚拟环境和 ROS 编译产物。先按 [README 的环境与外部资产表](README.md#环境与外部资产)准备依赖；部分脚本仍引用本机绝对路径。随后可在已配置机器上执行：

```bash
cd mgmd-linux
bash scripts/demo_live.sh phantom70 6 1349 rqt
```

窗口选择 `/mav/debug_image`，结束后运行 `bash scripts/demo_live.sh stop`。GPU 环境需核对日志里 LAC 和 LAD 均使用 `CUDAExecutionProvider`。本机摄像头已验证出图与窗口，但当时没有目标验证识别准确率。

## 后续工作

ID 保持/恢复、静止目标发现和高速目标的定量诊断仍待完成；性能优化候选按用户决定暂不实施。完整待办与验收边界见 [SPEC-ROS §34](mgmd-linux/docs/SPEC-ROS.md)。
