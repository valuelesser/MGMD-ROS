# ROS GPU 双路集成规格

版本 0.1.0，目标分支 `master`。本次从 `perf-single-stream@6e796ab` 选取 GPU 双路及其运行依赖。新入口 `track_gpu.launch` 默认用两路轮转接纳窗口，共用 TF 并按帧序更新和发布。现有 ROS 入口继续使用原有节点。

- GPU 灰度缓存按源帧号复用实际重叠图像。
- 每路的 BUF 八连通标记绑定自身 CUDA 流。双路解除应用共享锁，ORT 完成等待限制在各自 CUDA 流。
- LK 借用缓冲在 CUDA event 完成后释放。warp 后默认无显式等待。环境变量 `MGMD_SYNC_MFE_WARP=1` 可用于对照。

## 运行

沿用 Docker CUDA 环境与模型权重，从 `mgmd-linux` 编译 RTX 3050 所需两个库：

```bash
/usr/local/cuda/bin/nvcc -O3 -arch=sm_86 -shared -Xcompiler=-fPIC mgmd_onnx/cuda_buf_ccl.cu -o /tmp/mgmd-buf-ccl.so
/usr/local/cuda/bin/nvcc -O3 -arch=sm_86 --fmad=false -shared -Xcompiler=-fPIC mgmd_onnx/cuda_tf.cu -o /tmp/mgmd-tf-g3.so
roslaunch mav_tracker_nodes track_gpu.launch decoder_backend:=opencv warp_backend:=opencv \
  decoder_leading_frames:=25 profile_wall:=true
```

OpenCV Python 需包含 CUDA LK、remap 和 cudacodec，ORT 使用 CUDA EP。25 帧偏移仅适用于已验证的 `phantom30`。可选 `realtime:=false` 对照全帧吞吐。预热可通过 `warmup_frames:=12` 开启。

在已有 ROS Noetic 工作区重新运行 `catkin_make` 并加载 `devel/setup.bash`。将 `mgmd-linux` 和 CUDA OpenCV Python 模块目录加入 `PYTHONPATH`，将对应的 OpenCV、ORT/CUDA 和 NVDEC 动态库目录加入 `LD_LIBRARY_PATH`。设置 `MGMD_YOLOV5_DIR` 指向已有 YOLOv5 源码目录。可选 libav 解码库可用 `cc -O3 -shared -fPIC mgmd_onnx/cuda_decoder.c -o /tmp/mgmd-cuda-decoder.so $(pkg-config --cflags --libs libavformat libavcodec libavutil)` 编译。

## 维护与证据

参考源分支 `SPEC-PERF-SINGLE` §1.3、§1.5、§1.7–1.8、§1.13。历史全帧去同步对照为 20.49 → 23.53 FPS。之后每轮将代码、规格版本、验证结果同步提交，检查 diff 和提交范围，推送后核对远端。

0.1.0 集成验收：在现有容器中使用隔离的 ROS master、集成副本节点和现有 CUDA 动态库，按 1080p、rqt/HUD、预热 12 窗口、不限速跑完 `phantom30`。读取 1,498 帧，处理 1,496 个窗口，跳过 0 个，1,183 个窗口有检测。墙钟 64.423 秒，23.22 FPS。此轮核对运行和全片计数，未逐帧比较框和置信度。容器原始日志为 `/tmp/mgmd-master-integration.log`。
