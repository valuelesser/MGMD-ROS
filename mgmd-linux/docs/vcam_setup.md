# 虚拟摄像头安装与使用（SPEC §8.31 路径 3）

用于 **无实机无人机** 时验证实时追踪链路：把 ARD-MAV 视频推流成虚拟摄像头，
`live_track.py --camera` 读取它，行为与真实摄像头完全一致。

---

## 0. 前置检查（免 root）

```bash
cd /home/gaoduan/毕业设计/mgmd-linux
bash scripts/setup_vcam.sh --status
```

预期输出类似：

```
[vcam] 模块: 0
[vcam] 设备:
crw-rw----+ 1 root video 81, 0 ... /dev/video0
crw-rw----+ 1 root video 81, 1 ... /dev/video1
```

**本机现状（已核实）**：`v4l2loopback.ko` 内核模块**已存在**于
`/lib/modules/5.15.0-198-generic/kernel/v4l2loopback/`，
`vermagic` 与当前内核匹配，依赖模块 `videodev` 已加载。
**无需安装任何软件包，只需 root 权限加载模块。**

---

## 1. 加载虚拟摄像头（需 root，一条命令）

```bash
sudo bash scripts/setup_vcam.sh
```

这会执行：

```bash
modprobe v4l2loopback video_nr=10 card_label="MAV Test Camera" \
         exclusive_caps=1 max_buffers=8
```

成功后输出：

```
[vcam] ✅ 创建成功: /dev/video10
crw-rw---- 1 root video 81, 10 ... /dev/video10
```

### 参数说明

| 参数 | 作用 |
|---|---|
| `video_nr=10` | 设备号为 `/dev/video10`（避开已有的 video0/1） |
| `card_label` | 设备显示名 |
| `exclusive_caps=1` | 让应用正确识别为"输出型"设备（**必须**，否则 ffmpeg 写入报错） |
| `max_buffers=8` | 缓冲数，8 足够实时推流 |

### 自定义设备号

```bash
VIDEO_NR=11 sudo -E bash scripts/setup_vcam.sh
```

---

## 2. 推送视频流（普通权限即可）

**新开一个终端**（保持推流常驻）：

```bash
cd /home/gaoduan/毕业设计/mgmd-linux
bash scripts/push_vcam.sh                           # 默认 phantom70，循环
bash scripts/push_vcam.sh phantom63.mp4             # 指定视频
bash scripts/push_vcam.sh /path/to/v.mp4 --once     # 播一遍就停
```

脚本内部：

```bash
ffmpeg -re -stream_loop -1 -i <视频> -f v4l2 -pix_fmt yuv420p /dev/video10
```

**关键点 `-re`**：按原速推送。**不加 `-re` 会瞬间推完**，
摄像头那边只能读到最后一帧（表现为画面冻结）。

---

## 3. 运行实时追踪

**再开一个终端**：

```bash
cd /home/gaoduan/毕业设计/mgmd-linux
source scripts/env_setup.sh

# 弹出窗口实时显示
python3 scripts/live_track.py --camera 10 --display

# 或无头录制
python3 scripts/live_track.py --camera 10 --record vcam_out.mp4 --limit 600

# 加 TRT 加速
python3 scripts/live_track.py --camera 10 --display --lad-trt-engine "$MGMD_TRT_ENGINE"
```

**注意 `--camera 10`**：不是 `0`。`/dev/video0` 是笔记本内置摄像头（本机全黑）。

### 交互键

| 键 | 作用 |
|---|---|
| `q` / `ESC` | 退出 |
| `s` | 截图到 `outputs/live_snap_NNN.jpg` |
| `SPACE` | 暂停/继续 |
| `d` | 切换 MFE 原始候选框（黄细框）显示 |

---

## 4. 完整流程（三个终端）

```
终端 A（root，一次）     sudo bash scripts/setup_vcam.sh
终端 B（常驻推流）       bash scripts/push_vcam.sh phantom70.mp4
终端 C（运行追踪）       python3 scripts/live_track.py --camera 10 --display
```

---

## 5. 清理

```bash
# 停推流：终端 B 按 Ctrl-C
# 卸载虚拟摄像头
sudo bash scripts/setup_vcam.sh --remove

# 或重启后自动失效（模块不持久化）
```

### 开机自动加载（可选）

```bash
echo "v4l2loopback" | sudo tee /etc/modules-load.d/v4l2loopback.conf
echo 'options v4l2loopback video_nr=10 card_label="MAV Test Camera" exclusive_caps=1' \
  | sudo tee /etc/modprobe.d/v4l2loopback.conf
```

---

## 6. 常见问题

| 现象 | 原因 | 解决 |
|---|---|---|
| `modprobe: Operation not permitted` | 没有 root | 用 `sudo` |
| `modprobe: FATAL: Module v4l2loopback not found` | 模块文件缺失 | 本机已有，若真的缺失需装 `v4l2loopback-dkms` |
| `/dev/video10` 不存在 | 模块未加载成功 | `dmesg \| tail -20` 查原因 |
| ffmpeg 报 `Cannot open video device` | 未加 `exclusive_caps=1` | 重新加载模块（见上文参数） |
| live_track 画面冻结在一帧 | ffmpeg 缺 `-re` | 用 `push_vcam.sh`（已含 `-re`） |
| live_track 打开摄像头失败 | 设备号错了 | 确认是 `--camera 10` 而非 `0` |
| 推流中断后摄像头仍占用 | 前一个 ffmpeg 未退出 | `pgrep ffmpeg` 找 PID 后 `kill <PID>` |

---

## 7. 这个方案能验证什么、不能验证什么

### ✅ 能验证

| 项 | 说明 |
|---|---|
| **V4L2 采集链路** | 比"读视频文件"多验证了 OpenCV 的摄像头采集路径 |
| **实时性** | ffmpeg `-re` 按原速推流，能测真实吞吐 |
| **算法全链路** | MFE → LAC → LAD → 轨迹 → 显示 |
| **交互功能** | 截图、暂停、debug 切换 |

### ❌ 不能验证

| 项 | 原因 |
|---|---|
| **真实无人机检测** | 权重在 ARD-MAV（机载俯视）训练，普通场景必然失效 |
| **摄像头自身画质** | 虚拟摄像头绕过了真实传感器 |
| **真实环境光照/运动模糊** | 视频是既有的，不是现场采集 |

---

## 8. 与"方案 1（直接读视频文件）"的区别

| | 方案 1 视频文件 | 方案 3 虚拟摄像头 |
|---|---|---|
| 命令 | `--video x.mp4` | `--camera 10` + ffmpeg 推流 |
| 依赖 | 无 | 需 root 加载模块 |
| 验证范围 | 算法链路 | 算法链路 **+ V4L2 采集** |
| 实时性测试 | 受限（读文件太快） | ✅ 真实（ffmpeg 按原速推） |
| 适用 | 日常开发调试 | **验证摄像头采集链路 / 演示** |

**如果你只是想看检测效果，方案 1 更简单。**
**如果你想验证"摄像头输入"这条路径本身，方案 3 才有必要。**
