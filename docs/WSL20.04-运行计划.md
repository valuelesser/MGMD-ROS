# 在 WSL Ubuntu 20.04 上运行本项目：调研结论与实施计划

- 文档版本：v1（2026-11-04）
- 调研对象：`毕业设计/`（仓库根，git 仓库）
- 主目标子项目：`毕业设计/tracking/`（SPA2D × MVDT + MGMD 移植）
- 结论性质：**可行性调研 + 实施计划**。本文不含已完成的安装/运行结果，所有"实测"均标注了采集方式。
- 一句话结论：**可以跑，但要绕开 3 个坑**——Python 3.13 需自带（`uv`，已可用）、MVDT 视频需重新下载（仓库里没有）、4 GB 显存 / 7 GB 内存要降配。20.04 的 glibc 2.31 不是障碍，主流轮子都能装。

---

## 0. 结论摘要（TL;DR）

| 问题 | 结论 | 依据 |
|---|---|---|
| 20.04 能不能装 Python 3.13？ | **能**，用 `uv` 装 cpython-3.13.18，34.5 s 完成 | 已在目标机执行 `uv python install 3.13` |
| JAX / PyTorch 的 CUDA 轮子能不能装？ | **能**。jaxlib 0.11.2/0.12.0.1 是 `manylinux_2_27`，torch 2.11.0+cu128 是 `manylinux_2_28`，均 ≤ glibc 2.31 | `uv pip install --dry-run` 实测解析结果（附录 B） |
| GPU 通不通？ | **通**。`/dev/dxg` 存在，`nvidia-smi` 可用，旧环境 torch 2.5.1+cu121 报 `cuda.is_available()=True` | 目标机实测 |
| GUI（`--viz`）能不能用？ | **能**。WSLg 1.0.73.2 运行中，`DISPLAY=:0`、`WAYLAND_DISPLAY=wayland-0` | `wsl env` 实测 |
| 代码要不要改？ | **不需要改源码**。只改环境、数据路径、资源参数 | 源码未发现 3.13-only 语法、无 `distutils` 依赖 |
| 真实数据齐不齐？ | **不齐**。只有 GT CSV 和权重；MVDT 视频/元数据缺（约 1.8 GB 需下载） | `data_raw/` 只有 `mgmd_weights/` |
| 最大硬约束 | **显存 4 GB + 内存 7 GB**（README 验证环境是 16 GB / 5070 级）；D4/D3 与 SPA2D 全帧要降采样或降级 numpy | 目标机实测 |
| 建议方案 | 原生 20.04 + `uv` Python 3.13 虚拟环境（方案 A），Docker 作为兜底 | 第 6 节 |
| 预计总工时 | **1.5–3 小时**（其中依赖下载 10–25 min、MVDT 下载 15–45 min，取决于网络） | 第 7 节 |

---

## 1. 范围与假设

### 1.1 "这个项目"指什么

仓库根 `毕业设计/` 下有三个部分：

| 目录 | 内容 | 状态 | 是否本次主目标 |
|---|---|---|---|
| `tracking/` | SPA2D（MVDT 复现）+ MGMD 移植 + 可视化 + 报告链，RGENTS/README 齐全 | 主线、最完整 | ✅ **是** |
| `mgmd-linux/` | 早期 MGMD×ARD-MAV 工程（含 ROS1 节点） | 旧线；git 里 ROS 仿真路线已记"中止" | ❌ 附录 C 评估 |
| `paper/` | 论文 PDF/Markdown/资产 | 参考资料 | ❌ |

**假设**：用户要跑的是 `tracking/`（它同时包含 SPA2D 与 MGMD 两条流水线，是 README/AGENTS 描述的主工程）。
若实际只想跑 `mgmd-linux`（ARD-MAV，25 GB 数据在 `C:\Users\gaodu\Desktop\毕业设计\mgmd\data`），见附录 C，工作量与本节不同。

### 1.2 本次明确不做

- 不升级 / 不替换 WSL 发行版（不装 24.04、不跑 `do-release-upgrade`）。
- 不安装 CUDA Toolkit（README 明确：只需要驱动 + pip 轮子）。
- 不修改 `spa2d/`、`mgmd/` 的算法与指标语义；不改 `configs/*` 的检测参数（只允许按资源调 `runtime.downscale`、`gpu_chunk_bytes`）。
- 不 `git commit` / `git push`。
- 不在 `/mnt/c`（NTFS 9p）上建 venv、跑 pytest、解码视频（I/O 太慢）。

---

## 2. 调研方法与证据来源

| 类别 | 做法 | 产出 |
|---|---|---|
| 读文档 | `tracking/README.md`（733 行）、`AGENTS.md`、`docs/spa2d_mvdt_reproduction_spec.md`、`env/requirements.txt`、`env/requirements.lock`、`mgmd-linux/docs/SPEC.md` | 环境需求、版本锁定、已知坑 |
| 读代码 | `pyproject.toml`、`spa2d/config.py`、`mgmd/detect.py`、`scripts/run_mgmd.py`、`scripts/verify_env.py`、`scripts/fetch_mvdt_videos.sh`、`scripts/unpack_mvdt.sh` | 依赖边界、数据缺口、CLI |
| 探目标机 | `wsl -l -v`、`--status`、`--version`；在 Ubuntu-20.04 内查 OS/glibc/CPU/RAM/GPU/工具/网络/缓存 | 第 4 节实测表 |
| 依赖可行性 | `uv python install 3.13`；`uv pip install --dry-run` 解析 `jax[cuda13]`、`torch cu128` | 附录 B 轮子标签 |
| 资产盘点 | 统计 `tracking/` 文件数与体积、`data_raw/`、`data_derived/gt/`、`third_party/` | 第 3 节 |

采集时间：2026-11-04 04:40–05:10（Asia/Shanghai）。

---

## 3. 仓库与资产盘点

### 3.1 体积

| 路径 | 体积 | 说明 |
|---|---:|---|
| `毕业设计/` 全仓 | 2 155 MB | 其中 `.git` 占 2 079 MB（历史里含大产物） |
| `毕业设计/tracking/` | 48.3 MB / 202 文件 | 其中权重 41.2 MB |
| `毕业设计/mgmd-linux/` | 13.0 MB / 97 文件 | 旧线 |
| `毕业设计/paper/` | 14.8 MB | 文档 |

> 复制到 ext4 时**建议排除 `.git`**（省 2 GB，不影响运行）；只在需要提交时才保留。

### 3.2 已有 / 缺失

| 资产 | 状态 | 细节 |
|---|---|---|
| `data_raw/mgmd_weights/` | ✅ 有 | `Net_best.pth`、`yolov5s_GLAD-crop.pt`，另有 2 个 `.onnx`；合计 41 MB |
| `third_party/yolov5-v6.0/` | ✅ 有 | 99 文件；LAD 依赖 `models.*` / `utils.*` 顶层导入可用 |
| `data_derived/gt/*.csv` | ✅ 有 | D1 8 016 帧 / D2 4 749 / D3 14 196 / D4 10 933（按 CSV 行数，含表头） |
| `configs/thresholds_frozen.yaml` | ✅ 有 | SPA2D 运行必需 |
| `configs/mgmd_D{1..4}.yaml` | ✅ 有 | D4 已带 `downscale: 0.5` |
| `.venv/` | ❌ 无 | 必须重建（Windows 侧没有，也不应复制） |
| `data_raw/mvdt/`（元数据 + 视频） | ❌ 无 | 需 `fetch_data.sh`（约 3 MB）+ `fetch_mvdt_videos.sh`（约 1.56 GB 分卷） |
| `data_raw/synthetic/` | ❌ 无 | 可选，3.9 GB；`scripts/make_synthetic.py` 可重建 |
| `outputs/` | ❌ 无 | 运行时生成 |

### 3.3 git 状态（重要）

- 仓库根 = `毕业设计/`，HEAD = `95467aa`（mgmd-linux 的 ROS 收尾提交）。
- **`mgmd-linux/` 有约 20 个未提交的改动/删除**（`detector_node.py`、`env_setup.sh`、若干 outputs）。
- `tracking/` 已被 git 跟踪，无未提交改动。
- 含义：本次工作前不要顺手 `git checkout`/`git stash`，也不要复制 `mgmd-linux` 的半成品状态去做实验；正式跑之前建议只在 `tracking/` 内工作。

---

## 4. 目标环境实测

### 4.1 宿主（Windows）

| 项 | 值 |
|---|---|
| 版本 | Windows 11 专业版 Insider Preview，build **26340** |
| 物理内存 | 15.7 GB（采集时仅 1.8 GB 空闲） |
| CPU 线程 | 16 |
| C: 可用 | 277 GB / 752 GB |
| 驱动 | NVIDIA 616.92，CUDA UMD 13.4 |
| GPU | RTX 3050 Laptop，**4 096 MiB** |

### 4.2 WSL

| 项 | 值 | 影响 |
|---|---|---|
| WSL 版本 | 2.7.11.0，内核 6.18.33.2-1-microsoft-standard-WSL2 | 新，支持 GPU 直通 |
| WSLg | **1.0.73.2**，`/mnt/wslg` 存在，`/tmp/.X11-unix/X0` 存在 | `--viz` 可用 |
| 发行版 | Ubuntu 20.04.6 LTS（Focal） | 本文主题 |
| glibc | 2.31-0ubuntu9.18 | 轮子门槛见 §5 |
| 初始化 | systemd 已启用（`systemd-private-*` 存在） | 无需额外处理 |
| CPU / 内存 | 16 vCPU / **7 GB**（WSL 默认取宿主一半，无 `.wslconfig`） | 内存是瓶颈 |
| 磁盘 | `/` 1 007 GB，可用 942 GB | 充足 |
| 用户 | `gaodu`（在 sudo 组，但 sudo **需要密码**） | 见 §4.3 |
| Python | 系统 3.8.10（缺 `distutils`，uv 会告警）；`~/.local/bin/python3.11` | 需自备 3.13 |
| uv | **0.13.6**（`~/.local/bin/uv`） | 方案 A 的基础 |
| 旧环境 | `~/毕业设计/mgmd-linux/.venv`：Python 3.11.18 + torch 2.5.1+cu121，`cuda.is_available()=True` | GPU 直通已被证明 |

### 4.3 权限与工具

| 项 | 实测 |
|---|---|
| `sudo` | `sudo: a password is required`（不能非交互安装系统包） |
| **`wsl -u root`** | **可用**（`uid=0`），`apt-get update` 正常，`apt-cache policy` 能查到 `aria2 1.35.0`、`p7zip-full 16.02`、`libegl1 1.3.2` → **不需要 sudo 密码也能装系统包** |
| 已装 | `curl`、`ffmpeg`、`git`、`libGL.so.1`、`libglib-2.0`、`libX11` |
| 缺 | `aria2c`、`7z`、`rsync`、`gcc`（后两者本次非必需） |
| 缓存 | `~/.cache/uv` **11 GB**、`~/.cache/pip` 2 GB（可能已有部分轮子，能省下载） |

### 4.4 网络（从 WSL 内测）

| 目标 | 结果 | 备注 |
|---|---|---|
| `pypi.org/simple/` | 200，但耗时 ~15 s | 慢，建议镜像 |
| `download.pytorch.org/whl/cu128/` | 200，0.5 s | 快，torch 走官方索引没问题 |
| `api.github.com/rate_limit` | 200，0.48 s | MVDT 元数据/分卷下载走这里 |
| `raw.githubusercontent.com` | 200，0.59 s | 可用 |
| `github.com`（HTML） | 200，20 s | 慢但通 |
| `pypi.tuna.tsinghua.edu.cn` | 200，0.85 s | 可作 PyPI 镜像 |
| `uv python install 3.13` | 33.8 MiB / 34.5 s | 成功，说明 GitHub releases 可达 |

> 结论：**不存在"GitHub 完全被墙"的问题**，只是网页慢；`aria2` 多连接下载 MVDT 可行。

---

## 5. 依赖与兼容性核查

### 5.1 版本基线

`env/requirements.txt` 只给下限（`jax[cuda13]>=0.7`），真正验证过的版本在 `env/requirements.lock`：

| 包 | 锁定版本 |
|---|---|
| jax / jaxlib / jax-cuda13-plugin / jax-cuda13-pjrt | **0.11.2** |
| numpy | 2.5.3 |
| opencv-python | 5.0.0.93 |
| pandas | 3.0.6 |
| scipy | 1.18.1 |
| matplotlib | 3.11.2 |
| pytest | 9.1.1 |
| torch / torchvision（不在 lock 内，README 记录） | 2.11.0+cu128 / 0.26.0+cu128 |
| nvidia-*（CUDA 13.4 运行库） | 13.4.92 |

> ⚠️ **`env/requirements.lock` 被污染**：里面混入约 130 个 ROS Jazzy 包（`rclpy`、`ros2cli`、`nav2-*`、`ament-*` 等），因为当初是在 source 过 `/opt/ros/jazzy` 的 venv 里 `pip freeze` 的。
> **禁止 `pip install -r env/requirements.lock`**。它只能当"版本参考表"用；这也是建议后续单独修的一个小问题。

### 5.2 兼容性矩阵

| 需求 | 20.04 现状 | 判定 | 处置 |
|---|---|---|---|
| OS（验证环境 24.04） | 20.04.6 | 🟡 未验证组合，但 ABI 满足 | 接受；见 §5.3 |
| Python 3.13 | 系统 3.8；`uv` 可装 3.13.18 | 🟢 | `uv venv --python 3.13` |
| `libglib2.0-0t64` | 20.04 无此包名（24.04 的 t64 迁移） | 🟡 包名差异 | 用 `libglib2.0-0` |
| glibc ≥ 2.28（torch 2.11 轮子） | 2.31 | 🟢 | — |
| jaxlib cp313 轮子 | `manylinux_2_27_x86_64` | 🟢 | 直接装 |
| torch cp313 + cu128 | `manylinux_2_28_x86_64`（torchvision 同） | 🟢 | 直接装 |
| 驱动支持 CUDA 12.8+ | 616.92 / CUDA UMD 13.4 | 🟢 | — |
| GPU 直通 `/dev/dxg` | 存在；旧 venv `cuda=True` | 🟢 | 无需装 Toolkit |
| 显存 16 GB（README 建议） | **4 GB** | 🔴 | D4 `downscale 0.5→0.33`；SPA2D 开 `XLA_PYTHON_CLIENT_PREALLOCATE=false`、调小 `gpu_chunk_bytes`；D3/D4 本就不可用 SPA2D |
| 内存（README 未写） | **7 GB**（宿主仅 1.8 GB 空闲） | 🟡 | `.wslconfig memory=12GB swap=8GB`，并关掉 Windows 侧大程序 |
| 磁盘 3.5–7.4 GB | 942 GB 可用 | 🟢 | — |
| GUI / WSLg | 1.0.73，`DISPLAY=:0` | 🟢 | 显式 `export DISPLAY=:0` |
| OpenCV GUI 版 | 待装；`libGL.so.1` 已有 | 🟢 | 装 `opencv-python`（非 headless）；备用 `libegl1 libxkbcommon-x11-0` |
| `aria2` / `7z` | 缺 | 🟢 | `wsl -u root apt-get install` |
| MVDT 视频 1.56 GB | 缺；GitHub 可达 | 🟡 | 下载 + `7z t` 校验 + 解包 |
| sudo | 需密码 | 🟢 | 用 `wsl -u root` |

### 5.3 与 24.04 的实质差异（会不会出问题）

| 差异 | 影响 | 判断 |
|---|---|---|
| glibc 2.31 vs 2.39 | 轮子标签都 ≤ 2.28/2.27，不受影响 | 无影响 |
| `libglib2.0-0t64` 改名 | 仅 apt 命令不同 | 无影响 |
| Python 3.8 为系统默认 | 不能 `python3.13 -m venv` 直接用 | 用 uv 解决 |
| OpenCV 5 / numpy 2.5 需要更新的 GLib？ | 不；轮子是自带的 | 无影响 |
| WSL 内核 6.18 + WSLg | 比 24.04 时代更新 | 有利（GPU/WSLg 更稳） |
| 包管理器（apt focal） | 已确认能查到所需包 | 无影响 |

**唯一真正的风险不是 ABI，而是资源（4 GB 显存 / 7 GB 内存）与数据获取。**

---

## 6. 方案选型

| 方案 | 做法 | 优点 | 缺点 | 结论 |
|---|---|---|---|---|
| **A. 原生 20.04 + uv Python 3.13**（推荐） | `uv venv` + pip 轮子 | 最轻；不碰系统 Python；已验证可装 3.13 与轮子；`--viz` 走 WSLg | 需要 `.wslconfig` 调内存 | ✅ 主方案 |
| B. Miniforge/conda | 装 conda 后建 3.13 环境 | 与 README 原始做法一致 | 多下载 ~200 MB；conda 求解慢；无额外收益 | 备选（A 失败时） |
| C. Docker（仓库里有 `docker-desktop` 发行版） | 用 24.04 镜像复刻验证环境 | 环境完全一致；最稳 | 镜像/数据卷额外几 GB；GPU 需 `--gpus all`；WSLg 传 X 麻烦 | 兜底 |
| D. 升级到 24.04 | `do-release-upgrade` | 一次到位 | 破坏现有环境；20.04 是 ROS/旧工具链需要；耗时不可控 | ❌ 不推荐 |

选 A 的理由：3.13 与全部轮子已实测可解析；系统包只需 3 个小包；不污染系统 Python 3.8（可能有其他脚本依赖）。

**环境目录约定**（后文命令均以此为准）：

```
WSL 侧源码：~/proj/毕业设计/tracking     （ext4）
原仓库：    /mnt/c/Users/gaodu/Desktop/opencode/mgmd_project_20261104/毕业设计
```

---

## 7. 分阶段实施计划

> 每阶段都有：目标 / 命令 / 验收 / 回滚。命令默认在 `~/proj/毕业设计/tracking` 执行：
> `export PY=.venv/bin/python`

### P0 系统准备（约 5 min）

```bash
# 1) 系统包（注意 20.04 用 libglib2.0-0，不是 -0t64）
wsl -d Ubuntu-20.04 -u root -- bash -lc "apt-get update && apt-get install -y \
  aria2 p7zip-full libgl1 libglib2.0-0 libx11-6 libxtst6 fonts-dejavu-core rsync"

# 2) 内存上调（Windows 侧写 %USERPROFILE%\.wslconfig）
# [wsl2]
# memory=12GB
# processors=12
# swap=8GB
# 然后：wsl --shutdown（会关闭所有发行版，先保存工作）
```

验收：`aria2c --version | head -1`、`7z | head -2`、`wsl -d Ubuntu-20.04 -- free -g` 显示 ≈12 GB。
回滚：卸载上述包即可；`.wslconfig` 删除后 `wsl --shutdown`。

> 若不想动 `.wslconfig`：P3 全量 pytest 与 SPA2D 跑 D1 时内存可能吃紧，会以"pytest 中途被 OOM kill"或"XLA 编译极慢"表现出来。先按默认 7 GB 试，失败再改。

### P1 源码落地到 ext4（约 2 min）

```bash
mkdir -p ~/proj
rsync -a --info=progress2 \
  --exclude '.venv' --exclude '__pycache__' --exclude '.pytest_cache' --exclude '.git' \
  "/mnt/c/Users/gaodu/Desktop/opencode/mgmd_project_20261104/毕业设计/" ~/proj/毕业设计/
```

验收：`du -sh ~/proj/毕业设计/tracking` ≈ 48 MB；`ls ~/proj/毕业设计/tracking/configs | wc -l` = 11。
回滚：`rm -rf ~/proj/毕业设计`（源仓库不受影响）。

### P2 Python 3.13 环境（约 10–25 min，取决于下载）

```bash
cd ~/proj/毕业设计/tracking
export PY=.venv/bin/python

uv python install 3.13                     # 已在本机装好，重复执行是幂等的
uv venv .venv --python 3.13 --seed         # --seed 才带 pip/setuptools/wheel

# 2.1 主依赖：按 lock 钉版本（不要直接 -r env/requirements.txt，它会拉到 jax 0.12）
uv pip install --python $PY \
  "jax[cuda13]==0.11.2" "numpy==2.5.3" "scipy==1.18.1" \
  "opencv-python==5.0.0.93" "pandas==3.0.6" "PyYAML==6.0.3" \
  "matplotlib==3.11.2" "tqdm==4.70.1" "pytest==9.1.1"

# 2.2 PyTorch（官方 cu128 索引；下载量 3 GB 量级）
uv pip install --python $PY torch==2.11.0 torchvision==0.26.0 \
  --index-url https://download.pytorch.org/whl/cu128

# 2.3 YOLOv5 v6.0 源码运行期依赖
uv pip install --python $PY requests setuptools seaborn
```

要点：
- 当前 `uv` 解析 `jax[cuda13]` 会选 **0.12.0.1**，与 lock 记录的 0.11.2 不同 → **必须钉版本**，否则可能踩 API 变更。
- PyPI 慢时可 `export UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple`（torch 那条仍走官方索引）。
- `~/.cache/uv` 已有 11 GB，可能命中缓存、实际下载远小于 3 GB。

验收：

```bash
$PY scripts/verify_env.py                 # 期望 jax backend=gpu，退出码 0
$PY - <<'PY'
import torch, jax
print(torch.__version__, torch.version.cuda, torch.cuda.is_available(),
      torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).total_memory/2**30)
print(jax.__version__, jax.default_backend(), jax.devices())
PY
```

期望：`torch 2.11.0 12.8 True NVIDIA GeForce RTX 3050 Laptop GPU ~4.0`；`jax 0.11.2 gpu [CudaDevice(id=0)]`。
回滚：`rm -rf .venv`，重来。

### P3 无数据自检（约 5–10 min）

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS=--xla_gpu_autotune_level=0
$PY -m pytest -q                       # 基线：376 passed, 3 skipped（AGENTS.md 记录）
$PY -m pytest tests/test_mgmd_viz.py -q  # 41 passed，headless，不开窗
$PY -m pytest -m "not gpu" -q
SPA2D_RUN_ALL_MODE=smoke SPA2D_BACKEND=numpy bash scripts/run_all.sh
```

验收：
- pytest 通过数 ≥ 376（有无 3 skipped 均可），**无 collection error**；
- smoke 跑完生成 `outputs/report.md`；
- 若个别用例因缺 `data_raw/mvdt` 失败，记录用例名，P4 后再复跑。

回滚：无（只产生 `outputs/`、缓存）。

### P4 数据准备（约 15–45 min）

```bash
# 4.1 元数据（3 MB，含标定/检测标签；只有想重算 GT 才必需）
bash scripts/fetch_data.sh --source scripts/mvdt_urls.example.txt \
  --dest data_raw/mvdt --accept-license <LICENSE>

# 4.2 视频分卷（≈1.56 GB；aria2 4 连接 + 断点续传；走 GitHub contents API）
bash scripts/fetch_mvdt_videos.sh

# 4.3 解包前必须校验（多连接续传可能损坏分卷）
for f in data_raw/mvdt/_archives/*/*/*.zip; do 7z t "$f" >/dev/null || echo "BAD $f"; done
bash scripts/unpack_mvdt.sh
```

验收：

```bash
ls -l data_raw/mvdt/D{1,2,3,4}/cam*/cam*.mp4        # 4 个 mp4
ffprobe -v error -select_streams v:0 -show_entries stream=width,height,nb_frames \
  -of csv=p=0 data_raw/mvdt/D3/cam3/cam3.mp4        # 1920x1080（D4 为 4K）
```

说明：
- GT 已随仓库提供，**不需要**重跑 `make_derived_gt.py`；只在想复核时执行（需要 4.1）。
- `fetch_mvdt_videos.sh` 用未认证 GitHub API（60 次/小时）；脚本自带 `rate_limit` 等待逻辑，别并发跑。
- 若某分卷反复失败：删掉该文件重跑（`aria2 -c` 续传），或把损坏分卷单独重下。

### P5 MGMD 冒烟与全序列（约 5 min + 20–60 min）

```bash
# 5.1 短程冒烟（40 帧，含 torch/yolov5 首次加载；失败能最快暴露版本不兼容）
$PY scripts/run_mgmd.py --config configs/mgmd_D3.yaml \
  --gt data_derived/gt/D3_cam3.csv --out /tmp/d3_smoke \
  --modules mfe,tf,lac,lad --start 2808 --limit 40
cat /tmp/d3_smoke/run_meta.json | head -40

# 5.2 D2 全序列（4 749 帧，1080p）
$PY scripts/run_mgmd.py --config configs/mgmd_D2.yaml \
  --gt data_derived/gt/D2_cam1.csv --out outputs/runs_mgmd/D2_full \
  --modules mfe,tf,lac,lad
```

验收（指标对照 README 基线，@100 px）：

| 序列 | 期望 recall | 期望 precision | 容差 |
|---|---:|---:|---|
| D2/cam1 | 98.83 | 97.69 | ±1.0 pp |
| D1/cam2 | 95.96 | 90.08 | ±1.0 pp |
| D3/cam3 | 86.79 | 98.57 | ±1.0 pp |
| D4/cam3 | 83.39 | 97.76 | ±1.0 pp（需 `downscale 0.5`） |

回滚：删 `outputs/runs_mgmd/*`；不影响源码。
风险点：`torch 2.11` 默认 `weights_only=True` 与 YOLOv5 v6.0 旧权重的冲突，代码注释称已处理（`mgmd/detect.py`），冒烟就是验证这一点。

### P6 SPA2D（约 20–60 min）

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS=--xla_gpu_autotune_level=0
$PY scripts/run_sequence.py --config configs/D1_no_overlap.yaml \
  --gt data_derived/gt/D1_cam2.csv --out outputs/runs/D1_no_overlap \
  --backend jax --repeat 3
```

验收：`outputs/runs/D1_no_overlap/metrics.json` 里 recall/precision@100 = 100/100。
预期问题：
- 4 GB 显存下 JAX 可能 OOM → 依次尝试：`SPA2D_LIMIT=500` 先验证链路；`configs` 里 `search.gpu_chunk_bytes` 调小；`--backend numpy`（慢，但能出数）。
- **D3/D4 不要用 SPA2D**（README/AGENTS 明确：D3 recall 0、D4 4K OOM，不是本机问题）。
- `configs/thresholds_frozen.yaml` 缺失/冲突 → 退出码 5，仓库已带。

### P7 可视化（约 5 min）

```bash
export DISPLAY=:0
$PY - <<'PY'
import cv2, numpy as np
cv2.namedWindow("GUI probe", cv2.WINDOW_NORMAL)
cv2.imshow("GUI probe", np.zeros((120, 240, 3), np.uint8))
cv2.waitKey(500); cv2.destroyAllWindows(); print("GUI_OK")
PY

$PY scripts/run_mgmd.py --config configs/mgmd_D3.yaml \
  --gt data_derived/gt/D3_cam3.csv --out /tmp/viz_D3 \
  --modules mfe,tf,lac,lad --viz --start 2750 --limit 400
```

验收：窗口出现、叠加检测框/GT/状态；`Space` 暂停、`q` 退出；`run_meta.json.viz` 有 `aborted/abort_frame`。
若报 `E_VIZ_UNAVAILABLE`（退出码 4）：`ldd` 查缺库，补 `libegl1 libxkbcommon-x11-0`，或确认 `DISPLAY=:0`。
Qt 的 `QFontDatabase: Cannot find font directory` 是无害警告，不影响 `cv2.putText`。

### P8 汇总与收尾（约 10 min）

```bash
$PY scripts/make_report_mgmd.py --runs outputs/runs_mgmd --gt data_derived/gt
# SPA2D：make_tables / make_figures / make_report
```

收尾：把「20.04 上实际跑通的命令、版本、与基线的偏差、失败项」追加到 `docs/WSL20.04-运行计划.md` 的执行记录，或新建 `docs/WSL20.04-执行记录.md`。

### 7.1 阶段汇总

| 阶段 | 内容 | 预估 | 卡点 |
|---|---|---|---|
| P0 | 系统包 + 内存 | 5 min | 需 `wsl --shutdown` |
| P1 | 源码复制 | 2 min | — |
| P2 | venv + 依赖 | 10–25 min | 网络、3 GB 下载 |
| P3 | pytest + smoke | 5–10 min | 内存 |
| P4 | MVDT 数据 | 15–45 min | GitHub 限流/分卷损坏 |
| P5 | MGMD | 5 min + 20–60 min | torch/yolov5 兼容、4 GB 显存 |
| P6 | SPA2D | 20–60 min | JAX 显存 |
| P7 | 可视化 | 5 min | WSLg/Qt 库 |
| P8 | 报告 | 10 min | — |
| **合计** | | **1.5–3 h** | |

### 7.2 先做"最小验证"的替代路径

只想要"能跑起来"的证据、不想下载 1.8 GB 数据时，做到 **P3** 即可：
`pytest` 全量 + `SPA2D_RUN_ALL_MODE=smoke` + `tests/test_mgmd_pipeline.py`（合成帧，不依赖真实视频）。
这能证明 Python 3.13 / JAX / torch / OpenCV / pytest 在 20.04 上都正常，剩下的只是数据与算力问题。

---

## 8. 风险登记表

| ID | 风险 | 概率 | 影响 | 触发信号 | 应对 |
|---|---|---|---|---|---|
| R1 | 4 GB 显存 OOM（MGMD D4 / SPA2D 1080p） | 高 | 中 | `CUDA out of memory`；XLA `RESOURCE_EXHAUSTED` | D4 `downscale 0.33`；`XLA_PYTHON_CLIENT_PREALLOCATE=false`；`gpu_chunk_bytes` 减半；SPA2D 降 `--limit` 或 numpy |
| R2 | 7 GB 内存不足（pytest/XLA 编译/视频缓冲） | 中 | 中 | 进程被 kill、swap 抖动、编译极慢 | `.wslconfig memory=12GB`；关闭 Windows 侧程序；`pytest -x` 分段跑 |
| R3 | torch 2.11 与 YOLOv5 v6.0 不兼容 | 中 | 高 | 冒烟时 `torch.load` 反序列化失败、`np.float`/`torchvision.ops` 报错 | 代码已做 `weights_only` 处理；仍失败则回退 `torch 2.5.1+cu121`（旧 venv 已验证可跑，且 2.5 支持 3.13） |
| R4 | jax 版本漂移（不钉版本会装 0.12） | 高 | 中 | import 报 `AttributeError`、编译失败 | 严格 `jax[cuda13]==0.11.2` |
| R5 | PyPI 慢导致安装超时 | 中 | 低 | `uv` 卡住/重试 | 清华镜像；`~/.cache/uv` 复用；分两次装 |
| R6 | MVDT 分卷损坏 / GitHub 限流 | 中 | 中 | `7z t` 报错；API 返回 403 | 单文件重下；脚本自带限流等待；必要时改用浏览器下载 |
| R7 | OpenCV GUI 在 WSLg 下缺库 | 中 | 低 | `E_VIZ_UNAVAILABLE`、`libEGL.so.1` 缺失 | 装 `libegl1 libxkbcommon-x11-0`；确认 `DISPLAY=:0` |
| R8 | 指标与 README 基线不一致 | 低 | 中 | 偏差 >2 pp | 先查 cv2 版本/降采样/`frame_gap`；MGMD 是确定性推理，偏差通常来自 `downscale` 与 OpenCV 版本 |
| R9 | 全量跑时间过长（D3 14 196 帧） | 高 | 低 | 单帧 >300 ms | 先 `--limit` 出指标；全量放后台 `run_in_background`，按 PID 管理（**不要 `pkill -f`**） |
| R10 | 误用 `/mnt/c` 建环境 | 中 | 中 | 安装/导入极慢，`.venv` 权限异常 | 严格按 P1 复制到 ext4 |

---

## 9. 验收清单

- [ ] `wsl -d Ubuntu-20.04 -- bash -lc 'aria2c --version; 7z | head -2'` 均能输出
- [ ] `python -V` = 3.13.18（`~/proj/毕业设计/tracking/.venv/bin/python`）
- [ ] `scripts/verify_env.py` 退出码 0，`jax.devices()` 含 `CudaDevice`
- [ ] `torch.cuda.is_available()` = True，显存识别 ≈4 GB
- [ ] `pytest -q` 通过数 ≥376，无 collection error
- [ ] `SPA2D_RUN_ALL_MODE=smoke SPA2D_BACKEND=numpy bash scripts/run_all.sh` 生成 `outputs/report.md`
- [ ] `data_raw/mvdt/D{1..4}/cam*/cam*.mp4` 四个文件存在，`ffprobe` 分辨率正确
- [ ] MGMD D3 40 帧冒烟产出 `windows.jsonl` / `metrics.json` / `timing.json` / `run_meta.json`
- [ ] MGMD D2 全序列 recall/precision @100 px 与基线差 ≤1 pp
- [ ] SPA2D D1 recall/precision@100 = 100/100
- [ ] `--viz` 能开窗、能 `q` 退出，`run_meta.viz.aborted` 正确
- [ ] 偏差与失败项已记录到执行记录文档

---

## 10. 待确认事项（开工前需要拍板）

1. **范围**：只跑 `tracking/`（本文主目标），还是连 `mgmd-linux/`（ARD-MAV，25 GB 数据在 `C:\Users\gaodu\Desktop\毕业设计\mgmd\data`）一起？
2. **系统包**：允许用 `wsl -u root` 装 `aria2 / p7zip-full / libgl1 / libegl1` 吗？（不装就只能跳过真实数据下载）
3. **真实数据**：是否要下载 MVDT 视频（≈1.56 GB，GitHub）？不下载则止步于 P3（单测 + 合成数据）。
4. **资源调整**：是否允许写 `%USERPROFILE%\.wslconfig` 并把内存提到 12 GB？需要 `wsl --shutdown`（会关掉正在跑的 WSL 任务）。
5. **可视化**：本次是否需要 `--viz`（需要 WSLg 弹窗）？CI 式无头验证可以完全跳过。

---

## 附录 A：完整命令序列（可直接复制）

```bash
# ---------- Windows 侧 ----------
wsl -d Ubuntu-20.04 -u root -- bash -lc "apt-get update && apt-get install -y aria2 p7zip-full libgl1 libglib2.0-0 libx11-6 libxtst6 fonts-dejavu-core rsync"
# （可选）%USERPROFILE%\.wslconfig 写入 memory=12GB / swap=8GB 后：
# wsl --shutdown

# ---------- WSL 侧 ----------
mkdir -p ~/proj
rsync -a --info=progress2 --exclude '.venv' --exclude '__pycache__' --exclude '.pytest_cache' --exclude '.git' \
  "/mnt/c/Users/gaodu/Desktop/opencode/mgmd_project_20261104/毕业设计/" ~/proj/毕业设计/
cd ~/proj/毕业设计/tracking
uv venv .venv --python 3.13 --seed
PY=.venv/bin/python
uv pip install --python $PY "jax[cuda13]==0.11.2" "numpy==2.5.3" "scipy==1.18.1" \
  "opencv-python==5.0.0.93" "pandas==3.0.6" "PyYAML==6.0.3" "matplotlib==3.11.2" "tqdm==4.70.1" "pytest==9.1.1"
uv pip install --python $PY torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python $PY requests setuptools seaborn
$PY scripts/verify_env.py
$PY -m pytest -q
bash scripts/fetch_mvdt_videos.sh
bash scripts/unpack_mvdt.sh
$PY scripts/run_mgmd.py --config configs/mgmd_D3.yaml --gt data_derived/gt/D3_cam3.csv \
  --out /tmp/d3_smoke --modules mfe,tf,lac,lad --start 2808 --limit 40
```

## 附录 B：实测原始证据

```text
# 1) 发行版 / 内核 / glibc
Ubuntu 20.04.6 LTS (Focal Fossa)
Linux 6.18.33.2-microsoft-standard-WSL2
ldd (Ubuntu GLIBC 2.31-0ubuntu9.18) 2.31
WSL 2.7.11.0 | WSLg 1.0.73.2 | Windows 10.0.26340

# 2) 硬件
16 vCPU | Mem 7 GB | / 1007G avail 942G | RTX 3050 Laptop 4096 MiB
nvidia-smi: NVIDIA-SMI 616.92, CUDA UMD 13.4
/dev/dxg 存在；/dev/nvidia* 不存在（WSL 正常现象）
旧 venv: Python 3.11.18 + torch 2.5.1+cu121 → cuda.is_available()=True

# 3) WSLg
DISPLAY=:0  WAYLAND_DISPLAY=wayland-0  /tmp/.X11-unix/X0  /mnt/wslg

# 4) uv / Python 3.13
uv 0.13.6
Downloading cpython-3.13.18-linux-x86_64-gnu (33.8MiB) → Installed in 34.51s

# 5) 轮子解析（cp313 @ glibc 2.31，uv --dry-run）
jaxlib-0.12.0.1-cp313-cp313-manylinux_2_27_x86_64.whl
jax_cuda13_plugin-0.12.0.1-cp313-cp313-manylinux_2_27_x86_64.whl
torch-2.11.0+cu128-cp313-cp313-manylinux_2_28_x86_64.whl
torchvision-0.26.0+cu128-cp313-cp313-manylinux_2_28_x86_64.whl
triton-3.6.0-cp313-cp313-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl
nvidia_cudnn_cu12-9.19.0.56-py3-none-manylinux_2_27_x86_64.whl
# 钉 0.11.2 后：Resolved 23 packages（jax/jaxlib/jax-cuda13-{plugin,pjrt}==0.11.2）

# 6) 网络（WSL 内 curl）
pypi.org/simple/            200  15.0 s   （慢）
download.pytorch.org/whl/cu128/ 200  0.5 s
api.github.com/rate_limit   200  0.48 s
raw.githubusercontent.com   200  0.59 s
pypi.tuna.tsinghua.edu.cn   200  0.85 s

# 7) 系统包
apt: aria2 1.35.0-1build1 | p7zip-full 16.02+dfsg-7build1 | libegl1 1.3.2
```

## 附录 C：`mgmd-linux/`（ARD-MAV）与 ROS1 的评估

| 项 | 情况 | 结论 |
|---|---|---|
| 代码年龄 | 2026-10 的工程，`docs/SPEC.md` 168 KB 记录完整 | 可复用 |
| Python | 现成 `.venv` 已在新 WSL（3.11.18 + torch 2.5.1+cu121，GPU 可用） | 开箱能跑 |
| 数据 | ARD-MAV 25 GB（107 497 XML）在 `C:\Users\gaodu\Desktop\毕业设计\mgmd\data`，经 `/mnt/c` 只读访问 | 慢但可跑 |
| 主结果 | SPEC §10.12：d_min=0.02 时 P=0.897 / R=0.541 / F1=0.675 | 已冻结 |
| ROS1 | 系统无 `/opt/ros`；ROS Noetic 只支持 20.04，但 git 记录「仿真路线中止（LAC/LAD 无法验证）」 | **不建议**作为本次目标；只在这条线明确需要时再装 |
| 与 `tracking/` 关系 | `tracking/` 已是移植后的主线（MVDT + MGMD 四模块 + 可视化） | 先跑 `tracking/` |

若确认要跑 `mgmd-linux`，需要额外的 P4'：把 `mgmd/data` 从 `/mnt/c` 软链或复制到 ext4（25 GB，建议软链先试），然后在 `~/proj/毕业设计/mgmd-linux` 用现成 `.venv` 跑 `scripts/run_testset.py`。

---

## 附：本文未做的事

- 未在 WSL 内实际 `pip install`（只做了 `uv --dry-run` 解析与 `uv python install 3.13`）。
- 未下载 MVDT 数据、未解包、未跑 pytest/MGMD/SPA2D。
- 未写 `.wslconfig`、未装 apt 包（只验证了包可用与 root 权限）。
- 未修改仓库任何源码或配置。
