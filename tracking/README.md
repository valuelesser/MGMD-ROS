# spa2d_mvdt_repro — 在 MVDT 上复现 SPA2D 与 MGMD

本仓库在 **MVDT**（Multi-view Drone Tracking Dataset，ETH Zurich）上复现两种
few-pixel / 小目标无人机检测方法，并提供可直接运行的数据准备、训练/推理、评测、报告链
与 MGMD 实时可视化：

| 方法 | 思路 | 入口 |
|---|---|---|
| **SPA2D**（Shift-Plus-Add） | 运动补偿多帧累加（track-before-detect）：对候选速度做 shift-plus-add，峰值即目标位置，配合 PNR/连续性/状态机门控 | `scripts/run_sequence.py` |
| **MGMD**（MFE → TF → LAC → LAD） | 官方两帧差分运动特征增强 → SORT 轨迹过滤（论文 Algorithm 1）→ LAC 外观分类 → LAD（YOLOv5s-crop）局部精检；使用官方预训练权重，纯推理 | `scripts/run_mgmd.py` |

- **协议级别：Tier B（协议复现）**。未获取作者 artifact，不做论文表格逐值对照；MVDT 为
  **静态地面相机代理**，不能宣称 onboard / 运动相机复现。
- 所有原始数据与权重不随仓库分发（见 `data_raw/`，git-ignored）。

---

## 功能一览

| 能力 | 说明 | 入口 / 主要产物 |
|---|---|---|
| 数据准备 | 下载/解包 MVDT，生成点 GT | `scripts/fetch_*.sh`、`scripts/make_derived_gt.py` → `data_derived/gt/` |
| SPA2D | 运动补偿多帧累加 + 峰值/置信度/状态机 | `scripts/run_sequence.py` → `outputs/runs/` |
| MGMD | MFE → TF → LAC → LAD 全流水线 | `scripts/run_mgmd.py` → `outputs/runs_mgmd/` |
| 消融 | 逐级启用 MFE/TF/LAC/LAD | `run_mgmd.py --modules ...` |
| 实时可视化 | 逐帧叠加检测、GT、状态、速度、MFE 与分数 | `run_mgmd.py --viz` |
| 指标与报告 | recall/precision/F1、消融表、AP、Markdown 报告 | `scripts/make_tables.py`、`make_report*.py` |
| 自检与测试 | 环境检查、合成数据端到端、pytest | `scripts/verify_env.py`、`pytest` |

---

## 当前结果（Tier B，MVDT 点 GT + 半径匹配）

**SPA2D**（`outputs/report.md`）：

| 序列 | 配置 | recall@100 | precision@100 | 说明 |
|---|---|---:|---:|---|
| D1/cam2 | `D1_no_overlap`（matched） | **100.00%** | **100.00%** | 需匹配滤波 DoG + 尺度先验 |
| D2/cam1 | `D2_no_overlap`（matched） | **100.00%** | **100.00%** | 同上 |
| D3/cam3 | `D3_overlap` | 0.00% | 0.00% | stress 序列，已知未达标 |
| D4/cam3 | `D4_overlap` | — | — | 4K 全帧在 16 GB GPU OOM，未跑 |

**MGMD**（`outputs/report_mgmd.md`，全流水线 @100 px）：

| 序列 | recall | precision | F1 | 目标 |
|---|---:|---:|---:|---|
| D1/cam2 | **95.96%** | 90.08% | 92.92% | ≥95% ✅ |
| D2/cam1 | **98.83%** | 97.69% | 98.26% | ≥95% ✅ |
| D3/cam3 | **86.79%** | 98.57% | 92.31% | ≥50% ✅ |
| D4/cam3 | **83.39%** | 97.76% | 90.01% | 跑通出指标 ✅（downscale 0.5） |

> 指标口径：bbox 中心 → MVDT 点 GT 的半径匹配（10/20/50/100 px）；D1–D4 的 AC 按
> “各消融档最佳 recall”核验（D1 95.96、D2 98.83、D3 93.73、D4 84.71）。

---

## 目录结构

```text
.
├── spa2d/                  SPA2D 核心包（搜索/累加/峰值/置信度/状态机/指标/记录）
├── mgmd/                   MGMD 移植包（align/diff/track/classify/detect/pipeline/config/viz）
├── scripts/                CLI：数据准备、标定、运行、评测、报表、权重下载
├── configs/                SPA2D 6 份序列 config + MGMD 4 份 config + thresholds_frozen.yaml
├── tests/                  pytest（含合成端到端冒烟）
├── docs/                   SPEC 与论文 PDF（SPA2D / MGMD）
├── .autoresearch/issues/   开发任务单（issue-001…024）
├── data_raw/               原始数据与权重（git-ignored）
├── data_derived/           派生 GT（git-ignored）
├── outputs/                runs / metrics / figures / 报告（git-ignored）
├── third_party/            YOLOv5 v6.0 源码（MGMD/LAD 需要；git-ignored）
├── env/requirements.txt    依赖清单
└── AGENTS.md               面向 AI/开发者的约定与已知坑
```

数据布局（运行核心程序前需要准备）：

```text
data_raw/
├── mvdt/
│   ├── dataset1/ … dataset4/    # 官方 metadata（detections/trajectory/cameras）
│   ├── calibration/             # 相机标定（sonyG/p20pro/mate10/...）
│   ├── D1/cam2/cam2.mp4
│   ├── D2/cam1/cam1.mp4
│   ├── D3/cam3/cam3.mp4
│   ├── D4/cam3/cam3.mp4
│   └── _archives/...            # 可选：分卷 zip（重新解包用）
├── mgmd_weights/
│   ├── Net_best.pth             # LAC 官方权重
│   └── yolov5s_GLAD-crop.pt     # LAD 官方权重
└── synthetic/                   # 合成数据（可选，3.9GB；可重新生成）
data_derived/gt/
├── D1_cam2.csv  D2_cam1.csv  D3_cam3.csv  D4_cam3.csv
third_party/yolov5-v6.0/         # MGMD LAD 源码（可 fetch_yolov5_src.sh 重建）
```

---

## 迁移到另一台电脑

先把仓库源码复制/克隆到新机，再准备数据与权重；源码本身不包含被 git-ignore 的资产。
迁移前先按下面“环境配置 §2”安装系统依赖（至少 `rsync`/`curl`/`7z`/`aria2`）。

### 1. 必须带上的内容

`.venv/`、`data_raw/`、`data_derived/`、`outputs/`、`third_party/` 不在 git 里；只复制源码
会缺数据/权重。建议从旧机复制以下目录（保持相对路径）：

| 路径 | 是否必需 | 本机大小 | 说明 |
|---|---:|---:|---|
| `data_raw/mvdt/dataset1..4` + `calibration/` | 真实数据必需 | ~3MB | 官方 metadata/标定 |
| `data_raw/mvdt/D1..D4` | 真实数据必需 | ~1.8GB | 解包后的 4 个 mp4 |
| `data_raw/mvdt/_archives` | 可选 | ~1.8GB | 只在重新解包时需要 |
| `data_raw/mgmd_weights` | MGMD 必需 | 15MB | LAC/LAD 官方权重 |
| `data_derived/gt` | 必需 | 2.4MB | 4 个点 GT CSV |
| `third_party/yolov5-v6.0` | MGMD 必需 | 1.8MB | YOLOv5 v6.0 源码 |
| `data_raw/synthetic` | 可选 | 3.9GB | 合成测试/标定；可重新生成 |
| `configs/thresholds_frozen.yaml` | SPA2D 必需 | 1KB | 仓库已带；改过则一起复制 |

**不要复制 `.venv/`、`__pycache__/`、`.pytest_cache/`**：`.venv` 绑定解释器路径和 GPU
平台，必须在新机按“环境配置”重建。`outputs/` 只是历史结果，可不迁移。

### 2. 推荐：整目录复制

```bash
OLD=/path/to/old/tracking
NEW=/path/to/new/tracking
mkdir -p "$NEW/data_raw" "$NEW/data_derived" "$NEW/third_party"
rsync -a --info=progress2 "$OLD/data_raw/mvdt/" "$NEW/data_raw/mvdt/"
rsync -a --info=progress2 "$OLD/data_raw/mgmd_weights/" "$NEW/data_raw/mgmd_weights/"
rsync -a --info=progress2 "$OLD/data_derived/gt/" "$NEW/data_derived/gt/"
rsync -a --info=progress2 "$OLD/third_party/yolov5-v6.0/" "$NEW/third_party/yolov5-v6.0/"
# 若旧机改过 configs（尤其 thresholds_frozen.yaml），再复制：
# rsync -a "$OLD/configs/" "$NEW/configs/"
```

### 3. 从零下载（没有旧数据时）

```bash
cd /path/to/new/tracking   # 换成实际仓库根目录
curl -sL -o /tmp/MVDT_LICENSE.txt \
  https://raw.githubusercontent.com/CenekAlbl/drone-tracking-datasets/master/LICENSE
bash scripts/fetch_data.sh --source scripts/mvdt_urls.example.txt \
  --dest data_raw/mvdt --accept-license /tmp/MVDT_LICENSE.txt
bash scripts/fetch_mvdt_videos.sh
bash scripts/unpack_mvdt.sh

# 注意 --seq 用 MVDT 原始目录名 dataset1..4，不是 config 别名 D1..D4
for spec in "dataset1/cam2 D1_cam2" "dataset2/cam1 D2_cam1" \
            "dataset3/cam3 D3_cam3" "dataset4/cam3 D4_cam3"; do
  set -- $spec
  .venv/bin/python scripts/make_derived_gt.py --seq "$1" --raw data_raw/mvdt \
    --out "data_derived/gt/$2.csv"
done

bash scripts/fetch_mgmd_weights.sh
bash scripts/fetch_yolov5_src.sh
```

`scripts/mvdt_urls.example.txt` 已列出 D1–D4 的 metadata/标定；`fetch_mvdt_videos.sh`
下载 4 条序列的视频分卷。若 GitHub raw/API 不通或限流，优先用上面的复制方式
（`data_raw/mvdt`、`data_raw/mgmd_weights`、`third_party/yolov5-v6.0`）。

### 4. 迁移后自检

```bash
bash scripts/fetch_data.sh --dest data_raw/mvdt --check
.venv/bin/python scripts/verify_env.py
bash scripts/fetch_mgmd_weights.sh --check
bash scripts/fetch_yolov5_src.sh --check

for f in \
  data_raw/mvdt/dataset1/cameras.txt data_raw/mvdt/dataset2/cameras.txt \
  data_raw/mvdt/dataset3/cameras.txt data_raw/mvdt/dataset4/cameras.txt \
  data_raw/mvdt/calibration/sonyG/sonyG_1.json \
  data_raw/mvdt/calibration/sony5n_1440x1080/sony5n_1440x1080.json \
  data_raw/mvdt/calibration/p20pro/p20pro.json \
  data_raw/mvdt/calibration/mate10/mate10_1.json \
  data_raw/mvdt/dataset1/detections/cam2.txt \
  data_raw/mvdt/dataset2/detections/cam1.txt \
  data_raw/mvdt/dataset3/detections/cam3.txt \
  data_raw/mvdt/dataset4/detections/cam3.txt \
  data_raw/mvdt/D1/cam2/cam2.mp4 \
  data_raw/mvdt/D2/cam1/cam1.mp4 \
  data_raw/mvdt/D3/cam3/cam3.mp4 \
  data_raw/mvdt/D4/cam3/cam3.mp4 \
  data_derived/gt/D1_cam2.csv data_derived/gt/D2_cam1.csv \
  data_derived/gt/D3_cam3.csv data_derived/gt/D4_cam3.csv \
  data_raw/mgmd_weights/Net_best.pth \
  data_raw/mgmd_weights/yolov5s_GLAD-crop.pt \
  third_party/yolov5-v6.0/models/experimental.py \
  configs/thresholds_frozen.yaml; do
  test -s "$f" || { echo "MISSING $f"; exit 1; }
done
echo "ASSETS_OK"
```

`fetch_data.sh --check` 只校验 `checksums.sha256` 中列出的 metadata；视频/GT 用上面的
`test -s` 和 `7z t`（如保留 archives）补充校验。`verify_env.py` 在纯 CPU 环境返回 3
是正常的 numpy 回退提示，不是安装失败。

最后跑一次短程 D3 和无 GUI 可视化测试：

```bash
.venv/bin/python scripts/run_mgmd.py --config configs/mgmd_D3.yaml \
  --gt data_derived/gt/D3_cam3.csv --out /tmp/d3_smoke \
  --modules mfe,tf,lac,lad --start 2808 --limit 40
.venv/bin/python -m pytest tests/test_mgmd_viz.py -q
.venv/bin/python -m pytest -q
```

---

## 环境配置（含 MGMD 与可视化）

本节给出从空系统到运行 MGMD（含 `--viz`）的完整依赖；SPA2D 共用同一 Python/JAX
环境。仓库已带可用的 `.venv` 时，可跳过已完成的步骤。

### 1. 软硬件要求

| 项 | 验证环境 / 要求 | 说明 |
|---|---|---|
| OS | Ubuntu 24.04.1（WSL2） | 原生 Linux 亦可；WSL2 需要 WSLg（Windows 11） |
| Python | 3.13.x（本机 3.13.13） | 不要用 3.12 建新 venv |
| GPU | NVIDIA RTX 5060 Ti 16GB（sm_120） | 其他 NVIDIA 卡使用对应 CUDA 轮子 |
| 驱动 | 支持 CUDA 12.8+（本机 617.42） | `nvidia-smi` 能识别 GPU |
| 显存 | 16GB 建议 | D4 4K 使用 downscale 0.5 |
| 磁盘 | 真实数据约 3.5GB；含 synthetic 约 7.4GB | 下载/解包临时空间另留 2GB |
| GUI | WSLg / X11 | `--viz` 必需；无 GUI 不要加 |

本仓库使用 pip 的 CUDA wheels（JAX/PyTorch），**不需要单独安装 CUDA Toolkit**，
只需 NVIDIA 驱动；`nvidia-smi` 能正常输出即可。

### 2. 系统依赖

```bash
sudo apt-get update
sudo apt-get install -y \
  curl aria2 p7zip-full rsync \
  libgl1 libglib2.0-0t64 libx11-6 libxtst6 \
  fonts-dejavu-core
```

- `aria2` / `7z`：下载/解包 MVDT 分卷（`fetch_mvdt_videos.sh`、`unpack_mvdt.sh`）。
- `libgl1` / `libglib2.0` / `libx11`：OpenCV Qt 窗口与关窗检测。
- `libxtst6`：仅自动按键测试用；手动 `--viz` 不强制。
- `fonts-dejavu-core`：可选，便于系统字体渲染；OpenCV Qt 的字体目录警告仍可能出现，
  不影响 `cv2.putText`。
- Ubuntu 22.04 把 `libglib2.0-0t64` 换成 `libglib2.0-0`。

### 3. Python 3.13 与基础依赖

```bash
# 本机用 Miniforge 的 python3.13；也可用 uv/pyenv/系统包安装
python3.13 -m venv .venv
.venv/bin/python -m pip install -U pip setuptools wheel
.venv/bin/pip install -r env/requirements.txt
```

- 若提示 `ensurepip`/`venv` 缺失，安装 `python3.13-venv`，或用 conda/uv 创建环境。
- 系统没有 Python 3.13 时，用 Miniforge（推荐）或 `uv python install 3.13` 安装后再建 `.venv`。
- `env/requirements.txt` 包含 JAX、OpenCV、numpy/pandas、scipy/filterpy、pytest。
- 没有 GPU 时 SPA2D 加 `--backend numpy`；JAX CUDA 轮子不可用时按注释切换
  `jax[cuda12]`（RTX 50 系仍需 CUDA 12.8+ 轮子）。

SPA2D 在 GPU 上建议设置，防止 XLA 预分配/autotune 在 1080p/4K 下 OOM：

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS=--xla_gpu_autotune_level=0
```

### 4. MGMD 依赖（PyTorch + 权重 + YOLOv5）

`env/requirements.txt` 有意注释掉 torch/torchvision（CUDA wheel 需单独索引），
MGMD 的 LAC/LAD 必须安装：

```bash
# CUDA 12.8 轮子；本机验证 torch 2.11.0+cu128 / torchvision 0.26.0+cu128
.venv/bin/pip install torch torchvision \
  --index-url https://download.pytorch.org/whl/cu128

# YOLOv5 v6.0 源码运行时依赖
.venv/bin/pip install requests setuptools seaborn
```

纯 CPU 环境把索引换成 `https://download.pytorch.org/whl/cpu`；能跑但很慢。
JAX 与 PyTorch 的 CUDA 轮子相互独立，不需要版本一致，分别按各自索引安装即可。

```bash
.venv/bin/python - <<'PY'
import torch, torchvision
print("torch", torch.__version__, "cuda", torch.version.cuda,
      "available", torch.cuda.is_available())
print("torchvision", torchvision.__version__)
PY

# 官方权重与 YOLOv5 v6.0 源码（--check 只校验；缺失时去掉 --check 下载）
bash scripts/fetch_mgmd_weights.sh --check
bash scripts/fetch_yolov5_src.sh --check
```

### 5. 可视化依赖（WSLg / X11）

`--viz` 使用 OpenCV 的 GUI 版（`opencv-python`，不是 `opencv-python-headless`）。
若环境里误装了 headless 版，先替换：

```bash
.venv/bin/pip uninstall -y opencv-python-headless
.venv/bin/pip install "opencv-python>=4.11"
```

WSL2 下确认 WSLg 正常（若没有 WSLg，先在 Windows 执行 `wsl --update` 并重启 WSL）：

```bash
echo "DISPLAY=$DISPLAY WAYLAND_DISPLAY=$WAYLAND_DISPLAY"
# 本机应为 DISPLAY=:0；没有则手动 export DISPLAY=:0
```

GUI 自检（会弹出一个 0.5 秒的小窗口）：

```bash
.venv/bin/python - <<'PY'
import cv2, numpy as np
cv2.namedWindow("GUI probe", cv2.WINDOW_NORMAL)
cv2.imshow("GUI probe", np.zeros((120, 240, 3), np.uint8))
cv2.waitKey(500)
cv2.destroyAllWindows()
print("GUI_OK")
PY
```

- 出现 `QFontDatabase: Cannot find font directory ...` 只是 Qt 自带字体目录缺失的警告，
  通常无害，不影响 `cv2.putText` 和跟踪；可忽略（安装系统字体不保证消除该警告）。
- 无 GUI / CI 不要加 `--viz`；`tests/test_mgmd_viz.py` 使用 FakeWindow，不会开窗。
- 若 `--viz` 报 `E_VIZ_UNAVAILABLE`（退出码 4）：检查 `DISPLAY`、WSLg 或 X server；
  或直接去掉 `--viz`。

### 6. 一键环境自检

```bash
nvidia-smi
.venv/bin/python scripts/verify_env.py
.venv/bin/python -c 'import cv2, jax, torch; print("cv2", cv2.__version__, "jax", jax.__version__, "torch", torch.__version__)'
bash scripts/fetch_mgmd_weights.sh --check
bash scripts/fetch_yolov5_src.sh --check
ls data_raw/mvdt/D2/cam1/cam1.mp4 data_derived/gt/D2_cam1.csv
```

全部通过后再进入“运行步骤（最短路径）”。

### 7. 常见问题

| 现象 | 处理 |
|---|---|
| `nvidia-smi: command not found` | 在 Windows 安装 NVIDIA 驱动并确认 WSL2 GPU 直通；重启 WSL |
| JAX `backend=cpu` | 检查 `jax[cuda13]`/`jax[cuda12]` 与驱动；运行 `jax.devices()` |
| `torch.cuda.is_available()` 为 `False` | 卸载 CPU 版，按 cu128 索引重装 |
| `E_VIZ_UNAVAILABLE` | 检查 `DISPLAY`、WSLg/X server；无 GUI 时去掉 `--viz` |
| `libGL.so.1: cannot open` | `sudo apt-get install -y libgl1` |
| `aria2c` / `7z` 缺失 | `sudo apt-get install -y aria2 p7zip-full` |
| Qt 字体目录警告 | 可忽略，不影响 `cv2.putText`/跟踪 |

---

## 运行步骤（最短路径）

以下命令都在仓库根目录执行，统一使用 `.venv/bin/python`。参数全集见后面的
“核心程序”章节；先跑通 MGMD，再跑 SPA2D 最容易定位问题。

### 1. 确认环境与 MGMD 资产

```bash
.venv/bin/python scripts/verify_env.py
# 缺失时去掉 --check 下载（见“迁移到另一台电脑”）
bash scripts/fetch_mgmd_weights.sh --check
bash scripts/fetch_yolov5_src.sh --check

# 真实数据示例（D2）；完整资产检查见迁移章节
ls data_raw/mvdt/D2/cam1/cam1.mp4 data_derived/gt/D2_cam1.csv
```

### 2. 跑一条 MGMD 全流水线

```bash
.venv/bin/python scripts/run_mgmd.py \
  --config configs/mgmd_D2.yaml \
  --gt data_derived/gt/D2_cam1.csv \
  --out outputs/runs_mgmd/D2_full \
  --modules mfe,tf,lac,lad
```

`--out` 目录会生成 `windows.jsonl`、`metrics.json`、`timing.json`、`run_meta.json`；
字段说明见“MGMD §2 输出物”。

### 3. 查看结果与生成报告

```bash
.venv/bin/python - <<'PY'
import json
m = json.load(open("outputs/runs_mgmd/D2_full/metrics.json"))
print("recall@100 =", m["recall"]["100"])
print("precision@100 =", m["precision"]["100"])
PY

# 四档消融都跑完后，生成汇总/对比报告
.venv/bin/python scripts/make_report_mgmd.py \
  --runs outputs/runs_mgmd --gt data_derived/gt
```

### 4. 开实时可视化（D3 示例）

```bash
.venv/bin/python scripts/run_mgmd.py \
  --config configs/mgmd_D3.yaml \
  --gt data_derived/gt/D3_cam3.csv \
  --out /tmp/viz_D3 --modules mfe,tf,lac,lad \
  --viz --start 2750 --limit 400
```

窗口出现后按 `Space` 暂停查看；按 `q` 退出并保留已处理帧的指标。完整按键/参数见
“MGMD §4 实时可视化”。无 GUI 环境请去掉 `--viz`。

### 5. 跑 SPA2D（D1 示例）

```bash
.venv/bin/python scripts/run_sequence.py \
  --config configs/D1_no_overlap.yaml \
  --gt data_derived/gt/D1_cam2.csv \
  --out outputs/runs/D1_no_overlap \
  --backend jax --repeat 3
```

SPA2D 需要 `configs/thresholds_frozen.yaml`；仓库已带冻结阈值，标定/重新冻结见
“核心程序 1：SPA2D §2”。

### 6. 提交前测试

```bash
.venv/bin/python -m pytest tests/test_mgmd_viz.py -q   # 可视化（无 GUI）
.venv/bin/python -m pytest -q                          # 全量
```

---

## 快速自检（不需要真实数据，约 3–5 分钟）

```bash
# 1) 全量单测（约 2–3 分钟）
.venv/bin/python -m pytest

# 2) 跳过 GPU 用例
.venv/bin/python -m pytest -m "not gpu" -q

# 3) SPA2D 合成数据端到端冒烟（生成配置/GT/run/表/图/报告）
SPA2D_RUN_ALL_MODE=smoke SPA2D_BACKEND=numpy bash scripts/run_all.sh

# 4) MGMD 合成端到端（pytest 内用合成帧，不依赖真实视频）
.venv/bin/python -m pytest tests/test_mgmd_pipeline.py -q

# 5) MGMD 可视化单测（headless，不会打开真实窗口）
.venv/bin/python -m pytest tests/test_mgmd_viz.py -q
```

> `run_all.sh` 的 smoke 模式会写入 `outputs/`（`outputs/report.md` 等），如已有真实结果
> 需要保留，请先备份或改用独立输出目录。

---

## 核心程序 1：SPA2D

### 1. 准备数据与派生 GT

```bash
# 合成数据（标定 / 单测 / 演示）
.venv/bin/python scripts/make_synthetic.py --suite sanity    --out data_raw/synthetic --seed 0
.venv/bin/python scripts/make_synthetic.py --suite benchmark --out data_raw/synthetic --seed 0

# 真实 MVDT：先取元数据（2D 标签 / 标定），再取视频分卷并解包
curl -sL -o /tmp/MVDT_LICENSE.txt \
  https://raw.githubusercontent.com/CenekAlbl/drone-tracking-datasets/master/LICENSE
bash scripts/fetch_data.sh --source scripts/mvdt_urls.example.txt \
  --dest data_raw/mvdt --accept-license /tmp/MVDT_LICENSE.txt
bash scripts/fetch_mvdt_videos.sh          # 需 aria2；下载到 data_raw/mvdt/_archives
bash scripts/unpack_mvdt.sh                # 需 7z；产出 data_raw/mvdt/D*/cam*/cam*.mp4

# 派生 GT（MVDT 官方 2D 标签为 primary；投影 RMSE 记 N/A）
for spec in "dataset1/cam2 D1_cam2" "dataset2/cam1 D2_cam1" \
            "dataset3/cam3 D3_cam3" "dataset4/cam3 D4_cam3"; do
  set -- $spec
  .venv/bin/python scripts/make_derived_gt.py --seq "$1" --raw data_raw/mvdt \
    --out "data_derived/gt/$2.csv"
done
```

> `--seq` 必须使用 MVDT 原始目录名 `dataset1..4`；`D1/cam2` 等只是 config/运行别名。
> `mvdt_urls.example.txt` 已包含 D1–D4 的 metadata/标定。

### 2. 标定并冻结阈值

```bash
.venv/bin/python scripts/calibrate.py --synthetic data_raw/synthetic \
  --validation synthetic --configs configs/D1_no_overlap.yaml \
  --out configs/thresholds_frozen.yaml
```

`run_sequence.py` 强制要求 `configs/thresholds_frozen.yaml`（缺失/冲突 → 退出码 5）；
仅演示时可用 `--allow-unfrozen` 走 SPEC 默认阈值。

### 3. 单序列运行

```bash
.venv/bin/python scripts/run_sequence.py \
  --config configs/D1_no_overlap.yaml \
  --gt data_derived/gt/D1_cam2.csv \
  --out outputs/runs/D1_no_overlap \
  --backend jax --repeat 3
```

常用环境变量：

| 变量 | 作用 |
|---|---|
| `SPA2D_MATCH_SIGMA` / `SPA2D_MATCH_SIGMAS` | few-pixel 匹配滤波 DoG 尺度（单/多尺度） |
| `SPA2D_ACCEL=1` | 启用 4D (v,a) 加速度搜索（配 `SPA2D_A_*`） |
| `SPA2D_MIN_SPEED` | 排除 v≈0 静态候选 |
| `SPA2D_START_FRAME` / `SPA2D_LIMIT` | 起始帧 / 帧数上限 |
| `SPA2D_CHUNK_BYTES` | 显存分块大小（OOM 时调小） |
| `SPA2D_BACKEND` / `SPA2D_RUN_ALL_MODE` | 后端 / 一键脚本模式 |

### 4. 汇总与报告

```bash
.venv/bin/python scripts/make_tables.py  --runs outputs/runs --gt data_derived/gt --out outputs/metrics
.venv/bin/python scripts/make_figures.py --runs outputs/runs --gt data_derived/gt --out outputs/figures
.venv/bin/python scripts/make_report.py                      # → outputs/report.md
```

或一键全量（6 份 config × 3 次重复 + 基线 + 表/图/报告）：

```bash
SPA2D_RUN_ALL_MODE=real bash scripts/run_all.sh
```

---

## 核心程序 2：MGMD

### 1. 官方权重与 YOLOv5 v6.0 源码

```bash
bash scripts/fetch_mgmd_weights.sh        # Net_best.pth + yolov5s_GLAD-crop.pt
bash scripts/fetch_yolov5_src.sh          # → third_party/yolov5-v6.0（unpickle 依赖）
```

两个脚本都支持 `--check` 仅校验。权重缺失 → 退出码 3。

### 2. 运行单序列 / 消融档

```bash
# 全流水线
.venv/bin/python scripts/run_mgmd.py \
  --config configs/mgmd_D1.yaml \
  --gt data_derived/gt/D1_cam2.csv \
  --out outputs/runs_mgmd/D1_mfe-tf-lac-lad \
  --modules mfe,tf,lac,lad

# 四档消融（mfe / mfe,tf / mfe,tf,lac / mfe,tf,lac,lad）
for m in mfe mfe,tf mfe,tf,lac; do
  tag=${m//,/-}
  .venv/bin/python scripts/run_mgmd.py --config configs/mgmd_D1.yaml \
    --gt data_derived/gt/D1_cam2.csv \
    --out "outputs/runs_mgmd/D1_${tag}" --modules "$m"
done
```

输出物（`--out` 目录）：

| 文件 | 内容 |
|---|---|
| `windows.jsonl` | 每个窗口一条 JSON：索引、检测/状态、MFE 帧对、匹配结果、耗时 |
| `metrics.json` | recall/precision（10/20/50/100 px）、FP、correct、visible_den、原因统计 |
| `timing.json` | `t_pre/t_search/t_state/t_window` 等分层耗时与 FPS |
| `run_meta.json` | 命令、config 快照、权重 sha256、fps、`viz` 摘要（启用时） |

`--viz` 不影响以上指标；提前退出时保留已处理帧，并在 `run_meta.json.viz` 标记
`aborted/abort_frame`。

常用参数：`--limit N`（快速验证）、`--start N`（跳过前 N 帧）、`--modules`（消融）。
每序列的 `frame_gap / or_aggregation / overflow / motion_coherence / alg1_d_min / downscale`
已在 `configs/mgmd_D*.yaml` 与 SPEC D11–D16 中登记。

### 3. 汇总与报告

```bash
.venv/bin/python scripts/make_report_mgmd.py \
  --runs outputs/runs_mgmd --gt data_derived/gt \
  --spa2d outputs/metrics/summary.csv
# → outputs/report_mgmd.md + outputs/metrics_mgmd/{mgmd_summary,mgmd_ablation,mgmd_ap}.csv
```

### 4. 实时可视化（`--viz`）

在 MGMD 推理过程中弹出 OpenCV 窗口，用于观察检测/跟踪是否合理、定位 FP/miss 和调参；
它只做显示，**不改变检测、跟踪和指标计算**，默认关闭。目前仅 MGMD 支持 `--viz`，
不提供浏览器 UI 或 MP4/GIF 导出。

前置条件：可用的图形界面（本机 WSLg 下 `DISPLAY=:0`）。无 GUI 时请去掉 `--viz`；
强行请求会报 `E_VIZ_UNAVAILABLE`（退出码 4），不会静默降级。

D3 示例（D3 从第 2808 帧附近进入连续可见区间，从 2750 起跑便于看到进入过程）：

```bash
.venv/bin/python scripts/run_mgmd.py \
  --config configs/mgmd_D3.yaml \
  --gt data_derived/gt/D3_cam3.csv \
  --out /tmp/viz_D3 --modules mfe,tf,lac,lad \
  --viz --start 2750 --limit 400
```

窗口叠加内容：

- 检测框/中心点（绿）、GT（黄，`visible=1` 时）；
- 左上信息：帧号、状态转移/原因、模块、MFE 帧对、`score/pnr/margin/single`、outcome；
- 青色速度箭头；
- 右下状态栏：`PLAY/PAUSE`、FP/miss、缓冲区范围；
- 底部进度条：在最近 `--viz-buffer` 帧内回看。

按键：

| 键 | 作用 |
|---|---|
| Space | 播放 / 暂停 |
| ← / → | 暂停态 −1 / +1 帧 |
| ↑ / ↓ 或 a / d | 暂停态 −10 / +10 帧 |
| h / e | 跳到缓冲区首 / 尾 |
| n | 向前扫描到下一个 FP/miss 并暂停 |
| p | 回退到上一个 FP/miss（缓冲内） |
| q / ESC | 结束本轮（保留已处理记录/指标） |
| 拖动进度条 | 在最近 `--viz-buffer` 帧内回看 |

关闭窗口与 `q` 等价；暂停/回看只影响显示，不会重跑推理。

| 参数 | 默认 | 说明 |
|---|---:|---|
| `--viz-every N` | 1 | 每 N 帧渲染一次，降低显示开销 |
| `--viz-buffer N` | 200 | 回看缓冲帧数（10–2000） |
| `--viz-width N` | 1280 | 显示宽度（320–3840） |
| `--viz-scan-max N` | 5000 | 扫描下一个失败帧的上限 |

- `q`/ESC/关闭窗口会提前结束：保留已处理记录和 `metrics.json`，`run_meta.viz` 记录
  `aborted=true`、`abort_frame`。
- 只缓存缩放后的显示帧（默认 200 帧、1280 宽），不保存原始视频帧。
- 无 `--viz` 时不会创建窗口/缓冲区；`metrics.json` 与 `timing.json` 语义不变。

---

## CLI 速查

| 脚本 | 用途 |
|---|---|
| `scripts/verify_env.py` | 打印 Python/JAX/OpenCV 环境与 GPU 设备 |
| `scripts/make_synthetic.py` | 生成 `sanity` / `benchmark` 合成序列 |
| `scripts/fetch_data.sh` | MVDT 元数据下载 / sha256 校验 |
| `scripts/fetch_mvdt_videos.sh` | 分卷 zip（aria2 多连接+续传） |
| `scripts/unpack_mvdt.sh` | 7z 解包 → `data_raw/mvdt/D*/cam*/cam*.mp4` |
| `scripts/make_derived_gt.py` | 生成 `data_derived/gt/*.csv`（published2d/project3d） |
| `scripts/calibrate.py` | 7 步标定 + 冻结 `thresholds_frozen.yaml` |
| `scripts/run_sequence.py` | SPA2D 单序列端到端 |
| `scripts/run_mgmd.py` | MGMD 单序列端到端（模块子集 = 消融；`--viz` 实时可视化） |
| `scripts/baseline_bg_subtraction.py` | 静止相机背景建模基线（nn/kalman） |
| `scripts/benchmark_search.py` | S3 四策略搜索对照 |
| `scripts/make_tables.py` / `make_figures.py` | SPA2D 指标表 / 图 |
| `scripts/make_report.py` / `make_report_mgmd.py` | SPA2D / MGMD 报告 |
| `scripts/run_all.sh` | SPA2D 一键复跑（real / smoke） |
| `scripts/fetch_mgmd_weights.sh` / `fetch_yolov5_src.sh` | MGMD 权重 / YOLOv5 v6.0 源码 |

---

## 配置

- SPA2D：`configs/D{1,2}_no_overlap.yaml`、`D{1,2}_overlap.yaml`、`D3_overlap.yaml`、
  `D4_overlap.yaml`；统一由 `spa2d/config.py` 校验（非法 → 退出码 2）。
- MGMD：`configs/mgmd_D{1..4}.yaml`；由 `mgmd/config.py` 校验。
- 冻结阈值：`configs/thresholds_frozen.yaml`（SPA2D 专用；MGMD 不需要）。

### 退出码（全项目一致）

| 码 | 含义 |
|---:|---|
| 0 | 成功 |
| 2 | 配置非法 |
| 3 | 数据/权重缺失或校验失败 |
| 4 | 运行时失败 |
| 5 | 冻结阈值校验失败（SPA2D） |
| 130 | 用户中断 |

错误统一格式：`[spa2d] <code>: <msg> → <remediation>` / `[mgmd] <code>: <msg> → <remediation>`。

---

## 已知限制与注意事项

1. **Tier B 协议复现**：无作者 artifact，不做论文逐值对照；FPS 仅为桌面 GPU 相对值。
2. **数据域**：MVDT 是静态地面相机，不能外推 onboard / 运动相机结论。
3. **SPA2D**：D1/D2 需要匹配滤波 DoG + 尺度先验；D3 stress 仍 0 recall；D4 4K 全帧 OOM。
4. **MGMD**：LAC 官方权重在 MVDT 上失效（P(MAV)≈0），配置 `prob_thresh=0` 直通，LAD 承担
   外观筛选；各序列 `frame_gap / alg1_d_min` 按序适配（SPEC D11–D16），不是“零调参复现”。
5. **标定关联偏差（已知）**：`calibrate.py` 用 floor 关联 GT，评测用 `round_half_up`；
   N=8（D3/D4）差 1 帧。现有冻结阈值未重跑（详见 `docs/spa2d_mvdt_reproduction_spec.md` §5.12）。
6. **冻结阈值**：SPA2D 缺/冲突时退出码 5；`--allow-unfrozen` 仅用于 smoke。
7. **MVDT 许可**：CC BY-NC-SA 4.0（非商业/研究）；原始视频与派生帧不随仓库分发。
8. **分卷 zip**：多连接续传可能损坏分卷，解包前务必 `7z t` 校验。
9. **SPA2D 累加符号**：实现为 `A_v(x,y)=Σ_k I(x+k·v)`（与 SPEC 文本 `x−k·v` 差一个符号；
   物理自洽，见 AGENTS.md）。
10. 开发/代理约定与更多坑见 **`AGENTS.md`**。

---

## 文档与任务单

- SPA2D 技术规格：`docs/spa2d_mvdt_reproduction_spec.md`
- MGMD 技术规格：`docs/mgmd_mvdt_reproduction_spec.md`
- 可视化规格：`docs/spec-tracking-visualizer.md`
- 论文：`docs/ICRA27_5428_MS.pdf`（SPA2D）、
  `docs/Motion-guided small MAV detection in complex and non-planar scenes.pdf`（MGMD）
- 开发任务单：`.autoresearch/issues/issue-001…024`
- 模块/数据/算法约定：`AGENTS.md`

## 引用与许可

如使用本仓库的复现代码，请同时引用 MVDT 数据集、SPA2D 与 MGMD 原论文。MVDT 数据遵循
**CC BY-NC-SA 4.0**；本仓库代码仅用于研究与教学。
