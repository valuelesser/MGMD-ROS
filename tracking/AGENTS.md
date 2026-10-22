# AGENTS.md

SPA2D × MVDT（Tier B 协议复现）与 MGMD 移植工程。仓库根 = 工程根：

- `spa2d/`：SPA2D 核心包（搜索/累加/峰值/置信度/状态机/指标/记录）
- `mgmd/`：MGMD 移植包（align/diff/track/classify/detect/pipeline/config/viz）
- `scripts/`：CLI（数据、训练/推理、评测、报表、权重、可视化）
- `tests/`：pytest；`configs/`：SPA2D 6 份 + MGMD 4 份 + `thresholds_frozen.yaml`
- `docs/`：SPA2D/MGMD SPEC、可视化 SPEC、论文 PDF
- `.autoresearch/issues/`：issue-001…024 任务单
- `data_raw/`、`data_derived/`、`outputs/`：**git-ignored**

## Setup（Python 3.13 / WSL2）

仓库自带 `.venv`；重建时：

```bash
python3.13 -m venv .venv
.venv/bin/python -m pip install -U pip setuptools wheel
.venv/bin/pip install -r env/requirements.txt
.venv/bin/pip install torch torchvision \
  --index-url https://download.pytorch.org/whl/cu128
.venv/bin/pip install requests setuptools seaborn
```

- 统一用 `.venv/bin/python` 跑脚本与 pytest。
- SPA2D 无 GPU 可 `--backend numpy`；MGMD 的 LAC/LAD 依赖 PyTorch，CPU 可跑但很慢。
- 权重/YOLOv5 源码：`bash scripts/fetch_mgmd_weights.sh --check`、
  `bash scripts/fetch_yolov5_src.sh --check`（缺失时去掉 `--check` 下载）。
- WSLg 可视化需要 `DISPLAY=:0`；无 GUI 时不要加 `--viz`。Qt 的
  `QFontDatabase: Cannot find font directory ...` 通常无害，不影响 `cv2.putText`。
- 完整环境说明见 `README.md` 的“环境配置（含 MGMD 与可视化）”。

## 测试与校验

```bash
.venv/bin/python -m pytest                         # 当前 376 passed, 3 skipped（约 2.5 min）
.venv/bin/python -m pytest tests/test_mgmd_viz.py -q   # 41 passed；headless，不开真实窗口
.venv/bin/python -m pytest -m "not gpu" -q
.venv/bin/python scripts/verify_env.py
.venv/bin/python -m spa2d.config configs/*.yaml    # 配置校验（非法 → 退出码 2）
```

- `pyproject.toml` 的 `pythonpath=["."]` 和 `-p no:*` 用于屏蔽 `/opt/ros/jazzy`
  注入的 pytest11 插件；不要删，否则 pytest 会因 ROS 插件报错。
- `conftest.py` 设 `XLA_PYTHON_CLIENT_PREALLOCATE=false`；手跑 SPA2D 脚本也建议设置。
- `test_run_all_report.py` 在 `configs/thresholds_frozen.yaml` 存在时自动跳过 real 模式。

## 主要脚本

```bash
# ---------- SPA2D ----------
bash scripts/fetch_data.sh --source scripts/mvdt_urls.example.txt \
  --dest data_raw/mvdt --accept-license <LICENSE>
bash scripts/fetch_mvdt_videos.sh                 # aria2 分卷下载
bash scripts/unpack_mvdt.sh                       # 7z 解包
.venv/bin/python scripts/make_derived_gt.py --seq dataset1/cam2 \
  --raw data_raw/mvdt --out data_derived/gt/D1_cam2.csv
.venv/bin/python scripts/calibrate.py --synthetic data_raw/synthetic \
  --validation synthetic --configs configs/D1_no_overlap.yaml \
  --out configs/thresholds_frozen.yaml
.venv/bin/python scripts/run_sequence.py --config configs/D1_no_overlap.yaml \
  --gt data_derived/gt/D1_cam2.csv --out outputs/runs/X
# 可选追加：--limit N --start N --repeat 3 --backend jax --allow-unfrozen
.venv/bin/python scripts/make_tables.py --runs outputs/runs --gt data_derived/gt --out outputs/metrics
.venv/bin/python scripts/make_figures.py --runs outputs/runs --gt data_derived/gt --out outputs/figures
.venv/bin/python scripts/make_report.py
.venv/bin/python scripts/make_visuals_d1d2.py
.venv/bin/python scripts/baseline_bg_subtraction.py --config configs/D1_no_overlap.yaml \
  --gt data_derived/gt/D1_cam2.csv --out outputs/runs_baselines/bgsub_X --tracker nn
.venv/bin/python scripts/benchmark_search.py --backend jax
SPA2D_RUN_ALL_MODE=smoke SPA2D_BACKEND=numpy bash scripts/run_all.sh
SPA2D_RUN_ALL_MODE=real bash scripts/run_all.sh

# ---------- MGMD ----------
bash scripts/fetch_mgmd_weights.sh --check
bash scripts/fetch_yolov5_src.sh --check
.venv/bin/python scripts/run_mgmd.py --config configs/mgmd_D3.yaml \
  --gt data_derived/gt/D3_cam3.csv --out outputs/runs_mgmd/D3_full \
  --modules mfe,tf,lac,lad
# 可选追加：--start N --limit N

# 实时可视化（需要 WSLg/X11；从 2750 跑到末尾）
.venv/bin/python scripts/run_mgmd.py --config configs/mgmd_D3.yaml \
  --gt data_derived/gt/D3_cam3.csv --out /tmp/d3_viz \
  --modules mfe,tf,lac,lad --viz --start 2750 --limit 12000

.venv/bin/python scripts/make_report_mgmd.py --runs outputs/runs_mgmd --gt data_derived/gt
```

MGMD 消融四档：`mfe`、`mfe,tf`、`mfe,tf,lac`、`mfe,tf,lac,lad`。
输出：`windows.jsonl`、`metrics.json`、`timing.json`、`run_meta.json`。

## CLI / Env 开关

### SPA2D（`run_sequence` / `run_all`）

- `SPA2D_MATCH_SIGMA` / `SPA2D_MATCH_SIGMAS`：few-pixel DoG 匹配滤波尺度。
- `SPA2D_ACCEL=1`：4D `(v,a)` 加速度搜索；配 `SPA2D_A_*`。
- `SPA2D_MIN_SPEED`：排除 `v≈0` 静态候选。
- `SPA2D_START_FRAME`、`SPA2D_LIMIT`、`SPA2D_FRAME_LIMIT`、`SPA2D_CHUNK_BYTES`。
- `SPA2D_BACKEND`、`SPA2D_RUN_ALL_MODE`。
- **GPU 上必加**：`XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS=--xla_gpu_autotune_level=0`。

### MGMD（`run_mgmd`）

- `--modules`：`mfe` / `mfe,tf` / `mfe,tf,lac` / `mfe,tf,lac,lad`（消融）。
- `--start N`：跳过前 N 帧；`--limit N`：帧上限。
- `--viz`：打开实时 OpenCV 窗口；默认关闭。
- `--viz-every N`（默认 1）、`--viz-buffer N`（默认 200）、`--viz-width N`
  （默认 1280）、`--viz-scan-max N`（默认 5000）。
- 无 GUI 请求 `--viz` → `E_VIZ_UNAVAILABLE`，退出码 4；去掉 `--viz` 即可。
- 关闭窗口等价于 `q`：提前结束、保留已处理记录/metrics，`run_meta.viz` 记录
  `aborted/abort_frame`。

## 必须知道的约定 / 坑

### SPA2D

1. **累加符号**：`A_v(x,y)=Σ_k I(x + k·v)`，不是论文 Eq.7 的 `x−k·v`；与位置公式
   `p_c=p0+(N−1)/2·v` 和 S1 自洽。
2. **粗网格**：`2·vmax % delta_c == 0`（SPEC 原文误写 `vmax % delta_c==0`）；
   local 用 `floor(rl/delta_l) >= 1`。
3. **GT 语义**：MVDT 的 `(0,0)` 表示目标未出现 → `visible=0`；文件名
   `D{1..4}_cam{2,1,3,3}.csv`；config 用 `D1/cam2`，原始目录是 `dataset1/cam2`。
4. **冻结阈值**：`run_sequence` 必须有 `configs/thresholds_frozen.yaml`（缺/冲突 → 5）；
   `--allow-unfrozen` 仅用于演示。
5. **MVDT 分卷**：多连接续传可能损坏 zip；解包前务必 `7z t`。
6. **视频读取**：`io_mvdt.FrameSource` 常驻 capture 顺序前进；不要改成每帧 open/seek。
7. **标定/评测关联不一致（已知）**：`calibrate.py` 用 `floor`，评测用
   `round_half_up`；N=8（D3/D4）差 1 帧。当前冻结阈值未重跑；重标定应统一为
   `MET.associate_frame`（SPEC §5.12）。
8. SPA2D 结果：D1/D2 的 matched 配置（DoG+尺度先验）为 100%/100%；D3 stress
   recall 0；D4 4K 全帧在 16GB GPU OOM。**不要用 SPA2D 处理 D3/D4 并宣称可用。**

### MGMD

1. **流程**：MFE（帧差运动增强）→ TF（SORT + Algorithm 1）→ LAC（外观，官方权重）
   → LAD（YOLOv5s-crop）。
2. **LAC 域失效**：官方 `Net_best.pth` 在 MVDT 上 `P(MAV)≈0`，配置
   `prob_thresh=0` 直通；LAC 消融基本无变化。**LAD 才是主要提升来源。**
3. **调参**：`frame_gap`、`alg1_d_min`、`blur`、`overflow` 等按序列适配
   （D1 gap=32，D2 gap=3，D3 gap=2，D4 gap=1）；不是零调参复现。
4. **单应方向（已修复，勿回退）**：
   - `mgmd/align.py` 的 H 是 current→prev；`mgmd/pipeline.py` 的
     `_history_alignment_homography()` 会求逆后再用于 prev→current 历史对齐。
   - gap>1 时必须估计 t−1→t 的单应，不能直接用 t−gap→t 的 H。
   - 回归测试：`tests/test_mgmd_tf.py::test_history_alignment_is_prev_to_current`、
     `tests/test_mgmd_pipeline.py::test_history_alignment_homography_inverts_align_result`。
5. **加速度记录（已修复）**：`scripts/run_sequence.py::_record` 在 accel 模式下必须使用
   `center_frame_position_accel`；回归测试 `test_record_accel_estimate_includes_acceleration`。
6. **零方差背景 PNR（已修复）**：`mu==0 and sigma==0` 时用 sentinel，不计算误导性 PNR。
7. **性能小坑（已修复）**：不要恢复 `run_sequence.py` 的死 `pp` 缓冲；`spa2d/search.py`
   用 `_frame_hw()` 读取 `frames.shape`，避免 device→host 同步。
8. 指标：bbox 中心 → 点 GT 半径匹配（10/20/50/100 px）；`metrics.json` 是主要口径。

### 可视化（`--viz`）

1. 只支持 MGMD；不得修改检测/跟踪/指标语义；默认关闭。
2. `pipeline.run(on_frame=...)` 在计时结束后调用；`--viz` 开销不计入
   `t_pre/t_search/t_state/t_window`。
3. 回调返回 `False` 表示提前结束；保留已处理记录/metrics，并写
   `run_meta.viz={enabled,every,buffer,width,viewed,jumps,fp,miss,aborted,abort_frame}`。
4. 缓冲区只存缩放后的显示帧（默认 200 帧、1280 宽）；暂停/回看/拖动不重跑推理。
5. 关窗与 `q` 等价；`CvWindow` 用递归 X11 probe 检测 WM 包装窗口。无 GUI 时
   `E_VIZ_UNAVAILABLE`/4，不能静默降级。
6. GUI 相关单测使用 `FakeWindow`，**不要在 headless CI 打开真实窗口**。
7. 无 `--viz` 时 `metrics.json`/`timing.json` schema 与语义不变；修改
   `mgmd/viz.py` 或 pipeline hook 后必须跑 `tests/test_mgmd_viz.py` 和全量 pytest。

### 通用

- 退出码：0 成功 / 2 配置非法 / 3 数据/权重缺失 / 4 运行时 / 5 冻结校验 / 130 中断。
  错误格式：`[spa2d] <code>: <msg> → <remediation>`；MGMD 用 `[mgmd]`。
- 杀进程按 PID（`pgrep` 排除 `$$`）；**不要 `pkill -f`**，会匹配自身命令行导致挂起。
- 未获用户明确要求，不做 `git commit`/`git push`；不修改 `spa2d/` 语义来做 MGMD 可视化。

## 当前结果（Tier B，点 GT + 半径匹配）

SPA2D：

- D1/D2 matched：recall/precision **100%/100%**（验证窗口，非全序列）。
- D3 stress：recall 0；D4 4K：16GB GPU OOM。

MGMD 全流水线 @100px：

| 序列 | recall | precision | F1 | AC 最佳 recall |
|---|---:|---:|---:|---:|
| D1/cam2 | 95.96 | 90.08 | 92.92 | 95.96 ✅ |
| D2/cam1 | 98.83 | 97.69 | 98.26 | 98.83 ✅ |
| D3/cam3 | 86.79 | 98.57 | 92.31 | 93.73 ✅ |
| D4/cam3 | 83.39 | 97.76 | 90.01 | 84.71 ✅ |

- D3 的 400 帧短程样例：97.38/98.53（落在连续可见段，不能代替全量）。
- D3 可视化段（2750→末尾，11446 帧）：86.98/98.73。
- D3 全量主要瓶颈是漏检（可见 6368，命中 5527，漏 841）；precision 已很高。
- 结论：D1/D2 可用 SPA2D 或 MGMD；D3/D4 用 MGMD；Tier B 结论不外推到
  onboard/运动相机，也不做论文数值等价。

## 参考文档

- `README.md`：环境配置、运行步骤、可视化命令。
- `docs/spa2d_mvdt_reproduction_spec.md`：SPA2D 数据契约、算法、部署。
- `docs/mgmd_mvdt_reproduction_spec.md`：MGMD 移植与调参记录。
- `docs/spec-tracking-visualizer.md`：可视化 D1–D14、CLI、回调、键位与验收。
- 论文：`docs/ICRA27_5428_MS.pdf`、
  `docs/Motion-guided small MAV detection in complex and non-planar scenes.pdf`
- 任务单：`.autoresearch/issues/issue-001…024-*.md`
