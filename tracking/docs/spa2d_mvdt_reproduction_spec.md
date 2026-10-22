# SPEC: SPA2D × MVDT 协议复现系统（spa2d_mvdt_repro）

> 技术规格书，派生自：`tasks/mvdt_spa2d_reproduction_prd.md`
> 生成日期：2026-10-10 | 目标分支：待建（仓库尚未初始化 git） | Commit：n/a
> 论文参考：`docs/ICRA27_5428_MS.pdf`（SPA2D: Shift-Plus-Add for ABU Operations, ICRA 2027 #5428）

---

## 1. 概述

### 1.1 本 SPEC 覆盖范围

本 SPEC 将 PRD 的 Tier B 协议复现目标转化为可实现的工程契约：一个离线 Python 包 + CLI 工具链，在 MVDT 数据集的 4 条序列上实现 SPA2D（预处理 → 滑窗 → 粗到细/局部速度搜索 → 运动补偿累加 → 峰值提取 → 置信门控 → 状态机），并产出派生 GT、冻结阈值、指标表、敏感性分析、基线对比、失败案例与复现报告。硬件验收策略按用户决策调整为：在 RTX 5060 Ti（x86 开发机）上做算法正确性与相对性能分析；Jetson 部署为可选节。

### 1.2 PRD 引用

- 来源：`tasks/mvdt_spa2d_reproduction_prd.md`（v1.0, 2026-10-10）
- 论文：`docs/ICRA27_5428_MS.pdf`（已提取全文供本 SPEC 引用，见附录 A 关键引用）
- PRD 无 User Story 编号；本 SPEC 按 PRD 章节 §4–§13 逐节映射（映射见附录 B）
- 覆盖的 PRD 验收项：§10.1 S1–S4、§10.2 Tier B 验收（FPS 门槛按用户决策调整，见 1.3 决策 6）、§13 DoD

### 1.3 设计决策汇总

| # | 决策点 | 选择 | 理由 |
|---|--------|------|------|
| 1 | 计算后端 | JAX（`jax[cuda13]`，回退 `jax[cuda12]`）为主后端；NumPy 为参考后端用于单测/无 GPU/确定性对照 | 论文用 JAX 批处理假设搜索（论文 §V）；RTX 5060 Ti 为 Blackwell sm_120，需 CUDA 13 或 CUDA 12.8+ 轮子；NumPy 后端保证单测逐值可验证 |
| 2 | 双线性采样实现 | 手写双线性 gather（floor + 4 邻居 + 权重 + 越界置零 + valid mask），不用 `map_coordinates` | 需对每个假设批处理、需 BORDER_CONSTANT=0 与 valid mask（PRD §5.3）、需与 `cv2.remap` 交叉验证；`jax.scipy.ndimage.map_coordinates` 不支持批量语义与边界计数 |
| 3 | 计算图结构 | 两遍法：第一遍只算候选得分 `S(v)`；第二遍对 `v_best` 单独累加得到 `A`/`M` 再提峰值 | 论文 §V 明确「scoring 与 localization 分离」；搜索阶段不保留全量累加图，省显存 |
| 4 | 批处理 | 逻辑批 `batch_logical=512`，物理按显存预算自动分块 `gpu_chunk`，跨 chunk 做 max 归约；分块不改变结果 | PRD 约束 Batch=512；D4 3840×2160 下 512 假设无法整批驻留 16GB（见 §8.3 内存模型），必须分块 |
| 5 | 状态表示 | 规范状态 7 个：`INIT / TRACKED / TRACKED_LOW / REACQ / LOST / OUT_OF_FRAME / ABSENT`；`STAT`、`EDGE` 为布尔属性而非独立状态 | PRD §5.6「状态附加 STAT=True」「edge_candidate」语义为属性；消解枚举歧义，转换表见 §5.6 |
| 6 | FPS 验收 | RTX 5060 Ti 上只做相对分析（S3 四策略对照 + 分辨率/scaling 分析）；PRD §10.2 的绝对 FPS 下限（20/12/8）转为「报告项」；Jetson 章为可选 | 用户决策：无可用的 Jetson AGX Orin；PRD §7 本身允许非同类 GPU 只做正确性 |
| 7 | 阈值标定产物 | `thresholds_frozen.yaml` 存全局 θ；`width_ref` 为每序列派生量，取该序列前 `min(500, F)` 个 GT visible 帧的中位等效宽度（不读取得分，无泄漏） | PRD §9 只对 D1/D3 划验证片，D2/D4 也需要 `width_ref`；GT 派生量与测试调参无关，可预先固定规则（§9.6） |
| 8 | C1 CNN 退化基线 | 不在范围内 | 用户决策；PRD 标为可选 |
| 9 | 数据获取 | `scripts/fetch_data.sh`：支持本地目录/URL 清单，断点续传、sha256 校验、`--check` 模式；缺失时以合成数据先行开发 | 用户决策；MVDT 下载源可能失效（PRD §12 风险 1） |
| 10 | 帧缓存 | 设备端窗口帧缓存（保留当前窗口 N 帧的 device array，天然复用重叠帧）+ 主机端预处理器 LRU 缓存（默认 2GB） | 避免重叠窗口重复预处理/重复 H2D 传输；D4 单帧 33MB，窗口驻留仅 ~265MB |
| 11 | 半帧关联 | 默认 round-half-up；同时输出 floor 关联敏感性 | PRD §3 窗口中心时间定义 + §6 敏感性要求 |
| 12 | 环境管理 | miniforge python3.13 venv + pip；`env/requirements.txt`（直接依赖，带版本上限）+ `env/requirements.lock`（pip freeze 全量） | 本机现状：miniforge python 3.13.13、pip 26.1.2、无 uv；磁盘余量 767GB 足够数据 |
| 13 | 仓库初始化 | I-01 中 `git init`，`data_raw/ data_derived/ outputs/ *.pdf` 进 `.gitignore` | 隐私/许可；避免大体量文件入库 |
| 14 | 累加符号 | `A_v=Σ I(x + k·v)`（非论文 `x−k·v`） | 与 S1（返回 (0.7,0)）、Eq.9 位置公式、连续性模型自洽（§5.2 符号修正） |
| 15 | 候选表示 | 增加 few-pixel **DoG 匹配滤波 + 尺度先验**（`matched_filter(_multiscale)`，frozen `preprocess.matched_sigma=2.6667`） | 真实 MVDT 纯形态学暗增强被静止暗杂波锁死（recall 0）；DoG+尺度可分离小目标与大杂波，D1/D2 recall/precision@100=1.0 |
| 16 | 搜索扩展 | `min_speed`（运动相干，排除 v≈0）+ **4D (v,a) 粗到细全局捕获**（`global_search_accel`） | D3 stress 加速度使匀速假设失配；4D (v,a) 从捕获层解决（§5.3） |
| 17 | 真实数据形态 | MVDT 分卷 zip → 7z 解出 mp4；config 可选 `sequence.video`；GT `(0,0)` 视为 absent | 数据集不发布图像序列，只发分卷视频；MVDT 标注用 (0,0) 表目标缺席（§3.1/§3.2/§5.10） |

---

## 2. 架构

### 2.1 系统上下文

离线单机流水线，无服务端/数据库/Auth。数据流：

```text
MVDT 原始数据 ──► [派生物] ──►                ──► [实验运行] ──►     ──► [汇总]
data_raw/mvdt/     make_derived_gt.py          run_sequence.py        make_tables.py
  frames/calib/     data_derived/gt/*.csv       outputs/runs/<cfg>/    outputs/metrics/
  trajectory/         (cx,cy,visible,...)         windows.jsonl         make_figures.py
                                                 (每窗口记录)           outputs/figures/
合成数据            make_synthetic.py                                  outputs/report.md (人工)
data_raw/synthetic/ (GT 完全已知)              calibrate.py
                                                configs/thresholds_frozen.yaml
```

### 2.2 组件设计

| 模块 | 职责 | 不负责 |
|------|------|--------|
| `spa2d/config.py` | 加载/校验 config YAML 与 frozen 阈值；合并覆盖；schema 校验与约束断言 | 算法逻辑 |
| `spa2d/records.py` | 全部数据结构定义（窗口记录、GT 行、峰值、置信、计时、运行元数据）+ JSON/CSV 序列化 | 计算 |
| `spa2d/preprocess.py` | 灰度化 + 形态学闭背景 + 暗目标增强（PRD §5.1）+ **few-pixel DoG 匹配滤波/多尺度**（尺度先验） | 其他增强/学习式处理 |
| `spa2d/accumulate.py` | 运动补偿累加（**x+k·v**）、含**加速度累加**、双线性采样、BORDER_CONSTANT=0、valid mask、chunked 评分 | 得分聚合、峰值 |
| `spa2d/search.py` | 候选集合（global coarse-to-fine / local）、**4D (v,a) 全局/局部**、去重、top-K、`min_speed`、分块调度（PRD §5.4） | 累加本身 |
| `spa2d/peak.py` | 整数峰 + 二次曲面亚像素细化 + edge_candidate 判定（PRD §5.3） | 置信度 |
| `spa2d/confidence.py` | PNR、速度 margin、连续性残差（PRD §5.5） | 门控决策 |
| `spa2d/state_machine.py` | 门控规则、状态/属性转换、reason code 决策、prior 管理（PRD §5.6） | 指标 |
| `spa2d/metrics.py` | 窗口级 recall/precision@r、valid_outcome、RMSEp/RMSEv、敏感性（PRD §6） | 图表渲染 |
| `spa2d/timing.py` | 三层计时器、warmup、统计（mean/median/p95）、CUDA 同步点 | 指标 |
| `spa2d/io_mvdt.py` | 图像序列/视频解码（**常驻顺序 capture**）、帧排序、时间戳、标定/轨迹读取（含 MVDT 真实格式与空白表头）、哈希记录 | 预处理 |
| `spa2d/synthetic.py` | 合成序列生成（点目标/暗斑 + 云/虫/鸟干扰 + 已知 GT；benchmark 为平坦静止背景） | 真实数据接入 |
| `spa2d/frame_cache.py` | 设备端窗口帧缓存 + 主机端预处理 LRU 缓存（重叠窗口零重复预处理） | 磁盘缓存 |
| `spa2d/backends.py` | `jax`/`numpy` 后端选择与统一采样接口（shift_bilinear/accumulate(_accel)/score_batch(_accel)/stack/asarray/block_until_ready）；JAX jit 封装 | 算法逻辑 |

### 2.3 模块交互（单窗口处理时序）

```
run_sequence.py
  │ for t in窗口起点序列 (0, stride, 2*stride, ...):
  │   1. io/fetch → frames[t..t+N-1]（解码，计时排除）
  │   2. frame_cache → 取预处理帧（缺失则 preprocess.py 计算并缓存）
  │   3. state_machine.prepare(t) →
  │        search_mode ∈ {local, global}  +  disposition ∈ {INIT, REACQ, TRACKED, ...}
  │   4. search.generate(...) → 候选 v 集 V
  │   5. accumulate.score_batch(frames, V) → scores（分块、max 归约）→ v_best = argmax
  │   6. accumulate.accumulate(frames, v_best) → A, M 图
  │   7. peak.extract(A, M, v_best) → peak_value, p0_int, p0_sub, edge_candidate, M(peak)
  │   8. confidence.pnr(A, peak, rho)、confidence.margin(scores, v_best, δ_used)、
  │        state_machine.continuity(p_center vs p_pred) → PNR, margin, r_cont
  │   9. state_machine.step(...) → accepted、state_out、flags、reason
  │  10. 汇总 est = p0 + (N-1)/2*v_best（位置），v_best（速度）；写窗口记录
  │  11. timing 记账
  └─ 收尾：metrics.evaluate(records, gt)；metrics/timing 落盘
```

JAX 计算边界（必须遵守）：

- jit 函数一：`score_batch(frames[N,H,W], hyps[B,2]) -> scores[B]`（内部对 k 累加、末尾 max）。
- jit 函数二：`accumulate_one(frames[N,H,W], v[2]) -> (A[H,W], M[H,W])`。
- 预处理在 CPU（OpenCV）执行；帧上传为 float32 `[0,1]`。
- 计时在 jit 调用后必须 `jax.block_until_ready()` 再停表（`spa2d/timing.py` 提供包装）。

### 2.4 文件结构

```text
spa2d_mvdt_repro/
  README.md                         [新] 快速开始、数据获取、复跑命令
  .gitignore                        [新]
  env/
    requirements.txt                [新] 直接依赖（见 §8.4 附件 A）
    requirements.lock               [PRD] pip freeze 全量锁
    Dockerfile.jetson               [PRD] 可选 Jetson 节
    jetson_power_mode.md            [PRD] 可选 Jetson 节
  configs/
    D1_no_overlap.yaml  D1_overlap.yaml
    D2_no_overlap.yaml  D2_overlap.yaml
    D3_overlap.yaml     D4_overlap.yaml
    thresholds_frozen.yaml          [标定产物，初始不存在]
  spa2d/
    __init__.py
    config.py        records.py      backends.py    frame_cache.py
    preprocess.py    accumulate.py   search.py      peak.py
    confidence.py    state_machine.py metrics.py     timing.py
    io_mvdt.py       synthetic.py
  scripts/
    fetch_data.sh                   [新]
    fetch_mvdt_videos.sh            [新] 分卷 zip 下载（限流自适应）
    unpack_mvdt.sh                  [新] 7z 解包分卷 zip → mp4
    make_synthetic.py               [新]
    make_derived_gt.py              [PRD]
    run_sequence.py                 [PRD]
    run_all.sh                      [PRD] smoke/real 双模式
    calibrate.py                    [PRD]
    benchmark_search.py             [新] S3 四策略对照
    make_tables.py                  [PRD]
    make_figures.py                 [PRD]
    make_visuals_d1d2.py            [新] D1/D2 可视化
    baseline_bg_subtraction.py      [PRD]
    baseline_optical_flow_optional.py [PRD，可选]
    mvdt_urls.example.txt           [新] MVDT 元数据 URL 清单
    verify_env.py                   [新] 环境验证
  tests/
    conftest.py      test_config.py       test_preprocess.py
    test_accumulate_shift.py             test_search_grid.py
    test_peak_subpixel.py                test_confidence.py
    test_state_machine.py                test_metrics.py
    test_io_gt.py    test_synthetic.py     test_run_sequence.py
    test_calibration.py                  test_benchmark_search.py
    test_baseline_bg_subtraction.py      test_tables_figures.py
    test_run_all_report.py
  data_raw/       (git-ignored)
  data_derived/   (git-ignored)
  outputs/        (git-ignored)
    runs/ metrics/ figures/ report.md
  AGENTS.md                          [新] 仓库运行/约定手册
  docs/                              [论文 PDF、review_icra27_5428_en.md]
```

---

## 3. 数据模型（文件契约）

### 3.1 Config YAML Schema（configs/*.yaml）

```yaml
schema_version: 1
config_name: D1_no_overlap
sequence:
  name: D1/cam2
  frames_dir: data_raw/mvdt/D1/cam2/frames   # 图像序列目录；MVDT 实际以分卷 zip → unpack 得到 mp4
  video: data_raw/mvdt/D1/cam2/cam2.mp4      # 可选：视频路径（存在时优先于 frames_dir）
  frame_glob: "*.jpg"
  fps: 50.00
  resolution: [1920, 1080]     # [W, H]，校验解码一致性
  color_order: BGR             # 解码器原始顺序
  calibration: data_raw/mvdt/D1/cam2/calibration.json   # 可为 null
  trajectory: data_raw/mvdt/D1/cam2/trajectory.csv      # 可为 null
window:  {N: 5, stride: 5}
preprocess: {kernel_shape: ellipse, kernel_size: 5, normalize: divide_255}
search:
  mode: coarse_to_fine
  global: {vmax: 1.0, delta_c: 0.5}
  fine:   {rf: 0.2, delta_f: 0.1}
  local:  {rl: 0.3, delta_l: 0.1}
  top_k: 5
  batch_logical: 512
  gpu_chunk_bytes: 1610612736        # 1.5 GiB，物理分块预算
thresholds_file: configs/thresholds_frozen.yaml
cache: {host_max_bytes: 2147483648}  # 2 GiB LRU
metrics:
  radii_px: [10, 20, 50, 100]
  primary_radius_px: 100
  association: round_half_up         # 敏感性另算 floor
timing: {warmup_windows: 20, repeats: 3}
backend: jax                         # jax | numpy
seed: 0
```

校验规则（`config.py` 启动时断言，失败退出码 2）：

- 粗网格对称：`2*vmax` 能被 `delta_c` 整除（§3.1 原文写作 `vmax % delta_c == 0`，但 D3/D4 的 `(vmax=15, delta_c=2.0)` 不满足该式，而 `2*15/2=15` 为整数、与粗网格 16 点/轴一致；此处按实际几何修正）。
- fine 邻域：`rf % delta_f == 0`；local 邻域：`floor(rl / delta_l) >= 1`（D3 的 `0.6/0.25` 步数为 2，不要求整除）。
- `0 < stride <= N`；`N >= 2`。
- `kernel_size` 为奇数且 ∈ {3,5,7,9,11}；冻结后与 `thresholds_frozen.yaml` 的 `k` 一致。
- `primary_radius_px ∈ radii_px`。
- 六份 config 的搜索参数与 PRD §5.2/§5.4 表格逐值一致（单测 `test_config.py` 固化该表）。

### 3.2 派生 GT CSV（data_derived/gt/<Seq>_<cam>.csv）

Schema 与逐帧规则完全按 PRD §4.4（`frame,timestamp_s,visible,out_of_frame,cx,cy,bbox_x,bbox_y,bbox_w,bbox_h,area_px,width_px,width_source,gt_source,projection_rmse_px`）。SPEC 补充：

- `width_px`：area 解释 `sqrt(area_px)`（`width_source=sqrt_area`）；bbox 可得则 `width_source=bbox_w`；两者都有时 bbox 优先、area 同步保留。
- 每序列附 `data_derived/gt/README.md`：数据源、方法（`published2d` / `project3d`）、相机模型与畸变处理、投影 RMSE、帧哈希清单、生成命令。
- `make_derived_gt.py --method auto`：有发布 2D 位置 → `published2d` 为主并同时投影 3D 做 cross-check 写入 `projection_rmse_px`；否则 `project3d`。
- NaN 语义：`visible=0` 允许 `cx,cy` 为 NaN；CSV 写空串而非 "nan"。
- **MVDT 标注语义**：发布方 2D 标签（`detections/camN.txt`）以 `(cx,cy)=(0,0)` 表示“目标未出现/未标注”→ 派生 GT 将这类帧记为 `visible=0`（absent）。D1/D2/D3/D4 可见率分别约 49%/72%/45%/66%，D3 前 ~764 帧目标不出现。序列命名：configs 用 `D1/cam2`…，数据集目录为 `dataset1/cam2`…，GT 文件名 `D{1..4}_cam{2,1,3,3}.csv`。

### 3.3 窗口输出记录（outputs/runs/<config_name>/windows.jsonl）

每窗口一行 JSON（字段固定，dataclass `WindowRecord` in `records.py`）：

```json
{
  "window_index": 12, "t_start": 60, "t_center": 62.0, "assoc_frame": 62,
  "search_mode": "local",
  "state_in": "TRACKED", "state_out": "TRACKED",
  "accepted": true, "target_present": true, "reason": "OK",
  "est": {"x": 956.31, "y": 512.72, "vx": 0.7, "vy": -0.2},
  "peak": {"value": 3.42, "xy_int": [956, 512], "xy_sub": [956.31, 512.72],
            "subpixel_used": true, "valid_count": 5, "edge_candidate": false},
  "confidence": {"pnr": 12.7, "mu_bg": 0.31, "sigma_bg": 0.24, "margin": 0.08},
  "continuity": {"r_cont": 3.21, "r_cont_high": 25.0, "r_cont_low": 10.0, "exempt": false},
  "flags": {"stat": false, "edge": false},
  "miss_count": 0, "reacq_count": 0,
  "timing_ms": {"t_pre": 8.1, "t_search": 25.3, "t_state": 0.2, "t_total": 33.6}
}
```

同目录附带：

- `run_meta.json`：config 快照、frozen 阈值快照 + sha256、环境信息（`jax.devices()`、jaxlib 版本、XLA flags、`cv2.getBuildInformation()` 摘要、线程数、GPU 型号、驱动版本）、数据哈希、种子、命令行。
- `metrics.json` / `timing.json`：见 §5.8、§5.9。
- `repeat_<i>/`：`--repeat 3` 时每次重复的 windows.jsonl 与 metrics.json（主目录存第 1 次 + 三次极差汇总）。

### 3.4 合成数据（data_raw/synthetic/）

```text
data_raw/synthetic/
  manifest.json          # 全部序列参数、种子、哈希
  seq_000/ frames/*.png  # 或 .npy；PNG 保持 8-bit，GT 独立
  seq_000/gt.csv         # 与 §3.2 同 schema（width_source=bbox）
  seq_000/meta.json      # 速度真值、SNR 参数、干扰物清单
  benchmark/             # S3 专用：单序列 + 全网格真值（见 §5.11）
```

### 3.5 冻结阈值（configs/thresholds_frozen.yaml）

```yaml
schema_version: 1
frozen_at: "2026-10-10T00:00:00Z"
calibration:
  synthetic_manifest_sha256: "..."
  validation: ["D1/cam2:first500", "D3/cam3:first500"]
  configs_sha256: {D1_no_overlap: "...", ..., D4_overlap: "..."}
  code_commit: "..."            # git rev-parse HEAD
  protocol_version: "prd-v1.0"
  scan_trace: outputs/metrics/calibration_trace.json   # 每步扫描的完整记录
thresholds:
  theta_high_pnr: 8.0   # 标定后回填
  theta_low_pnr: 5.0
  theta_margin: 0.02
  theta_edge_pnr: 10.0       # 固定规则 theta_high_pnr + 2
  v_stat: 0.05
  m_lost: 3
  reacq_max: 5
  gap_prior_windows: 2
  rho_formula: "max(8, ceil(2*width_ref))"
  r_cont_high_formula: "max(25, 3*width_ref)"
  r_cont_low_formula: "max(10, 1.5*width_ref)"
width_ref: {D1/cam2: null, D2/cam1: null, D3/cam3: null, D4/cam3: null}
preprocess: {kernel_size: null, matched_sigma: null}    # k 标定第 1 步回填；matched_sigma = clamp(width_ref/3, 1.2, 4.0)（DoG 尺度先验）
```

冻结规则：`run_sequence.py` 加载后计算文件 sha256 并写入 `run_meta.json`；文件一旦生成，除 `calibrate.py --force` 外任何脚本不得改写；`--force` 必须在 `scan_trace` 中记录并触发报告中的「已重标定」声明。

---

## 4. 接口设计（CLI 与 Python API）

### 4.1 脚本 CLI 契约

| 脚本 | 用法 | 输入 | 输出 | 退出码 |
|------|------|------|------|--------|
| `fetch_data.sh` | `--source <dir\|urls.txt> --dest data_raw/mvdt [--check] [--resume]` | 本地目录或 URL 清单 | 数据落盘 + `checksums.sha256` + `provenance.json` | 0/3 |
| `make_synthetic.py` | `--suite sanity\|benchmark --out data_raw/synthetic [--seqs 30] [--seed 0]` | 生成参数 | §3.4 目录 | 0/2 |
| `make_derived_gt.py` | `--seq D1/cam2 --raw data_raw/mvdt --out data_derived/gt/D1_cam2.csv [--method auto\|published2d\|project3d] [--fps 50.0]` | 标定/轨迹 | §3.2 CSV + README | 0/2/3/4 |
| `run_sequence.py` | `--config configs/D1_no_overlap.yaml --gt ... --out outputs/runs/D1_no_overlap [--limit N] [--repeat 3] [--backend jax]` | config+GT | §3.3 | 0/2/3/4/5 |
| `calibrate.py` | `--synthetic data_raw/synthetic --validation D1/cam2:500,D3/cam3:500 --configs configs/*.yaml --out configs/thresholds_frozen.yaml [--force]` | 合成+验证片 | §3.5 + `calibration_trace.json` | 0/2/3/4/5 |
| `benchmark_search.py` | `--synthetic data_raw/synthetic/benchmark --out outputs/metrics/s3_search.csv --backend jax` | benchmark | S3 对照表（FPS、成功率） | 0/4 |
| `run_all.sh` | 无参数；按序执行下述全部命令 | — | 全部 outputs | 首个失败码 |
| `make_tables.py` | `--runs outputs/runs --gt data_derived/gt --out outputs/metrics` | 运行目录 | `summary.md/csv`、`sensitivity.csv`、`paper_vs_ours.md` | 0/3 |
| `make_figures.py` | `--runs outputs/runs --out outputs/figures [--failures 10]` | 运行目录 | Fig7 风格柱状图、Fig8 风格轨迹图、失败拼图（≥10 FA/≥10 miss，标注 reason code）；`--no-images` 供 license 禁用图像时使用 | 0/3 |
| `baseline_bg_subtraction.py` | `--config ... --gt ... --out outputs/runs_baselines/bgsub_<cfg> [--tracker nn\|kalman]` | config+GT | 与 §3.3 同结构（部分字段置 null） | 0/2/3/4 |
| `baseline_optical_flow_optional.py` | 同上 | — | 诊断用 | 0/4 |
| `fetch_mvdt_videos.sh` | 抓 MVDT 分卷 zip（aria2 多连接+断点续传，GitHub contents API + Accept raw，含限流自适应等待） | 网络 | `data_raw/mvdt/_archives/` | 0/1 |
| `unpack_mvdt.sh` | 7z 解包分卷 zip → mp4 | 归档 | `data_raw/mvdt/D{n}/cam{m}/camN.mp4` | 0/1 |
| `make_visuals_d1d2.py` | D1/D2 轨迹 + 逐窗误差 + 帧上 GT/估计叠加图 | 运行目录 | `outputs/figures/fig8_*.png、overlay_*.png` | 0 |

`run_sequence.py` 支持的 env 开关：`SPA2D_MATCH_SIGMA`（DoG 单尺度 σ）、`SPA2D_MATCH_SIGMAS`（多尺度列表）、`SPA2D_ACCEL=1`（4D 加速度搜索）+ `SPA2D_A_ACC/SPA2D_DA_COARSE/SPA2D_A_RADIUS/SPA2D_A_DA/SPA2D_A_TOPK`、`SPA2D_MIN_SPEED`、`SPA2D_START_FRAME`、`SPA2D_LIMIT`、`SPA2D_FRAME_LIMIT`、`SPA2D_CHUNK_BYTES`、`SPA2D_BACKEND`、`SPA2D_RUN_ALL_MODE`（`smoke` 合成冒烟 / `real` 真实 MVDT）。GPU 上建议 `XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS=--xla_gpu_autotune_level=0`。

`run_all.sh` 固定顺序（每步失败即停；real 需数据 + frozen；smoke 自带合成配置，不再覆盖真实 GT，写 `smoke_D1`）：

```bash
make_derived_gt.py ×4 序列        # 依赖数据已就位（--check 通过）
run_sequence.py ×6 configs        # 依赖 thresholds_frozen.yaml 存在
baseline_bg_subtraction.py ×6
make_tables.py && make_figures.py
```

`run_all.sh` 开始时执行 `fetch_data.sh --check` 与 frozen 阈值存在性检查，并打印确认为 Tier B 运行。

### 4.2 Python 包 API（公开函数签名）

```python
# spa2d/config.py
@dataclass(frozen=True)
class Config: ...   # §3.1 全字段 + load_config(path) -> Config + validate(cfg)
load_frozen(path: Path) -> FrozenThresholds   # 含 sha256 计算与一致性校验
effective_thresholds(cfg: Config, frozen: FrozenThresholds, width_ref: float) -> Thresholds

# spa2d/preprocess.py
build_kernel(k: int, shape: str = "ellipse") -> np.ndarray
to_gray(frame: np.ndarray, color_order: str = "BGR") -> np.ndarray      # float32 [0,255]
preprocess_frame(frame, kernel, color_order="BGR") -> np.ndarray        # float32 [0,1]
matched_filter_frame(frame, sigma, color_order="BGR") -> np.ndarray     # DoG 匹配滤波（尺度先验）
matched_filter_multiscale(frame, sigmas, color_order="BGR") -> np.ndarray  # 多尺度（限制大尺度）

# spa2d/accumulate.py  (backend dispatch via backends.py)
shift_bilinear(img: Array, dx: float, dy: float) -> tuple[Array, Array]      # (采样值, valid 权重)
accumulate_window(frames: Array, v: Array) -> tuple[Array, Array]            # (A, M)
accumulate_window_accel(frames, v, a) -> tuple[Array, Array]                 # 加速度 (A, M)
score_batch(frames: Array, hyps: Array) -> Array                             # [B]
score_batch_accel(frames, hyps: Array) -> Array                              # [B]，hyps (B,4)=[vx,vy,ax,ay]
score_batch_chunked(frames, hyps, chunk_size, xp) -> Array                   # 分块拼接

# spa2d/backends.py
available_backends() -> tuple[str, ...]
get_backend(name: str | None = None) -> Backend   # shift_bilinear/accumulate_window(+accel)/score_batch(+accel)/score_batch_chunked/asarray/stack/block_until_ready

# spa2d/search.py
coarse_grid(vmax: float, delta_c: float) -> Array            # ((2vmax/δc+1)^2, 2)
fine_neighborhoods(top: Array, rf: float, delta_f: float) -> Array
local_candidates(v_prev: Array, rl: float, delta_l: float) -> Array
dedupe(cands: Array, eps: float = 1e-9) -> Array
top_k_indices(scores, cands, k: int) -> np.ndarray
gpu_chunk_size(gpu_chunk_bytes, height, width, batch_logical) -> int
run_search(frames, cands, batch_logical, gpu_chunk_bytes, backend=None, min_speed=0.0) -> SearchResult
global_search(frames, vmax, dc, rf, df, top_k, batch_logical, gpu_chunk_bytes, backend=None, min_speed=0.0) -> SearchResult
local_search(frames, v_prev, rl, delta_l, batch_logical, gpu_chunk_bytes, backend=None, min_speed=0.0) -> SearchResult
brute_force(frames, vmax, step, batch_logical, gpu_chunk_bytes, backend=None, min_speed=0.0) -> SearchResult
local_search_accel(frames, v_prev, a_prev, rv, dv, ra, da, batch_logical, gpu_chunk_bytes, backend=None)
global_search_accel(frames, vmax, dc, amax, da, rf, df, ra, daa, top_k, batch_logical, gpu_chunk_bytes, backend=None, min_speed=0.0)
accel_grid(vmax, dc, amax, da) -> np.ndarray          # (B,4)=[vx,vy,ax,ay]
accel_neighborhoods(top, rf, df, ra, daa) -> np.ndarray

# spa2d/peak.py
extract_peak(A: Array, M: Array, N: int, edge_margin: int) -> PeakResult
subpixel_refine(A: Array, xy_int: tuple[int,int]) -> tuple[float,float] | None

# spa2d/confidence.py
background_stats(A: Array, M: Array, center: tuple[int,int], rho: int) -> tuple[float,float]
pnr(peak_value: float, mu: float, sigma: float) -> float
velocity_margin(scores: Array, cands: Array, v_best: Array, delta_used: float) -> tuple[float, bool]

# spa2d/state_machine.py
class StateMachine:
    def prepare(self, t_start: int) -> WindowPlan          # search_mode, disposition
    def continuity(self, p_center_xy: np.ndarray, t_center: float) -> Continuity
    def step(self, obs: WindowObservation) -> WindowOutcome # 见 §5.6

# spa2d/metrics.py
associate_frame(t_start: int, N: int, mode: str) -> int
evaluate_windows(records: list[WindowRecord], gt: GTTable,
                 radii: list[int], assoc_mode: str) -> MetricsResult
gt_velocity(gt: GTTable) -> np.ndarray      # 中心差分

# spa2d/io_mvdt.py
open_sequence(frames_dir: str | None, video: str | None, glob: str) -> FrameSource  # 迭代 (idx, bgr)
frame_timestamps(fps: float, n: int, dataset_ts: np.ndarray | None) -> np.ndarray
load_calibration(path) -> CameraModel | None
load_trajectory(path) -> Trajectory3D | None
project_3d_to_2d(traj: Trajectory3D, cam: CameraModel, fps: float) -> (np.ndarray, float)  # 2D pts, rmse

# spa2d/timing.py
class Timer:  # 三层计时；sync_jax() 内部在停表前 block_until_ready
    def tick(self, layer: str) -> contextmanager
summary(times_ms: list[float], warmup: int) -> TimingStats   # mean/median/p95

# spa2d/frame_cache.py
class FrameCache:  # host LRU（max_bytes）+ device 窗口驻留
    def get_preprocessed(self, idx: int) -> np.ndarray
    def device_window(self, idxs: list[int]) -> list[Array]
```

### 4.3 错误与退出码

| 退出码 | 含义 | 触发示例 |
|-------:|------|----------|
| 0 | 成功 | — |
| 2 | 配置非法 | YAML 缺字段、`vmax % delta_c != 0`、kernel_size 非法 |
| 3 | 数据缺失/损坏 | 帧数不足、hash 不符、GT 行缺失、`fetch_data.sh --check` 失败 |
| 4 | 运行时失败 | 解码失败、帧含 NaN/Inf、JAX 设备不可用且未指定 numpy 回退、OOM 分块仍失败 |
| 5 | 冻结校验失败 | frozen 阈值 sha256 缺失、与 config 的 k 冲突、标定产物被篡改 |
| 130 | 用户中断 | SIGINT |

所有脚本 `--help` 输出与上表一致；错误信息统一格式 `[套件] <code>: <message> → <remediation>`。

---

## 5. 业务逻辑（算法规范）

### 5.1 预处理（对齐论文 Eq.6，PRD §5.1）

```text
G = gray(frame)                              # BGR→GRAY 或 RGB→GRAY，float32 [0,255]
C = morphologyEx(G, MORPH_CLOSE, B, border=默认)
D = max(C - G, 0)
I = D / 255.0                                # float32 [0,1]
```

- `B = getStructuringElement(MORPH_ELLIPSE, (k,k))`，k 来自冻结阈值（默认 5）。
- 禁用一切全图直方图均衡/CLAHE/学习式增强（PRD 明令）；敏感性实验独立报告节。
- OpenCV closing/remap 使用默认边界（closing 默认 reflect101/constant 由 OpenCV 决定，记录版本即可）；累加边界单独定义为 constant 0（§5.2）。

**补充（真实数据所需）：few-pixel 暗斑匹配滤波 + 尺度先验**（新增，不在论文主链路）。真实 MVDT 上，纯形态学暗增强的全局 max 累加会被更强的静止大杂波锁死（D1 recall 0）。为此增加第二种候选表示：

- `matched_filter_frame(frame, sigma)`：对灰度做 DoG = `GaussianBlur(1.6σ) − GaussianBlur(σ)`（宽-窄差分），暗小斑正响应、大尺度暗结构被抑制；`sigma` 为尺度先验（按目标等效宽度自动选，冻结 `preprocess.matched_sigma = clamp(width_ref/3, 1.2, 4.0)`，当前 2.6667）。
- `matched_filter_multiscale(frame, sigmas)`：尺度自适应（取各 σ 的最大响应；**限制大尺度**以防大杂波主导，如 `{1.5,2.5,4.0}`）。
- 接入：`run_sequence` 在 `SPA2D_MATCH_SIGMA>0`（单尺度）或 `SPA2D_MATCH_SIGMAS`（多尺度）时改用该表示；run_meta 记录 `matched_sigma`。真实 D1/D2 用 σ=2.667 达到 recall/precision@100=1.0。

### 5.2 运动补偿累加（对齐论文 Eq.7，PRD §5.3）

对候选 `v=(vx,vy)`，窗口帧 `frames[k] = I_{t+k}`：

```text
A_v(x,y) = Σ_{k=0}^{N-1} frames[k] (x + k·vx, y + k·vy)     # 双线性，常量 0 边界
M_v(x,y) = Σ_{k=0}^{N-1} ones    (x + k·vx, y + k·vy)     # valid 计数 ∈ [0,N]
```

**符号修正**：论文 Eq.7 写作 `x − k·v`，与 S1（目标以 (0.7,0) 运动必须返回 (0.7,0)）、Eq.9（`p_c = p0 + (N−1)/2·v`）及连续性模型 `p_pred = p_prev + v·Δt` 不自洽（`x−k·v` 实际在 v = −w 处相干）。本实现采用 `x + k·v`，保证 `v_best` 即目标像素速度。

**加速度变体（新增，§5.3 的 4D 搜索用）**：

```text
A_{v,a}(x,y) = Σ_{k=0}^{N-1} frames[k] (x + k·vx + ½k²·ax, y + k·vy + ½k²·ay)
```

（`accumulate_window_accel` / `score_batch_accel`；注意与上面 `x+k·v` 的符号方向一致：移项即后帧按 `-(k v + ½k² a)` 搬回首帧。）

手写双线性采样（所有后端逐位同构）：

```text
sx = x + k·vx;  sy = y + k·vy
x0 = floor(sx); fx = sx - x0;  y0 = floor(sy); fy = sy - y0
for (ax, ay) in {(0,0),(1,0),(0,1),(1,1)}:
    w = (fx if ax else 1-fx) * (fy if ay else 1-fy)
    in = 0 ≤ x0+ax < W and 0 ≤ y0+ay < H
    val += w * (frames[k][y0+ay, x0+ax] if in else 0)
    msk += w if in else 0
```

- float32 计算；不做任何预平滑/模糊。
- 与 `cv2.remap(INTER_LINEAR, BORDER_CONSTANT=0)` 对齐且容差 ≤ 1e-4（float32 输入时），由 `test_accumulate_shift.py` 验证（OpenCV 对 8bit 可能固定点加速，测试一律用 float32 随机图）。
- 得分 `S(v) = max_{x,y} A_v(x,y)`（raw sum，不加归一化；`S_norm` 仅敏感性报告，不进主链路）。
- `edge_margin = max(2, ceil((N-1)·max(|vx|,|vy|)) + 1)`；`edge_candidate = peak 在边带内或 M_v(peak) < N`。
- 分块：`chunk = clamp(floor(gpu_chunk_bytes / (2·H·W·4)), 1, batch_logical)`（2 份缓冲：shifted + accumulator；见 §8.3）；跨块结果做逐候选 max，语义不变。

### 5.3 候选速度集合（PRD §5.4）

统一约定：候选集为 float32 `[B,2]`，去重（Chebyshev 距离 ≤ 1e-9 视为重复），去重后排序保证确定性（字典序：先 vx 后 vy）。

```text
Global coarse-to-fine（首次/丢失后）:
  C = coarse_grid(vmax, delta_c)                    # 每轴 m=(2vmax/δc+1) 个值
  scores = score_batch(frames, C)
  T = top_k(C, scores, K=5)                         # 并列按 (|v|, vx, vy) 字典序稳定取前 5
  V = dedupe(∪_{vi∈T} fine_neighborhoods(vi, rf, δf))
  scores = score_batch(frames, V)                   # 注意：V 独立评分，不继承 coarse 分数
  v_best = argmax(scores)                           # 并列取 sort 后第一个
Local（有效 prior）:
  V = local_candidates(v_prev, rl, δl)              # (2·floor(rl/δl)+1)^2 个
  v_best = argmax(score_batch(frames, V))
```

各 config 的候选规模（供单测与性能预估）：

| Config | coarse 数 | fine/种子 | fine 并集上限 | local 数 | 全局候选总数（去重前） |
|--------|----------:|----------:|--------------:|---------:|-----------------------:|
| D1_* | 25 | 25 | 125 | 49 | 150 |
| D2_no_overlap | 289 | 25 | 125 | 49 | 414 |
| D2_overlap | 441 | 25 | 125 | 49 | 566 |
| D3_overlap | 256 | 25 | 125 | 25 | 381 |
| D4_overlap | 256 | 121 | 605 | 81 | 861 |

`SearchResult = {cands, scores, v_best, second_best, chunk_count}`。top-K 与 argmax 均为确定性的稳定选择（避免 GPU 非确定性转导为选择抖动）。

**新增（实现后补充）：**

- **运动相干约束 `min_speed`**：`run_search`/`global_search`/`local_search`/`brute_force` 支持 `min_speed>0` 时先排除 `speed < min_speed` 的候选（惩罚 v≈0 静态峰；全排除则回退全集）。`run_sequence` 经 env `SPA2D_MIN_SPEED` 启用。用于剔除静止杂波（D1/D3 曾锁定静止暗结构）。
- **4D (v,a) 粗到细全局捕获**（stress 序列加速度适配，替代"先匀速再局部"）：
  - `accel_grid(vmax, δc, amax, δa)`：`v∈[−vmax,vmax]² × a∈[−amax,amax]²` 笛卡尔积候选（4 维去重）；`accel_neighborhoods(top, rf, δf, ra, δa)`：top-K 的 4D fine 邻域并集。
  - `global_search_accel(frames, vmax, dc, amax, da, rf, df, ra, da, top_k, ...)`：coarse(v,a) → top-K → fine(v,a) 邻域独立评分 → argmax → `(v_best, a_best)`。
  - `local_search_accel(frames, v_prev, a_prev, rv, dv, ra, da, ...)`：局部 (v,a) 精细搜索（跟踪窗）。
  - 累加/评分：`accumulate_window_accel` / `score_batch_accel`（`A_{v,a}=Σ I(x−k·v−½k²·a)`）；中心帧位置 `center_frame_position_accel(p0, v, a, N) = p0 + (N−1)/2·v + ½·((N−1)/2)²·a`。
  - 接入：`run_sequence` 在 `SPA2D_ACCEL=1` 时，**全局窗用 4D coarse-to-fine、局部窗用 local_search_accel**；env：`SPA2D_A_ACC`（amax 默认 1.5）、`SPA2D_DA_COARSE`（δa 默认 1.5）、`SPA2D_A_RADIUS`（ra 默认 1.0）、`SPA2D_A_DA`（δa 默认 0.25）、`SPA2D_A_TOPK`（默认 8）。margin 走 1.0（θ_margin 适配）。

### 5.4 峰值提取与亚像素（PRD §5.3）

```text
1. (x0,y0) = 整数 argmax（并列取 (y,x) 字典序最小）
2. 若 3×3 邻域 M_v ≥ 1 全部成立且 A(x0,y0) 为严格最大：
     fx = 0.5·(A(x0-1,y0) - A(x0+1,y0)) / (A(x0-1,y0) - 2A(x0,y0) + A(x0+1,y0))
     fy 同理；|分母| ≤ 1e-12 或 |f| > 0.5 → 该轴取 0（不细化）
     p0 = (x0 + fx, y0 + fy)（clip 到 [0, W-1]×[0, H-1]）；subpixel_used = True
   否则 p0 = (x0, y0)，subpixel_used = False
3. est 中心：p_center = p0 + (N-1)/2 · v_best     （论文 Eq.9 / PRD §5.3）
4. 记录 peak_value = A(x0,y0)、M(peak)、edge_candidate
```

### 5.5 置信度与连续性（PRD §5.5）

```text
PNR:
  ρ = max(8, ceil(2·width_ref))，width_ref 来自 frozen 表（§3.5）
  背景集合 = {M_v(p) ≥ 1} 去掉以整数峰为中心的圆盘半径 ρ
  μ_bg, σ_bg = mean/std（float32→float64 聚合）；若背景像素 < 100 → PNR = 0 并警告
  PNR = (peak_value - μ_bg) / max(σ_bg, 1e-6)
margin:
  δ_used = delta_f（global 模式）或 delta_l（local 模式）
  排除 |vx-vx_best| ≤ max(2δ_used, 0.2) 且 |vy-vy_best| ≤ max(2δ_used, 0.2) 的候选
  若排除后无候选 → margin = 1.0, single_candidate = True
  否则 margin = (S - S2)/max(S, 1e-6)
连续性（仅当存在 last_accepted）:
  p_pred = p_prev + v_prev·(c - c_prev)          # c 为当前窗口中心时间
  r_cont = ||p_center - p_pred||₂
  r_cont_high = max(25, 3·width_ref)；r_cont_low = max(10, 1.5·width_ref)
  无 prior → r_cont = NaN, exempt = True（仅 INIT/REACQ 首窗允许）
```

### 5.6 状态机（PRD §5.6 的精确化）

**规范状态**（7 个）：`INIT, TRACKED, TRACKED_LOW, REACQ, LOST, OUT_OF_FRAME, ABSENT`。
**布尔属性**：`stat`（|vx|,|vy| ≤ v_stat 且接受）、`edge`（edge_candidate=True 且接受）。
**prior 定义**：`last_accepted = (t_start_prev, c_prev, p_prev, v_prev)`；`prior_ok = last_accepted 存在 ∧ state∉{LOST, OUT_OF_FRAME, ABSENT} ∧ (t_start - t_start_prev)/stride ≤ gap_prior_windows`。

**窗口计划 `prepare(t)`**：

- `prior_ok` → `search_mode=local`，disposition 保持当前跟踪状态。
- 否则 → `search_mode=global`，disposition = `REACQ`（有过接受历史）或 `INIT`（从未接受）。

**接受判定（按序）**：

```text
1. edge 路径: 若 edge_candidate: 接受 ⟺ PNR ≥ θ_edge_PNR ∧ cont_ok_high
2. 高置信路径: 若 PNR ≥ θ_high_PNR: 接受 ⟺ margin ≥ θ_margin ∧ cont_ok_high
3. 低置信路径: 若 θ_low_PNR ≤ PNR < θ_high_PNR: 接受 ⟺ margin ≥ θ_margin ∧ prior_ok ∧ cont_ok_low
4. 否则拒绝

cont_ok_high/low: exempt ? True : r_cont ≤ r_cont_high/low
接受后输出状态: 高置信/edge → TRACKED；低置信 → TRACKED_LOW；stat 属性照记；
  接受即更新 last_accepted、miss_count=0、reacq_count=0。
```

**reason code 决策（拒绝时，按序首个命中；接受恒为 OK）**：

```text
1. edge_candidate ∧ PNR < θ_edge_PNR                        → EDGE_REJECT
2. PNR < θ_low_PNR                                          → LOW_PNR
3. margin < θ_margin                                        → LOW_MARGIN
4. 阈值均过 ∧ 非 exempt ∧ r_cont > 上限（高/低路径各自上限） → DISCONTINUOUS
5. PNR ≥ θ_low ∧ margin 过 ∧ ¬prior_ok（低路径需要 prior）   → NO_PRIOR_GLOBAL
6. state_in == LOST 且本窗为 global 尝试失败                 → LOST
7. state_in == OUT_OF_FRAME 且本窗未执行搜索（周期跳过）     → OUT_OF_FRAME
8. 其余拒绝                                                  → ABSENT
```

**转换表（state_in × 本窗结果 → state_out）**：

| state_in | 接受（高/edge） | 接受（低） | 拒绝 |
|----------|----------------|-----------|------|
| INIT | TRACKED | TRACKED_LOW | INIT（reason 1–4,6→NO_PRIOR_GLOBAL/ABSENT） |
| TRACKED / TRACKED_LOW | TRACKED | TRACKED_LOW | miss_count++；< m_lost → 原状态；≥ m_lost → LOST |
| REACQ | TRACKED | —（低路径需 prior_ok，REACQ 无） | reacq_count++；< reacq_max → REACQ；≥ reacq_max → 预测位置出帧则 OUT_OF_FRAME 否则 LOST |
| LOST | TRACKED（清 miss/reacq） | — | 预测位置（若可外推）出帧 → OUT_OF_FRAME；否则保持 LOST（每窗都做 global） |
| OUT_OF_FRAME | TRACKED（清计数） | — | `window_index % reacq_max != 0` → 跳过搜索（省算力），reason=OUT_OF_FRAME；执行搜索仍失败 → 预测位置回帧内则转 LOST，否则保持 OUT_OF_FRAME |

补充规则：

- `ABSENT` 仅作为「无输出且无更具体原因」的落点（上表 reason 8）；同时作为 `target_present=False` 的语义标签写入记录，不参与搜索模式判定。
- `OUT_OF_FRAME` 周期的定义（决策）：每 `reacq_max` 个窗口执行一次 global 搜索，其余窗口直接输出 `target_present=False`，timing 记为 T_state（近零算力），记录 `skipped_by_period=true`（新增字段，写入记录）。
- 所有接受/拒绝必须 log 完整决策上下文（PNR/margin/r_cont/阈值/属性/计数器），供审计与前缀分析。
- `min_speed`（§5.3 补充）作用于候选层：静态杂波（v≈0）若取胜，接受与否仍经 PNR/margin/continuity 门控；`min_speed` 是表示/候选侧的运动相干手段，不改变门控规则。

### 5.7 单窗口主循环伪代码（run_sequence.py 核心）

```python
sm = StateMachine(thresholds, width_ref)
for t in range(start_frame, F - N + 1, stride):      # start_frame 由 SPA2D_START_FRAME 指定（默认 0）
    plan = sm.prepare(t)
    if plan.skip_search:                       # OUT_OF_FRAME 周期跳过
        rec = make_skip_record(t, plan); continue
    # 预处理表示：default preprocess_frame；或 SPA2D_MATCH_SIGMA(S) → matched_filter(_multiscale)
    frames = cache.device_window(range(t, t + N))
    if os.environ.get("SPA2D_ACCEL") == "1":
        if plan.search_mode == "global":
            v_best, a_best, _, _ = search.global_search_accel(frames, ..., min_speed=SPA2D_MIN_SPEED)
        else:
            v_best, a_best, _, _ = search.local_search_accel(frames, v_prev, a_prev, ...)
        A, M = accumulate.accumulate_window_accel(frames, v_best, a_best)
        p_center = peak.center_frame_position_accel(peak.xy_sub, v_best, a_best, N)
        margin = 1.0
    else:
        res = search.local_search(frames, v_prev, ...) if plan.search_mode == "local" \
              else search.global_search(frames, ..., min_speed=SPA2D_MIN_SPEED)
        v_best = res.v_best
        A, M = accumulate.accumulate_window(frames, v_best)
        p_center = peak.center_frame_position(peak.xy_sub, v_best, N)
        margin, single = confidence.velocity_margin(res.scores, res.cands, v_best, delta_used)
    pk = peak.extract_peak(A, M, N, peak.edge_margin_for(N, v_best))
    cont = sm.continuity(p_center, t_center)
    obs = WindowObservation(pnr, margin, pk.edge_candidate, cont, p_center, v_best)
    outcome = sm.step(obs, plan)
    write_window_record(...)
```

### 5.8 指标计算（PRD §6 的精确化）

```text
assoc_frame(t) = floor(t + (N-1)/2 + 0.5)        # round-half-up；敏感性 mode=floor
对每条窗口记录 r（共 W_out 条）与 GT 表：
  若 r.target_present:
      f = assoc_frame(r.t_start); g = gt[f]
      hit@R = g.visible==1 ∧ ||est_xy - (g.cx,g.cy)||₂ ≤ R
      对每个 R ∈ radii: correct[R] += hit@R; fp[R] += ¬hit@R
      primary: correct@100, fp@100
      若 g.visible==1: 记录 all-visible 误差（诊断 RMSE）
  否则:
      valid_absent += (g.visible==0 ∨ g.out_of_frame==1)
visible_den = #{窗口记录的 assoc_frame 上 GT visible==1}
recall@R = correct[R]/visible_den
precision@R = correct[R]/(correct[R]+fp[R])
valid_outcome = (correct@100 + valid_absent)/W_out
RMSEp = sqrt(mean_{correct@100} ||est_xy-gt_xy||₂²)
RMSEv = sqrt(mean_{correct@100} ||est_v-gt_v||₂²)   # gt_v 中心差分（PRD §4.4）
```

输出（`outputs/metrics/` 与每 run 的 `metrics.json`）：

- 主表：每 config 一行（FPS、valid_outcome、recall@100、precision@100、RMSEp、RMSEv + 论文对照列）。
- 敏感性表：recall/precision@{10,20,50,100}px；正常距离归一化 `d/max(width_ref,1)` 分布（p50/p90/p95）；round-half-up vs floor 关联差异；3 次重复运行的指标极差。
- 诊断：all-accepted-visible RMSE；reason code 分布；FP 来源（insect/bird/cloud/clutter/edge/reacq/unknown，支持人工后标注 CSV 覆盖，不改变指标）。
- `make_tables.py` 同时生成 `paper_vs_ours.md`，自动区分 Tier A/B（Tier A 列在无作者 artifact 时置 N/A）。

### 5.9 计时（PRD §7）

- `T_pre`（预处理）、`T_search`（候选生成+累加+得分+峰值）、`T_state`（门控+状态机+日志）；`T_window = T_pre+T_search+T_state`。
- 主 FPS：`window_fps = 1/mean(T_window)`，`effective_frame_fps = stride/mean(T_window)`；排除解码与磁盘 IO；另报含解码的 end-to-end FPS（同一计时器框架，层名 `T_decode`）。
- warmup 排除前 20 窗口；报告 mean/median/p95；`--repeat 3` 报告三跑极差。
- JIT 编译时间单独记录（`run_meta.json.jit_compile_ms`），不计入主 FPS，但注明是否包含在后缀使用中。
- 硬件记录：GPU 型号/驱动、`nvpmodel -q`/`jetson_clocks`（存在时）、是否锁频、是否独占；RTX 5060 Ti 上另报 `pixels×hypotheses/window` 的 scaling 表（PRD §7 的跨硬件外推依据）。

### 5.10 派生 GT 算法（make_derived_gt.py）

```text
输入: 序列标定 + 轨迹（2D 发布位姿 或 3D GNSS）；支持 MVDT 目录布局（datasetN/{detections,trajectory} + calibration/<model>）
方法 auto:
  A. 发布方 2D 位置存在 → primary=published2d；同时用 3D+标定投影做 cross-check，
     写 projection_rmse_px；两者不一致（RMSE>20px）→ 警告并在 README 记录
  B. 仅 3D 轨迹 → project3d: x = K[R|t]X（含已声明畸变模型）；默认不矫正 rolling shutter
时间戳: 优先数据集时间戳；缺失则 timestamp_s = frame/fps
速度: 中心差分 v(f)=(p(f+1)-p(f-1))/2；端点单侧差分；不平滑
目标尺寸: bbox 可得→bbox_w；仅 area→sqrt；均记 width_source
MVDT 标注语义: 发布 2D 用 (0,0) 表“目标未出现/未标注” → 该帧记 visible=0（absent）
序列命名: configs 用 D1/cam2…；数据集目录 dataset1/cam2…；GT 文件名 D{1..4}_cam{2,1,3,3}.csv
输出: §3.2 CSV + README（方法、RMSE、哈希、命令）；--fps 与序列表一致性断言
```

敏感性产物（报告用，不改变主 GT）：`data_derived/gt/sensitivity/<Seq>_cam_shift_pm0.5.csv`（曝光中点 ±0.5 帧时间偏移的 GT 中心）由脚本开关 `--emit-sensitivity` 生成。

MVDT 输入形态：实际**视频为分卷 zip**（`camN.z01`+`camN.zip`），需 `unpack_mvdt.sh`（7z 多卷）解出 `camN.mp4`；configs 用可选 `sequence.video` 指向该 mp4（`io_mvdt.open_sequence` 直接解码视频；无图像序列目录）。MVDT **不发布相机朝向**（仅部分数据集给相机位置），故 `projection_rmse_px = N/A`（无 cross-check）。

### 5.11 合成数据生成（make_synthetic.py + synthetic.py）

参数模型（全部确定性，`numpy.random.Generator(seed)`）：

- 画布：`(H,W) ∈ {(1080p), (1440×1080), (2160p 降采样抽查)}`；背景为天空渐变 + Perlin-like 平滑云层 + 高斯噪声，噪声 σ 由 SNR 档位决定。
- 目标：暗高斯斑（预处理前的暗目标），等效宽度 w ∈ {2,4,8,16} px，对比度 c 由 SNR ∈ {low, mid, high} 决定；运动为匀速（可选固定加速度 < 0.02 px/frame²）。
- 速度覆盖：D1/D2 范围 [−5,5]、D3/D4 范围 [−15,15]，含整数与半整数速度点；30 条短序列（每条 ≥ 120 帧）。
- hard negative：鸟/虫（更大/更快的暗斑）、云（大尺度缓慢暗区）——确保 GT `visible=0` 时系统输出会构成 FP。
- `benchmark/`：单序列，目标 `(0.7, 0)` px/frame，200 帧，含背景纹理与噪声；附全网格真值表（供 S1/S3）。
- 输出 manifest 记录：每条序列参数、seed、目标轨迹真值、帧哈希。

S1–S4 复核流程（PRD §10.1）：

- S1：点目标静止背景 + `(0.7,0)`；brute-force `vx,vy∈[−1,1], 步长 0.1`（441 假设）必须返回 `(0.7,0)`，误差 ≤ 1e-6 或 ≤ 1 个 grid step。
- S2：正确假设 PNR 显著大于错误假设；输出 `PNR_correct − PNR_wrong` 分布（正确=与真值 Chebyshev ≤ max(2δf,0.2)；错误=其余中的最高分假设）。
- S3：`benchmark_search.py` 在 benchmark 序列上跑 4 策略（brute-force 全网格 0.1；multiple hill-climbing 9 起点、步长 0.1、邻居爬山、并行、访问缓存；coarse-to-fine 按 §5.3；coarse-seeded fine-hill-climbing），输出每个策略的 FPS 与是否命中真值。验收：coarse-to-fine 命中且 GPU FPS ≥ 两种 hill-climbing（brute-force 仅作精度上限与 FPS 参照）。
- S4：全部单测通过。

### 5.12 标定协议（calibrate.py，PRD §9 的可执行化）

标定输入与扫描顺序固定（全部写 `calibration_trace.json`，不参看测试序列指标）：

```text
1. k ∈ {3,5,7,9,11}: 在合成 sanity set 上计算 separation(k) =
     median(PNR_correct) − median(PNR_wrong_hypothesis)
   PNR_correct: 运行完整搜索，v_best 与真值 Chebyshev ≤ max(2δf,0.2) 时的 PNR
   PNR_wrong :  候选中排除近真值者，对最高分假设的 PNR（若不存在则跳过该序列）
   tie（差 < 1e-6）→ 取更小 k
2. θ_high_PNR: 在验证片（D1/cam2 前 500 帧、D3/cam3 前 500 帧）上扫描
   1.0..20.0 步长 0.5，最大化 accepted-correct 的 F1@100px；
   多阈值 F1 差 < 0.2% → 取 precision 更高者；仍并列 → 取更大阈值
3. θ_low_PNR: 固定 θ_high 后扫描 1.0..θ_high 步长 0.5，最大化 valid_outcome 且
   precision@100 相对步骤 2 配置下降 ≤ 0.5 个百分点；否则 θ_low = θ_high
4. θ_margin ∈ {0, 0.01, 0.02, 0.05, 0.1}: 诊断运行（仅禁用 margin 门控），
   统计 false accept 中「r_cont > r_cont_low」者数量 N_bad；
   取使 N_bad = 0 的最小 θ；若全部非零 → 取 0.1 并记录
5. θ_edge_PNR = θ_high_PNR + 2（公式固定，不扫描）
6. width_ref(seq) = median(width_px | visible=1 且 frame < min(500,F))，全序列适用；
   由此派生 ρ、r_cont_high、r_cont_low（公式固定）
7. 验证片泄漏检查: 对 D1/D3 同时计算「后段 held-out tail（frame ≥ 500）」指标，
   写入 trace；若 tail 与全序列指标差异 > 5 个百分点，报告显著标注
冻结: 写 thresholds_frozen.yaml（§3.5），含全部输入哈希；此后只读
```

禁止事项（协议代码级保障）：测试序列 D2/cam1、D4/cam3 的任何帧不得出现在步骤 1–7 的扫描输入中；`calibrate.py` 内部对序列名断言白名单 `{D1/cam2, D3/cam3, synthetic}`。

**已知偏差（2026-10-16 代码审查，未重跑）**：`calibrate.py` 用 `f = t + (N−1)//2`（floor）关联 GT，而 `run_sequence.py`/`metrics.py` 按 config 的 `association: round_half_up` 关联。N=5（D1/D2）两者一致；N=8（D3/D4）差 1 帧（floor=3 vs round_half_up=4）。现有 `configs/thresholds_frozen.yaml` 按原协议生成，未重跑；若重新标定，应改用 `MET.associate_frame(t, N, association)` 以与评测口径一致。

### 5.13 静止相机基线（baseline_bg_subtraction.py）

```text
背景模型: 对每序列做滑动中位背景（窗口 50 帧，冷启动用前 50 帧中位）
前景: |I_t − B| 或 (B − I_t)（暗目标模式，默认暗目标） > μ_f + κ·σ_f（κ 默认 3）
连通域: cv2.connectedComponentsWithStats，面积过滤 [1, 4·max_target_area]
跟踪: nearest-neighbor（默认，门限 50px）或 Kalman（--tracker kalman，匀速模型）
输出: 与 §3.3 同 schema 的子集（conf 字段置 null，reason 记 BASELINE_*）
评测: 同一 assoc_frame 关联（公平对比）+ 另报全帧级指标；SPA2D_FRAME_LIMIT 可限帧
备注: D4 4K 下基线逐帧中位背景也较慢/内存受限（本机超时），需降帧数/分辨率
```

分层对比（PRD §10.2）：按运动速度、目标宽度、clutter 等级分层，SPA2D vs 基线 的分层 recall/precision 表由 `make_tables.py` 生成（分层标签从 GT 与合成参数计算；真实序列的 clutter 等级由人工标注 CSV 提供，缺省 unknown）。

---

## 6. 错误处理

### 6.1 错误分类

| Code | 退出码 | 条件 | 恢复行为 |
|------|-------:|------|----------|
| E_CONFIG_INVALID | 2 | schema/约束失败 | 列出全部违规项后退出 |
| E_DATA_MISSING | 3 | 帧目录/视频/标定/轨迹缺失或帧数 < N | 提示 `fetch_data.sh` 或 `make_synthetic.py` |
| E_DATA_HASH | 3 | 帧哈希与 `checksums.sha256` 不符 | `--check` 模式继续报全部差异后退出 |
| E_GT_INVALID | 3 | GT 行数/帧号不连续、visible 行缺 cx/cy | 指出首个违规行号 |
| E_DECODE | 4 | 单帧解码失败（None） | 输出帧索引后终止（不静默跳过） |
| E_NAN | 4 | 帧/preproc 输出含 NaN/Inf | 报帧索引；`--allow-nan-skip` 显式开关才跳过并记日志 |
| E_JAX_DEVICE | 4 | `jax.devices()` 无 GPU 且 backend=jax 且未 `--backend numpy` | 提示换后端或修复驱动 |
| E_OOM_CHUNK | 4 | chunk=1 仍 OOM | 提示降低分辨率/换 numpy 后端并降级重试一次 |
| E_FROZEN_MISSING | 5 | thresholds_frozen.yaml 不存在/哈希冲突/k 不一致 | 提示先跑 `calibrate.py` |
| E_FROZEN_TAMPER | 5 | run_meta 校验不符 | 拒绝运行，除非 `--allow-unfrozen`（会打入报告警告） |

### 6.2 重试策略

- 数据下载：`fetch_data.sh --resume` 支持断点续传；网络错误重试 3 次（指数退避 2s/4s/8s），校验失败不重试同 URL。
- 运行层无自动重试（复现实验必须显式、可审计）；`--repeat N` 是显式重复而非重试。

### 6.3 失败模式

- MVDT 源不可用：合成数据链路完整可用（开发不被阻塞）；`run_all.sh` 在数据缺失时给出明确 Tier B 阻塞声明。
- JAX 版本/驱动不兼容（Blackwell）：提供 `env/requirements.txt` 中 `jax[cuda13]` 与 `jax[cuda12]` 两行备选，安装脚本失败时回退 numpy 后端保证单测与算法验证可跑。
- 16GB 显存不足：自动分块（§8.3）；仍失败 → 建议 `--backend numpy`（慢但正确）。**额外手段**：`SPA2D_CHUNK_BYTES` 显式降分块；`XLA_PYTHON_CLIENT_PREALLOCATE=false`（防 75% 预分配 OOM）；`XLA_FLAGS=--xla_gpu_autotune_level=0`（防 XLA autotune 探测型大分配）。D4 4K 全帧累加在 16GB 显存/15GB 内存上**实测仍 OOM**（见 §11 现状）。
- 分卷 zip 下载损坏：GitHub raw 阻断大文件、未认证 API 限流（60 请求/小时）、aria2 多连接/断点续传可能损坏分卷（曾致 D3 `7z t` Data Error）。必须 `7z t` 校验通过后解包；必要时干净重下或单连接慢速下载。

---

## 7. 安全与合规

### 7.1 数据许可

- `fetch_data.sh` 首次运行需 `--accept-license <LICENSE.md 路径>` 或环境变量 `MVDT_LICENSE_ACCEPTED=1`；provenance.json 记录许可文件哈希。
- 派生标注发布策略（PRD §2.2）：若 license 禁止派生发布，交付物仅含生成脚本、manifest 与哈希；`make_figures.py --no-images` 输出纯统计图（不含原始像素）；失败案例拼图在 license 不明时默认禁用。
- 不发布任何 MVDT 原始帧。

### 7.2 输入校验

- 所有 YAML/JSON 以严格模式解析（未知字段报错）。
- 帧序号、GT 帧号连续性校验；尺寸与 config.resolution 一致性断言。
- 路径遍历防护：输出路径必须位于 `outputs/`、`data_derived/` 或用户显式 `--out`。

### 7.3 数据保护

- 无网络服务、无凭据、无个人数据。URL 清单为纯文本，可包含 token 时提醒用户不要入库（.gitignore 覆盖 `*.urls`）。

---

## 8. 性能

### 8.1 预期负载（实测量级）

| Config | 窗口数 `floor((F−N)/stride)+1` | 全局候选/次 | local 候选/次 | 备注 |
|--------|-------------------------------:|------------:|--------------:|------|
| D1_no_overlap | 1603 | 150（仅首窗） | 49 | F=8016, N=5, s=5 |
| D1_overlap | 8012 | 150 | 49 | s=1 |
| D2_no_overlap | 949 | 414 | 49 | F=4749 |
| D2_overlap | 4745 | 566 | 49 | s=1 |
| D3_overlap | 3548 | 381 | 25 | F=14196, N=8, s=4 |
| D4_overlap | 2732 | 861 | 81 | F=10933, 4K |

像素-假设吞吐（用于 scaling 分析）：每候选每窗 = `N×H×W` 次双线性采样；D4 全局窗 ≈ 861×8×8.3M ≈ 57 Gpx。RTX 5060 Ti 的实际 FPS 全部记录并生成 scaling 表（`pixels×hypotheses/s`），不设绝对门槛（决策 6）。

### 8.2 优化策略

- JIT：`score_batch`/`accumulate_one` 各一次编译；编译时间单列不计 FPS。
- 分块与向量化：候选批 `[B]`，帧 `[N]` 循环在 jit 内展开（N ≤ 8 静态 shape）；`gpu_chunk_bytes` 控制物理批；`SPA2D_CHUNK_BYTES` 可显式覆盖（D1/D2/D3 默认 1.5GiB → chunk≈121/97；过小 chunk 会显著增加 jit 调用次数、拖慢速度——D3 曾因此降到 ~0.08 窗/秒）。
- 帧缓存：设备端当前窗口驻留 + 主机 LRU（默认 2GB）；重叠窗口零重复预处理。
- 预处理多线程：OpenCV `setNumThreads` 记录并固定（config 或环境变量 `SPA2D_CV_THREADS`，默认 4），保证计时可比。
- GPU 稳定运行（RTX 5060 Ti / Blackwell sm_120）：`XLA_PYTHON_CLIENT_PREALLOCATE=false`（关 75% 预分配）、`XLA_FLAGS=--xla_gpu_autotune_level=0`（关 autotune 探测大分配）。缺这两样时，1080p/4K 下易出现 `CUDA_ERROR_OUT_OF_MEMORY`。

### 8.3 内存模型

- 单帧 float32：1080p=8.3MB、1440×1080=6.2MB、4K=33.2MB。
- 窗口驻留：N×帧（D4: ~266MB 设备 + 主机缓存）。
- 搜索分块公式：`chunk = clamp(floor(gpu_chunk_bytes/(2·H·W·4)), 1, 512)`。D4 默认 1.5GiB → chunk≈22；1080p → chunk=512 封顶。
- 主机 LRU：按预算驱逐，记录命中率到 `run_meta.json.cache_stats`。

### 8.4 环境与依赖（env/requirements.txt 基线）

```text
python>=3.13,<3.14
jax[cuda13]>=0.7          # 备选: jax[cuda12]>=0.7（sm_120 需 CUDA 12.8+ 轮子）
numpy>=2.1
opencv-python>=4.11
pyyaml>=6
pandas>=2.2
matplotlib>=3.9
tqdm>=4
pytest>=8
```

安装后 `scripts/`（或 README）提供验证脚本：`python -c "import jax; print(jax.devices())"` 与 `cv2.getBuildInformation()` 摘要写入 `run_meta.json`。lock 文件由 `pip freeze > env/requirements.lock` 生成并在每次环境变更时更新。

---

## 9. 测试策略

### 9.1 单元测试（tests/）

| 文件 | 覆盖 | 关键断言 |
|------|------|----------|
| `test_config.py` | schema 校验、六配置参数对照 PRD 表 | 违规 YAML 全被拒；六配置逐值匹配 |
| `test_preprocess.py` | 预处理器 | 闭运算减原图 ≥0；kernel k∈{3..11} 形状；输出范围 [0,1] |
| `test_accumulate_shift.py` | 双线性 vs `cv2.remap` | 随机图 + 随机亚像素位移，max|Δ|≤1e-4；valid mask 与手算一致 |
| `test_search_grid.py` | 网格/邻域/去重/top-K | 候选数量=§5.3 表；并列稳定选择；去重语义 |
| `test_peak_subpixel.py` | 峰值/亚像素 | 合成抛物面恢复已知偏移（≤1e-3）；degenerate 情形回退整数 |
| `test_confidence.py` | PNR/margin | 手算小样本逐一对照；背景不足保护；single_candidate |
| `test_state_machine.py` | 转换表 + reason 优先级 | 表驱动：每状态 × 接受/拒绝组合断言 state_out/reason/flags；OUT_OF_FRAME 周期跳过 |
| `test_metrics.py` | 指标 golden | 构造 10 窗口小记录集，手算 recall/precision/RMSEp/RMSEv/valid_outcome；半帧两模式 |
| `test_io_gt.py` | GT 读取/关联 | NaN 行、out_of_frame、assoc_frame 两种模式 |
| `test_synthetic.py` | 生成器 | 同 seed 逐位可复现；GT 与像素一致（目标位置注入校验） |
| `test_metrics.py` | 指标 golden | 构造 10 窗口小记录集，手算 recall/precision/RMSEp/RMSEv/valid_outcome/FP 来源；半帧两模式 |
| `test_io_gt.py` | GT 读取/关联 | NaN 行、out_of_frame、assoc_frame 两种模式；真实 MVDT 格式（K-matrix/distCoeff、detections/rtk、空白表头）、fetch 全链、make_derived_gt（published2d/project3d/auto-crosscheck/sensitivity/(0,0)→absent）、真实 MVDT 集成（skipif） |
| `test_run_sequence.py` | 端到端 | numpy 全链、windows 结构/timing 四层/run_meta/assoc_frame/reason、--limit、--repeat 3 极差、frozen 退出码 5、--allow-unfrozen、config 非法退出码 2、数据缺失退出码 3、stride=1 缓存命中率 |
| `test_calibration.py` | 标定 | 7 步产物 frozen+trace、白名单拒绝、--force 守卫 |
| `test_benchmark_search.py` | S3 | climb 恢复速度、visited 缓存、run_benchmark 行结构/CTF hit、CLI numpy 警告+CSV、缺失 benchmark 退出码 3 |
| `test_baseline_bg_subtraction.py` | 基线 | nn 检出目标、kalman、缺 GT 退出码 3 |
| `test_tables_figures.py` | 表/图 | summary/sensitivity/paper_vs_ours/stratified 产出、fig7/fig8、--no-images |
| `test_run_all_report.py` | 一键复跑 | smoke 端到端（report 九节 + Tier B）、real 缺 frozen 退出码 5（存在时 skip） |
| 补充（后续新增） | 匹配滤波/加速度/运动相干 | `test_matched_filter_selects_small_over_large_blob`（小斑 vs 大斑选中）；`test_accumulate_window_accel_matches_manual`、`test_local_search_accel_recovers`、`test_min_speed_motion_coherence`；`test_no_forbidden_enhancement_in_source` 仅主链路禁滤波、放行 matched_filter_frame 的 DoG |

### 9.2 集成测试（pytest -m integration，或用小序列）

- 合成 120 帧序列端到端：`make_synthetic → run_sequence` 输出窗口记录、metrics.json 结构合法、FPS 计时层齐全。
- S1（brute-force 0.7 恢复）、S2（PNR 分离分布）、S3（四策略 bench，RTX 5060 Ti 上跑，断言 coarse-to-fine 命中且 FPS ≥ hill-climbing 两变体——若 GPU 上不稳定，允许标记 flaky 并在报告中以 CSV 为验收证据）。

### 9.3 边界用例

- N 偶数的半帧中心（N=8 → c=t+3.5）在 metrics 两种关联模式。
- 目标出帧/再入：状态机 LOST→OUT_OF_FRAME→TRACKED 全链。
- BORDER_CONSTANT 边缘：峰值落在 edge_margin 内必须走 EDGE 路径。
- GT 含 NaN、全不可见序列（metrics 除零保护：visible_den=0 → recall=NaN 并警告）。
- 去重后候选恰为 1（margin=1.0 路径）。
- D4 分块（chunk<batch）与 D1 整批结果一致（同一合成序列两 chunk 大小对比）。

### 9.4 验收映射（PRD → 实现/测试）

| PRD 项 | SPEC 落实 | 验证 |
|--------|-----------|------|
| S1 | §5.2/§5.3 + benchmark brute-force | `test_search_grid` + integration S1 |
| S2 | §5.5 PNR | integration S2 分布报告 |
| S3 | §5.11 + benchmark_search.py | outputs/metrics/s3_search.csv 断言 |
| S4 | §9.1 | `pytest` 全绿 |
| Tier B 精度门槛（D1/D2 recall≥95% prec≥97%；D3 88%/90%；D4 80%/90%） | §5 全链路 + 标定 | `make_tables.py` 输出对照表 + report.md 显式断言 |
| FPS 门槛 | 决策 6：相对分析 + scaling 表 | s3_search.csv + timing 表（绝对 FPS 为报告项） |
| 敏感性（{10,20,50,100}px、半帧、重复极差） | §5.8 | sensitivity.csv |
| 基线（中位背景+连通域+NN/Kalman） | §5.13 | runs_baselines + 分层表 |
| 失败案例 ≥10 FA / ≥10 miss | make_figures | figures 产出 + reason 标注 |
| 无测试集调参 | 冻结阈值 + 白名单断言 | run_meta 哈希 + calibrate 白名单单测 |
| DoD 全项 | §10 阶段完成 | run_all.sh 一键复跑 |

---

## 10. 实施计划

### 10.1 阶段

```text
Phase 0 脚手架（I-01, I-02）: 仓库/环境/config schema/records/合成生成器
Phase 1 核心算法（I-03..I-06）: preprocess / accumulate / peak / search + 单测 + S1/S2
Phase 2 数据与门控（I-07..I-10）: S3 bench / MVDT 获取+派生GT / confidence / 状态机
Phase 3 端到端（I-11..I-13）: run_sequence / 标定冻结 / metrics
Phase 4 交付（I-14..I-16）: tables+figures / baselines / run_all+报告
```

增量原则：Phase 1 结束前不接触真实数据；D1/cam2 是首个真实序列冒烟对象；D2/D4 在标定冻结后运行（协议禁止用它们调参）。

### 10.2 Issue 映射

| Issue | 标题 | SPEC 章节 | 优先级 | 依赖 |
|-------|------|-----------|--------|------|
| I-01 | 仓库脚手架 + 环境锁 + config/records schema | §2.4, §3.1, §8.4 | high | — |
| I-02 | 合成数据生成器 + manifest + 单测 | §5.11, §3.4 | high | I-01 |
| I-03 | 预处理模块 + 单测 | §5.1, §9.1 | high | I-01 |
| I-04 | 累加/双线性模块（JAX+NumPy）+ cv2 对照测试 | §5.2, §2.3 | high | I-01 |
| I-05 | 峰值提取/亚像素/边缘保护 + 单测 | §5.4 | high | I-04 |
| I-06 | 搜索（global/local/去重/分块）+ 单测 + S1/S2 | §5.3, §5.7 | high | I-02, I-04 |
| I-07 | S3 四策略基准脚本 | §5.11 | medium | I-06 |
| I-08 | MVDT 获取脚本 + io_mvdt + 派生 GT | §4.1, §5.10, §3.2 | high | I-01 |
| I-09 | 置信度模块（PNR/margin/连续性）+ 单测 | §5.5 | high | I-05 |
| I-10 | 状态机 + 转换表测试 | §5.6 | high | I-09 |
| I-11 | run_sequence 端到端 + 计时 + run_meta | §2.3, §3.3, §5.9 | high | I-06, I-08, I-10 |
| I-12 | calibrate + 冻结阈值 | §5.12, §3.5 | high | I-02, I-03, I-06, I-09, I-11 |
| I-13 | metrics 模块 + golden 测试 | §5.8, §9.1 | high | I-11 |
| I-14 | make_tables + make_figures | §4.1, §5.8 | medium | I-13 |
| I-15 | 基线（背景建模；光流可选） | §5.13 | medium | I-08, I-13 |
| I-16 | run_all + 三跑复现 + report.md（含 Tier 声明） | §4.1, §6, PRD §11 | high | 全部 |

### 10.3 增量交付

1. 单测 + 合成 S1–S4 全绿（可在无 MVDT 数据时完成）。
2. D1_no_overlap + D1_overlap 真实序列跑通并出指标。
3. 标定冻结 → D2/D3/D4 全量运行（冻结后一次跑完，不回头改阈值）。
4. 基线、敏感性、失败拼图、报告齐备；`run_all.sh` 从零一键复跑验证。

---

## 11. 开放问题与风险

### 11.1 未决问题

- MVDT 官方下载地址/镜像可用性未验证（需执行 I-08 时确认；URL 以用户提供为准，不猜测）。
- MVDT 发布物是否含 2D 标注：`--method auto` 两条路径都实现，实际以数据为准。
- D1/D2 “Target size 90–400/200–850” 的解释（bbox 宽 vs 面积）：报告按 PRD §4.2 双解释 + 敏感性表处理，不与 GT 生成强绑定。
- Jetson 节（`env/Dockerfile.jetson`、50W 模式、绝对 FPS）是否在本轮实施：暂标记为可选，确认硬件后再启用。

### 11.2 技术风险

| 风险 | 影响 | 缓解 |
|------|------|------|
| JAX 对 sm_120（Blackwell）轮子适配 | 主后端不可用 | `jax[cuda13]`/`jax[cuda12]` 双备选 + NumPy 后端保底；安装验证脚本 |
| 16GB 显存 vs D4 4K 全帧累加 | D4 运行失败/极慢 | 分块公式 + chunk=1 降级 + numpy 回退；**实测即便分块 + autotune=0 + 预分配关闭，D4 4K 全帧仍 OOM**（16GB GPU/15GB RAM）→ D4 需更大显存/内存或 ROI/降分辨率 |
| 相机不同步/rolling shutter（ad-hoc 数据集） | GT 时间偏差进入 2D 中心 | GT 敏感性（±0.5 帧）+ 投影 RMSE 报告；结论限定 static ground-camera proxy |
| 阈值全局 vs 序列差异 | D2/D4 指标偏低 | 协议禁止测试集调参；失败先查阈值/状态机（PRD §12）；如实报告 |
| 窗口关联半帧（N=8） | 指标关联歧义 | 双模式报告；主指标 round-half-up |
| GPU 归约非确定性 | 3 次运行指标抖动 | `--repeat 3` 极差写入报告；选择逻辑（argmax/top-K）确定性化 |
| **真实数据静止暗杂波主导** | 纯形态学暗增强锁杂波（D1 recall 0） | few-pixel DoG 匹配滤波 + 尺度先验（已实现，D1/D2 recall/precision@100=1.0） |
| **D3 stress（快目标+杂波+间歇可见）** | recall 仍 0（含 min_speed/4D (v,a)；oracle 完美对齐仅在部分段可分离） | 需背景抑制（时序中位背景减除）/ROI 先验；记录为未完成（见附录 C） |
| **分卷 zip 下载损坏** | `7z t` Data Error、mp4 残缺 | `7z t` 必验；干净重下/单连接慢速下载 |

### 11.3 假设

- PRD 为唯一需求源；本 SPEC 的精确化解释（状态属性化、reason 优先级、OUT_OF_FRAME 周期、width_ref 规则、θ_margin 操作化）如与后续作者 artifact 冲突，进入 Tier A 时重新对齐。
- 用户决策已知：无 Jetson（FPS 相对化）、数据先合成后真实、单 SPEC、C1 出范围。
- 硬件：x86_64 + RTX 5060 Ti 16GB、767GB 空闲磁盘、miniforge python3.13，均已在 2026-10-10 核实。

---

## 附录 C：实现现状（与 SPEC 对齐，2026-10-12）

**已按 SPEC 实现**：I-01..I-16 全部交付；`spa2d/`（config/records/preprocess/accumulate/search/peak/confidence/state_machine/metrics/timing/io_mvdt/synthetic/frame_cache/backends）、全部 CLI、单测 **228 passed / 1 skipped**（GPU 标记在可用时启用；real-mode run_all 在 frozen 存在时 skip）。

**相对 SPEC 的实现补充（已实现，已并入正文）**：
- **符号修正**：累加用 `x + k·v`（§5.2）。
- **few-pixel DoG 匹配滤波 + 尺度先验**：`preprocess.matched_filter(_multiscale)`；`SPA2D_MATCH_SIGMA(S)`；frozen `preprocess.matched_sigma = clamp(width_ref/3,1.2,4.0)=2.6667`（§5.1、§3.5）。
- **运动相干 `min_speed`** 与 **4D (v,a) 加速度搜索**：`global_search_accel/local_search_accel/accumulate_window_accel/score_batch_accel/center_frame_position_accel`；`SPA2D_ACCEL` + `SPA2D_A_*`（§5.2/§5.3/§5.7、§4.2）。
- **真实数据适配**：MVDT 分卷 zip → `unpack_mvdt.sh`（7z）得 mp4；config 可选 `sequence.video`；`io_mvdt.FrameSource` 常驻顺序读取（§3.1/§5.10、§11）；GT `(0,0)` 视为 absent（§3.2/§5.10）；`fetch_mvdt_videos.sh`（限流自适应下载）。
- **GPU 运行**：`SPA2D_CHUNK_BYTES` 显式分块、`SPA2D_START_FRAME/LIMIT/FRAME_LIMIT`、`XLA_PYTHON_CLIENT_PREALLOCATE=false`、`XLA_FLAGS=--xla_gpu_autotune_level=0`（§6/§8）。
- **真实冻结阈值**：θ_high=20, θ_low=1.0, θ_margin=0（静止目标）, θ_edge=22；k=5；matched_sigma=2.6667（`configs/thresholds_frozen.yaml`）。
- **环境补充**：`pyproject.toml` 屏蔽 ROS 注入的 pytest 插件（`/opt/ros/jazzy`）；`AGENTS.md`（仓库运行/约定手册）；`docs/review_icra27_5428_en.md`（论文评审）。

**实测结果（真实 MVDT，RTX 5060 Ti，有界窗口）**：
- D1/cam2（匹配滤波 σ=2.667，前 500 帧/100 窗）：recall@100=1.0，precision@100=1.0，valid_outcome=1.0，RMSEp=2.84px（全 OK）；σ=1.6/3.5 同为 1.0。
- D2/cam1（匹配滤波 σ=2.667，前 600 帧/120 窗）：recall@100=1.0，precision@100=1.0，valid_outcome=1.0，RMSEp=17.13px。
- D3/cam3（stress，目标快+杂波+间歇可见）：recall@100=0（含 min_speed 与 4D (v,a)；oracle 完美 GT 对齐仅在部分段可分离）→ **未完成**，需背景抑制/ROI。
- D4/cam3（3840×2160）：4K 全帧累加在 16GB GPU/15GB RAM **OOM**，未产出（需更大显存/内存或 ROI/降分辨率）。
- 对比论文目标值：D1/D2 子集达标（FPS 为桌面 GPU 相对值，不做绝对对照）；D3/D4 未达标 → **Tier B 结论以 D1/D2 为准**。

**已知未完成/风险（与 §11.2 一致）**：D3 杂波/运动、D4 显存、MVDT 分卷下载不稳、阈值/预处理与论文差异（缺作者 artifact）。

---

## 附录 A：论文关键引用（→ 实现锚点）

- Eq.1/7 累加公式、Eq.2/8 得分、Eq.9 中心帧位置 → §5.2/§5.4（**符号修正**：实现用 `x + k·v`）
- Eq.6 预处理、Eq.10 初始化候选集、Eq.11 局部候选 → §5.1/§5.3（实现另加 **DoG 匹配滤波** 与 **4D (v,a)**，超出论文主链路）
- §III 四策略（brute-force / multiple hill-climbing / coarse-to-fine / coarse-seeded fine-hill-climbing）与 batch=512、coarse δ=0.5、top-5 → §5.3/§5.11
- Table I 序列参数、Table II 目标指标与 N/stride/搜索配置 → §3.1/§8.1/§5.8
- Jetson AGX Orin 50W、16–28 FPS（ABU-like）→ 决策 6 与 §5.9

## 附录 B：PRD 映射索引

| PRD 章节 | SPEC 章节 |
|----------|-----------|
| §3 术语 | §3（数据模型）、§5（算法约定） |
| §4 数据与派生 GT | §3.2, §4.1, §5.10 |
| §5 算法规范 | §2.3, §5.1–§5.7 |
| §6 指标 | §5.8 |
| §7 性能与计时 | §5.9, §8 |
| §8 交付物 | §2.4, §4.1 |
| §9 标定协议 | §5.12, §3.5 |
| §10 验收 | §9.4, §10 |
| §11 报告模板 | §4.1（make_tables/figures）、I-16 |
| §12 风险 | §11 |
| §13 DoD | §9.4, §10.3 |
