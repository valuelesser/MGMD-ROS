# SPEC: MGMD × MVDT 复现（MGMD 检测器在本仓库的移植与评测）

> 技术规格，派生自: `tasks/mgmd_mvdt_reproduction_prd.md`
> Generated: 2026-10-14 | Target branch: main | Commit: （仓库尚无提交）
> 官方参考实现: https://github.com/WindyLab/Global-Local-MAV-Detection @ `main`
> （MOD2.py sha1-blob 656bf36、Functions.py 182b08a、GLAD.py a71262f，常数已逐行提取）

---

## 1. Summary

### 1.1 本 SPEC 覆盖范围

MGMD 三模块流水线（MFE 运动特征增强 → TF 轨迹过滤 → LAC/LAD 外观精检）在本仓库的
工程化移植：新增 `mgmd/` Python 包（镜像 `spa2d/` 布局）、4 份序列配置、
`scripts/run_mgmd.py` CLI、权重获取脚本、消融与报告脚本。使用官方预训练权重做
纯推理，不训练。产出与 `spa2d.records` / `make_tables.py` 兼容的 run 记录，
在 MVDT D1–D4 上给出半径匹配 P/R/F1 与论文 Table 3 风格消融。

### 1.2 PRD 引用

- Source: `tasks/mgmd_mvdt_reproduction_prd.md`
- User Stories 覆盖: US-001 ~ US-008（全部）
- Functional Requirements 覆盖: FR-1 ~ FR-14（全部）

### 1.3 关键设计决策（含对 PRD 的两处偏差）

| # | 决策 | 选择 | 理由 |
|---|------|------|------|
| D1 | MFE 算法口径 | **官方代码口径**（两帧差分 + 自适应阈值），**偏差于 PRD/论文**的三帧差分 OR 描述 | 用户裁决（2026-10-14）：官方 MOD2.py 是论文唯一可核对实现；论文三帧差分无对应官方代码。PRD US-003 的「三帧差分/OR/k 可配」条款以本 SPEC §5.2 为准 |
| D2 | 帧差阈值 | `T = 5 + int(mean(diff))`（官方自适应），base=5 可配 | 官方 MOD2.py:30；论文阈值 5 即 base |
| D3 | yolov5 依赖 | ~~pip 固定 tag v6.0~~ → **codeload tarball + `third_party/yolov5-v6.0/`**（`scripts/fetch_yolov5_src.sh`，sha256 预 pin，`mgmd.detect` 运行时 sys.path 注入） | 实施期修正：v6.0 tag 无 setup.py 不能 pip 安装；且权重 pickle 引用顶层 `models.*`/`utils.*` 路径，PyPI yolov5 7.x（`yolov5.models.*`）unpickle 不兼容。官方 INSTALL.md 明确 YOLOv5 v6.0；pin tarball 可复现且离线可用 |
| D4 | SORT 依赖 | 新增 `scipy`（Hungarian）+ `filterpy`（Kalman） | 用户裁决；官方仓库无 SORT 代码，按论文+PRD 参数实现 |
| D5 | run 记录 | 逐帧伪窗口复用 `WindowRecord`（N=1, stride=1），`spa2d` 零改动 | PRD FR-9；`metrics._N_from`/`associate_frame` 对 N=1 自洽；`make_tables.py` 免改 |
| D6 | 每帧输出上限 | 至多 1 个最终框（选择规则见 §5.6） | MVDT 单目标 + 评测「每帧至多一个 TP」约定 |
| D7 | D4 4K | config `runtime.downscale: 0.5`（→1920×1080 处理），输出坐标 ×2 映射回原图 | 16GB GPU 约束；对 recall 影响列入实验（§11.2） |
| D8 | diff 掩膜减法 | `cv2.subtract`（饱和）替代官方 numpy uint8 回绕减法 | 官方 `thresh - mask` 在 0−255 处回绕为 1 属实现瑕疵；静态相机 mask 仅边缘窄带，影响可忽略，报告中标注 |
| D9 | torch 加载旧权重 | `mgmd.detect` 内封装 `torch.load(..., weights_only=False)` | torch≥2.6 默认 `weights_only=True` 会拒绝 yolov5 v6.0 时代 .pt；sm_120 需 torch≥2.7(cu128)，无法降级回避 |
| D10 | 权重 sha256 | 实值预 pin（§3.4），非 TOFU 占位 | 本次 SPEC 阶段已从官方仓库下载并核对大小+zip 完整性 |
| D11 | 差分帧间隔 k | 新增 `mfe.frame_gap`（默认 1=官方两帧口径；D1–D4 配 32/3/2/1） | issue-024 实跑证据：MVDT 目标位移中位 1.13–7.56 px/帧，k=1 时 D1 GT 处差分峰值仅 3 < T=5（亚像素位移），官方口径结构性失效；PRD US-003 本就预留「k 可配」。k>1 时前 k 帧无差分对 → ABSENT。仍属官方算法结构（阈值/形态学/一致性不变） |
| D12 | 论文口径 OR 聚合 | 新增 `mfe.or_aggregation`（默认 false；D1 启用）：`binary = th(|It−Ît−k|) OR th(|It−Ît+k|)`，离线用未来帧（PRD §7 允许 k 帧延迟，报告标注）；逐框一致性取各对最大 score；裁剪扩框量 = `enlarge_px + 2·(frame_gap−1)` | D1 两帧口径上限 ~92%，OR 提升至 99%+（±k 双向运动证据）；属 PRD 原始论文描述 |
| D13 | 候选溢出策略 | 新增 `mfe.overflow`（drop=官方整帧弃检 / topk=按面积截断；D1 用 topk） | OR 使噪声 blob 翻倍触发 >max_boxes 整帧弃检（11/295 帧），topk 保留最大候选避免丢帧（D1 98%→含弃检损失） |
| D14 | 逐框运动一致性 | 新增 `mfe.motion_coherence`（默认 true=官方；MVDT 全序 false） | 官方向 ShiTomasi+LK 一致性为移动相机视差设计；MVDT 静态相机弱纹理小目标无稳定角点 → 误杀（D1 GT 命中 48%→100%）。假阳性交由 TF + LAD 过滤；score 改用 blob 差分峰值 |
| D15 | LAC 域失效处理 | 新增 `lac.prob_thresh=0` 直通档（MVDT 全序）；此时保持上游排序键，仅执行概率计算 | 官方 Net_best.pth 在 MVDT 上 P(MAV)≈0（GT 裁剪 0.000 vs 背景 0.004），硬阈值 0.5 → 全流水线零输出；LAD 域内有效（8/8, conf≈0.89）承接外观筛选 |
| D16 | SORT 滑行输出 | 已确认轨迹在未匹配帧（tsu≤max_age）输出 Kalman 预测框 | 与 D15 同源：MVDT 帧差在近静止相位无信号，MFE 无候选 → 轨迹 tsu>0；经典 SORT 仅输出本帧匹配轨迹，导致可见帧漏检 5.6%（D1 实测 219/3935）。轨迹一旦确认（Algorithm 1 粘性）即持续输出，帧内平滑 |

---

## 2. Architecture

### 2.1 系统上下文

```
data_raw/mvdt/D*/cam*/camN.mp4 ──► mgmd.io_mvdt (包装 spa2d.io_mvdt.FrameSource)
                                        │ 逐帧 BGR（可 downscale）
                                        ▼
┌─ MFE ─ align.py: 网格LK+RANSAC单应对齐（Î_{t-1}→t）
│        diff.py : absdiff→自适应阈值→掩膜→中值→形态学→连通域→候选框
│                  + 逐框运动一致性过滤（ShiTomasi+LK）
├─ TF ── track.py: SORT(scipy/filterpy) + 论文 Algorithm 1 轨迹分类
├─ LAC ─ classify.py: Net(32×32 CNN, Net_best.pth) MAV 概率过滤        [懒加载 torch]
└─ LAD ─ detect.py: 320×320 裁剪 + yolov5s_GLAD-crop.pt 精炼框          [懒加载 torch/yolov5]
                                        ▼
              pipeline.py（--modules 子集开关）→ WindowRecord(N=1) 逐帧
                                        ▼
   outputs/runs_mgmd/<run>/{windows.jsonl, run_meta.json, timing.json}
                                        ▼
   scripts/make_tables.py（复用） + scripts/make_report_mgmd.py（新增）
```

### 2.2 组件职责

| 模块 | 文件 | 职责 | 依赖 |
|------|------|------|------|
| config | `mgmd/config.py` | YAML schema 加载校验，`MgmdError`（`[mgmd] code: msg → remediation`） | pyyaml |
| io | `mgmd/io_mvdt.py` | 包装 `spa2d.io_mvdt.FrameSource`（常驻 capture 顺序读帧），附加 downscale | spa2d, cv2 |
| align | `mgmd/align.py` | 网格关键点 LK + RANSAC 单应；失败回退恒等 + warning（FR-3）；产出对齐帧与掩膜 | cv2, numpy |
| diff | `mgmd/diff.py` | 两帧差分、自适应阈值、形态学、连通域、逐框运动一致性 → 候选框 | cv2, numpy |
| track | `mgmd/track.py` | SORT + Algorithm 1 | scipy, filterpy（均懒加载） |
| classify | `mgmd/classify.py` | LAC：Net 定义 + 推理 | torch/torchvision（懒加载） |
| detect | `mgmd/detect.py` | LAD：320×320 裁剪 + YOLOv5s-crop 推理 | torch + yolov5@v6.0（懒加载） |
| pipeline | `mgmd/pipeline.py` | 逐帧编排、模块子集、每帧 ≤1 框选择、WindowRecord 生成 | 以上全部 |

### 2.3 模块交互（数据流）

逐帧 `t`：`read(t)` →（downscale）→ `align.align(prev_gray, cur_gray) → (Î_prev, mask, H)`
→ `diff.candidates(cur, Î_prev, mask) → [Rect]+coherence_score`
→（可选）`track.update(rects) → confirmed_tracks（Algorithm 1）`
→（可选）`classify.mav_prob(crop) ≥ θ_lac`
→（可选）`detect.refine(frame, center) → bbox+conf`
→ 选择规则（§5.6）→ `WindowRecord`（坐标经 1/downscale 映射回原图）。

模块关闭时该级为恒等直通（§5.7 消融语义）。帧缓冲仅需 `prev` 一帧（官方两帧差分），
无未来帧依赖——PRD/论文的 `It+k` 延迟在 D1 口径下不存在（官方 MOD2 即因果两帧）。

### 2.4 文件结构

```
mgmd/                          [NEW PACKAGE]
├── __init__.py                # 版本/导出
├── config.py                  # schema + MgmdError + load_config（退出码 2）
├── align.py                   # 网格 LK + RANSAC 单应 + 掩膜
├── diff.py                    # 差分 + 形态学 + 连通域 + 运动一致性
├── track.py                   # SORT + Algorithm 1
├── classify.py                # LAC Net（torch 懒加载）
├── detect.py                  # LAD YOLOv5s-crop（torch/yolov5 懒加载）
├── pipeline.py                # 编排 + WindowRecord 适配
└── io_mvdt.py                 # FrameSource 包装 + downscale
configs/
├── mgmd_D1.yaml … mgmd_D4.yaml [NEW ×4]
scripts/
├── fetch_mgmd_weights.sh      [NEW] US-001
├── run_mgmd.py                [NEW] US-006 CLI
├── make_report_mgmd.py        [NEW] US-007/US-008 消融+对比报告
└── make_tables.py             [MODIFY: 仅当 paper_vs_ours 对未知 config 报错时加防御分支]
tests/
├── test_mgmd_config.py        [NEW]
├── test_mgmd_mfe.py           [NEW]
├── test_mgmd_tf.py            [NEW]
├── test_mgmd_appearance.py    [NEW]
└── test_mgmd_pipeline.py      [NEW] smoke/integration
env/requirements.txt           [MODIFY: 分组追加 mgmd 依赖]
data_raw/mgmd_weights/         [NEW, git-ignored] Net_best.pth / yolov5s_GLAD-crop.pt / README.md
outputs/runs_mgmd/             [NEW, git-ignored]
outputs/report_mgmd.md         [NEW, git-ignored]
```

`spa2d/` 包**不修改**（PRD Non-Goal）；records/metrics 原样复用。

---

## 3. Data Model

### 3.1 配置 schema（`mgmd/config.py`，schema_version=1）

```yaml
schema_version: 1
config_name: mgmd_D1
sequence:                # 语义与 spa2d.SequenceConfig 完全一致
  name: D1/cam2          # D{1..4}_cam{2,1,3,3}；决定 GT 文件映射 D1_cam2.csv
  frames_dir: data_raw/mvdt/D1/cam2/frames
  frame_glob: "*.jpg"
  fps: 50.00
  resolution: [1920, 1080]
  color_order: BGR
  video: data_raw/mvdt/D1/cam2/cam2.mp4   # 可选，与 frames_dir 二选一（同 spa2d 约定）
mfe:                     # 默认值 = 官方 MOD2.py/Functions.py 常数（§5.1/§5.2）
  blur_kernel: 11
  grid_spacing: [96, 72]        # 官方 gridSizeW=32*3, gridSizeH=24*3
  lk_win_size: 15
  lk_max_level: 3
  max_flow_px: 50               # 官方运动点 >50px 丢弃
  min_inliers: 9                # 官方 <9 回退恒等（FR-3）
  ransac_thresh: 3.0
  frame_gap: 1                  # 差分帧间隔 k（D11；1=官方两帧，D1–D4 配 32/3/2/1）
  or_aggregation: false         # 论文口径 OR 聚合（D12；D1 启用）
  overflow: drop                # 候选溢出策略（D13；drop=官方，D1 topk）
  motion_coherence: true        # 逐框运动一致性（D14；官方 true，MVDT 全序 false）
  diff_thresh_base: 5           # T = base + int(mean(diff))
  diff_adaptive: true           # false → T=base 固定（论文字面口径，对照用）
  median_blur: 5
  open_iter: 1
  close_iter: 3
  morph_kernel: 3
  area_min: 30                  # 官方 contourArea 下界（开区间）
  area_max: 3000
  ratio_min: 0.6                # w/h 开区间
  ratio_max: 3.0
  max_boxes: 50                 # 超过则整帧弃检（官方 'too much bboxes'）
  enlarge_px: 2                 # 官方 a=2
  motion_avg_dist_min: 1.0      # 运动一致性：avg_dist 下界
  motion_ratio_theta_max: 0.8   # std/mean(θ) 上界
  motion_ratio_dist_max: 0.8    # std/mean(dist) 上界
tf:
  iou_thresh: 0.3
  max_age: 3
  min_hits: 1
  alg1_age: 3                   # 轨迹分类最小 age
  alg1_window: 3                # D(s_{N-w}, s_N) 的 w
  alg1_d_min: 1.0               # ρ/d 阈值（FR-5 可配）
lac:
  weights: data_raw/mgmd_weights/Net_best.pth
  input_size: 32
  prob_thresh: 0.5              # 官方为 argmax，2 类时等价 P≥0.5
  device: auto                  # auto|cpu|cuda
lad:
  weights: data_raw/mgmd_weights/yolov5s_GLAD-crop.pt
  crop_size: 320                # 官方 enlarge_region2 a=160 → 320×320
  conf_thresh: 0.5
  imgsz: 320
  device: auto
metrics:                       # 与 spa2d.MetricsConfig 同构
  radii_px: [10, 20, 50, 100]
  primary_radius_px: 100
  association: round_half_up
runtime:
  downscale: 1.0                # D4 配 0.5；输出坐标 ÷downscale 映射回原图
  limit: null                   # 帧上限（--limit 可覆盖）
backend: numpy                  # MFE/TF 计算后端标记（记录用；外观模块独立 device）
seed: 0
```

校验规则（非法 → 退出码 2，`E_CONFIG_INVALID`）：未知字段拒绝；必填缺失拒绝；
`0<ratio_min<ratio_max`、`area_min<area_max`、`0≤prob_thresh≤1`、`0≤conf_thresh≤1`、
`0<downscale≤1`、`iou_thresh∈(0,1)`、`max_age≥1`、`min_hits≥1`、`alg1_age≥2`、
`alg1_window≥1`、`crop_size≥64 且为 32 倍数`、`primary_radius_px∈radii_px`、
`association∈(round_half_up, floor)`、`device∈(auto,cpu,cuda)`。
CLI 校验入口：`.venv/bin/python -m mgmd.config configs/mgmd_*.yaml`。

### 3.2 Run 记录映射（WindowRecord 逐帧伪窗口，D5）

| WindowRecord 字段 | MGMD 取值 |
|---|---|
| window_index / t_start / t_center / assoc_frame | 帧索引 f（`metrics._N_from` ⇒ N=1，`associate_frame` ⇒ f） |
| search_mode | `"global"`（MFE 始终全帧） |
| state_in / state_out | 有输出且有确认轨迹: `"TRACKED"`；有输出无轨迹（消融 mfe 单档）: `"INIT"`；无输出: `"ABSENT"`；state_in=上一帧 state_out |
| accepted / target_present | 本帧是否输出框（两者同值） |
| reason | 输出: `"OK"`；无输出: `"ABSENT"` |
| est | `Estimate(cx, cy, vx, vy)`：bbox 中心（原图坐标）+ Kalman 速度；无 TF 时 vx=vy=0 |
| peak / continuity | `None`（schema 允许） |
| confidence | 有输出时 `Confidence(pnr=key, margin=key, mu_bg=0, sigma_bg=0, single_candidate)`：`pnr` 承载**选择得分**（依模块档：MFE motion score / LAC P(MAV) / LAD conf），供半径匹配口径 AP（§9.3 stretch）排序；无输出为 `None` |
| flags | 默认 `Flags()`（stat/edge 均 False） |
| miss_count / reacq_count | 轨迹 miss 计数 / reacq 计数（无 TF 时 0） |
| skipped_by_period | False |
| timing_ms | `t_pre`=MFE ms，`t_search`=TF+LAC+LAD ms，`t_state`=选择+记录 ms，`t_total` |

`run_meta.json` 兼容 `make_tables.load_runs` 所需键：`config.name`、
`config.snapshot.sequence.name`、`backend`；另增 `mgmd` 段：`modules`、
权重文件名+sha256、torch/yolov5 版本、downscale、官方代码 blob sha。
`timing.json` 含 `fps.window_fps` / `fps.effective_frame_fps`。

### 3.3 权重清单（`data_raw/mgmd_weights/`，git-ignored）

| 文件 | 大小 (bytes) | 用途 | 必需性 |
|---|---|---|---|
| Net_best.pth | 248841 | LAC | 必需 |
| yolov5s_GLAD-crop.pt | 14454645 | LAD | 必需 |
| yolov5s_GLAD.pt | 14359541 | 可选全局检测器基线 | 可选 |

### 3.4 sha256 预 pin（D10，写入 fetch 脚本常量）

```
Net_best.pth            0d3a4631060339ee72551a89e95887fc49a2d50f9f3473ef354cf3db2830faf5
yolov5s_GLAD-crop.pt    031813a6c7ef8446080528e198b4f4e44ee936f066d1b12612adc4ef8af0d8a0
yolov5s_GLAD.pt         9e8edde080cf9e1eac0fcd68c12d7073bb23dac4059f2929ed3a06f301537a0c
```

（2026-10-14 经 GitHub contents API `Accept: application/vnd.github.raw` 下载，
大小与 API 清单一致，zip 容器完整性校验通过。已知坑：该通道可能静默截断——
`yolov5s_GLAD.pt` 首次下载仅 7.6MB——故脚本必须先验大小再验 sha256。）

### 3.5 迁移/兼容

- `data_raw/mgmd_weights/`、`outputs/runs_mgmd/` 加入 `.gitignore`（沿用 data_raw/outputs 既有规则，确认即可）。
- GT 沿用 `data_derived/gt/D{1..4}_cam{2,1,3,3}.csv`，无 schema 变化。
- 无数据库、无破坏性变更。

---

## 4. API 设计（Python 契约 + CLI；本工程无网络服务）

### 4.1 模块函数契约

```python
# mgmd/align.py
def align(prev_gray: np.ndarray, cur_gray: np.ndarray, cfg: MfeConfig) -> AlignResult
# AlignResult(compensated: np.ndarray, mask: np.ndarray, homography: np.ndarray,
#             inliers: int, fallback: bool)
# 单应失败/prev 为 None → 恒等矩阵 + fallback=True + warnings.warn（FR-3，不崩溃）

# mgmd/diff.py
def candidates(cur_bgr: np.ndarray, compensated: np.ndarray, mask: np.ndarray,
               cfg: MfeConfig) -> list[Candidate]
# Candidate(rect=(x,y,w,h), score=float)  # score=运动一致性 avg_dist（选择规则用）
# len(候选) > cfg.max_boxes → 返回 []（官方行为）

# mgmd/track.py
class SortTracker:
    def __init__(self, cfg: TfConfig): ...
    def update(self, rects: list[tuple[int,int,int,int]], frame_idx: int) -> list[Track]
    # Track(id, rect, age, hits, velocity=(vx,vy), confirmed: bool)
    # confirmed = age ≥ alg1_age 且 D(s_{N−w}, s_N)=ρ/d ≥ alg1_d_min（Algorithm 1）

# mgmd/classify.py（torch 懒加载；缺 torch → ImportError 带补救提示）
class LacClassifier:
    def __init__(self, weights: Path, device: str): ...   # 权重缺失 → MgmdError(退出码3)
    def mav_prob(self, crop_bgr: np.ndarray) -> float     # ∈[0,1]

# mgmd/detect.py（torch+yolov5 懒加载）
class LadDetector:
    def __init__(self, weights: Path, cfg: LadConfig): ... # 权重缺失 → 退出码3
    def refine(self, frame_bgr: np.ndarray, center_xy: tuple[float,float],
               downscale: float) -> RefinedBox | None
    # RefinedBox(rect=(x,y,w,h), conf=float)；坐标为原图尺度

# mgmd/pipeline.py
def run(source: FrameSource, cfg: Config, modules: frozenset[str],
        out_dir: Path, limit: int | None) -> int  # 退出码
```

### 4.2 CLI

```bash
.venv/bin/python scripts/run_mgmd.py --config configs/mgmd_D1.yaml \
    --gt data_derived/gt/D1_cam2.csv --out outputs/runs_mgmd/D1_full \
    [--limit N] [--modules mfe,tf,lac,lad] [--backend numpy]
.venv/bin/python -m mgmd.config configs/mgmd_D1.yaml          # 配置校验
bash scripts/fetch_mgmd_weights.sh [--with-global] [--check]  # US-001
.venv/bin/python scripts/make_report_mgmd.py --runs outputs/runs_mgmd \
    --gt data_derived/gt --spa2d outputs/metrics/summary.csv --out outputs/report_mgmd.md
.venv/bin/python scripts/make_tables.py --runs outputs/runs_mgmd --gt data_derived/gt \
    --out outputs/metrics_mgmd                                # 复用现有脚本
```

错误格式统一 `[mgmd] <code>: <msg> → <remediation>`（FR-13）。

### 4.3 退出码（沿用仓库约定，FR-14）

| 码 | 含义 | 触发 |
|---|---|---|
| 0 | 成功 | — |
| 2 | 配置非法 | schema 校验失败 |
| 3 | 数据缺失 | 权重缺失/sha256 不匹配/GT 缺失/视频缺失 |
| 4 | 运行时 | 解码失败、torch 运行期错误等 |
| 5 | （保留，MGMD 无冻结阈值机制，不使用） | — |
| 130 | 中断 | KeyboardInterrupt |

---

## 5. Business Logic

### 5.1 MFE · 帧对齐（`align.py`，移植 `Functions.motion_compensate`）

1. 预处理（在 `diff` 入口做）：`GaussianBlur(11×11, σ=0)` → `cvtColor(BGR2GRAY)`。
2. 网格关键点：间距 `grid_spacing=(96,72)`，点 `(i·96+48, j·72+36)`，
   `i∈[0, W/96−1), j∈[0, H/72−1)`（1920×1080 → 19×14；D4 downscale 后同此）。
   （论文「30×20」为另一分辨率下的表述；本工程以官方间距常数为准，D1 已记。）
3. LK：`calcOpticalFlowPyrLK(prev, cur, pts)`，`winSize=(15,15)`、`maxLevel=3`、
   `criteria=(EPS|COUNT, 30, 0.01)`；`st==1` 保留；位移 >`max_flow_px`(50) 丢弃。
4. 单应：有效点 <`min_inliers`(9) → `H=[[0.999,0,0],[0,0.999,0],[0,0,1]]`，
   `fallback=True` + `warnings.warn`（FR-3）；否则
   `findHomography(good_new, good_old, RANSAC, ransac_thresh=3.0)`；
   `findHomography` 返回 None / `np.linalg.inv` 奇异 → 同样回退恒等 + warning。
5. 补偿：`warpPerspective(prev_gray, H, (W,H), INTER_LINEAR|WARP_INVERSE_MAP)`。
6. 掩膜：图像四角经 `inv(H)` 变换 → fillPoly → `mask=255−im`（无效边缘区）。

### 5.2 MFE · 差分与候选提取（`diff.py`，移植 `MOD2_global`）

1. `frameDiff = absdiff(cur_gray, compensated)`。
2. 自适应阈值（D2）：`T = diff_thresh_base + int(mean(frameDiff))`（`diff_adaptive=false`
   时 `T=diff_thresh_base`）；`threshold(·, T, 255, BINARY)`。
3. 掩膜扣除：`cv2.subtract(thresh, mask)`（D8，饱和；官方 numpy 回绕减法不采用）。
4. `medianBlur(·, 5)` → `MORPH_OPEN 3×3 RECT iter=1` → `MORPH_CLOSE 3×3 RECT iter=3`。
5. `findContours(RETR_LIST, CHAIN_APPROX_SIMPLE)` → `boundingRect`；
   保留 `area_min < contourArea < area_max` 且 `ratio_min < w/h < ratio_max`。
6. 候选数 > `max_boxes`(50) → 返回 `[]`（整帧弃检，官方行为）。
7. 逐框运动一致性（官方 "motion classifier"，不含 Net）：
   - `enlargebox(rect, a=enlarge_px)`，边界 clamp；扩大后 `w/h∉(0.6,3)` → 弃。
   - 当前灰度裁剪上 `goodFeaturesToTrack(maxCorners=30, qualityLevel=0.15,
     minDistance=3, blockSize=3)`；无点 → 弃。
   - `calcOpticalFlowPyrLK(crop_compensated, crop_current, pts)`（winSize 15,
     maxLevel 3, criteria (30,0.03)），`st==1` 点数 <1 → 弃。
   - 逐点位移 `dist_j` 与方向 `θ_j=57.3·atan2(Δy,Δx)`；
     `mean(dist) < motion_avg_dist_min(1.0)` 或 `std(θ)/mean(θ) > 0.8` 或
     `std(dist)/mean(dist) > 0.8` → 弃。
   - 通过者 `score = mean(dist)`。

### 5.3 TF · SORT（`track.py`，PRD FR-4）

- 状态向量 `[cx, cy, s, r, vx, vy, vs]`（SORT 标准 7 维，r=宽高比），匀速模型，
  `filterpy.KalmanFilter`（dim_x=7, dim_z=4）；过程/测量噪声取 SORT 论文默认
  （`std` 超参按 sort.py 经典实现：位置 1/20·s 加权等）。
- 关联：`scipy.optimize.linear_sum_assignment` 于 IoU 代价矩阵；
  匹配门限 `iou_thresh=0.3`。
- 生命周期：`min_hits=1`（首检即出），`max_age=3`（连续 3 帧未匹配删除）。

### 5.4 TF · 轨迹分类 Algorithm 1（FR-5）

```
对每个活跃轨迹 s（位置经累计单应对齐到当前帧坐标系）：
    confirmed(s) ⟺ age(s) ≥ alg1_age(3)
               且  D(s_{N−w}, s_N) = ρ/d ≥ alg1_d_min(1.0)
其中 ρ = ‖align(s_N) − align(s_{N−w})‖₂   （w=alg1_window=3；轨迹不足 w 帧用最早点）
     d = √(w_N² + h_N²)                  （当前框对角线）
```

未 confirmed 的轨迹不输出。ρ/d 中 d 的精确定义（对角线 vs √面积）论文未明示，
本 SPEC 取对角线并列入假设（§11.3 A2），报告做敏感性对照。

### 5.5 LAC / LAD（`classify.py` / `detect.py`）

- **LAC**（移植 `Functions.Mynet_infer`，注意官方实际用 `Net` 非 `MyNet`）：
  网络 `Net`：`Conv(3,6,k5)→MaxPool2→Conv(6,16,k5)→MaxPool2→Flatten(16·5·5)
  →Linear(120)→Linear(84)→Linear(2)`；输入 crop `resize(32×32)` +
  `torchvision.transforms.ToTensor`（/255）；`load_state_dict(torch.load(
  Net_best.pth, map_location))`；softmax 后取 `P(class=1)` ≥ `prob_thresh`(0.5)。
- **LAD**（对应官方 `detector3` + `enlarge_region2(a=160)`）：
  以候选中心裁剪 `crop_size×crop_size`(320)，边界不足时平移窗口 clamp
  （`enlarge_region2` 语义）；yolov5 v6.0 `attempt_load` 等价物加载
  `yolov5s_GLAD-crop.pt`（D9：`torch.load(weights_only=False)`）；
  推理 `imgsz=320`，`conf > conf_thresh`(0.5)；多框取**离裁剪中心最近**者
  （对齐官方 `detector3.detect(region, x_center, y_center)` 的最近中心语义）；
  坐标加回裁剪偏移并 `÷downscale` 映射原图。

### 5.6 每帧最终选择规则（D6，至多 1 框）

依启用模块取候选池，按优先级选 1 个：

| 启用模块 | 候选池 | 选择键（降序，并列取面积小者） |
|---|---|---|
| mfe | MFE candidates | score（运动一致性 avg_dist） |
| mfe,tf | confirmed 轨迹 | age → score |
| mfe,tf,lac | LAC 通过 | mav_prob |
| mfe,tf,lac,lad | LAD 精炼 | conf |

跳过模块 = 直通。任一级输出为空 → 本帧无输出（reason=ABSENT）。

### 5.7 消融档位（FR-8，对齐论文 Table 3 结构）

`--modules` 四档：`mfe` / `mfe,tf` / `mfe,tf,lac` / `mfe,tf,lac,lad`。
run 目录命名 `outputs/runs_mgmd/{SEQ}_{modules连线}`，如 `D1_mfe-tf-lac-lad`。

### 5.8 边界与已知边角

- 序列首帧无 prev → 无输出（reason=ABSENT），自第 2 帧起检测。
- `enlargebox` 后 `w1 或 h1 ≤ 0`（贴边极小框）→ 弃。
- LAD 裁剪遇边界：`enlarge_region2` 平移 clamp，保证 320×320（分辨率 <320 时
  上采样到 320 并记录 warning——MVDT 不触发）。
- downscale≠1 时所有坐标（候选、轨迹、LAD 输出）在写记录前 `÷downscale`。
- `It±k` 未来帧：D1 口径（官方两帧）无此依赖；若日后恢复论文三帧口径，
  需 k 帧延迟并在报告中注明（PRD §7 条款保留）。

---

## 6. Error Handling

### 6.1 错误分类

| code | exit | 条件 | remediation 示例 |
|---|---|---|---|
| E_CONFIG_INVALID | 2 | schema 违规/未知字段/缺字段 | 指出字段与合法取值 |
| E_WEIGHTS_MISSING | 3 | 权重文件不存在 | `bash scripts/fetch_mgmd_weights.sh` |
| E_WEIGHTS_HASH | 3 | sha256 不匹配 | 删除残留文件后重跑 fetch（提示已知静默截断问题） |
| E_DATA_MISSING | 3 | GT/视频/帧目录缺失 | 指向 fetch_mvdt_videos/unpack/make_derived_gt |
| E_DEP_MISSING | 4（运行期）或测试 skip | torch/yolov5/scipy/filterpy 缺失 | `pip install -r env/requirements.txt`；仅 LAC/LAD/TF 相关受影响 |
| E_DECODE | 4 | 帧解码失败 | 检查视频完整性（7z t / ffprobe） |
| E_RUNTIME | 4 | 其他运行异常 | traceback + 补救 |

### 6.2 降级策略

- 单应估计失败 → 恒等 + warning（FR-3），**不中断**；warning 计数写入 run_meta。
- torch/yolov5 缺失且未启用 lac/lad → MFE/TF 正常运行；启用时才报 E_DEP_MISSING。
- 整帧候选 >50 → 当帧弃检（非错误）。

### 6.3 重试

离线批处理无自动重试；权重下载脚本内 `--retry 3 -C -`（断点续传）+ 大小/sha256 双验。

---

## 7. Security

- 权重仅本地下载使用，MIT license（官方仓库），来源与 hash 记录于
  `data_raw/mgmd_weights/README.md`（US-001 AC4）；不入 git。
- `torch.load(weights_only=False)` 仅限本脚本 sha256 校验通过的官方权重；
  校验失败拒绝加载（D9 与 6.1 联动，避免 pickle 反序列化不可信文件）。
- 无网络服务、无密钥、无用户输入注入面；YAML 用 `yaml.safe_load`。

---

## 8. Performance

### 8.1 负载

D1–D4 离线批处理，1080p（D4 4K downscale 0.5→1080p 等效）；无 FPS 门槛
（论文 28FPS@RTX3070 仅供参考）；smoke 模式 ≤5 min。

### 8.2 优化与显存

- 逐帧流式处理，常驻 capture（`FrameSource`，禁止逐帧 open/seek）。
- 全帧运算仅 CPU/OpenCV（blur/LK/形态学）→ 显存压力只在 LAC(32×32)/LAD(320×320)
  裁剪推理，天然有界；D4 风险在全帧预处理，经 downscale 0.5 消解（D7）。
- GPU 环境沿用 `XLA_PYTHON_CLIENT_PREALLOCATE=false`（jax 路径不变，仅共存）。
- LAC/LAD 批量：同帧多候选可拼 batch 推理（实现可选优化，非验收项）。

### 8.3 资源上限

`runtime.downscale` 为唯一降载旋钮；D4 默认 0.5。若仍 OOM → 退出码 4 并提示
调小 downscale（remediation 内嵌）。

---

## 9. Testing Strategy

### 9.1 单元测试（全部 `.venv/bin/python -m pytest`，numpy 级可无 GPU 运行）

| 文件 | 覆盖 |
|---|---|
| `tests/test_mgmd_config.py` | 4 份 config 加载通过；未知字段/非法值 → 退出码 2；CLI `-m mgmd.config` |
| `tests/test_mgmd_mfe.py` | 合成帧：静态背景零候选；已知位移点目标检出且框心误差 ≤5px；均匀纹理帧触发恒等回退+warning（FR-3）；掩膜减法无回绕伪前景（D8）；>max_boxes 整帧弃检 |
| `tests/test_mgmd_tf.py` | 静止/抖动轨迹被 Algorithm 1 过滤；匀速轨迹 confirmed；断帧 ≤2 续接（max_age=3）；IoU 关联正确性 |
| `tests/test_mgmd_appearance.py` | `pytest.importorskip("torch")`；合成张量验证 Net 输出形状/P∈[0,1]；LAD 用 monkeypatch 假推理验证裁剪数学+坐标映射+最近中心选择；权重缺失 → 退出码 3 |
| `tests/test_mgmd_pipeline.py` | smoke：合成序列 modules=mfe 端到端 → windows.jsonl schema 往返一致 + `evaluate_windows` 可算；模块子集四档冒烟 |

### 9.2 依赖缺失矩阵（FR-11）

| 缺失 | 影响 |
|---|---|
| torch/torchvision/yolov5 | 仅 `test_mgmd_appearance.py` skip；其余全绿 |
| scipy/filterpy | 仅 `test_mgmd_tf.py` skip |
| 无 GPU | 全部可跑（LAC/LAD device=cpu 路径单测覆盖） |

### 9.3 验收映射

| US/FR | 测试/脚本 | 类型 |
|---|---|---|
| US-001 / FR-1 | `fetch_mgmd_weights.sh` 幂等+sha256+失败退出码（手动+脚本自验） | 集成 |
| US-002 / FR-13/14 | test_mgmd_config.py | 单元 |
| US-003 / FR-2/3 | test_mgmd_mfe.py | 单元 |
| US-004 / FR-4/5 | test_mgmd_tf.py | 单元 |
| US-005 / FR-6/7/11 | test_mgmd_appearance.py | 单元 |
| US-006 / FR-8/9/12 | test_mgmd_pipeline.py + D1 smoke + D4 实跑 | 集成 |
| US-007 / FR-10 | make_report_mgmd.py 产出 P/R/F1@r + 消融表 | 集成 |
| US-008 | outputs/report_mgmd.md + 全量 pytest | 验收 |

---

## 10. Implementation Plan

### 10.1 阶段与 Issue 映射（沿用 `.autoresearch/issues/` 编号续排）

| Issue | 内容 | SPEC 节 | 依赖 |
|---|---|---|---|
| issue-017-fetch-mgmd-weights | US-001 权重脚本（sha256 §3.4 预 pin） | 3.3/3.4, 6.1 | — |
| issue-018-mgmd-scaffolding-config-cli | US-002 包骨架+schema+CLI 骨架+依赖分组 | 2.4, 3.1, 4.2 | — |
| issue-019-mfe-align-diff | US-003 MFE（align+diff+一致性） | 5.1/5.2 | 018 |
| issue-020-tf-sort-alg1 | US-004 SORT+Algorithm 1 | 5.3/5.4 | 018 |
| issue-021-lac-lad | US-005 外观模块（含 D9 加载兼容） | 5.5 | 017, 018 |
| issue-022-runner-records | US-006 runner+WindowRecord 适配+run_meta | 3.2, 4.1, 5.6 | 019–021 |
| issue-023-metrics-ablation | US-007 指标+消融表（make_report_mgmd.py） | 5.7, 9.3 | 022 |
| issue-024-full-eval-report | US-008 D1–D4 全量+对比报告 | 8, 9.3 | 023 |

### 10.2 增量交付

每 issue 独立可合：骨架（018）落地后即保证 pytest 全绿（懒加载）；外观模块
权重未下载时对应测试 skip。评测（023/024）依赖真实视频解包完成。

---

## 11. Open Questions & Risks

### 11.1 未决问题

- 无阻塞项。（PRD 原 Open Questions 中 k 值与 LAC 阈值经 D1/D2 裁决以官方代码为准：
  k 等效 1、LAC argmax≡0.5；权重 sha256 已实值预 pin。）

### 11.2 技术风险

| 风险 | 影响 | 缓解 |
|---|---|---|
| yolov5 v6.0 代码 + torch≥2.7 兼容（weights_only/旧 API） | LAD 加载失败 | D9 封装加载；备选 `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`；最终 fallback torch.hub 拉 v6.0（PRD 原方案） |
| 官方两帧 MFE 在 D3 达 recall≥50% 不确定 | US-008 指标 | D3 失败案例入报告；`diff_adaptive:false` 对照档辅助诊断 |
| D4 downscale 0.5 损 recall | 指标偏差 | config 可调；报告做 0.5/1.0 对照（若显存允许） |
| Algorithm 1 ρ/d 定义与论文原文出入 | TF 行为偏差 | A2 假设+敏感性对照 |
| contents API 下载静默截断 | 权重损坏 | 大小+sha256 双验（§3.4 已实测一次截断） |

### 11.3 假设

- A1：官方 MOD2.py/Functions.py 常数即论文实现真值（D1 裁决背书）。
- A2：Algorithm 1 的 d 取当前框对角线 √(w²+h²)；论文原文若不同，改 `track.py`
  单点常量并重跑 TF 相关档位即可。
- A3：`yolov5s_GLAD-crop.pt` 输入分辨率按 320 处理（官方 trt 引擎未给源码，
  imgsz 可配，加载后打印模型 stride 核验）。
- A4：MVDT GT 点坐标为全分辨率坐标系；downscale 输出统一映射回原图后评测。
- A5：`make_tables.py` 对未知 config_name 的 paper 对照行为需在 023 验证；
  若报错则加防御分支（§2.4 唯一允许的 make_tables.py 改动）。
