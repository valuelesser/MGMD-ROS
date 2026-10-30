# 代码审查报告：mgmd-linux 全包问题清单

**审查范围**：`mgmd/`（tf, kalman, mfe, lac, lad, metrics, pipeline, split, config, annotations, io_video）、`scripts/` 关键路径、`tests/`
**审查日期**：2026-10-29
**方法**：逐模块阅读 + 针对性探针脚本实测（全部在工程 `.venv` 下运行）
**基线状态**：`pytest tests/` → 60 passed, 2 failed（2 个失败均为离线环境无法下载 `Arial.ttf`，与代码无关）

---

## 总览

| 编号 | 严重度 | 位置 | 一句话 |
|---|---|---|---|
| **B-01** | **高** | `tf.py` `TrackState.window` | 帧号跳跃时取窗跨度可达 49 帧，`D` 被放大 16 倍 |
| **B-02** | **高** | `metrics.py` `_bucket` | `n_gt=0` 的空帧被错分进 `"4+"` 桶，分组 precision 失真 |
| **B-03** | 中 | `tf.py` / `pipeline.py` | SORT 关联断链（详见 `BUG_sort_association_break.md`） |
| **B-04** | 中 | `scripts/*` vs `metrics.py` | IoU 门限 `>` 与 `>=` 口径不一致 |
| **B-05** | 中 | `pipeline.py` | `scores` 字段在一条链路里换了三种量纲 |
| **B-06** | 低 | `pipeline.py` | 未启用 LAC 时 `lad_topk` 退化为"按面积取 top-k" |
| **B-07** | 低 | `tf.py` | `_age` 与 `lifetime` 语义重复；`min_hits` 在默认值下恒不生效 |
| **B-08** | 低 | `scripts/calibrate_d.py` | `--percentiles` 无范围校验，>100 会索引越界 |
| **B-09** | 信息 | `scripts/calibrate_d.py` | 用 GT 框标定 `d_min`，与生产环境 MFE 框分布不匹配 |

已排除（实测无问题）：`KalmanBox` 滤波方程、MFE 单应方向、`crop_region` 边界、AP 计算、`VideoReader` 帧号、split 自洽性。

---

## B-01【高】`TrackState.window` 的取窗在帧号跳跃时失效

**位置**：`mgmd/tf.py:119-133`

```python
def window(self, span: int = 3):
    """取 [s_{N-span}, s_N] 对。返回 (older, newest) 或 None。"""
    if len(self.boxes) < 2:
        return None
    newest_frame, newest_box = self.boxes[-1]
    target = newest_frame - span          # ← 按帧号，不是按缓冲位置
    older = None
    for f, b in self.boxes[:-1]:
        if f <= target and (older is None or f > older[0]):
            older = (f, b)
    if older is None:
        older = self.boxes[0]
        if older[0] >= newest_frame:
            return None
    return older, (newest_frame, newest_box)
```

**问题**：文档写"取 `[s_{N-span}, s_N]`"（即倒数第 `span+1` 个框），实现却是"帧号 ≤ `N-span` 的最近一个框"。
两者只有在**帧号连续**时等价。而 `lifetime` 是按**更新次数**计的，它与帧号无关。

**实测**：

```
加入帧 (1, 2, 3, 52)，buffer_size=4 → 全部保留
boxes = [1, 2, 3, 52]   lifetime = 4   （≥3，通过寿命门控）

window(3) 取到 (帧3, 帧52)，实际跨度 = 49 帧，而不是 3 帧

D(self) 实测 = 3.465
若按真正的 3 帧跨度应为 ≈ 0.212
```

**影响**：`D = ρ/d`，`ρ` 是两框中心距。跨度从 3 帧变 49 帧，匀速运动下 `ρ` 放大约 16 倍，
`D` 同步放大。方向是**让 D 偏大** → 静止/慢速目标被误判为"超出道路的运动目标" → **假阳性确认**。
这是比"漏检"更危险的错误方向，因为它会制造并不存在的目标。

**触发条件**：`frame` 参数出现跳跃。当前 `pipeline.py` 用 `t = idx - k` 且逐帧连续读取，
所以**正常路径不会触发**；但以下情况会：
- 视频解码丢帧（`VideoReader.read` 在 `ok=False` 时直接返回 `None` 终止，但部分编码器会返回索引跳变）；
- 未来接入跳帧读取或 ROI 裁剪；
- 单元测试里任何不连续的帧号。

**修复**：需要注意，**单纯改成按缓冲位置取并不能解决问题**，实测如下：

```
连续帧 1..10（buffer=4，保留 [7,8,9,10]）
  原实现 window(3) = (7, 10)     修复（boxes[-(span+1)]）= (7, 10)   一致 ✓

跳跃帧 (1,2,3,52)
  原实现          → (3, 52)   跨度 49
  按缓冲位置取     → (1, 52)   跨度 51     ← 更差
```

正确做法是**显式检查窗口内帧号是否连续**：

```python
def window(self, span: int = 3):
    if len(self.boxes) < span + 1:
        return None
    older, newest = self.boxes[-(span + 1)], self.boxes[-1]
    if newest[0] - older[0] != span:       # 帧号不连续 → 拒绝
        return None
    return older, newest
```

并在 `classify` 中把 `None` 归因到 `reason="gap_in_window"`，与 `no_window` 区分，
便于统计有多少轨迹因丢帧而无法判定。

---

## B-02【高】`_bucket` 把空帧错分进 `"4+"` 桶

**位置**：`mgmd/metrics.py:159-163` + `mgmd/config.py:63`

```python
# config.py
GT_COUNT_BUCKETS = ((1, 1), (2, 3), (4, 10**9))

# metrics.py
def _bucket(n: int) -> str:
    for lo, hi in C.GT_COUNT_BUCKETS:
        if lo <= n <= hi:
            return f"{lo}-{hi}" if lo != hi else f"{lo}"
    return "4+"                       # ← 兜底
```

**问题**：两个缺陷叠加。

1. `n = 0` 不匹配任何区间（`1 ≤ 0` 为假），落入兜底 `return "4+"`。
   **空帧被归到"4 个及以上目标"组。**
2. `n = 4..10⁹` 命中第三区间，标签是 `"4-1000000000"` 而不是 `"4+"`。
   于是 **`"4+"` 桶只会包含空帧**，真正的多目标帧在另一个名字奇怪的桶里。

**实测**：

```
_bucket(0)  = '4+'
_bucket(1)  = '1'
_bucket(2)  = '2-3'
_bucket(4)  = '4-1000000000'
_bucket(50) = '4-1000000000'
```

**影响**：`summary()["by_gt_count"]` 是写进 `metrics.json` 的正式报告字段（SPEC §5.3）。
空帧（MFE 全空、大量 FP）会被统计成一个叫 `4+` 的组，而报告读者会把它理解成"4 个以上目标"。
如果空帧数量可观，这一组的 precision 会被严重低估，且**真实的 4+ 组被隐藏在 `4-1000000000` 这个非法标签下**。

**修复**：

```python
GT_COUNT_BUCKETS = ((0, 0), (1, 1), (2, 3), (4, 10**9))
BUCKET_LABELS = ("0", "1", "2-3", "4+")

def _bucket(n: int) -> str:
    for (lo, hi), label in zip(C.GT_COUNT_BUCKETS, C.BUCKET_LABELS):
        if lo <= n <= hi:
            return label
    return "4+"
```

**注意**：这会改变 `metrics.json` 的键名，历史结果不可直接对比。

---

## B-03【中】SORT 关联断链

已在 `reports/BUG_sort_association_break.md` 详述。要点：10×10 框、速度 > 5.4 px/帧 时第 2 帧断链，
`track_id` 每帧递增，Algorithm 1 因 `lifetime_lt_3` 永不输出。

---

## B-04【中】IoU 门限的 `>` 与 `>=` 不一致

**位置**：`mgmd/metrics.py:72`（`>=`）vs `scripts/sweep_morph.py:54,56`、`run_tf.py:63,65`、
`run_mfe.py:74`、`ablate_extras.py:80,82`、`exp_normalize.py:146,148`、`make_video.py:201`（`>`）

```python
# metrics.py — 评测口径
if gain[i, j] >= iou_thresh:        # 等于阈值算命中

# scripts/* — 各脚本自己的命中判定
if best > C.IOU_THRESH:             # 等于阈值不算命中
```

**实测**：构造 IoU 恰好 = 0.25 的一对框（位移 d = 6.0000000000）：

```
match_frame(thresh=0.25)     → tp=[0]   计入
best > C.IOU_THRESH          → False    不计入
```

**影响**：浮点恰好等于阈值的概率极低，单次评测的偏差可以忽略。
但**报告里的数字来自两套口径**（`metrics.py` 的正式评测 vs 脚本里的快速统计），
两者对同一份数据可能给出不同召回率。一旦有人拿脚本数字和 `metrics.json` 对比，就会对不上。

**修复**：脚本统一改为调用 `match_frame` / `evaluate`，不再自己写命中判定；或至少统一为 `>=`。

---

## B-05【中】`scores` 字段在一条链路里换了三种量纲

**位置**：`mgmd/pipeline.py:97, 118, 121-123, 137, 152-153`

```
第 97 行   cand_scores = [float(w*h)]          → 面 积（可能上千）
第 118 行  pool = [(b, p) for ...]             → P(MAV)，0~1（prefilter_lac 生效时）
第 121 行  tracker.update(..., scores=...)     → 透传
第 123 行  pool = [(tr.box, tr.score)]         → 透传
第 137 行  pool = [(b, p) for (b,_s), p ...]   → P(MAV)，0~1（LAC 阶段）
第 152 行  refined = [(rb.box, rb.conf)]       → YOLO conf，0~1
最终输出   scores=[s for _, s in pool]
```

**实测**：TF 的 `score` 确实是未归一化的面积：

```
TF 输出 score = 1200.0   (传入 scores=[1200.0]，40x30 框)
```

**影响**：
1. 最终写入 `write_jsonl` 的 `scores` 字段，其含义取决于启用了哪些模块，
   下游（`run_testset.py`、`table3_compare.py`、AP 排序）无法知道它代表什么。
2. `AP` 需要 `conf` 做排序，不同类型的分数量纲会导致 AP 不可比。
3. 若将来把 `prefilter_lac` 与 `lad_topk` 组合使用，排序依据会在两个分支间跳变。

**修复**：`pool` 里的分数改为显式的多字段结构，或在每个阶段用一个明确命名的中间变量，
最终输出时统一到一个约定的分数（建议：`conf`，且写明来源）。

---

## B-06【低】未启用 LAC 时 `lad_topk` 语义退化为"按面积取 top-k"

**位置**：`mgmd/pipeline.py:143-146`

```python
# lad_topk>0 时只精修 score 最高的前 k 个候选（SPEC §8.14）
# 依据：实测 GT 有 83% 概率落在 LAC 排序 top-3 内
if lad_topk and len(pool) > lad_topk:
    pool = sorted(pool, key=lambda x: -x[1])[:lad_topk]
```

注释的依据是"LAC 排序 top-3"。但若 `modules` 不含 `lac`，`pool` 的 score 仍是 MFE 面积，
此处变成"按面积取最大的 k 个"。代码没有任何分支区分这两种情形。

**影响**：不会崩溃，但结论与注释不符。若有人在消融实验里"关闭 LAC 但保留 lad_topk"，
会得到一个未被声明的第三种行为。

**修复**：仅在 `clf is not None and clf 已执行过` 时启用 top-k；否则跳过并 warn。

---

## B-07【低】`tf.py` 的重复/失效字段

**位置**：`mgmd/tf.py:214-216, 297`

1. **`_age` 与 `lifetime` 语义重复**：`_age[tid]` 每帧 +1（含未匹配帧），
   `TrackState.total_updates` 只在匹配到检测时 +1。两者都叫"年龄"，数值不同，
   但 `_age` **仅用于填充 `Track.age` 输出字段，不参与任何判定**。
2. **`min_hits` 在默认值下恒不生效**：新轨迹创建时 `hits=1`，
   `if self._hits.get(tid,0) < self.min_hits` 在 `min_hits=1` 时恒为 False。
   真正拦住"刚出生轨迹"的是 `TrackState.classify` 的 `lifetime < 3` 门控。
   `tf.py` 头部声明"min_hits 1"容易被误读为"SORT 的 1 帧确认 = Algorithm 1 确认"。
3. **`step_H` 无调用方**：`align=True` 名义上启用单应对齐，但工程内无一处传入 `step_H`，
   每帧 `classify` 都走 `reason="align_missing"`。（上一轮已记，此处并列。）

---

## B-08【低】`calibrate_d.py --percentiles` 无范围校验

**位置**：`scripts/calibrate_d.py:83`

```python
idx = int(round(p / 100.0 * (n - 1)))
thr = vals[idx]
```

`p ≤ 100` 时 `idx ≤ n-1`，安全。但 `p > 100` 会直接越界：

```
--percentiles 120 → idx = 119, 列表长度 100 → IndexError
```

**修复**：`if not 0 < p <= 100: raise MgmdError(...)`。

---

## B-09【信息】`d_min` 用 GT 框标定，与生产环境的 MFE 框分布不匹配

**位置**：`scripts/calibrate_d.py:30-62`

docstring 已写明是"从 GT 轨迹收集 D 值（替代真实检测轨迹，用于阈值先验）"，
属于**知情的设计选择**，不是 bug。但报告里必须声明其后果：

- GT 框是精确目标框；
- MFE 候选框是运动区域连通域的外接框（经开闭运算，尺寸/形状不同）；
- `D = ρ/d`，`d` 是框对角线 → 框变大则 `D` 变小；
- 因此用 GT 标定的 `d_min` 套用到 MFE 轨迹上会**系统性偏严**（更多轨迹被拒）。

建议：报告里同时给出"GT 标定"与"MFE 自标定"两组阈值及其对应召回，
并用 `configs/d_calibration.json` 的 `per_video` 字段展示跨视频离散度。

---

## 已验证无问题的部分

以下项目我做了针对性验证，结论是**代码正确**，列出以便后续审查不重复投入：

| 项目 | 验证方式 | 结论 |
|---|---|---|
| `KalmanBox` 滤波方程 | 匀速输入 1~20 px/帧 | 速度估计误差 0.03~0.31 px/帧，正确 |
| MFE 单应方向 | 已知平移 (+20,+10) | 对齐后 `absdiff` 均值 0.000，方向正确 |
| `_valid_region` | 平移场景 | 有效区 95%，掩膜正确 |
| `crop_region` | 4 个边界用例（含小于 crop_size 的帧） | 无越界，clamp 正确 |
| AP 计算（11point / allpoint） | 手算对照 | 完全一致 |
| `match_frame` Hungarian 一对一 | IoU=1.0 边界 | 正确判 TP |
| `VideoReader` 帧号 | 30 帧实测视频 | 连续 `read` 返回 1,2,3...,确为 1 基 |
| `split_ard.json` 自洽性 | 与 `config.TEST_IDS` 对照 | 37/8/15，无交集，条件组全在测试集内 |
| `annotations.parse_xml` | 逆序坐标、退化框 | 有规范化与丢弃保护 |

---

## 建议的处理顺序

1. **B-01**：改动小、风险明确、后果严重（假阳性）。可作为第一个修复。
2. **B-02**：改动小，但会改变历史 `metrics.json` 的键名，需同步更新已产出的报告。
3. **B-03**：涉及关联判据，需要消融实验配合，按 `BUG_sort_association_break.md` 的修复计划走。
4. **B-04 / B-05**：口径与语义问题，建议在所有实验重跑前统一，否则新旧数字不可比。
5. **B-06 ~ B-09**：清理与文档层面，可批量处理。

---

## 说明

本报告只做诊断，**未修改任何业务代码**。
所有实测输出均可用工程 `.venv` 复现：

```bash
cd /home/gaoduan/毕业设计/mgmd-linux
PYTHONPATH=. .venv/bin/python <探针脚本>
```

B-09 关于真实数据分布的结论无法在当前环境验证——ARD 数据集所在卷
（`/media/gaoduan/0C1CFCAF1CFC94C4*`）未挂载，因此真实序列上的
召回率、帧间位移分布、IDSW 等指标均未测量。
