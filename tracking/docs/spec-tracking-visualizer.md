# SPEC: MGMD 实时跟踪可视化（Live Viewer）

> 技术规格，派生自：`tasks/prd-tracking-visualizer.md`
> Generated: 2026-10-20 | Target branch: main | Commit: （仓库尚无提交）

## 1. Summary

### 1.1 本 SPEC 覆盖范围

为 `scripts/run_mgmd.py` 增加 `--viz`：在 MGMD 推理主循环中逐帧读取处理帧，用 OpenCV 窗口
叠加检测框/中心、GT、状态机信息、速度、MFE 帧对与选择分数，并支持播放/暂停、单帧/±10 帧、
缓冲区内拖动、跳转到下一个/上一个 FP/miss。关闭 `--viz` 时零行为变化；开启时可视化不参与
`timing.json` 计时，也不改变 `windows.jsonl` / `metrics.json` 的内容。

### 1.2 PRD 引用

- Source: `tasks/prd-tracking-visualizer.md`
- User Stories: US-001 … US-008（全部）
- Functional Requirements: FR-1 … FR-16（全部）

### 1.3 设计决策摘要

| # | 决策 | 选择 | 理由 |
|---|------|------|------|
| D1 | 可视化代码位置 | 新增 `mgmd/viz.py`，不改 `spa2d/` | MGMD 专用，保持 spa2d 零耦合 |
| D2 | 与管线的接入方式 | `pipeline.run(..., on_frame=None)` 可选回调 | 不传时零开销；测试可注入收集器 |
| D3 | 回调签名 | `on_frame(index, frame_bgr, record, pick_rect) -> bool` | `frame` 为处理尺度帧；`pick_rect` 为最终框（处理坐标）；返回 `False` 请求提前结束 |
| D4 | 提前退出语义 | `q`/关窗 → 回调返回 False → 管线停止读帧，写出已处理记录与指标（exit 0），`run_meta.viz.aborted=true` | PRD US-006；保留已产出的结果 |
| D5 | 回看与实时 | 缓冲区内回看/拖动**不重跑推理**；恢复播放从当前实时位置继续 | 视频顺序解码约束；避免昂贵的 seek/重启 |
| D6 | 缓冲区内容 | 缩放后的显示帧 + `FrameView` 摘要；`deque(maxlen=N)` | 4K 原帧不进内存；显示帧固定宽度 |
| D7 | GT 来源 | 复用 `run_mgmd.py --gt` 的路径；`R.read_gt_csv` | 与评测同源，保证 FP/miss 口径一致 |
| D8 | FP/miss 判定 | 原图坐标 + `primary_radius_px`，逻辑镜像 `spa2d.metrics` | 与报告口径一致 |
| D9 | 坐标映射 | 显示坐标 = 原图坐标 × `downscale` × `display_w/frame_w`；`pick_rect` 只乘后者 | D4 downscale=0.5 无 2× 偏移 |
| D10 | GUI 抽象 | `CvWindow` 适配器 + `FakeWindow`（测试） | 单测不启动真实 GUI |
| D11 | 键位兼容 | 方向键 ±1；`a/d` ±10（cv2 无法可靠读 Shift）；`h/e` 作为 Home/End 备份 | OpenCV waitKey 平台差异 |
| D12 | 失败帧快进 | “静默扫描”：继续推理但不绘制，命中即暂停；上限 `--viz-scan-max` 帧或 120 s | 实时流无法向后跳，向前扫描可控 |
| D13 | 缓冲内存默认 | 200 帧 × 1280 宽（1080p 约 550 MB），`--viz-buffer` 可调 | PRD 假设；可调低 |
| D14 | 早退指标 | 部分 records 上照常计算 `metrics.json`，并打 warning + `run_meta.viz` 标记 | 不伪造完整 run 指标 |

---

## 2. Architecture

### 2.1 系统上下文

```text
run_mgmd.py CLI
  ├─ load_config / parse_modules / GT 存在性检查（现有）
  ├─ 新增：--viz* 参数校验 + Visualizer 构造（仅 --viz）
  └─ pipeline.run(..., on_frame=visualizer.update)
        ├─ MgmdFrameSource（视频/帧目录，downscale）
        ├─ MFE/TF/LAC/LAD（现有）
        ├─ _frame_record → WindowRecord（现有）
        └─ on_frame(index, frame, record, pick_rect)
              └─ Visualizer
                    ├─ classify(record, gt, radius)
                    ├─ RingBuffer(display_frame, FrameView)
                    ├─ Navigator(key/scan/pause)
                    └─ CvWindow / FakeWindow → draw_overlay
```

### 2.2 组件职责

| 组件 | 文件 | 职责 |
|---|---|---|
| `Visualizer` | `mgmd/viz.py` [NEW] | 生命周期、窗口、GT 缓存、缓冲/导航、渲染与统计 |
| `FrameView` | 同上 | `(index, record, rect, gt, outcome)` 的不可变视图 |
| `RingBuffer` | 同上 | 固定容量显示帧 + 视图的滚动缓冲 |
| `Navigator` | 同上 | 播放/暂停/单帧/拖动/失败帧扫描状态机 |
| `draw_overlay` | 同上 | 纯函数：在显示帧上画全部叠加 |
| `CvWindow` / `FakeWindow` | 同上 | OpenCV 窗口适配器 / 测试替身 |
| `pipeline.run` | `mgmd/pipeline.py` [MODIFY] | 新增 `on_frame`、`_emit()`、提前停止、`source.close()` |
| CLI | `scripts/run_mgmd.py` [MODIFY] | `--viz*` 参数、构造与关闭 Visualizer、错误码 |
| `MgmdFrameSource.close` | `mgmd/io_mvdt.py` [MODIFY] | 释放底层 capture |

### 2.3 文件结构

```text
mgmd/
├── viz.py                  [NEW] Visualizer / RingBuffer / Navigator / draw_overlay / CvWindow
├── pipeline.py             [MODIFY] on_frame 回调、_emit、_VizStop、source.close
├── io_mvdt.py              [MODIFY] MgmdFrameSource.close()
scripts/
└── run_mgmd.py             [MODIFY] --viz* CLI 与生命周期
tests/
└── test_mgmd_viz.py        [NEW]
README.md                   [MODIFY] MGMD 章节增加 --viz 与按键表
```

---

## 3. Data Model

### 3.1 新增数据结构（`mgmd/viz.py`）

```python
@dataclass(frozen=True)
class FrameView:
    index: int                          # 绝对帧号
    record: WindowRecord
    rect: tuple[float, float, float, float] | None   # 处理尺度最终框
    gt: GTFrame | None
    outcome: str                        # "correct" | "fp" | "miss" | "absent"

class RingBuffer:
    def __init__(self, capacity: int): ...
    def append(self, view: FrameView, display_frame: np.ndarray) -> None
    def get(self, index: int) -> tuple[FrameView, np.ndarray] | None
    @property
    def first(self) -> int | None
    @property
    def last(self) -> int | None
    def __len__(self) -> int

class Navigator:
    mode: str                           # "play" | "pause" | "scan"
    cursor: int | None                  # 缓冲内正在查看的绝对帧号
    def handle_key(self, key: int) -> str   # 动作名（见 §5.3）
    def decide(self, view: FrameView | None, every: int) -> bool
        # True=管线下继续；False=停止本轮（q/关窗）
    def should_render(self, view: FrameView) -> bool
```

### 3.2 run_meta 扩展（仅 `--viz`）

```json
"viz": {
  "enabled": true, "every": 1, "buffer": 200, "width": 1280,
  "viewed": 123, "jumps": 2, "fp": 10, "miss": 5,
  "aborted": false, "abort_frame": null
}
```

`windows.jsonl`、`metrics.json`、`timing.json` 的 schema 均不变。

---

## 4. API / CLI Design

### 4.1 `scripts/run_mgmd.py` 新增参数

| 参数 | 类型/默认 | 校验 | 说明 |
|---|---|---|---|
| `--viz` | flag / off | — | 启用实时窗口 |
| `--viz-every` | int / 1 | ≥1，否则 exit 2 | 每 N 帧渲染一次（仍处理/缓冲每帧） |
| `--viz-buffer` | int / 200 | 10…2000，否则 exit 2 | 回看缓冲帧数 |
| `--viz-width` | int / 1280 | 320…3840，否则 exit 2 | 显示帧宽度（等比缩放） |
| `--viz-scan-max` | int / 5000 | ≥1，否则 exit 2 | 失败帧向前扫描上限（帧） |

以上参数仅在 `--viz` 启用时生效；无论是否启用，均做范围校验以保持可预期。

### 4.2 `pipeline.run` 签名变化

```python
def run(cfg, modules, gt_path, out_dir, limit, backend, start=0,
        on_frame: Callable[[int, np.ndarray, R.WindowRecord, tuple | None], bool] | None = None) -> int
```

- `on_frame` 返回 `False` → 停止继续读帧；`run()` 返回 `EXIT_OK`。
- `on_frame` 抛异常 → 作为 `E_RUNTIME`（exit 4），但已处理记录仍写出。

### 4.3 错误响应

| Code | Exit | 条件 | 处理 |
|---|---:|---|---|
| `E_VIZ_UNAVAILABLE` | 4 | `cv2.namedWindow/imshow` 失败或 `DISPLAY` 不可用 | 打印 WSLg/X11 修复建议 |
| `E_CONFIG_INVALID` | 2 | `--viz-*` 越界 | 指出参数与合法区间 |
| `E_RUNTIME` | 4 | 绘制/窗口异常 | 关闭窗口后按现有异常路径 |
| `E_INTERRUPT` | 130 | Ctrl-C | finally 关闭窗口/capture |

---

## 5. Business Logic

### 5.1 每帧流程（回调内）

```text
1. 读取 record.assoc_frame 的 GT（越界→None）；classify:
   visible = gt and gt.visible==1 and gt.cx/cy not None
   if record.target_present:
       outcome = correct if visible and dist(est,gt) <= primary_radius else fp
   else:
       outcome = miss if visible else absent
2. display_frame = resize(frame, width=min(viz_width, frame.w))
   （高度按比例；BGR）
3. view = FrameView(...)；buffer.append(view, display_frame)
4. navigator.decide(view, every):
   - mode=pause：阻塞循环（waitKey 30ms）直到键事件；若 q/关窗→return False
   - mode=scan：若 outcome ∈ {fp, miss}→冻结到该帧、mode=pause、jumps++；
              否则不渲染，继续；超过 scan_max 或 120s→提示并 pause
   - mode=play：cursor=index；if index % every==0 → render
5. 渲染时 draw_overlay(buffer 当前帧或实时帧) + imshow + waitKey(1)
6. return True
```

### 5.2 叠加规则

- 检测：`rect` 存在时画框（颜色按 outcome：绿=correct、红=fp）；中心画实心点；
  无 `rect` 但有 `record.est` 时只画中心点（ABSENT 不画）。
- GT：可见时画黄色十字；`bbox_*` 完整时画黄框。
- 文本面板（左上）：`frame=<idx> (<k>/<total>)`、`state_in→state_out`、`reason`、
  `modules=mfe,tf,...`、`MFE pair [i-k,i]`（OR 时 `+[i,i+k]`）、`score=…`、`pnr=…`、`margin=…`。
- 速度箭头：从检测中心沿 `(vx,vy)`；`arrow_len = clamp(10*|v|, 10, 200)` 显示像素。
- 右下：`PLAY/PAUSE/STEP`、`fp=<n> miss=<n>`、`buffer=<first>..<last>`。
- 颜色（BGR）：correct `(0,200,0)`、fp `(0,0,255)`、miss `(0,165,255)`、GT `(0,255,255)`、
  absent 灰 `(160,160,160)`、箭头青 `(255,255,0)`。

### 5.3 键位 / 动作

| 键 | 动作 |
|---|---|
| Space (32) | play ↔ pause |
| ←/→ (81/83) | 暂停态 −1/+1 帧（缓冲内）；播放态忽略 |
| ↑/↓ (82/84) | 暂停态 −10/+10 帧 |
| a / d | 暂停态 −10/+10 帧（Shift+方向键的兼容替代） |
| h / e（以及 Home/End 若可用） | 跳到缓冲首/尾 |
| n | 从实时位置静默扫描到下一个 FP/miss 并暂停 |
| p | 在缓冲内回退到上一个 FP/miss |
| q / ESC (113/27) | 请求停止本轮；窗口关闭同样视为 q |
| trackbar 拖动 | 暂停并显示缓冲内对应帧 |

### 5.4 状态机

```text
PLAY --Space--> PAUSE
PAUSE --Space--> PLAY（cursor 丢弃，回到实时）
PAUSE --←/→/↑/↓/a/d/h/e/trackbar--> PAUSE（cursor 在缓冲内移动）
PAUSE --n--> SCAN
SCAN --命中 fp/miss--> PAUSE（cursor=命中帧，jumps++）
SCAN --超限--> PAUSE（cursor=实时帧，状态提示）
PAUSE/PLAY --p--> PAUSE（缓冲内上一个失败帧；无则提示）
任意 --q/关窗--> STOP（回调返回 False）
```

### 5.5 边界条件

- `--start > 0`：缓冲与显示从 start 开始；绝对帧号保留。
- `limit` 尾部 `ABSENT`：`_absent` 若 `frame is None`，使用缓冲最后一帧（或黑帧）绘制。
- 缓冲区未满时 trackbar 范围按实际长度；满后滚动。
- `--viz-every>1`：仍缓冲每一帧以支持单帧回看；仅减少渲染次数。
- D4 `downscale=0.5`：检测/GT/速度乘 `0.5`，`rect` 已为处理尺度。
- 关闭窗口（waitKey 返回 -1 或 `getWindowProperty` 不可见）→ 视为 q。
- GT 帧越界/不可见：不画 GT；miss 仅在 GT 可见时计。

---

## 6. Error Handling

### 6.1 错误分类

| Code | Exit | 场景 | 行为 |
|---|---:|---|---|
| `E_VIZ_UNAVAILABLE` | 4 | 无 GUI/`cv2.namedWindow` 失败 | 关闭已建资源；stderr 提示 `WSLg`/`DISPLAY`；不写新 run |
| `E_CONFIG_INVALID` | 2 | `--viz-*` 非法 | 参数名+合法区间 |
| `E_RUNTIME` | 4 | 渲染/窗口异常 | `finally` 关闭窗口，按 run_mgmd 现有路径 |
| `E_INTERRUPT` | 130 | Ctrl-C | finally 关闭窗口 + source |

### 6.2 失败模式

- 回调中止：保留已处理 records；`metrics.json` 为部分指标；`run_meta.viz.aborted=true` +
  `abort_frame`，并在 stderr 打 warning。
- 窗口被用户关闭：与 q 相同。
- GT 读取失败：在 `run_mgmd` 启动前已按 `E_DATA_MISSING`（3）处理，viz 不重复实现。

---

## 7. Security

无新增外部输入/网络/权限。仅本地调试工具；不写用户指定路径以外的文件。

---

## 8. Performance

### 8.1 开销目标

- `--viz` 关闭：与当前版本吞吐差异 < 1%（仅多一次 `on_frame is None` 判断）。
- `--viz` 开启：每帧一次 `resize`（仅到 ≤`viz-width`）+ 轻量绘制 + `imshow`；
  `t_pre/t_search/t_state` 不变；可视化时间单独统计并在退出时打印，不写入 `timing.json`。
- 缓冲内存：`buffer × width × round(width*H/W) × 3` 字节；默认约 550 MB（1280×720×200）。
- 单键响应 < 100 ms（1080p 显示帧）。

### 8.2 优化

- 只在渲染时 resize；缓冲区存显示帧，不存处理帧/原帧。
- `draw_overlay` 只读 `FrameView`，不拷贝大数组（`frame.copy()` 一次用于绘制）。
- 扫描模式跳过绘制，仅做检测分类。

---

## 9. Testing Strategy

### 9.1 Unit

- `classify`: correct/fp/miss/absent 四类；GT 不可见、`cx/cy=None`、距离恰等于 radius。
- `RingBuffer`: 容量、滚动、按绝对帧 get、越界返回 None。
- `Navigator`: play/pause/step±1/±10/seek/next/prev/scan 超限/quit。
- `draw_overlay`: D4 式 downscale 坐标；检测框/GT 位置与颜色；空 record 不崩。
- `FakeWindow`: 队列键事件驱动完整键位表。

### 9.2 Integration

- `pipeline.run` 传入收集回调（无 GUI）：回调次数 == records 数；`rect` 在 `target_present` 时非 None。
- 同一合成序列 + 相同 `--limit`：带/不带回调的 `metrics.json` 文本完全一致。
- 回调在第 5 帧返回 False：`records` 长度 6、exit 0、`metrics.json` 存在、`run_meta.viz.aborted`。
- `run_mgmd --viz` 参数非法 → exit 2；模拟 GUI 初始化失败 → `E_VIZ_UNAVAILABLE`（exit 4）。

### 9.3 Acceptance mapping

| US/FR | Test | Type |
|---|---|---|
| US-001/FR-1/2/3 | callback count + metrics invariance + CLI args | integration |
| US-002/FR-4 | overlay detect/GT/state pixels | unit |
| US-003/FR-5/15 | overlay velocity/pair/score + downscale geometry | unit |
| US-004/FR-6/7/8 | Navigator + FakeWindow keys + trackbar | unit |
| US-005/FR-9 | Navigator scan/prev fail | unit |
| US-006/FR-10/11 | abort run + no-GUI error | integration |
| US-007/FR-12/13 | metrics byte equality + timing keys unchanged | integration |
| US-008/FR-16 | README/`--help` 文案人工核对 | manual |
| FR-14 | RingBuffer memory/滚动 | unit |

---

## 10. Implementation Plan

### 10.1 Phases

1. **P1** `mgmd/viz.py`：`FrameView`/`classify`/`RingBuffer`/`Navigator`/`draw_overlay` + 单测（无 GUI）。
2. **P2** `pipeline.run` 回调 + `_emit` + `_VizStop` + `MgmdFrameSource.close()` + `run_meta.viz`。
3. **P3** `run_mgmd.py --viz*` CLI、`Visualizer` 生命周期、GUI 错误处理、`CvWindow`。
4. **P4** 集成测试、README 按键表、真实 D2 `--limit 200` 手动 GUI 冒烟。

### 10.2 Issue mapping（建议）

| Issue | SPEC 章节 | 依赖 |
|---|---|---|
| V1 overlay/buffer/navigator + 单测 | §3, §5.2–5.4, §9.1 | — |
| V2 pipeline callback/abort/run_meta | §4.2, §5.1, §5.5 | V1 |
| V3 CLI + CvWindow + 错误处理 | §4.1, §4.3, §6 | V2 |
| V4 集成测试 + README + GUI 冒烟 | §9.2, §9.3 | V3 |

---

## 11. Open Questions & Risks

### 11.1 Unresolved

- 是否需要在暂停时把“当前帧/总帧”显示为绝对帧号（`start+limit`）还是处理序号？→ 默认绝对帧号。
- `--viz-every` 与扫描模式同时使用时，扫描命中帧是否强制渲染？→ 是。

### 11.2 Risks

| Risk | Impact | Mitigation |
|---|---|---|
| WSL2 无 WSLg/X11，窗口无法创建 | `--viz` 不可用 | 明确 `E_VIZ_UNAVAILABLE` + 修复建议；不静默降级 |
| cv2 无法区分 Shift，键位不一致 | ±10 帧不可用 | 增加 `a/d` 兼容键 |
| 缓冲内存过大 | OOM/卡顿 | 默认 200 帧；`--viz-buffer` 可调；仅存显示帧 |
| 早退指标为部分结果 | 误读 | `run_meta.viz.aborted/abort_frame` + warning |
| D1 全流水线仅 ~4 fps，扫描慢 | 跳转耗时长 | `--viz-scan-max` + 120 s 上限；提示 |
| 暂停阻塞管线被误认为卡死 | 使用困惑 | 标题/状态栏显示 PAUSE；按键提示 |

### 11.3 Assumptions

- OpenCV HighGUI 可用（WSLg 或 X server）；否则按错误路径处理。
- `WindowRecord.est` 为原图坐标、`pick_rect` 为处理尺度坐标（与当前 `_frame_record`/`_process` 一致）。
- `run_meta.json` 中的 `gt` 与 `run_mgmd.py --gt` 相同。
- 可视化仅用于调试，不进入 CI 的 GUI 测试。
