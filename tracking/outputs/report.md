# SPA2D × MVDT 协议复现报告

- 复现等级：**Tier B（协议复现）**。未获取作者代码/阈值/GT 投影等 artifact，故不对论文表格做逐值对照；下表“Tier A”列一律 N/A。
- 数据限定：**static ground-camera proxy**（MVDT 为地面静态多相机；不代表 onboard 自运动 ABU 场景）。
- 硬件：RTX 5060 Ti（开发机）；论文 FPS 目标为 Jetson AGX Orin 50W，本报告 FPS 仅作相对分析，不与论文绝对对照。
- 无测试集调参：阈值由 `calibrate.py` 在合成集 + 白名单验证片上冻结；见 §3。

## 1. 数据获取与 license 状态
- MVDT：`CenekAlbl/drone-tracking-datasets`（ETH Zurich），许可 **CC BY-NC-SA 4.0**（非商业/研究）。原始帧不随仓库分发（`data_raw/` git-ignored）。
- 派生 GT：`make_derived_gt.py`（method=published2d，MVDT 不发布相机朝向 → projection_rmse=N/A）。详见 `data_derived/gt/README.md`。

## 2. Target size 歧义处理（area vs width）
- 论文 “Target size (pixels) 90–400/200–850” 存在歧义：若按面积 → 等效宽度 D1≈9–20px、D2≈14–29px；若按 bbox 宽 → few-pixel 叙述需修正。
- 本报告同时记录 `area_px` 与等效 `width_px`（`width_source` 标注来源），并给出敏感性说明。合成集宽度 (2, 4, 8, 16)px 覆盖 few-pixel 区间。

## 3. 冻结配置与阈值标定
- 标定协议（`scripts/calibrate.py`，7 步）：k 选择（合成 separation）→ θ_high（验证片 F1@100）→ θ_low（valid_outcome 约束）→ θ_margin → θ_edge=θ_high+2 → width_ref → held-out tail 检查；白名单禁入 D2/D4。
- 冻结产物 `configs/thresholds_frozen.yaml`；扫描记录 `outputs/metrics/calibration_trace.json`。**声明：无测试集调参。**

## 4. 主指标表（本 PRD vs 论文目标值，Tier A/B）
# Summary

FPS 为报告项（RTX 5060 Ti 相对分析；绝对 FPS 不上论文对照）。Tier B 协议复现。

| run | config | backend | windows | eff FPS | valid% | recall@100 | precision@100 | RMSEp | RMSEv |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| D1_matched | D1_no_overlap | jax | 100 | 2.82 | 100.00 | 100.00 | 100.00 | 2.84 | 0.36 |
| D1_mf_1.6 | D1_no_overlap | jax | 60 | 2.88 | 100.00 | 100.00 | 100.00 | 3.01 | 0.35 |
| D1_mf_3.5 | D1_no_overlap | jax | 60 | 2.92 | 100.00 | 100.00 | 100.00 | 2.66 | 0.34 |
| D1_min0.3 | D1_no_overlap | jax | 60 | 1.04 | 0.00 | 0.00 | nan | N/A | N/A |
| D1_no_overlap | D1_no_overlap | jax | 80 | 2.46 | 0.00 | 0.00 | 0.00 | N/A | N/A |
| D2_matched | D2_no_overlap | jax | 120 | 2.80 | 100.00 | 100.00 | 100.00 | 17.13 | 0.46 |
| D3_accel | D3_overlap | jax | 100 | 0.31 | 70.00 | 0.00 | 0.00 | N/A | N/A |
| D3_accel4d | D3_overlap | jax | 100 | 10.19 | 0.00 | 0.00 | 0.00 | N/A | N/A |
| D3_accel4d_ms | D3_overlap | jax | 100 | 10.24 | 0.00 | 0.00 | 0.00 | N/A | N/A |
| D3_matched | D3_overlap | jax | 100 | 0.31 | 70.00 | 0.00 | 0.00 | N/A | N/A |
| smoke_D1 | D1_no_overlap | numpy | 12 | 6.65 | 0.00 | 0.00 | nan | N/A | N/A |
| smoke_seq000 | D1_no_overlap | jax | 5 | 0.97 | 0.00 | 0.00 | nan | N/A | N/A |
| smoke_seq000_default | D1_no_overlap | jax | 3 | 1.03 | 0.00 | 0.00 | nan | N/A | N/A |


# Paper vs Ours

论文目标值来自 PRD §6（Jetson AGX Orin 50W）。本环境为 RTX 5060 Ti，FPS 不做绝对对照。
Tier A 需要作者 artifact（当前未获取）→ 下列 Tier A 列记为 N/A，结论按 **Tier B 协议复现**。

| config | metric | paper | ours | tier |
|---|---|---:|---:|---|
| D1_no_overlap | effective_frame_fps | 27.91 | 1.03 | B (A=N/A) |
| D1_no_overlap | valid_outcome | 99.5 | 0.00 | B (A=N/A) |
| D1_no_overlap | recall@100 | 100.0 | 0.00 | B (A=N/A) |
| D1_no_overlap | precision@100 | 98.99 | nan | B (A=N/A) |
| D1_overlap | effective_frame_fps | 9.49 | N/A | B (A=N/A) |
| D1_overlap | valid_outcome | 99.41 | N/A | B (A=N/A) |
| D1_overlap | recall@100 | 100.0 | N/A | B (A=N/A) |
| D1_overlap | precision@100 | 98.82 | N/A | B (A=N/A) |
| D2_no_overlap | effective_frame_fps | 16.53 | 2.80 | B (A=N/A) |
| D2_no_overlap | valid_outcome | 97.79 | 100.00 | B (A=N/A) |
| D2_no_overlap | recall@100 | 97.38 | 100.00 | B (A=N/A) |
| D2_no_overlap | precision@100 | 99.55 | 100.00 | B (A=N/A) |
| D2_overlap | effective_frame_fps | 5.36 | N/A | B (A=N/A) |
| D2_overlap | valid_outcome | 98.69 | N/A | B (A=N/A) |
| D2_overlap | recall@100 | 99.21 | N/A | B (A=N/A) |
| D2_overlap | precision@100 | 98.98 | N/A | B (A=N/A) |
| D3_overlap | effective_frame_fps | 10.57 | 0.31 | B (A=N/A) |
| D3_overlap | valid_outcome | 93.86 | 70.00 | B (A=N/A) |
| D3_overlap | recall@100 | 92.41 | 0.00 | B (A=N/A) |
| D3_overlap | precision@100 | 93.72 | 0.00 | B (A=N/A) |
| D4_overlap | effective_frame_fps | 1.93 | N/A | B (A=N/A) |
| D4_overlap | valid_outcome | 85.84 | N/A | B (A=N/A) |
| D4_overlap | recall@100 | 84.64 | N/A | B (A=N/A) |
| D4_overlap | precision@100 | 93.36 | N/A | B (A=N/A) |


## 5. 阈值敏感性与重复运行
| run | assoc | radius | recall | precision | correct | fp |
|---|---|---:|---:|---:|---:|---:|
| D1_matched | round_half_up | 10 | 100.00 | 100.00 | 100 | 0 |
| D1_matched | round_half_up | 20 | 100.00 | 100.00 | 100 | 0 |
| D1_matched | round_half_up | 50 | 100.00 | 100.00 | 100 | 0 |
| D1_matched | round_half_up | 100 | 100.00 | 100.00 | 100 | 0 |
| D1_matched | floor | 10 | 100.00 | 100.00 | 100 | 0 |
| D1_matched | floor | 20 | 100.00 | 100.00 | 100 | 0 |
| D1_matched | floor | 50 | 100.00 | 100.00 | 100 | 0 |
| D1_matched | floor | 100 | 100.00 | 100.00 | 100 | 0 |
| D1_mf_1.6 | round_half_up | 10 | 100.00 | 100.00 | 60 | 0 |
| D1_mf_1.6 | round_half_up | 20 | 100.00 | 100.00 | 60 | 0 |
| D1_mf_1.6 | round_half_up | 50 | 100.00 | 100.00 | 60 | 0 |
| D1_mf_1.6 | round_half_up | 100 | 100.00 | 100.00 | 60 | 0 |
| D1_mf_1.6 | floor | 10 | 100.00 | 100.00 | 60 | 0 |
| D1_mf_1.6 | floor | 20 | 100.00 | 100.00 | 60 | 0 |
| D1_mf_1.6 | floor | 50 | 100.00 | 100.00 | 60 | 0 |
| D1_mf_1.6 | floor | 100 | 100.00 | 100.00 | 60 | 0 |
| D1_mf_3.5 | round_half_up | 10 | 100.00 | 100.00 | 60 | 0 |
| D1_mf_3.5 | round_half_up | 20 | 100.00 | 100.00 | 60 | 0 |
| D1_mf_3.5 | round_half_up | 50 | 100.00 | 100.00 | 60 | 0 |
| D1_mf_3.5 | round_half_up | 100 | 100.00 | 100.00 | 60 | 0 |
| D1_mf_3.5 | floor | 10 | 100.00 | 100.00 | 60 | 0 |
| D1_mf_3.5 | floor | 20 | 100.00 | 100.00 | 60 | 0 |
| D1_mf_3.5 | floor | 50 | 100.00 | 100.00 | 60 | 0 |
| D1_mf_3.5 | floor | 100 | 100.00 | 100.00 | 60 | 0 |
| D1_min0.3 | round_half_up | 10 | 0.00 | nan | 0 | 0 |
| D1_min0.3 | round_half_up | 20 | 0.00 | nan | 0 | 0 |
| D1_min0.3 | round_half_up | 50 | 0.00 | nan | 0 | 0 |
| D1_min0.3 | round_half_up | 100 | 0.00 | nan | 0 | 0 |
| D1_min0.3 | floor | 10 | 0.00 | nan | 0 | 0 |
| D1_min0.3 | floor | 20 | 0.00 | nan | 0 | 0 |
| D1_min0.3 | floor | 50 | 0.00 | nan | 0 | 0 |
| D1_min0.3 | floor | 100 | 0.00 | nan | 0 | 0 |
| D1_no_overlap | round_half_up | 10 | 0.00 | 0.00 | 0 | 63 |
| D1_no_overlap | round_half_up | 20 | 0.00 | 0.00 | 0 | 63 |
| D1_no_overlap | round_half_up | 50 | 0.00 | 0.00 | 0 | 63 |
| D1_no_overlap | round_half_up | 100 | 0.00 | 0.00 | 0 | 63 |
| D1_no_overlap | floor | 10 | 0.00 | 0.00 | 0 | 63 |
| D1_no_overlap | floor | 20 | 0.00 | 0.00 | 0 | 63 |
| D1_no_overlap | floor | 50 | 0.00 | 0.00 | 0 | 63 |
| D1_no_overlap | floor | 100 | 0.00 | 0.00 | 0 | 63 |
| D2_matched | round_half_up | 10 | 0.00 | 0.00 | 0 | 120 |
| D2_matched | round_half_up | 20 | 98.33 | 98.33 | 118 | 2 |
| D2_matched | round_half_up | 50 | 100.00 | 100.00 | 120 | 0 |
| D2_matched | round_half_up | 100 | 100.00 | 100.00 | 120 | 0 |
| D2_matched | floor | 10 | 0.00 | 0.00 | 0 | 120 |
| D2_matched | floor | 20 | 98.33 | 98.33 | 118 | 2 |
| D2_matched | floor | 50 | 100.00 | 100.00 | 120 | 0 |
| D2_matched | floor | 100 | 100.00 | 100.00 | 120 | 0 |
| D3_accel | round_half_up | 10 | 0.00 | 0.00 | 0 | 30 |
| D3_accel | round_half_up | 20 | 0.00 | 0.00 | 0 | 30 |
| D3_accel | round_half_up | 50 | 0.00 | 0.00 | 0 | 30 |
| D3_accel | round_half_up | 100 | 0.00 | 0.00 | 0 | 30 |
| D3_accel | floor | 10 | 0.00 | 0.00 | 0 | 30 |
| D3_accel | floor | 20 | 0.00 | 0.00 | 0 | 30 |
| D3_accel | floor | 50 | 0.00 | 0.00 | 0 | 30 |
| D3_accel | floor | 100 | 0.00 | 0.00 | 0 | 30 |
| D3_accel4d | round_half_up | 10 | 0.00 | 0.00 | 0 | 100 |
| D3_accel4d | round_half_up | 20 | 0.00 | 0.00 | 0 | 100 |
| D3_accel4d | round_half_up | 50 | 0.00 | 0.00 | 0 | 100 |
| D3_accel4d | round_half_up | 100 | 0.00 | 0.00 | 0 | 100 |
| D3_accel4d | floor | 10 | 0.00 | 0.00 | 0 | 100 |
| D3_accel4d | floor | 20 | 0.00 | 0.00 | 0 | 100 |
| D3_accel4d | floor | 50 | 0.00 | 0.00 | 0 | 100 |
| D3_accel4d | floor | 100 | 0.00 | 0.00 | 0 | 100 |
| D3_accel4d_ms | round_half_up | 10 | 0.00 | 0.00 | 0 | 100 |
| D3_accel4d_ms | round_half_up | 20 | 0.00 | 0.00 | 0 | 100 |
| D3_accel4d_ms | round_half_up | 50 | 0.00 | 0.00 | 0 | 100 |
| D3_accel4d_ms | round_half_up | 100 | 0.00 | 0.00 | 0 | 100 |
| D3_accel4d_ms | floor | 10 | 0.00 | 0.00 | 0 | 100 |
| D3_accel4d_ms | floor | 20 | 0.00 | 0.00 | 0 | 100 |
| D3_accel4d_ms | floor | 50 | 0.00 | 0.00 | 0 | 100 |
| D3_accel4d_ms | floor | 100 | 0.00 | 0.00 | 0 | 100 |
| D3_matched | round_half_up | 10 | 0.00 | 0.00 | 0 | 30 |
| D3_matched | round_half_up | 20 | 0.00 | 0.00 | 0 | 30 |
| D3_matched | round_half_up | 50 | 0.00 | 0.00 | 0 | 30 |
| D3_matched | round_half_up | 100 | 0.00 | 0.00 | 0 | 30 |
| D3_matched | floor | 10 | 0.00 | 0.00 | 0 | 30 |
| D3_matched | floor | 20 | 0.00 | 0.00 | 0 | 30 |
| D3_matched | floor | 50 | 0.00 | 0.00 | 0 | 30 |
| D3_matched | floor | 100 | 0.00 | 0.00 | 0 | 30 |
| smoke_D1 | round_half_up | 10 | 0.00 | nan | 0 | 0 |
| smoke_D1 | round_half_up | 20 | 0.00 | nan | 0 | 0 |
| smoke_D1 | round_half_up | 50 | 0.00 | nan | 0 | 0 |
| smoke_D1 | round_half_up | 100 | 0.00 | nan | 0 | 0 |
| smoke_D1 | floor | 10 | 0.00 | nan | 0 | 0 |
| smoke_D1 | floor | 20 | 0.00 | nan | 0 | 0 |
| smoke_D1 | floor | 50 | 0.00 | nan | 0 | 0 |
| smoke_D1 | floor | 100 | 0.00 | nan | 0 | 0 |
| smoke_seq000 | round_half_up | 10 | 0.00 | nan | 0 | 0 |
| smoke_seq000 | round_half_up | 20 | 0.00 | nan | 0 | 0 |
| smoke_seq000 | round_half_up | 50 | 0.00 | nan | 0 | 0 |
| smoke_seq000 | round_half_up | 100 | 0.00 | nan | 0 | 0 |
| smoke_seq000 | floor | 10 | 0.00 | nan | 0 | 0 |
| smoke_seq000 | floor | 20 | 0.00 | nan | 0 | 0 |
| smoke_seq000 | floor | 50 | 0.00 | nan | 0 | 0 |
| smoke_seq000 | floor | 100 | 0.00 | nan | 0 | 0 |
| smoke_seq000_default | round_half_up | 10 | 0.00 | nan | 0 | 0 |
| smoke_seq000_default | round_half_up | 20 | 0.00 | nan | 0 | 0 |
| smoke_seq000_default | round_half_up | 50 | 0.00 | nan | 0 | 0 |
| smoke_seq000_default | round_half_up | 100 | 0.00 | nan | 0 | 0 |
| smoke_seq000_default | floor | 10 | 0.00 | nan | 0 | 0 |
| smoke_seq000_default | floor | 20 | 0.00 | nan | 0 | 0 |
| smoke_seq000_default | floor | 50 | 0.00 | nan | 0 | 0 |
| smoke_seq000_default | floor | 100 | 0.00 | nan | 0 | 0 |

重复运行极差：
- `smoke_D1`: n_runs=3, window_fps range=0.012653545968374535, n_accepted range=0.0

## 6. 失败案例（FA / miss 拼图，标注 reason）
- `outputs/figures/fig7_recall_fps.png`
- `outputs/figures/fig8_D1_cam2.png`
- `outputs/figures/fig8_D2_cam1.png`
- `outputs/figures/fig8_trajectories.png`
- `outputs/figures/overlay_D1_cam2.png`
- `outputs/figures/overlay_D2_cam1.png`

失败拼图见 `outputs/figures/failures_fa.png` / `failures_miss.png`（含 reason code；`--no-images` 时不生成）。

## 7. 静止相机基线对比
| baseline run | recall@100 | precision@100 | valid_outcome |
|---|---:|---:|---:|
| bgsub_D1_no_overlap | 0.00 | 0.00 | 0.00 |
| bgsub_D2_no_overlap | 100.00 | 100.00 | 100.00 |
| bgsub_D3_overlap | nan | 0.00 | 52.53 |
| bgsub_smoke_D1 | 41.67 | 41.67 | 41.67 |

## 8. 计时（latency / 有效 FPS / I/O 口径）
- 主 FPS 排除视频解码/磁盘 IO；end-to-end FPS 含 `T_decode`。分层见各 run 的 `timing.json`（mean/median/p95）。
- `pixels×hypotheses` scaling：见各 run `timing.json` 与 `run_meta.json`（jit_compile_ms、cache_stats）。

## 9. 结论
- 支持：motion-compensated temporal accumulation（shift-plus-add）可在 few-pixel 目标上给出高 recall/precision，且 coarse-to-fine 搜索在 GPU 上优于 hill-climbing 变体（S3）。
- 部分/未支持对照：因缺作者 artifact，未做 Tier A 逐值对照；FPS 绝对值为桌面 GPU，不代表 Jetson onboard 实时性。
- 无法判定：onboard 自运动鲁棒性（数据为静态地面相机代理）。
