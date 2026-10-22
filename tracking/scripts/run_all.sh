#!/usr/bin/env bash
# scripts/run_all.sh — 一键复跑（SPEC §4.1）。
#
# 模式：
#   SPA2D_RUN_ALL_MODE=real  (默认) 真实 MVDT 全量：fetch --check → frozen 检查 →
#       make_derived_gt ×4 → run_sequence ×6 (--repeat 3) → baseline ×6 →
#       make_tables → make_figures → make_report。前置失败即停（Tier B）。
#   SPA2D_RUN_ALL_MODE=smoke 合成数据端到端冒烟（不依赖 MVDT 视频帧）。
#
# 环境：SPA2D_BACKEND=jax|numpy（默认 jax）
set -euo pipefail
cd "$(dirname "$0")/.."

PY=.venv/bin/python
MODE="${SPA2D_RUN_ALL_MODE:-real}"
BACKEND="${SPA2D_BACKEND:-jax}"

echo "════════════════════════════════════════════════════════"
echo " SPA2D × MVDT 复现 — Tier B（协议复现） | mode=$MODE backend=$BACKEND"
echo " 声明：未获取作者 artifact，不做 Tier A 逐值对照；FPS 为桌面 GPU 相对分析。"
echo "════════════════════════════════════════════════════════"

run() { echo "+ $*"; "$@"; }

if [ "$MODE" = "smoke" ]; then
  echo "[smoke] 合成数据端到端"
  run "$PY" scripts/make_synthetic.py --suite sanity --out data_raw/synthetic_smoke \
      --seqs 2 --frames 60 --width 240 --height 180 --seed 0 --overwrite
  run "$PY" scripts/make_synthetic.py --suite benchmark --out data_raw/synthetic_smoke \
      --frames 40 --width 240 --height 180 --seed 0 --overwrite
  "$PY" - <<'PYEOF'
import yaml, pathlib
cfg = yaml.safe_load(pathlib.Path("configs/D1_no_overlap.yaml").read_text())
cfg["sequence"]["video"] = None
cfg["sequence"]["name"] = "smoke/D1"
cfg["sequence"]["frames_dir"] = "data_raw/synthetic_smoke/seq_000/frames"
cfg["sequence"]["resolution"] = [240, 180]
cfg["sequence"]["fps"] = 25.0
cfg["window"] = {"N": 5, "stride": 5}
cfg["search"]["gpu_chunk_bytes"] = 134217728
cfg["backend"] = "numpy"
pathlib.Path("outputs").mkdir(exist_ok=True)
pathlib.Path("outputs/smoke_D1.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
pathlib.Path("data_derived/gt").mkdir(parents=True, exist_ok=True)
pathlib.Path("data_derived/gt/smoke_D1.csv").write_text(
    pathlib.Path("data_raw/synthetic_smoke/seq_000/gt.csv").read_text())
print("smoke config + GT written")
PYEOF
  run "$PY" scripts/run_sequence.py --config outputs/smoke_D1.yaml \
      --gt data_derived/gt/smoke_D1.csv --out outputs/runs/smoke_D1 \
      --backend "$BACKEND" --limit 12 --repeat 3 --allow-unfrozen
  run "$PY" scripts/baseline_bg_subtraction.py --config outputs/smoke_D1.yaml \
      --gt data_derived/gt/smoke_D1.csv --out outputs/runs_baselines/bgsub_smoke_D1 --tracker nn
  run "$PY" scripts/make_tables.py --runs outputs/runs --gt data_derived/gt --out outputs/metrics
  run "$PY" scripts/make_figures.py --runs outputs/runs --gt data_derived/gt --out outputs/figures
  run "$PY" scripts/make_report.py
  echo "✅ smoke 完成：outputs/report.md, outputs/metrics/, outputs/figures/"
  exit 0
fi

# ---- real ----
run bash scripts/fetch_data.sh --dest data_raw/mvdt --check
if [ ! -f configs/thresholds_frozen.yaml ]; then
  echo "❌ 缺少冻结阈值 configs/thresholds_frozen.yaml → 先运行 calibrate.py" >&2
  exit 5
fi
declare -A SEQ=( [D1_no_overlap]=D1_cam2 [D1_overlap]=D1_cam2 [D2_no_overlap]=D2_cam1 \
                 [D2_overlap]=D2_cam1 [D3_overlap]=D3_cam3 [D4_overlap]=D4_cam3 )
for s in "D1/cam2" "D2/cam1" "D3/cam3" "D4/cam3"; do
  run "$PY" scripts/make_derived_gt.py --seq "$s" --raw data_raw/mvdt \
      --out "data_derived/gt/${s//\//_}.csv"
done
for c in D1_no_overlap D1_overlap D2_no_overlap D2_overlap D3_overlap D4_overlap; do
  run "$PY" scripts/run_sequence.py --config "configs/$c.yaml" \
      --gt "data_derived/gt/${SEQ[$c]}.csv" --out "outputs/runs/$c" \
      --backend "$BACKEND" --repeat 3
done
for c in D1_no_overlap D1_overlap D2_no_overlap D2_overlap D3_overlap D4_overlap; do
  run "$PY" scripts/baseline_bg_subtraction.py --config "configs/$c.yaml" \
      --gt "data_derived/gt/${SEQ[$c]}.csv" --out "outputs/runs_baselines/bgsub_$c"
done
run "$PY" scripts/make_tables.py --runs outputs/runs --gt data_derived/gt --out outputs/metrics
run "$PY" scripts/make_figures.py --runs outputs/runs --gt data_derived/gt --out outputs/figures
run "$PY" scripts/make_report.py
echo "✅ real 完成：outputs/report.md"
