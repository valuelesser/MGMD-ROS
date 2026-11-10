#!/usr/bin/env bash
# 全 15 个测试视频评测（SPEC T11b）。
# 用法：
#   bash scripts/run_testset.sh                 # GPU（自动）
#   bash scripts/run_testset.sh --cpu           # 强制 CPU
#   bash scripts/run_testset.sh --video phantom09
set -e
cd "$(dirname "$0")/.."
source scripts/env_setup.sh

DEVICE="auto"
ARGS=()
for a in "$@"; do
  case "$a" in
    --cpu) DEVICE="cpu" ;;
    --gpu) DEVICE="cuda" ;;
    *) ARGS+=("$a") ;;
  esac
done

# 自检 GPU
$PY - <<'PYCHK'
import torch, sys
ok = torch.cuda.is_available()
print(f"[env] torch={torch.__version__} cuda_available={ok}")
if ok:
    print(f"[env] device={torch.cuda.get_device_name(0)} "
          f"vram={torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")
PYCHK

exec $PY -u scripts/run_testset.py --device "$DEVICE" "${ARGS[@]}"
