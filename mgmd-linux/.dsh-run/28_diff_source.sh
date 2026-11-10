#!/usr/bin/env bash
# 对比「工作区源项目」与「WSL 里实际运行的项目」，排除生成物/依赖/辅助脚本
set +u
SRC="/mnt/c/Users/gaodu/Desktop/opencode/mgmd_project_20261104/毕业设计/mgmd-linux"
DST="/home/gaodu/毕业设计/mgmd-linux"

echo "== diff（只列差异；无输出=完全一致） =="
diff -rq \
  --exclude=.dsh-run \
  --exclude=third_party \
  --exclude=outputs \
  --exclude=.venv38 \
  --exclude=build \
  --exclude=devel \
  --exclude=__pycache__ \
  --exclude=.pytest_cache \
  --exclude='*.engine' \
  "$SRC" "$DST"
echo "diff_rc=$?"

echo
echo "== 源文件数量/校验和（两棵树各自统计） =="
for d in "$SRC" "$DST"; do
  n=$(find "$d" -type f \
        -not -path '*/.dsh-run/*' -not -path '*/third_party/*' \
        -not -path '*/outputs/*' -not -path '*/.venv38/*' \
        -not -path '*/ros1_ws/build/*' -not -path '*/ros1_ws/devel/*' \
        -not -path '*/__pycache__/*' | wc -l)
  h=$(find "$d" -type f \
        -not -path '*/.dsh-run/*' -not -path '*/third_party/*' \
        -not -path '*/outputs/*' -not -path '*/.venv38/*' \
        -not -path '*/ros1_ws/build/*' -not -path '*/ros1_ws/devel/*' \
        -not -path '*/__pycache__/*' -print0 \
      | sort -z | xargs -0 sha256sum 2>/dev/null | sha256sum | cut -d' ' -f1)
  echo "$d"
  echo "  files=$n  aggregate_sha256=$h"
done
echo "DIFF_DONE"
