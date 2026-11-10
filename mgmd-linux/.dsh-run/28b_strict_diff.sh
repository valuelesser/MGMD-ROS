#!/usr/bin/env bash
# 严格对比：排除生成物、依赖、辅助脚本、catkin 自动生成的两个文件
set +u
SRC="/mnt/c/Users/gaodu/Desktop/opencode/mgmd_project_20261104/毕业设计/mgmd-linux"
DST="/home/gaodu/毕业设计/mgmd-linux"
out=$(diff -rq \
  --exclude=.dsh-run --exclude=third_party --exclude=outputs \
  --exclude=.venv38 --exclude=build --exclude=devel \
  --exclude=__pycache__ --exclude=.pytest_cache --exclude='*.engine' \
  --exclude=CMakeLists.txt --exclude=.catkin_workspace \
  "$SRC" "$DST")
rc=$?
if [ -z "$out" ] && [ "$rc" -eq 0 ]; then
  echo "IDENTICAL: 所有源码文件逐字节一致"
else
  echo "DIFF rc=$rc:"
  echo "$out"
fi
