#!/usr/bin/env bash
# scripts/fetch_data.sh — MVDT 数据获取/校验（SPEC §4.1 / §7.1）。
#
# 用法：
#   fetch_data.sh --source <dir|urls.txt> --dest data_raw/mvdt [--resume] \
#                 [--accept-license <LICENSE>]
#   fetch_data.sh --dest data_raw/mvdt --check
#
# 本地目录模式：rsync 复制目录树；URL 清单模式：逐行 <url> [relpath]，curl 下载（可 -C - 续传）。
# 之后校验/生成 checksums.sha256，并写 provenance.json。退出码：0 成功 / 2 用法·许可 / 3 数据错误。
set -uo pipefail

SUITE="spa2d"
SOURCE=""
DEST=""
CHECK=0
RESUME=0
LICENSE=""

usage() {
  sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'
}

err() { # code message remedy
  echo "[$SUITE] $1: $2 → $3" >&2
  exit "$1"
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --source) SOURCE="$2"; shift 2 ;;
    --dest) DEST="$2"; shift 2 ;;
    --check) CHECK=1; shift ;;
    --resume) RESUME=1; shift ;;
    --accept-license) LICENSE="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) err 2 "未知参数 $1" "见 --help" ;;
  esac
done

[ -n "$DEST" ] || err 2 "缺少 --dest" "见 --help"
[ "$CHECK" = 1 ] || [ -n "$SOURCE" ] || err 2 "缺少 --source" "见 --help"

DEST="${DEST%/}"

gen_checksums() {
  ( cd "$DEST" \
    && find . -type f ! -name provenance.json ! -name 'checksums.sha256*' -print0 \
       | sort -z | xargs -0 -r sha256sum | sed 's#  \./#  #' > checksums.sha256.tmp \
    && mv checksums.sha256.tmp checksums.sha256 )
}

verify_checksums() { # returns nonzero on mismatch; prints diff
  ( cd "$DEST" && sha256sum -c checksums.sha256 2>&1 | grep -v ': OK$' ) || true
}

if [ "$CHECK" = 1 ]; then
  [ -d "$DEST" ] || err 3 "目标目录不存在: $DEST" "先运行获取流程"
  [ -f "$DEST/checksums.sha256" ] || err 3 "缺少 checksums.sha256: $DEST" "重新获取或生成校验和"
  BAD="$(verify_checksums)"
  if [ -n "$BAD" ]; then
    echo "$BAD" >&2
    err 3 "校验失败（见上）" "重新获取损坏文件"
  fi
  echo "[$SUITE] checksums OK: $DEST"
  exit 0
fi

# ---- 获取模式：需要许可接受（SPEC §7.1）----
LICENSE_SHA=""
if [ -n "$LICENSE" ]; then
  [ -f "$LICENSE" ] || err 2 "许可文件不存在: $LICENSE" "提供有效 LICENSE 路径"
  LICENSE_SHA="$(sha256sum "$LICENSE" | awk '{print $1}')"
elif [ "${MVDT_LICENSE_ACCEPTED:-0}" = "1" ]; then
  LICENSE_SHA="env:MVDT_LICENSE_ACCEPTED=1"
else
  err 2 "未接受数据许可" "用 --accept-license <LICENSE> 或设置 MVDT_LICENSE_ACCEPTED=1"
fi

mkdir -p "$DEST" || err 3 "无法创建目录: $DEST" "检查权限"

if [ -d "$SOURCE" ]; then
  MODE="local"
  rsync -a --exclude checksums.sha256 --exclude provenance.json \
        "$SOURCE"/ "$DEST"/ || err 3 "本地复制失败: $SOURCE" "检查源目录权限"
elif [ -f "$SOURCE" ]; then
  MODE="urls"
  while IFS= read -r line || [ -n "$line" ]; do
    case "$line" in ''|'#'*) continue ;; esac
    url="$(echo "$line" | awk '{print $1}')"
    rel="$(echo "$line" | awk '{print $2}')"
    [ -n "$rel" ] || rel="$(basename "$url")"
    out="$DEST/$rel"
    mkdir -p "$(dirname "$out")"
    curl_args=(-fSL)
    if [ "$RESUME" = 1 ] && [ -f "$out" ]; then curl_args+=(-C -); fi
    ok=0
    for attempt in 1 2 3; do
      if curl "${curl_args[@]}" -o "$out" "$url" 2>/dev/null; then ok=1; break; fi
      case "$attempt" in 1) sleep 2 ;; 2) sleep 4 ;; 3) sleep 8 ;; esac
    done
    [ "$ok" = 1 ] || err 3 "下载失败: $url" "检查网络/URL 后重试 --resume"
  done < "$SOURCE"
else
  err 2 "source 既非目录也非 URL 清单: $SOURCE" "提供本地目录或 urls.txt"
fi

# 若源数据自带 checksums.sha256，先做一次校验
if [ -f "$DEST/checksums.sha256" ]; then
  BAD="$(verify_checksums)"
  if [ -n "$BAD" ]; then
    echo "$BAD" >&2
    err 3 "源校验失败（见上）" "重新获取损坏文件"
  fi
fi

gen_checksums
BAD="$(verify_checksums)"
[ -z "$BAD" ] || { echo "$BAD" >&2; err 3 "生成后校验失败" "重新获取"; }

python3 - "$DEST" "$SOURCE" "$MODE" "$LICENSE_SHA" <<'PY'
import hashlib, json, pathlib, sys, datetime
dest, source, mode, lic = sys.argv[1:5]
root = pathlib.Path(dest)
files = {}
for p in sorted(root.rglob("*")):
    if p.is_file() and p.name != "provenance.json":
        files[str(p.relative_to(root))] = hashlib.sha256(p.read_bytes()).hexdigest()
prov = {
    "schema_version": 1,
    "mode": mode,
    "source": source,
    "dest": dest,
    "accepted_license_sha256": lic or None,
    "n_files": len(files),
    "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "files": files,
}
(root / "provenance.json").write_text(json.dumps(prov, indent=2, sort_keys=True))
PY

echo "[$SUITE] fetched ($MODE): ${DEST} ($(find "$DEST" -type f | wc -l) files)"
exit 0
