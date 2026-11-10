#!/usr/bin/env bash
set +u
LATEST=/home/gaodu/.ros/log/latest
echo "== latest dir =="
ls -l "$LATEST" 2>/dev/null | head -3
echo "== image_view logs =="
ls -t /home/gaodu/.ros/log/*/image_view*.log 2>/dev/null | head -3
for f in $(ls -t /home/gaodu/.ros/log/*/image_view*.log 2>/dev/null | head -1); do
  echo "--- $f ---"
  tail -40 "$f"
done
echo "== node log dirs =="
ls -t /home/gaodu/.ros/log/ | head -4
echo "== image_view proc env (DISPLAY/WAYLAND) =="
pid=$(pgrep -x image_view | head -1)
echo "pid=$pid"
tr '\0' '\n' < "/proc/$pid/environ" 2>/dev/null | grep -Ei 'display|wayland|qt|gdk' || echo "no env"
echo "DIAG_DONE"
