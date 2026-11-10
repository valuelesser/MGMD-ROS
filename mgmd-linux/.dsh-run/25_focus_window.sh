#!/usr/bin/env bash
set +u
export DISPLAY=:0
W=$(xdotool search --name "MGMD-ROS Qt" | head -1)
echo "win=$W"
if [ -n "$W" ]; then
  xdotool windowsize "$W" 1000 760
  xdotool windowmove "$W" 40 40
  xdotool windowactivate "$W"
  xdotool windowraise "$W"
  sleep 1
  echo "title1: $(xdotool getwindowname "$W")"
  xdotool getwindowgeometry "$W"
  sleep 3
  echo "title2: $(xdotool getwindowname "$W")"
fi
echo "FOCUS_DONE"
