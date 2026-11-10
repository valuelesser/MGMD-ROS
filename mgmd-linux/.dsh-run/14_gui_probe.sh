#!/usr/bin/env bash
set +u
export DISPLAY=:0

echo "== x tools =="
command -v xwininfo || echo "no xwininfo"
command -v wmctrl || echo "no wmctrl"

echo "== all windows (xdotool, no visible filter) =="
for w in $(xdotool search --name '.' 2>/dev/null); do
  echo "win=$w name=[$(xdotool getwindowname "$w" 2>/dev/null)]"
done

echo "== opencv gui probe (system python3) =="
python3 - <<'PY'
import cv2, numpy as np, time
print("cv2", cv2.__version__)
img = np.zeros((240, 320, 3), np.uint8)
img[:] = (0, 128, 255)
cv2.namedWindow("probe_window", cv2.WINDOW_NORMAL)
cv2.imshow("probe_window", img)
for _ in range(30):
    cv2.waitKey(100)
cv2.destroyAllWindows()
print("PROBE_OK")
PY

echo "== windows after probe =="
for w in $(xdotool search --name '.' 2>/dev/null); do
  echo "win=$w name=[$(xdotool getwindowname "$w" 2>/dev/null)]"
done
echo "GUI_PROBE_DONE"
