#!/usr/bin/env bash
set +u
export DISPLAY=:0
echo "== env =="
env | grep -Ei 'qt_qpa|gdk|display|wayland|libgl'
echo "== xdotool windows =="
for w in $(xdotool search --name '.' 2>/dev/null); do
  echo "win=$w name=[$(xdotool getwindowname "$w" 2>/dev/null)]"
done
echo "== GTK 3s window =="
python3 - <<'PY'
import cv2, numpy as np
img = np.zeros((200, 300, 3), np.uint8); img[:] = (0, 128, 255)
cv2.namedWindow("gtk_test", cv2.WINDOW_NORMAL)
cv2.imshow("gtk_test", img)
for _ in range(30): cv2.waitKey(100)
cv2.destroyAllWindows()
print("GTK_DONE")
PY
echo "== Qt 3s window =="
python3 - <<'PY'
import sys
from PyQt5.QtWidgets import QApplication, QLabel
app = QApplication(sys.argv)
l = QLabel("QT TEST WINDOW"); l.setWindowTitle("QT TEST"); l.resize(400,200); l.show()
from PyQt5.QtCore import QTimer
QTimer.singleShot(4000, app.quit)
app.exec_()
print("QT_DONE")
PY
echo "== weston recent =="
grep -iE 'gtk_test|QT TEST|appId' /mnt/wslg/weston.log 2>/dev/null | tail -8
echo "PROBE_DONE"
