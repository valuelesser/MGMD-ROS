#!/usr/bin/env bash
# 打开一个纯橙色 OpenCV 窗口并保持 12 秒，用于像素级验证 WSLg 弹窗
set +u
export DISPLAY=:0
python3 - <<'PY'
import cv2, numpy as np, time
img = np.zeros((300, 500, 3), np.uint8)
img[:] = (0, 128, 255)          # BGR -> 橙色
cv2.namedWindow("probe_orange", cv2.WINDOW_NORMAL)
cv2.resizeWindow("probe_orange", 500, 300)
cv2.imshow("probe_orange", img)
cv2.waitKey(200)
cv2.moveWindow("probe_orange", 80, 80)
for _ in range(120):
    cv2.waitKey(100)
cv2.destroyAllWindows()
print("PROBE_DONE")
PY
