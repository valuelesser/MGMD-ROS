#!/usr/bin/env python3
"""21_qt_viewer.py — PyQt5 版 /mav/debug_image 查看器（WSLg 下比 OpenCV GTK 稳）。

用法：
    DISPLAY=:0 python3 21_qt_viewer.py
窗口标题：MGMD-ROS Qt
"""
import sys
import threading

import cv2
import numpy as np
import rospy
from sensor_msgs.msg import Image

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QLabel

latest = {"img": None, "n": 0}


def cb(msg: Image):
    arr = np.frombuffer(msg.data, dtype=np.uint8)
    if arr.size != msg.height * msg.width * 3:
        return
    latest["img"] = arr.reshape((msg.height, msg.width, 3)).copy()
    latest["n"] += 1


def main():
    rospy.init_node("qt_viewer", anonymous=True)
    rospy.Subscriber("/mav/debug_image", Image, cb, queue_size=1,
                     buff_size=2 * 1920 * 1080 * 3 + 1024)
    threading.Thread(target=rospy.spin, daemon=True).start()

    app = QApplication(sys.argv)
    label = QLabel("waiting for /mav/debug_image ...")
    label.setWindowTitle("MGMD-ROS Qt")
    label.setAlignment(Qt.AlignCenter)
    label.resize(960, 720)
    label.show()

    def tick():
        img = latest["img"]
        if img is None:
            return
        h, w = img.shape[:2]
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # Qt 5.12 没有 Format_BGR888
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
        label.setPixmap(QPixmap.fromImage(qimg).scaled(
            label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        label.setWindowTitle("MGMD-ROS Qt  frames=%d" % latest["n"])

    timer = QTimer()
    timer.timeout.connect(tick)
    timer.start(33)
    print("Qt viewer 启动，等待图像…", flush=True)
    app.exec_()


if __name__ == "__main__":
    main()
