#!/usr/bin/env python3
"""17_viewer.py — 主线程 cv2 窗口，订阅 /mav/debug_image 实时显示。

为什么不用 rosrun image_view：它在回调线程里 imshow，在 WSLg 的 GTK/Wayland 下
窗口不映射（实测 weston 日志无该 appId）。这里把 imshow 放在主线程。
"""
import cv2
import numpy as np
import rospy
from sensor_msgs.msg import Image

latest = {"img": None, "n": 0}


def cb(msg: Image):
    arr = np.frombuffer(msg.data, dtype=np.uint8)
    if arr.size != msg.height * msg.width * 3:
        return
    latest["img"] = arr.reshape((msg.height, msg.width, 3))
    latest["n"] += 1


def main():
    rospy.init_node("debug_viewer")
    rospy.Subscriber("/mav/debug_image", Image, cb, queue_size=1,
                     buff_size=2 * 1920 * 1080 * 3 + 1024)
    cv2.namedWindow("MGMD-ROS debug", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("MGMD-ROS debug", 960, 720)
    rospy.loginfo("[viewer] 等待 /mav/debug_image ...")
    while not rospy.is_shutdown():
        img = latest["img"]
        if img is not None:
            cv2.imshow("MGMD-ROS debug", img)
        key = cv2.waitKey(30) & 0xFF
        if key in (27, ord("q")):
            break
    cv2.destroyAllWindows()
    print("viewer 退出，显示帧数 =", latest["n"])


if __name__ == "__main__":
    main()
