#!/usr/bin/env python3
"""tt_driver — Tello/TT 视频源与飞控指令桥。

S1 骨架：仅保证节点能起、话题能发。真正的 tello/file/gazebo 三模式在 S4/S5.5 实现。
"""
import rospy
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
import numpy as np

W, H = 960, 720


def main():
    rospy.init_node("tt_driver")
    pub_img = rospy.Publisher("/camera/image_raw", Image, queue_size=2)
    pub_state = rospy.Publisher("/tt/state", Float32MultiArray, queue_size=2)
    rospy.Subscriber("/tt/cmd_vel", Twist, lambda m: None)

    rate = rospy.Rate(30)
    img = Image()
    img.height, img.width = H, W
    img.encoding = "bgr8"
    img.is_bigendian = 0
    img.step = W * 3
    img.data = bytes(H * W * 3)     # 全黑图（S1 骨架）

    seq = 0
    while not rospy.is_shutdown():
        now = rospy.Time.now()
        img.header.stamp = now
        img.header.frame_id = "camera"
        img.header.seq = seq
        pub_img.publish(img)
        st = Float32MultiArray(data=[0.0] * 10)
        pub_state.publish(st)
        seq += 1
        rate.sleep()


if __name__ == "__main__":
    main()
