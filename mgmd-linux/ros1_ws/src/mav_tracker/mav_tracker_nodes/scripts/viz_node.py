#!/usr/bin/env python3
"""viz_node — 画面叠加与 HUD。

S1 骨架：原图直接转发到 /mav/debug_image。
"""
import rospy
from sensor_msgs.msg import Image


def main():
    rospy.init_node("viz_node")
    pub = rospy.Publisher("/mav/debug_image", Image, queue_size=2)

    def on_image(msg):
        pub.publish(msg)

    rospy.Subscriber("/camera/image_raw", Image, on_image, queue_size=1)
    rospy.spin()


if __name__ == "__main__":
    main()
