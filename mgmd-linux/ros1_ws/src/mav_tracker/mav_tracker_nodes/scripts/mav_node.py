#!/usr/bin/env python3
"""mav_node — 帧发给 sidecar，检测转 ROS 消息。

S1 骨架：发空 DetectionArray。socket 协议在 S3 实现。
"""
import rospy
from sensor_msgs.msg import Image
from mav_tracker_msgs.msg import DetectionArray


def main():
    rospy.init_node("mav_node")
    pub = rospy.Publisher("/mav/detections", DetectionArray, queue_size=2)
    pub_c = rospy.Publisher("/mav/candidates", DetectionArray, queue_size=2)

    def on_image(msg):
        out = DetectionArray()
        out.header = msg.header
        out.image_width, out.image_height = msg.width, msg.height
        out.n_candidates = 0
        pub.publish(out)
        pub_c.publish(out)

    rospy.Subscriber("/camera/image_raw", Image, on_image, queue_size=1)
    rospy.spin()


if __name__ == "__main__":
    main()
