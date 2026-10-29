#!/usr/bin/env python3
"""tracker_node — SORT 轨迹关联与 ID 管理。

S1 骨架：收空检测，发空 TrackArray。真 SORT 在 S3 接入。
"""
import rospy
from mav_tracker_msgs.msg import DetectionArray, TrackArray


def main():
    rospy.init_node("tracker_node")
    pub = rospy.Publisher("/mav/tracks", TrackArray, queue_size=2)

    def on_det(msg):
        out = TrackArray()
        out.header = msg.header
        pub.publish(out)

    rospy.Subscriber("/mav/detections", DetectionArray, on_det, queue_size=1)
    rospy.spin()


if __name__ == "__main__":
    main()
