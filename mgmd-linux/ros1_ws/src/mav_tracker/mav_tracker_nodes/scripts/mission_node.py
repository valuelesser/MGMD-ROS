#!/usr/bin/env python3
"""mission_node — 跟踪任务编排、视觉伺服、失控保护。

S1 骨架：发零速度。控制律在 S8 实现。
"""
import rospy
from geometry_msgs.msg import Twist
from mav_tracker_msgs.msg import TrackArray


def main():
    rospy.init_node("mission_node")
    pub = rospy.Publisher("/tt/cmd_vel", Twist, queue_size=2)

    def on_tracks(msg):
        cmd = Twist()          # 全零：不动
        pub.publish(cmd)

    rospy.Subscriber("/mav/tracks", TrackArray, on_tracks, queue_size=1)
    rospy.spin()


if __name__ == "__main__":
    main()
