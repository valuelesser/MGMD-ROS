#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""target_mover — 目标轨迹驱动节点（SPEC-ROS §11.6）

通过 `/gazebo/set_model_state` 按预设轨迹移动目标模型，
使 MFE 的三帧差分能检测到运动（静态目标候选恒为 0）。

轨迹复用 ARD-MAV 风格（§11.6）：
    - 直线匀速
    - 变速（正弦调制）
    - 尺寸衰减（远离，模拟目标变小）

用法：
    rosrun mav_tracker_nodes target_mover.py _model:=target _traj:=line

参数：
    ~model    目标模型名，默认 target
    ~traj     轨迹类型: line | sine | recede，默认 line
    ~speed    基础速度 m/s，默认 0.8
    ~rate     更新频率 Hz，默认 30
"""
from __future__ import annotations

import math

import rospy
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState


class TargetMover:
    def __init__(self):
        self.model = rospy.get_param("~model", "target")
        self.traj = rospy.get_param("~traj", "line")
        self.speed = float(rospy.get_param("~speed", 0.8))
        self.rate = float(rospy.get_param("~rate", 30.0))

        rospy.loginfo("[target_mover] 等待 /gazebo/set_model_state ...")
        rospy.wait_for_service("/gazebo/set_model_state", timeout=30)
        self.set_state = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)

        # 起点：从 Gazebo 读实际位置，避免硬编码与 world 不一致
        self.base_x, self.base_y, self.base_z = self._read_start_pose()

        self.t0 = rospy.Time.now().to_sec()

    def _read_start_pose(self):
        """读目标当前位姿作为轨迹基准点。"""
        try:
            from gazebo_msgs.srv import GetModelState
            rospy.wait_for_service("/gazebo/get_model_state", timeout=15)
            get_state = rospy.ServiceProxy("/gazebo/get_model_state", GetModelState)
            st = get_state(self.model, "world")
            p = st.pose.position
            rospy.loginfo("[target_mover] 基准点 = (%.2f, %.2f, %.2f)",
                          p.x, p.y, p.z)
            return p.x, p.y, p.z
        except Exception as exc:            # noqa: BLE001
            rospy.logwarn("[target_mover] 读起点失败(%s)，回退 (5,0,1.6)", exc)
            return 5.0, 0.0, 1.6

    def _pose(self, t: float):
        """返回 (x, y, z)。"""
        if self.traj == "line":
            # 横向匀速往返，幅度 ±1.2 m
            span = 1.2
            period = 4.0 * span / self.speed
            ph = (t % period) / period
            y = -span + 2.0 * span * (ph if ph < 0.5 else 1.0 - ph)
            return self.base_x, y, self.base_z

        if self.traj == "sine":
            # 正弦变速（ARD-MAV 风格）
            y = 1.2 * math.sin(2.0 * math.pi * t / 6.0)
            z = self.base_z + 0.25 * math.sin(2.0 * math.pi * t / 4.0)
            return self.base_x, y, z

        if self.traj == "recede":
            # 尺寸衰减：沿 x 远离，同时横向摆动
            x = self.base_x + 1.5 * math.sin(2.0 * math.pi * t / 10.0)
            y = 1.0 * math.sin(2.0 * math.pi * t / 5.0)
            return x, y, self.base_z

        rospy.logwarn_once("[target_mover] 未知轨迹 %s，回退 line", self.traj)
        return self.base_x, 0.0, self.base_z

    def spin(self):
        r = rospy.Rate(self.rate)
        n = 0
        while not rospy.is_shutdown():
            t = rospy.Time.now().to_sec() - self.t0
            x, y, z = self._pose(t)
            msg = ModelState()
            msg.model_name = self.model
            msg.pose.position.x = x
            msg.pose.position.y = y
            msg.pose.position.z = z
            msg.pose.orientation.w = 1.0
            msg.reference_frame = "world"
            try:
                self.set_state(msg)
            except rospy.ServiceException as exc:
                rospy.logwarn_throttle(5.0, "[target_mover] 调用失败: %s", exc)
            n += 1
            if n % 60 == 0:
                rospy.loginfo("[target_mover] %s t=%.1fs pos=(%.2f, %.2f, %.2f)",
                              self.traj, t, x, y, z)
            r.sleep()


if __name__ == "__main__":
    rospy.init_node("target_mover")
    TargetMover().spin()
