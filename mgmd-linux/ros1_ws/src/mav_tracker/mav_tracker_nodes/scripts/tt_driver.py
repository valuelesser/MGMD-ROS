#!/usr/bin/env python3
"""tt_driver — 视频源与飞控指令桥（SPEC-ROS §9 / S3.5）。

三种 source 模式：
    source:=stub    全黑假帧（S1 骨架，用于纯话题连通性测试）
    source:=file    回放本地视频（S3.5 新增，用于无硬件联调）
    source:=tello   真机 Tello/TT SDK 取流（S6 实现，当前未实现）

参数：
    source:=file path:=<视频名或路径> [rate:=30] [loop:=true] [limit:=0]

    视频名可用 ARD-MAV 短名（如 phantom30），也可给绝对路径。
    rate  目标发布频率（Hz）；0 表示按视频原始帧率尽快发布。
    loop  回放结束后是否循环。
    limit 最多回放的帧数（0 = 不限），用于快速验证。

发布时间戳：header.stamp 用**墙钟时间**（与真实相机一致），
不使用视频原始 PTS —— 消费者（mav_node）按时间戳算延迟，需要墙钟语义。
"""
from pathlib import Path

import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray

import numpy as np


def _resolve_video(name: str) -> Path:
    """短名 → ARD 视频路径；否则按字面路径。"""
    p = Path(name)
    if p.is_file():
        return p
    import sys
    _root = Path(__file__).resolve().parents[5]        # mgmd-linux/
    if (_root / "mgmd").is_dir() and str(_root) not in sys.path:
        sys.path.insert(0, str(_root))
    try:
        from mgmd import config as C
        cand = C.ARD_VIDEOS / f"{name}.mp4"
        if cand.is_file():
            return cand
    except Exception as exc:      # noqa: BLE001
        rospy.logwarn(f"[tt_driver] 无法导入 mgmd.config: {exc}")
    raise SystemExit(f"[tt_driver] 找不到视频: {name}")


class FileSource:
    """回放本地视频文件。"""

    def __init__(self, path: Path, loop: bool):
        self.path = path
        self.loop = loop
        self.cap = None
        self._open()

    def _open(self):
        import cv2
        self.cap = cv2.VideoCapture(str(self.path))
        if not self.cap.isOpened():
            raise SystemExit(f"[tt_driver] 打不开视频: {self.path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.n_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.seq = 0

    def read(self):
        """返回 (bgr_frame, None) 或 (None, 'eof')。"""
        ok, frame = self.cap.read()
        if not ok:
            if not self.loop:
                return None, "eof"
            self.cap.release()
            self._open()
            ok, frame = self.cap.read()
            if not ok:
                return None, "eof"
        self.seq += 1
        return frame, None

    def release(self):
        if self.cap is not None:
            self.cap.release()


class StubSource:
    """全黑假帧。"""

    def __init__(self, w: int, h: int):
        self.path = ""
        self.frame = np.zeros((h, w, 3), dtype=np.uint8)
        self.fps = 30.0

    def read(self):
        return self.frame, None

    def release(self):
        pass


class TelloSource:
    """真机取流（S6 实现）。"""

    def __init__(self):
        raise SystemExit(
            "[tt_driver] source:=tello 尚未实现（S6 实机联调阶段）。\n"
            "  当前可用: source:=stub | source:=file path:=phantom30"
        )


def make_source(rospy_module) -> object:
    gp = rospy_module.get_param
    source = gp("~source", "stub")
    if source == "stub":
        return StubSource(int(gp("~width", 960)), int(gp("~height", 720)))
    if source == "file":
        path = gp("~path", "")
        if not path:
            raise SystemExit("[tt_driver] source:=file 需要 path:=<视频名或路径>")
        return FileSource(_resolve_video(path), bool(gp("~loop", True)))
    if source == "tello":
        return TelloSource()
    raise SystemExit(f"[tt_driver] 未知 source: {source}")


def main():
    rospy.init_node("tt_driver")
    src = make_source(rospy)

    rate_hz = float(rospy.get_param("~rate", 0.0))
    limit = int(rospy.get_param("~limit", 0))

    pub_img = rospy.Publisher("/camera/image_raw", Image, queue_size=2)
    pub_state = rospy.Publisher("/tt/state", Float32MultiArray, queue_size=2)
    rospy.Subscriber("/tt/cmd_vel", Twist, lambda m: None)

    hz = rate_hz if rate_hz > 0 else getattr(src, "fps", 30.0)
    rate = rospy.Rate(hz)
    rospy.loginfo(
        "[tt_driver] source=%s path=%s hz=%.1f loop=%s limit=%s",
        rospy.get_param("~source", "stub"), getattr(src, "path", ""),
        hz, rospy.get_param("~loop", True), limit or "inf",
    )

    import time
    sent = 0
    t0 = time.time()
    while not rospy.is_shutdown():
        frame, err = src.read()
        if err == "eof":
            rospy.loginfo("[tt_driver] 回放结束，共 %d 帧，耗时 %.1fs",
                          sent, time.time() - t0)
            break

        h, w = frame.shape[:2]
        msg = Image()
        msg.header.stamp = rospy.Time.now()       # 墙钟时间，不用视频 PTS
        msg.header.frame_id = "camera"
        msg.header.seq = sent
        msg.height, msg.width = h, w
        msg.encoding = "bgr8"
        msg.is_bigendian = 0
        msg.step = w * 3
        msg.data = frame.tobytes()
        pub_img.publish(msg)
        pub_state.publish(Float32MultiArray(data=[0.0] * 10))

        sent += 1
        if limit and sent >= limit:
            rospy.loginfo("[tt_driver] 到达 limit=%d，停止", limit)
            break
        rate.sleep()

    src.release()
    rospy.loginfo("[tt_driver] 退出，共发布 %d 帧", sent)


if __name__ == "__main__":
    main()
