#!/usr/bin/env python3
"""video_source — 视频源节点（SPEC-ROS §17.2 / S3.5）。

只负责"把视频帧发出去"，**不管飞控**（本项目只做视频上的目标跟踪）。

三种 source 模式：
    source:=stub    全黑假帧（话题连通性测试）
    source:=file    回放本地视频（主要用法）
    source:=camera  本地摄像头（cv2.VideoCapture(索引)）

参数：
    source:=file path:=<视频名或路径> [rate:=30] [loop:=true] [limit:=0]
    source:=camera device:=0

    视频名可用 ARD-MAV 的短名（如 phantom30），也可给绝对路径。
    rate  目标发布频率（Hz）；0 表示按视频原始帧率尽快发布。
    width/height  输出分辨率（0 = 保持原始）。非 0 时用 INTER_AREA 缩放。
    loop  回放结束后是否循环。循环时帧号 (header.seq) 会从 0 重新开始。
    limit 最多发布的帧数（0 = 不限）。

**帧号定义（§17.4）**：`header.seq` = 视频内的帧序号（0 起，逐帧 +1）。
下游 detector_node 用它作为 MFE 的帧号，所以必须与视频严格对应。

**时间戳**：`header.stamp` 用墙钟时间（与真实相机一致），便于算端到端延迟。
"""
from pathlib import Path

import rospy
from sensor_msgs.msg import Image

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
        rospy.logwarn(f"[video_source] 无法导入 mgmd.config: {exc}")
    raise SystemExit(f"[video_source] 找不到视频: {name}")


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
            raise SystemExit(f"[video_source] 打不开视频: {self.path}")
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


class CameraSource:
    """本地摄像头。"""

    def __init__(self, index: int):
        import cv2
        self.path = f"camera:{index}"
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise SystemExit(f"[video_source] 打不开摄像头 {index}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0

    def read(self):
        ok, frame = self.cap.read()
        return (frame, None) if ok else (None, "eof")

    def release(self):
        self.cap.release()


def make_source(rospy_module) -> object:
    gp = rospy_module.get_param
    source = gp("~source", "stub")
    if source == "stub":
        return StubSource(int(gp("~width", 960)), int(gp("~height", 720)))
    if source == "file":
        path = gp("~path", "")
        if not path:
            raise SystemExit("[video_source] source:=file 需要 path:=<视频名或路径>")
        return FileSource(_resolve_video(path), bool(gp("~loop", True)))
    if source == "camera":
        return CameraSource(int(gp("~device", 0)))
    raise SystemExit(f"[video_source] 未知 source: {source}")


def main():
    rospy.init_node("video_source")
    src = make_source(rospy)

    rate_hz = float(rospy.get_param("~rate", 0.0))
    limit = int(rospy.get_param("~limit", 0))
    loop = bool(rospy.get_param("~loop", True))
    out_w = int(rospy.get_param("~width", 0))
    out_h = int(rospy.get_param("~height", 0))

    pub_img = rospy.Publisher("/camera/image_raw", Image, queue_size=2)

    hz = rate_hz if rate_hz > 0 else getattr(src, "fps", 30.0)
    rate = rospy.Rate(hz)
    rospy.loginfo(
        "[video_source] source=%s path=%s hz=%.1f loop=%s limit=%s res=%s",
        rospy.get_param("~source", "file"), getattr(src, "path", ""),
        hz, loop, limit or "inf",
        f"{out_w}x{out_h}" if out_w > 0 else "原始",
    )

    import time
    sent = 0
    t0 = time.time()
    while not rospy.is_shutdown():
        frame, err = src.read()
        if err == "eof":
            rospy.loginfo("[video_source] 回放结束，共 %d 帧，耗时 %.1fs",
                          sent, time.time() - t0)
            break

        # 分辨率缩放（0 = 原始）。INTER_AREA 缩小时质量最好（避免摩尔纹）
        if out_w > 0 and out_h > 0:
            import cv2
            frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
        h, w = frame.shape[:2]
        msg = Image()
        msg.header.stamp = rospy.Time.now()       # 墙钟时间，不用视频 PTS
        msg.header.frame_id = "camera"
        # 文件循环时 FileSource.seq 会归零；累计发送数 sent 只用于 limit/日志。
        # 其余源没有视频内帧号，使用累计发送数。
        msg.header.seq = src.seq - 1 if isinstance(src, FileSource) else sent
        msg.height, msg.width = h, w
        msg.encoding = "bgr8"
        msg.is_bigendian = 0
        msg.step = w * 3
        msg.data = frame.tobytes()
        pub_img.publish(msg)

        sent += 1
        if limit and sent >= limit:
            rospy.loginfo("[video_source] 到达 limit=%d，停止", limit)
            break
        rate.sleep()

    src.release()
    rospy.loginfo("[video_source] 退出，共发布 %d 帧", sent)


if __name__ == "__main__":
    main()
