#!/usr/bin/env python3
"""visualize_node — 把检测框画到视频上，输出带框视频文件（SPEC-ROS §17.3）。

**只做画框和写视频，不参与检测**。分开的原因：
    - 画框是纯 I/O 工作，放在检测线程里会拖慢算法
    - 关掉这个节点，检测性能不受影响（验收用）

订阅：
    /camera/image_raw   原图
    /mav/detections     检测结果（按 header.seq 与图像配对）

发布：
    /mav/debug_image    叠加后的图（可选，供 rqt_image_view 实时看）

写文件：
    ~out 指定的 mp4 路径（默认 outputs/ros_viz/<视频名>.mp4）

参数：
    ~out          输出视频路径；空字符串则不写文件
    ~fps          输出视频帧率，默认 30
    ~encoder      nvenc|cpu   默认 nvenc（不可用时自动回落 cpu，见 §18）
    ~show_tid     是否显示 track_id，默认 true
    ~show_src     是否显示来源模块（lad/lac），默认 false
    ~draw_history 是否画轨迹历史，默认 true
    ~history_len  轨迹历史长度，默认 30

配对策略：
    图像的 header.seq 是视频帧号。检测不一定每帧都有（算法 ~14 Hz，视频 30 Hz），
    所以用"最近一次检测"贴到当前帧上。这样视频 30 fps 全帧都有框，观感连续。
"""
import os
import time
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np
import rospy
from sensor_msgs.msg import Image

from mav_tracker_msgs.msg import DetectionArray

try:
    from nvenc_writer import NvencWriter
except ImportError:                       # 脚本目录不在 sys.path 时
    import sys as _sys
    from pathlib import Path as _P
    _sys.path.insert(0, str(_P(__file__).resolve().parent))
    from nvenc_writer import NvencWriter

# 每个 track_id 用固定颜色，便于肉眼追踪
_COLORS = [
    (56, 56, 255), (151, 157, 255), (31, 112, 255), (29, 178, 255),
    (49, 210, 207), (10, 249, 72), (23, 204, 146), (134, 219, 61),
    (52, 147, 26), (187, 212, 0), (168, 153, 44), (255, 194, 0),
    (147, 69, 52), (255, 115, 100), (236, 24, 0), (255, 56, 132),
    (133, 0, 82), (255, 56, 203), (200, 149, 255), (199, 55, 255),
]


def color_for(tid: int):
    return _COLORS[tid % len(_COLORS)] if tid >= 0 else (200, 200, 200)


def draw_detections(frame: np.ndarray, dets, show_tid: bool, show_src: bool,
                    history: dict, history_len: int, info: str) -> np.ndarray:
    out = frame
    for d in dets:
        x1, y1, x2, y2 = int(d.x1), int(d.y1), int(d.x2), int(d.y2)
        c = color_for(d.track_id)
        cv2.rectangle(out, (x1, y1), (x2, y2), c, 2)

        # 标签：conf 和可选 track_id / source
        label = f"{d.confidence:.2f}"
        if show_tid and d.track_id >= 0:
            label = f"#{d.track_id} {label}"
        if show_src:
            label = f"{d.source} {label}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), c, -1)
        cv2.putText(out, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # 轨迹历史：画中心点连线
        if d.track_id >= 0:
            hist = history[d.track_id]
            hist.append((int(d.cx), int(d.cy)))
            while len(hist) > history_len:
                hist.popleft()
            pts = list(hist)
            for i in range(1, len(pts)):
                cv2.line(out, pts[i - 1], pts[i], c, 1, cv2.LINE_AA)

    # HUD
    if info:
        lines = info.splitlines()
        sizes = [cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
                 for line in lines]
        line_h = max(th for _, th in sizes) + 7
        cv2.rectangle(out, (8, 8),
                      (16 + max(tw for tw, _ in sizes), 12 + line_h * len(lines)),
                      (0, 0, 0), -1)
        for i, line in enumerate(lines):
            cv2.putText(out, line, (12, 8 + line_h * (i + 1)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
    return out


def main():
    rospy.init_node("visualize_node")

    gp = rospy.get_param
    out_path = gp("~out", "outputs/ros_viz/ros_track.mp4")
    fps = float(gp("~fps", 30.0))
    show_tid = bool(gp("~show_tid", True))
    show_src = bool(gp("~show_src", False))
    draw_history = bool(gp("~draw_history", True))
    history_len = int(gp("~history_len", 30))
    publish = bool(gp("~publish", True))

    encoder_mode = gp("~encoder", "nvenc")
    writer = None
    lock_frame = {"w": None, "h": None}
    latest = {"dets": [], "seq": -1, "n_cand": 0, "mfe": 0.0, "lac": 0.0, "lad": 0.0}
    det_times = deque(maxlen=60)
    history = defaultdict(lambda: deque(maxlen=history_len))
    n_written = 0
    n_images = 0

    pub = rospy.Publisher("/mav/debug_image", Image, queue_size=1) if publish else None

    def on_det(msg: DetectionArray):
        det_times.append(time.monotonic())
        latest["dets"] = msg.detections
        latest["seq"] = msg.header.seq
        latest["n_cand"] = msg.n_candidates
        latest["mfe"] = msg.mfe_ms
        latest["lac"] = msg.lac_ms
        latest["lad"] = msg.lad_ms
        if not draw_history:
            history.clear()

    def ensure_writer(w, h):
        nonlocal writer
        if writer is not None or not out_path:
            return
        force_cpu = (encoder_mode == "cpu")
        try:
            writer = NvencWriter(out_path, w, h, fps=fps, force_cpu=force_cpu)
            rospy.loginfo("[visualize_node] 写视频 %s  %dx%d @%.0ffps  encoder=%s",
                          out_path, w, h, fps, writer.backend)
        except Exception as exc:                       # noqa: BLE001
            rospy.logerr("[visualize_node] 无法创建视频文件: %s (%s)", out_path, exc)
            writer = None

    def on_image(msg: Image):
        nonlocal n_written, n_images
        n_images += 1
        arr = np.frombuffer(msg.data, dtype=np.uint8)
        if arr.size != msg.height * msg.width * 3:
            return
        frame = arr.reshape((msg.height, msg.width, 3))

        dets = latest["dets"]
        now = time.monotonic()
        while det_times and now - det_times[0] > 2.0:
            det_times.popleft()
        processing_fps = (len(det_times) - 1) / (det_times[-1] - det_times[0]) \
            if len(det_times) >= 2 and det_times[-1] > det_times[0] else 0.0
        info = ""
        if latest["seq"] >= 0:
            info = (f"Processing {processing_fps:.1f} FPS\n"
                    f"frame {msg.header.seq}  det {len(dets)}  "
                    f"cand {latest['n_cand']}  "
                    f"MFE {latest['mfe']:.0f} LAC {latest['lac']:.1f} "
                    f"LAD {latest['lad']:.1f} ms")

        out = draw_detections(frame.copy(), dets, show_tid, show_src,
                              history, history_len, info)

        ensure_writer(out.shape[1], out.shape[0])
        if writer is not None:
            writer.write(out)
            n_written += 1

        if pub is not None:
            m = Image()
            m.header = msg.header
            m.height, m.width = out.shape[:2]
            m.encoding = "bgr8"
            m.is_bigendian = 0
            m.step = out.shape[1] * 3
            m.data = out.tobytes()
            pub.publish(m)

        if n_written % 100 == 0 and n_written > 0:
            rospy.loginfo("[visualize_node] 已写 %d 帧（收到 %d 张图）",
                          n_written, n_images)

    rospy.Subscriber("/camera/image_raw", Image, on_image, queue_size=1,
                     buff_size=2 * 1920 * 1080 * 3 + 1024)
    rospy.Subscriber("/mav/detections", DetectionArray, on_det, queue_size=4)

    _released = {"done": False}

    def _shutdown(*_a):
        # 幂等：on_shutdown 与 signal handler 可能都触发
        if _released["done"]:
            return
        _released["done"] = True
        if writer is not None:
            writer.close()
            rospy.loginfo("[visualize_node] 视频已保存: %s（%d 帧）",
                          out_path, n_written)

    rospy.on_shutdown(_shutdown)

    # rospy.on_shutdown 只在 ros::shutdown 时触发；直接 kill (SIGTERM/SIGINT)
    # 不会走到那里，VideoWriter 没 release 会导致 mp4 的 moov atom 缺失、文件损坏。
    # 所以额外挂 signal handler。
    import signal

    def _sig(signum, _frame):
        rospy.loginfo("[visualize_node] 收到信号 %d，释放视频写入器", signum)
        _shutdown()
        raise SystemExit(0)

    for _s in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(_s, _sig)
        except ValueError:
            pass        # 非主线程时无法注册，忽略

    rospy.loginfo("[visualize_node] 就绪，输出 %s", out_path or "（不写文件）")
    rospy.spin()
    _shutdown()


if __name__ == "__main__":
    main()
