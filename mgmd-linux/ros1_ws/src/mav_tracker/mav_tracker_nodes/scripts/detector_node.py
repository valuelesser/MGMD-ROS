#!/usr/bin/env python3
"""detector_node — 单进程内跑 MGMD 全链路，输出检测（SPEC-ROS §17 / S4）。

架构（三节点方案，§17.2）:
    video_source ──► /camera/image_raw ──► detector_node ──► /mav/detections
                                                          └─► visualize_node

**帧序号定义（§17.4，本次修正）**
    帧序号 = **视频帧号**，取自 `msg.header.seq`（video_source 发的原始帧号）。
    不是"收到的第几个帧"。

    为什么必须这样：
      - MFE 的三帧差分依赖**真实时间间隔**，用接收计数在丢帧时会错位
      - `/mav/detections.header.seq` 必须能回溯到视频的哪一帧，否则 S5 的
        rosbag 对拍（中心差 < 2px）无从谈起

**帧缓冲（6 帧，§17.5）**
    MFE 需要 (t-k, t, t+k)。帧缓冲固定 6 帧，凑满即处理最早的可处理帧。
    不再用"收到计数 - k"猜中心帧（那是旧版的 bug，三帧永远凑不齐）。

**丢帧策略**
    相机 30 Hz、算法 ~14 Hz。单槽位覆盖旧帧，保证延迟有上界、不堆积。

**参数**
    ~device       auto|cuda|cpu        默认 auto
    ~d_min        0.02                 与阶段一最终口径一致
    ~lac_thresh   0.5
    ~lad_conf     0.5
    ~mfe_preset   recommended|revised|official
    ~fast_tf      true                 使用向量化 FastSortTracker
    ~buffer_size  6                    MFE 帧缓冲深度
"""
import threading
import time
from collections import OrderedDict, deque

import numpy as np
import rospy
from sensor_msgs.msg import Image

from mav_tracker_msgs.msg import Detection, DetectionArray


class LatestFrameSlot:
    """单槽位：写覆盖旧值，读取走。用于解耦 30 Hz 输入与 ~14 Hz 处理。

    若用有界队列，算法跟不上时队列会单调增长，延迟无限变大。
    覆盖式槽位保证：处理完永远取最新帧，延迟有上界。
    """

    def __init__(self):
        self._cond = threading.Condition()
        self._item = None
        self.recv = 0
        self.dropped = 0

    def put(self, item):
        with self._cond:
            if self._item is not None:
                self.dropped += 1        # 旧帧被覆盖 = 丢弃
            self._item = item
            self.recv += 1
            self._cond.notify()

    def get(self, timeout=0.5):
        with self._cond:
            if self._item is None:
                self._cond.wait(timeout)
            item, self._item = self._item, None
            return item


def imgmsg_to_bgr(msg: Image) -> np.ndarray:
    """sensor_msgs/Image(bgr8) → numpy HxWx3。"""
    buf = np.frombuffer(msg.data, dtype=np.uint8)
    expected = msg.height * msg.width * 3
    if buf.size != expected:
        raise ValueError(f"图像数据长度 {buf.size} != {expected}")
    return buf.reshape((msg.height, msg.width, 3))


def build_detection_array(header, w, h, boxes, confs, sources,
                          n_cand, t_mfe, t_tf, t_lac, t_lad,
                          track_ids=None) -> DetectionArray:
    out = DetectionArray()
    out.header = header
    out.image_width = w
    out.image_height = h
    out.n_candidates = n_cand
    out.mfe_ms = float(t_mfe)
    out.tf_ms = float(t_tf)
    out.lac_ms = float(t_lac)
    out.lad_ms = float(t_lad)
    for i, (b, c) in enumerate(zip(boxes, confs)):
        d = Detection()
        d.header = header
        d.x1, d.y1, d.x2, d.y2 = (float(v) for v in b)
        d.cx = (d.x1 + d.x2) / 2.0
        d.cy = (d.y1 + d.y2) / 2.0
        d.confidence = float(c)
        d.source = sources[i] if i < len(sources) else "lad"
        d.track_id = int(track_ids[i]) if track_ids and i < len(track_ids) else -1
        out.detections.append(d)
    return out


class Pipeline:
    """持有跨帧状态：MFE 帧缓冲 + SORT 轨迹。

    帧缓冲以**视频帧号**为 key。凑满 `buffer_size` 帧后取最早可处理的一帧。
    """

    def __init__(self, device: str, cfg, d_min: float, lac_thresh: float,
                 lad_conf: float, fast_tf: bool, buffer_size: int = 6):
        from mgmd.lac import crop_box
        from mgmd_onnx import LacOnnx, LadOnnx

        self._crop_box = crop_box
        self.cfg = cfg
        self.k = cfg.k
        self.device = device
        self.d_min = d_min
        self.lac_thresh = lac_thresh
        self.lad_conf = lad_conf
        self.buffer_size = max(buffer_size, 2 * self.k + 1)

        if fast_tf:
            from mgmd_onnx.tf_fast import FastSortTracker
            self.tracker = FastSortTracker(iou_thresh=0.3, max_age=3, min_hits=1,
                                           span=3, d_mode="self", d_min=d_min)
            rospy.loginfo("[detector_node] tracker = FastSortTracker（向量化，§14.7）")
        else:
            from mgmd.tf import SortTracker
            self.tracker = SortTracker(iou_thresh=0.3, max_age=3, min_hits=1,
                                       span=3, d_mode="self", d_min=d_min)
            rospy.loginfo("[detector_node] tracker = SortTracker（基线）")

        self.lac = LacOnnx(device=device)
        self.lad = LadOnnx(conf_thresh=lad_conf, device=device)

        # 视频帧号 -> (header, BGR)。用 OrderedDict 保证按帧号顺序淘汰
        self.frames: "OrderedDict[int, tuple]" = OrderedDict()
        self.header_of = {}
        self.n_processed = 0
        self.overhead: dict = {}
        self.last_pub = 0.0
        self.n_waiting = 0         # 因缓冲不足而等待的次数
        self.last_done = -1        # 最近一次处理的视频帧号（防止重复处理）
        self.n_dropped_evict = 0   # 因缓冲淘汰而未处理的帧数
        rospy.loginfo(
            "[detector_node] LAC provider=%s  LAD provider=%s  buffer=%d  k=%d",
            self.lac.session.get_providers()[0], self.lad.session.get_providers()[0],
            self.buffer_size, self.k)

    # ---------- 帧缓冲 ----------

    def push(self, frame_id: int, header, arr: np.ndarray) -> None:
        """存入一帧。同一帧号重复推送则覆盖（循环回放时会重复）。"""
        if self.frames and frame_id <= next(reversed(self.frames)):
            # 帧号回绕（循环视频）→ 清空缓冲重新开始，避免 key 冲突
            self.frames.clear()
            self.header_of.clear()
        self.frames[frame_id] = arr
        self.header_of[frame_id] = header
        while len(self.frames) > self.buffer_size:
            old, _ = self.frames.popitem(last=False)
            self.header_of.pop(old, None)

    def ready_frame_id(self):
        """返回缓冲中**下一个可处理**的帧号（要求 t-k 与 t+k 都在）。

        关键：必须跳过已经处理过的帧，否则会重复处理同一帧。
        按帧号升序找第一个 > last_done 且前驱后继都在的帧。
        """
        if len(self.frames) < 2 * self.k + 1:
            return None
        for t in list(self.frames.keys()):
            if t <= self.last_done:
                continue
            if (t - self.k) in self.frames and (t + self.k) in self.frames:
                return t
        return None

    # ---------- 单帧处理 ----------

    def step(self, t: int):
        """处理视频帧 t。返回结果字典或 None。"""
        from mgmd.mfe import enhance

        if (t - self.k) not in self.frames or (t + self.k) not in self.frames:
            return None

        cur = self.frames[t]
        header = self.header_of[t]
        t0 = time.perf_counter()
        res = enhance(self.frames[t - self.k], cur, self.frames[t + self.k], self.cfg, t)
        t_mfe = (time.perf_counter() - t0) * 1000.0
        n_cand = len(res.boxes)

        pool = [((float(x), float(y), float(x + w), float(y + h)), float(w * h))
                for (x, y, w, h) in res.boxes]

        # ---- TF。SORT 用 _miss 计数判超龄，**不依赖帧号绝对值**（§16.7/§19）：
        # 实测 stride=4 下传视频帧号(2,6,10..) 与处理序号(1,2,3..) 输出为
        # 8.36 vs 8.47 个/帧，差异在噪声内。所以传哪个都可以，这里传视频帧号。
        t0 = time.perf_counter()
        tracks = self.tracker.update([b for b, _ in pool], t,
                                     scores=[s for _, s in pool])
        t_tf = (time.perf_counter() - t0) * 1000.0
        pool = [(tr.box, tr.score, tr.track_id) for tr in tracks]

        # ---- LAC
        t0 = time.perf_counter()
        if pool:
            crops, keep = [], []
            for b, s, tid in pool:
                c = self._crop_box(cur, b)
                if c is not None:
                    crops.append(c)
                    keep.append((b, s, tid))
            if crops:
                probs = self.lac.mav_prob_batch(crops)
                pool = [(b, p, tid) for (b, _s, tid), p in zip(keep, probs)
                        if p >= self.lac_thresh]
            else:
                pool = []
        t_lac = (time.perf_counter() - t0) * 1000.0

        # ---- LAD
        t0 = time.perf_counter()
        out_boxes, out_conf, out_src, out_tid = [], [], [], []
        for b, _p, tid in pool:
            rb = self.lad.refine(cur, ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2))
            if rb is not None and rb.conf >= self.lad_conf:
                out_boxes.append(rb.box)
                out_conf.append(rb.conf)
                out_src.append("lad")
                out_tid.append(tid)
        t_lad = (time.perf_counter() - t0) * 1000.0

        self.n_processed += 1
        self.last_done = t
        return dict(frame=t, header=header, boxes=out_boxes, confs=out_conf,
                    sources=out_src, track_ids=out_tid, n_cand=n_cand,
                    shape=cur.shape[:2],
                    t_mfe=t_mfe, t_tf=t_tf, t_lac=t_lac, t_lad=t_lad)

    def evict_below(self, t: int) -> None:
        """处理完 t 后，淘汰 t-k 及更早的帧（再也不会用到）。"""
        cutoff = t - self.k
        for key in [k for k in self.frames if k <= cutoff]:
            if key > self.last_done:
                self.n_dropped_evict += 1     # 这帧还没处理就被淘汰了
            del self.frames[key]
            self.header_of.pop(key, None)


def main():
    rospy.init_node("detector_node")

    gp = rospy.get_param
    device = gp("~device", "auto")
    d_min = float(gp("~d_min", 0.02))
    lac_thresh = float(gp("~lac_thresh", 0.5))
    lad_conf = float(gp("~lad_conf", 0.5))
    mfe_preset = gp("~mfe_preset", "recommended")
    fast_tf = bool(gp("~fast_tf", True))
    buffer_size = int(gp("~buffer_size", 6))

    import sys
    from pathlib import Path
    _root = Path(__file__).resolve().parents[5]
    if (_root / "mgmd").is_dir() and str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

    from mgmd.mfe import MFEConfig
    cfg = MFEConfig()
    fn = getattr(cfg, f"apply_{mfe_preset}", None)
    if fn is None:
        valid = [n[6:] for n in dir(cfg) if n.startswith("apply_")]
        raise SystemExit(f"[detector_node] 未知 mfe_preset={mfe_preset}，可选 {valid}")
    fn()
    cfg.validate()
    rospy.loginfo("[detector_node] MFE preset=%s k=%d mode=%s",
                  mfe_preset, cfg.k, cfg.mode)

    if device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:            # noqa: BLE001
            device = "cpu"

    pipe = Pipeline(device, cfg, d_min, lac_thresh, lad_conf, fast_tf, buffer_size)

    pub_det = rospy.Publisher("/mav/detections", DetectionArray, queue_size=2)

    slot = LatestFrameSlot()
    stop = threading.Event()

    _n_img = {"n": 0}

    def on_image(msg: Image):
        _n_img["n"] += 1
        if _n_img["n"] % 100 == 1:
            rospy.loginfo("[detector_node] 收到第 %d 张图 seq=%d %dx%d",
                          _n_img["n"], msg.header.seq, msg.width, msg.height)
        t_cb = time.perf_counter()
        try:
            arr = imgmsg_to_bgr(msg)
        except ValueError as exc:
            rospy.logwarn_throttle(5.0, "[detector_node] 图像解析失败: %s", exc)
            return
        t_dec = time.perf_counter()
        # **帧号 = 视频帧号**（video_source 发的 header.seq），不是接收计数
        slot.put((int(msg.header.seq), msg.header, arr, t_cb, t_dec))

    def worker():
        rospy.loginfo("[detector_node] 工作线程启动 device=%s d_min=%.3f", device, d_min)
        n = 0
        hz_win = deque(maxlen=50)
        t_start = time.time()
        while not stop.is_set() and not rospy.is_shutdown():
            item = slot.get(timeout=0.2)
            t_pop = time.perf_counter()
            if item is None:
                continue
            frame_id, header, arr, t_cb, t_dec = item
            pipe.push(frame_id, header, arr)

            # 缓冲里可能有不止一帧可处理（处理慢时会积压），一次处理到追平
            while True:
                t = pipe.ready_frame_id()
                if t is None:
                    pipe.n_waiting += 1
                    break
                t_step0 = time.perf_counter()
                try:
                    r = pipe.step(t)
                except Exception as exc:                     # noqa: BLE001
                    rospy.logerr("[detector_node] 处理异常 (frame %d): %s", t, exc)
                    pipe.evict_below(t)
                    break
                t_step1 = time.perf_counter()
                pipe.evict_below(t)
                if r is None:
                    break

                h, w = r["shape"]
                msg = build_detection_array(
                    r["header"], w, h, r["boxes"], r["confs"], r["sources"],
                    r["n_cand"], r["t_mfe"], r["t_tf"], r["t_lac"],
                    r["t_lad"], r["track_ids"])
                t_build = time.perf_counter()
                pub_det.publish(msg)
                t_pub = time.perf_counter()

                # ---- 端到端拆解（SPEC-ROS §20.7）----
                o = pipe.overhead
                # 回调被 rospy 调用的时刻 - 上一帧发布完成的时刻
                # = ROS 传输 + 线程交接 + 队列等待（这是我们怀疑的大头）
                if pipe.last_pub:
                    gap = (t_cb - pipe.last_pub) * 1000.0
                    if gap >= 0:
                        o["frame_gap"] = o.get("frame_gap", 0.0) + gap
                o["decode"] = o.get("decode", 0.0) + (t_dec - t_cb) * 1000.0
                o["queue_to_work"] = o.get("queue_to_work", 0.0) + (t_pop - t_dec) * 1000.0
                o["step"] = o.get("step", 0.0) + (t_step1 - t_step0) * 1000.0
                o["build"] = o.get("build", 0.0) + (t_build - t_step1) * 1000.0
                o["publish"] = o.get("publish", 0.0) + (t_pub - t_build) * 1000.0
                o["n"] = o.get("n", 0) + 1
                pipe.last_pub = t_pub

                n += 1
                hz_win.append(time.time())
                if n <= 5:
                    rospy.loginfo(
                        "[detector_node] 分段明细#%d: MFE %.1f LAC %.1f LAD %.1f "
                        "候选 %d 检测 %d",
                        n, r["t_mfe"], r["t_lac"], r["t_lad"],
                        r["n_cand"], len(r["boxes"]))
                if n % 50 == 0:
                    hz = 0.0
                    if len(hz_win) >= 2:
                        hz = (len(hz_win) - 1) / max(1e-6, hz_win[-1] - hz_win[0])
                    if pipe.overhead.get("n"):
                        k = float(pipe.overhead["n"])
                        parts = " ".join(
                            f"{key} {pipe.overhead.get(key, 0.0) / k:.2f}"
                            for key in ("frame_gap", "decode",
                                        "queue_to_work", "step", "build",
                                        "publish"))
                        rospy.loginfo("[detector_node] 端到端拆解/帧: %s (ms)", parts)
                        pipe.overhead = {}
                    rospy.loginfo(
                        "[detector_node] 已处理 %d 帧  瞬时 %.2f Hz  收到 %d "
                        "入口丢 %d  缓冲淘汰 %d  等待 %d  frame=%d  候选 %d  "
                        "检测 %d  MFE %.1f ms",
                        n, hz, slot.recv, slot.dropped, pipe.n_dropped_evict,
                        pipe.n_waiting, r["frame"], r["n_cand"],
                        len(r["boxes"]), r["t_mfe"])
        rospy.loginfo("[detector_node] 工作线程退出，共处理 %d 帧", n)

    th = threading.Thread(target=worker, name="det_worker", daemon=True)
    th.start()

    rospy.Subscriber("/camera/image_raw", Image, on_image, queue_size=1,
                     buff_size=2 * 1920 * 1080 * 3 + 1024)
    rospy.on_shutdown(stop.set)
    rospy.loginfo("[detector_node] 就绪，等待 /camera/image_raw")
    rospy.spin()
    stop.set()
    th.join(timeout=2.0)


if __name__ == "__main__":
    main()
