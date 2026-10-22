"""mgmd.viz — 运行中实时可视化（SPEC tasks/spec-tracking-visualizer.md）。

组件：
- ``FrameView``：一帧的检测记录 + 最终框 + GT + 分类结果（不可变）；
- ``RingBuffer``：显示帧与视图的固定容量滚动缓冲；
- ``Navigator``：播放/暂停/单帧/失败帧扫描状态机；
- ``draw_overlay``：纯函数叠加（不依赖 OpenCV HighGUI，可无头单测）；
- ``CvWindow`` / ``FakeWindow``：窗口适配器（真实 GUI / 测试替身）；
- ``Visualizer``：生命周期、键位/轨迹条、缓冲与统计。

设计约束（SPEC D1–D14）：
- 可视化不参与 ``pipeline`` 的 t_pre/t_search/t_state 计时；
- `--viz` 关闭时零开销；缓冲区只存缩放后的显示帧；
- 检测/GT 坐标按 ``downscale`` 映射回处理尺度，再按显示缩放绘制。
"""
from __future__ import annotations

import os
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Iterable

import cv2
import numpy as np

from spa2d.records import GTFrame, WindowRecord

__all__ = [
    "VizUnavailable",
    "FrameView",
    "RingBuffer",
    "Navigator",
    "OverlayState",
    "classify",
    "draw_overlay",
    "CvWindow",
    "FakeWindow",
    "Visualizer",
]

# 键码（OpenCV HighGUI；Windows 特殊键做归一化）
KEY_ESC = 27
KEY_SPACE = 32
KEY_LEFT, KEY_UP, KEY_RIGHT, KEY_DOWN = 81, 82, 83, 84
KEY_HOME, KEY_END = 80, 87
KEY_A, KEY_D = 97, 100
KEY_E, KEY_H = 101, 104
KEY_N, KEY_P, KEY_Q = 110, 112, 113

_WIN_KEYS = {2424832: KEY_LEFT, 2490368: KEY_UP,
             2555904: KEY_RIGHT, 2621440: KEY_DOWN}

# BGR 颜色（SPEC §5.2）
COLOR_CORRECT = (0, 200, 0)
COLOR_FP = (0, 0, 255)
COLOR_MISS = (0, 165, 255)
COLOR_GT = (0, 255, 255)
COLOR_ABSENT = (160, 160, 160)
COLOR_ARROW = (255, 255, 0)
COLOR_TEXT = (255, 255, 255)
COLOR_PANEL = (24, 24, 24)


class VizUnavailable(RuntimeError):
    """OpenCV HighGUI 不可用（无 DISPLAY / WSLg / Qt）。"""


def _norm_key(key: int) -> int:
    return _WIN_KEYS.get(int(key), int(key))


class _X11WindowProbe:
    """按标题跟踪 X11 window id，用于可靠检测 WM_DELETE 关窗。

    OpenCV 的 ``getWindowProperty`` 在 Qt/WSLg 下对已关闭窗口可能仍返回 1，
    因此用 XQueryTree 检查原始 window id 是否仍存在。
    """

    def __init__(self, title: str) -> None:
        self.title = title
        self.dpy = None
        self.wid: int | None = None
        try:
            import ctypes

            self._ct = ctypes
            x11 = ctypes.CDLL("libX11.so.6")
            x11.XOpenDisplay.restype = ctypes.c_void_p
            x11.XOpenDisplay.argtypes = [ctypes.c_char_p]
            self.dpy = x11.XOpenDisplay(None)
            if not self.dpy:
                return
            self._x11 = x11
            x11.XDefaultRootWindow.restype = ctypes.c_ulong
            x11.XDefaultRootWindow.argtypes = [ctypes.c_void_p]
            x11.XQueryTree.argtypes = [
                ctypes.c_void_p, ctypes.c_ulong,
                ctypes.POINTER(ctypes.c_ulong), ctypes.POINTER(ctypes.c_ulong),
                ctypes.POINTER(ctypes.POINTER(ctypes.c_ulong)),
                ctypes.POINTER(ctypes.c_uint)]
            x11.XQueryTree.restype = ctypes.c_int
            x11.XFree.argtypes = [ctypes.c_void_p]
            x11.XFetchName.argtypes = [
                ctypes.c_void_p, ctypes.c_ulong,
                ctypes.POINTER(ctypes.c_char_p)]
            x11.XFetchName.restype = ctypes.c_int
            x11.XCloseDisplay.argtypes = [ctypes.c_void_p]
            self.root = x11.XDefaultRootWindow(self.dpy)
            self.wid = self._find()
        except Exception:  # noqa: BLE001 - 无 X11 时退回 OpenCV property
            self.dpy = None

    def _children_of(self, window: int) -> list[int]:
        ct, x11 = self._ct, self._x11
        root_ret = ct.c_ulong()
        parent = ct.c_ulong()
        children = ct.POINTER(ct.c_ulong)()
        n = ct.c_uint()
        ok = x11.XQueryTree(self.dpy, int(window), ct.byref(root_ret),
                            ct.byref(parent), ct.byref(children), ct.byref(n))
        if not ok:
            return []
        try:
            return [int(children[i]) for i in range(n.value)]
        finally:
            x11.XFree(children)

    def _children(self) -> list[int]:
        return self._children_of(self.root)

    def _find_in(self, window: int, depth: int = 0) -> int | None:
        """递归查找窗口（WSLg 会把 client 包在 WM frame 下）。"""
        if depth > 8:
            return None
        prefix = self.title.split(" | ")[0]
        for child in self._children_of(window):
            name = self._ct.c_char_p()
            if self._x11.XFetchName(self.dpy, child, self._ct.byref(name)):
                try:
                    text = name.value.decode("utf-8", "replace") if name.value else ""
                finally:
                    self._x11.XFree(name)
                if text == self.title or text.startswith(prefix):
                    return child
            found = self._find_in(child, depth + 1)
            if found is not None:
                return found
        return None

    def _find(self) -> int | None:
        if not self.dpy:
            return None
        return self._find_in(self.root)

    def _contains(self, window: int, target: int, depth: int = 0) -> bool:
        if depth > 8:
            return False
        for child in self._children_of(window):
            if child == target:
                return True
            if self._contains(child, target, depth + 1):
                return True
        return False

    def is_open(self) -> bool | None:
        """True/False=确定；None=无法判断（退回 OpenCV property）。"""
        if not self.dpy:
            return None
        if self.wid is None:
            self.wid = self._find()
            if self.wid is None:
                return None
        return self._contains(self.root, self.wid)

    def close(self) -> None:
        if self.dpy:
            self._x11.XCloseDisplay(self.dpy)
            self.dpy = None


def _display_available() -> bool:
    """检查 DISPLAY 是否可连接（在 cv2.namedWindow 前防 Qt qFatal 崩溃）。"""
    display = os.environ.get("DISPLAY", "")
    if not display:
        return False
    try:
        import ctypes

        x11 = ctypes.CDLL("libX11.so.6")
        x11.XOpenDisplay.restype = ctypes.c_void_p
        x11.XOpenDisplay.argtypes = [ctypes.c_char_p]
        handle = x11.XOpenDisplay(display.encode())
        if not handle:
            return False
        x11.XCloseDisplay.argtypes = [ctypes.c_void_p]
        x11.XCloseDisplay(handle)
        return True
    except Exception:  # noqa: BLE001 - 非 X11 环境按不可用处理
        return False


@dataclass(frozen=True)
class FrameView:
    """一帧可视化视图（不可变）。"""

    index: int
    record: WindowRecord
    rect: tuple[float, float, float, float] | None
    gt: GTFrame | None
    outcome: str  # correct | fp | miss | absent


def classify(record: WindowRecord, gt: GTFrame | None, radius_px: float) -> str:
    """按 spa2d.metrics 口径分类：correct / fp / miss / absent。

    ``radius_px`` 为原图尺度的 ``primary_radius_px``。
    """
    visible = bool(gt is not None and gt.visible == 1
                   and gt.cx is not None and gt.cy is not None)
    present = bool(record.target_present)
    if present:
        est = record.est
        if est is None or not visible:
            return "fp"
        dist = float(np.hypot(est.x - gt.cx, est.y - gt.cy))
        return "correct" if dist <= float(radius_px) else "fp"
    return "miss" if visible else "absent"


class RingBuffer:
    """固定容量的显示帧 + FrameView 滚动缓冲。"""

    def __init__(self, capacity: int) -> None:
        if int(capacity) < 1:
            raise ValueError(f"capacity 必须 ≥ 1，实际 {capacity}")
        self.capacity = int(capacity)
        self._items: deque[tuple[FrameView, np.ndarray]] = deque(maxlen=self.capacity)

    def append(self, view: FrameView, display_frame: np.ndarray) -> None:
        self._items.append((view, display_frame))

    def get(self, index: int) -> tuple[FrameView, np.ndarray] | None:
        for view, frame in self._items:
            if view.index == index:
                return view, frame
        return None

    @property
    def first(self) -> int | None:
        return self._items[0][0].index if self._items else None

    @property
    def last(self) -> int | None:
        return self._items[-1][0].index if self._items else None

    def __len__(self) -> int:
        return len(self._items)


class Navigator:
    """播放/暂停/扫描状态机（SPEC §5.4）。"""

    def __init__(self, scan_max: int = 5000, scan_timeout_s: float = 120.0,
                 clock=time.monotonic) -> None:
        self.scan_max = int(scan_max)
        self.scan_timeout_s = float(scan_timeout_s)
        self._clock = clock
        self.mode = "play"          # play | pause | scan | stop
        self.cursor: int | None = None
        self.jumps = 0
        self.scan_count = 0
        self.scan_exhausted = False
        self._scan_t0 = 0.0

    # ---- keys ----
    def on_key(self, key: int) -> str:
        """处理键并返回动作名。"""
        key = _norm_key(key)
        if key in (KEY_Q, KEY_ESC):
            self.mode = "stop"
            return "quit"
        if key == KEY_SPACE:
            if self.mode == "play":
                self.mode = "pause"
            elif self.mode in ("pause", "scan"):
                self.mode = "play"
                self.cursor = None
            return "toggle"
        if key == KEY_N:
            self.begin_scan()
            return "scan"
        if key == KEY_P:
            return "prev_fail"
        if key == KEY_LEFT:
            return "back1"
        if key == KEY_RIGHT:
            return "forward1"
        if key in (KEY_UP, KEY_A):
            return "back10"
        if key in (KEY_DOWN, KEY_D):
            return "forward10"
        if key in (KEY_HOME, KEY_H):
            return "start"
        if key in (KEY_END, KEY_E):
            return "end"
        return "none"

    # ---- scan / render ----
    def begin_scan(self) -> None:
        self.mode = "scan"
        self.cursor = None
        self.scan_count = 0
        self.scan_exhausted = False
        self._scan_t0 = self._clock()

    def decide(self, view: FrameView, every: int) -> bool:
        """每帧调用：True=继续管线；False=请求停止（q/关窗）。"""
        if self.mode == "stop":
            return False
        if self.mode == "scan":
            self.scan_count += 1
            if view.outcome in ("fp", "miss"):
                self.mode = "pause"
                self.cursor = view.index
                self.jumps += 1
                return True
            if (self.scan_count >= self.scan_max
                    or self._clock() - self._scan_t0 >= self.scan_timeout_s):
                self.mode = "pause"
                self.cursor = None
                self.scan_exhausted = True
            return True
        if self.mode == "play":
            self.cursor = view.index
        return True

    def should_render(self, view: FrameView, every: int) -> bool:
        if self.mode == "pause":
            return True
        if self.mode == "scan":
            return view.outcome in ("fp", "miss")
        if every <= 1:
            return True
        return view.index % int(every) == 0


@dataclass(frozen=True)
class OverlayState:
    mode: str
    fp: int
    miss: int
    total: int
    start: int = 0
    buffer_first: int | None = None
    buffer_last: int | None = None
    status: str = ""


def _put_text(img: np.ndarray, text: str, y: int, x: int = 8,
              color=COLOR_TEXT, scale=0.5) -> None:
    cv2.putText(img, text, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, scale,
                color, 1, cv2.LINE_AA)


def _fmt(value, digits: int = 3) -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def draw_overlay(frame: np.ndarray, view: FrameView, cfg: Any, modules: Iterable[str],
                 state: OverlayState) -> np.ndarray:
    """在显示帧上绘制全部叠加（纯函数，不调用任何 GUI）。"""
    out = np.asarray(frame).copy()
    if out.ndim == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    h, w = out.shape[:2]
    downscale = float(getattr(cfg.runtime, "downscale", 1.0))
    processing_w = float(cfg.sequence.resolution[0]) * downscale
    scale = (w / processing_w) if processing_w > 0 else 1.0
    to_disp = downscale * scale

    rec = view.record
    gt = view.gt
    est = rec.est
    outcome = view.outcome
    color = {"correct": COLOR_CORRECT, "fp": COLOR_FP,
             "miss": COLOR_MISS, "absent": COLOR_ABSENT}.get(outcome, COLOR_ABSENT)

    # 文本面板（先画，避免遮住检测/GT）
    gap = int(getattr(cfg.mfe, "frame_gap", 1))
    or_mode = bool(getattr(cfg.mfe, "or_aggregation", False))
    pair = f"[{view.index - gap},{view.index}]"
    if or_mode:
        pair += f" + [{view.index},{view.index + gap}]"
    conf = rec.confidence
    lines = [
        f"frame={view.index} ({view.index - state.start + 1}/{state.total})",
        f"{rec.state_in}->{rec.state_out}  reason={rec.reason}",
        f"modules={','.join(sorted(modules))}  MFE pair {pair}",
        f"score={_fmt(getattr(conf, 'pnr', None))}  pnr={_fmt(getattr(conf, 'pnr', None))}"
        f"  margin={_fmt(getattr(conf, 'margin', None))}"
        f"  single={getattr(conf, 'single_candidate', '—')}",
        f"outcome={outcome}  gt={'visible' if gt is not None and gt.visible == 1 else 'absent'}",
    ]
    panel_h = 14 + 18 * len(lines)
    panel_w = min(w - 8, 520)
    overlay = out.copy()
    cv2.rectangle(overlay, (4, 4), (4 + panel_w, panel_h), COLOR_PANEL, -1)
    cv2.addWeighted(overlay, 0.55, out, 0.45, 0, out)
    for i, line in enumerate(lines):
        _put_text(out, line, 20 + 18 * i)

    # 检测框 + 中心
    if rec.target_present and est is not None:
        if view.rect is not None:
            x, y, rw, rh = (float(v) * scale for v in view.rect)
            cv2.rectangle(out, (int(round(x)), int(round(y))),
                          (int(round(x + rw)), int(round(y + rh))), color, 2)
        cx, cy = int(round(est.x * to_disp)), int(round(est.y * to_disp))
        cv2.circle(out, (cx, cy), 3, color, -1)
        # 速度箭头（原始尺度 px/frame → 显示 px）
        vx, vy = float(est.vx) * to_disp, float(est.vy) * to_disp
        mag = float(np.hypot(vx, vy))
        if mag > 1e-9:
            length = min(200.0, max(10.0, 10.0 * mag))
            ex = int(round(cx + length * vx / mag))
            ey = int(round(cy + length * vy / mag))
            cv2.arrowedLine(out, (cx, cy), (ex, ey), COLOR_ARROW, 1, tipLength=0.25)

    # GT 点/框
    if gt is not None and gt.visible == 1 and gt.cx is not None and gt.cy is not None:
        gx, gy = int(round(gt.cx * to_disp)), int(round(gt.cy * to_disp))
        cv2.drawMarker(out, (gx, gy), COLOR_GT, cv2.MARKER_CROSS, 10, 1)
        if all(v is not None for v in (gt.bbox_x, gt.bbox_y, gt.bbox_w, gt.bbox_h)):
            bx, by = gt.bbox_x * to_disp, gt.bbox_y * to_disp
            bw, bh = gt.bbox_w * to_disp, gt.bbox_h * to_disp
            cv2.rectangle(out, (int(round(bx)), int(round(by))),
                          (int(round(bx + bw)), int(round(by + bh))), COLOR_GT, 1)

    # 右下状态栏
    status = (f"{state.mode.upper()}  fp={state.fp} miss={state.miss}"
              f"  buffer={state.buffer_first}..{state.buffer_last}")
    if state.status:
        status += f"  | {state.status}"
    tw = cv2.getTextSize(status, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0][0]
    _put_text(out, status, h - 8, x=max(8, w - tw - 8), color=COLOR_TEXT,
              scale=0.45)
    return out


class CvWindow:
    """OpenCV HighGUI 适配器；不可用时抛 ``VizUnavailable``。"""

    def __init__(self, title: str, width: int | None = None, height: int | None = None) -> None:
        self.title = title
        self._probe: _X11WindowProbe | None = None
        if not _display_available():
            raise VizUnavailable(
                "DISPLAY/WSLg 不可用：无法创建 OpenCV 窗口；请确认 WSLg/X server "
                "或去掉 --viz 重跑")
        try:
            cv2.namedWindow(title, cv2.WINDOW_NORMAL)
            if width and height:
                cv2.resizeWindow(title, int(width), int(height))
        except cv2.error as exc:  # pragma: no cover - 依赖真实 GUI
            raise VizUnavailable(
                f"无法创建 OpenCV 窗口（{exc}）；请检查 DISPLAY/WSLg 或使用 --viz 之外的模式"
            ) from exc
        self.closed = False
        self._shown = False
        self._probe = _X11WindowProbe(title)

    def show(self, frame: np.ndarray) -> None:
        cv2.imshow(self.title, frame)
        self._shown = True

    def wait_key(self, delay_ms: int = 1) -> int:
        key = int(cv2.waitKey(int(delay_ms)))
        if self._shown and not self.is_open():
            self.closed = True
        return -1 if key < 0 else key

    def create_trackbar(self, name: str, maxval: int, callback=None) -> None:
        cv2.createTrackbar(name, self.title, 0, int(maxval), callback or (lambda _v: None))

    def get_trackbar(self, name: str) -> int:
        try:
            return int(cv2.getTrackbarPos(name, self.title))
        except cv2.error:
            return 0

    def set_trackbar(self, name: str, pos: int) -> None:
        try:
            cv2.setTrackbarPos(name, self.title, int(pos))
        except cv2.error:
            pass

    def is_open(self) -> bool:
        state = self._probe.is_open() if self._probe is not None else None
        if state is not None:
            return state
        try:
            return bool(cv2.getWindowProperty(self.title, cv2.WND_PROP_VISIBLE) >= 1)
        except cv2.error:
            return False

    def close(self) -> None:
        self.closed = True
        if self._probe is not None:
            self._probe.close()
            self._probe = None
        try:
            cv2.destroyWindow(self.title)
            cv2.waitKey(1)
        except cv2.error:
            pass


class FakeWindow:
    """无 GUI 测试替身：队列键 + 记录显示帧 + 内存 trackbar。"""

    def __init__(self, keys: Iterable[int] | None = None) -> None:
        self._keys: deque[int] = deque(keys or [])
        self.frames: list[np.ndarray] = []
        self.trackbars: dict[str, int] = {}
        self.closed = False

    def show(self, frame: np.ndarray) -> None:
        self.frames.append(frame.copy())

    def wait_key(self, delay_ms: int = 1) -> int:
        if self._keys:
            return int(self._keys.popleft())
        if delay_ms >= 20:          # 暂停循环空转：自动关闭，避免测试挂起
            self.closed = True
        return -1

    def create_trackbar(self, name: str, maxval: int, callback=None) -> None:
        self.trackbars.setdefault(name, 0)

    def get_trackbar(self, name: str) -> int:
        return int(self.trackbars.get(name, 0))

    def set_trackbar(self, name: str, pos: int) -> None:
        self.trackbars[name] = int(pos)

    def is_open(self) -> bool:
        return not self.closed

    def close(self) -> None:
        self.closed = True


class Visualizer:
    """实时可视化主类；作为 ``pipeline.run(on_frame=...)`` 回调。"""

    def __init__(self, cfg: Any, modules: Iterable[str], gt_frames: list[GTFrame],
                 total: int, start: int = 0, every: int = 1, buffer: int = 200,
                 width: int = 1280, scan_max: int = 5000,
                 window: CvWindow | FakeWindow | None = None,
                 clock=time.monotonic) -> None:
        self.cfg = cfg
        self.modules = list(modules)
        self.gt_frames = list(gt_frames)
        self.total = int(total)
        self.start = int(start)
        self.every = max(1, int(every))
        self.width = int(width)
        self.buffer = RingBuffer(buffer)
        self.nav = Navigator(scan_max=scan_max, clock=clock)
        self.fp = 0
        self.miss = 0
        self.viewed = 0
        self.abort_frame: int | None = None
        self.status = ""
        self.closed = False
        self._tb_name = "frame"
        self._last_tb = 0
        self._seek_request: int | None = None
        if window is None:
            window = CvWindow(
                f"MGMD {getattr(cfg, 'config_name', '')} | {','.join(sorted(modules))}",
                width=min(self.width, 1600), height=900)
        self.window = window
        try:
            self.window.create_trackbar(self._tb_name, max(9, buffer - 1),
                                        self._on_trackbar)
        except Exception:  # noqa: BLE001 - 某些后端不支持 trackbar
            pass

    # ---- helpers ----
    def _gt_at(self, index: int) -> GTFrame | None:
        return self.gt_frames[index] if 0 <= index < len(self.gt_frames) else None

    def _to_display(self, frame: np.ndarray | None) -> np.ndarray:
        if frame is None:
            if len(self.buffer):
                return self.buffer.get(self.buffer.last)[1].copy()  # type: ignore[index]
            return np.zeros((1, 1, 3), dtype=np.uint8)
        target_w = min(self.width, frame.shape[1])
        if target_w == frame.shape[1]:
            return frame
        target_h = max(1, int(round(frame.shape[0] * target_w / frame.shape[1])))
        return cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)

    def _state(self) -> OverlayState:
        return OverlayState(
            mode=self.nav.mode, fp=self.fp, miss=self.miss, total=self.total,
            start=self.start, buffer_first=self.buffer.first,
            buffer_last=self.buffer.last, status=self.status)

    def _latest(self) -> FrameView | None:
        item = self.buffer.get(self.buffer.last) if self.buffer.last is not None else None
        return item[0] if item else None

    def _current(self) -> FrameView | None:
        if self.nav.cursor is not None:
            item = self.buffer.get(self.nav.cursor)
            if item is not None:
                return item[0]
        return self._latest()

    def _render(self, view: FrameView) -> None:
        item = self.buffer.get(view.index)
        if item is None:
            return
        out = draw_overlay(item[1], view, self.cfg, self.modules, self._state())
        self.window.show(out)
        first = self.buffer.first
        if first is not None:
            self._last_tb = max(0, int(view.index) - int(first))
            self.window.set_trackbar(self._tb_name, self._last_tb)

    def _on_trackbar(self, pos: int) -> None:
        """OpenCV trackbar 回调：记录拖动请求（不直接改状态，避免 GUI 线程竞态）。"""
        if self.buffer.first is None:
            return
        if int(pos) != self._last_tb:
            self._seek_request = int(pos)

    def _apply_seek(self) -> bool:
        """消费 trackbar 拖动请求；播放态拖动自动暂停。"""
        if self._seek_request is None or self.buffer.first is None:
            return False
        target = int(self.buffer.first) + int(self._seek_request)
        self._seek_request = None
        item = self.buffer.get(target)
        if item is None or self.nav.mode not in ("play", "pause"):
            return False
        self.nav.cursor = target
        if self.nav.mode == "play":
            self.nav.mode = "pause"
        return True

    def _poll_trackbar(self) -> None:
        first = self.buffer.first
        if first is None:
            return
        pos = self.window.get_trackbar(self._tb_name)
        if pos != self._last_tb:
            target = int(first) + int(pos)
            if self.buffer.get(target) is not None:
                self.nav.cursor = target
                self._last_tb = pos

    def _move(self, delta: int) -> None:
        first, last = self.buffer.first, self.buffer.last
        if first is None or last is None:
            return
        base = self.nav.cursor if self.nav.cursor is not None else last
        target = min(int(last), max(int(first), int(base) + int(delta)))
        if target == base:
            self.status = "buffer boundary"
        else:
            self.status = ""
        self.nav.cursor = target

    def _prev_fail(self) -> None:
        first, last = self.buffer.first, self.buffer.last
        if first is None or last is None:
            return
        start = self.nav.cursor if self.nav.cursor is not None else last
        for idx in range(int(start) - 1, int(first) - 1, -1):
            item = self.buffer.get(idx)
            if item is not None and item[0].outcome in ("fp", "miss"):
                self.nav.cursor = idx
                self.status = ""
                return
        self.status = "no previous fp/miss in buffer"

    def _pause_loop(self) -> bool:
        while self.nav.mode == "pause":
            view = self._current()
            if view is None:
                break
            if getattr(self.window, "closed", False):
                self.abort_frame = view.index
                return False
            opener = getattr(self.window, "is_open", None)
            if callable(opener) and not opener():
                self.abort_frame = view.index
                return False
            self._render(view)
            self._apply_seek()
            key = self.window.wait_key(30)
            if getattr(self.window, "closed", False):
                self.abort_frame = view.index
                return False
            if key == -1:
                if getattr(self.window, "closed", False):
                    self.abort_frame = view.index
                    return False
                self._poll_trackbar()
                continue
            action = self.nav.on_key(key)
            if action == "quit":
                self.abort_frame = view.index
                return False
            if action == "toggle":
                return True
            if action == "scan":
                return True
            if action == "back1":
                self._move(-1)
            elif action == "forward1":
                self._move(1)
            elif action == "back10":
                self._move(-10)
            elif action == "forward10":
                self._move(10)
            elif action == "start" and self.buffer.first is not None:
                self.nav.cursor = self.buffer.first
            elif action == "end" and self.buffer.last is not None:
                self.nav.cursor = self.buffer.last
            elif action == "prev_fail":
                self._prev_fail()
            self._poll_trackbar()
        return True

    # ---- public ----
    def update(self, index: int, frame: np.ndarray | None, record: WindowRecord,
               rect: tuple[float, float, float, float] | None) -> bool:
        if self.closed:
            return False
        gt = self._gt_at(int(record.assoc_frame))
        view = FrameView(int(index), record, tuple(rect) if rect is not None else None,
                         gt, classify(record, gt, self.cfg.metrics.primary_radius_px))
        self.buffer.append(view, self._to_display(frame))
        self.viewed += 1
        if view.outcome == "fp":
            self.fp += 1
        elif view.outcome == "miss":
            self.miss += 1

        key = self.window.wait_key(1)
        if getattr(self.window, "closed", False):
            self.abort_frame = view.index
            return False
        if key != -1:
            action = self.nav.on_key(key)
            if action == "quit":
                self.abort_frame = view.index
                return False
            if action == "prev_fail":
                self.nav.cursor = view.index      # 当前实时帧；从它的前一帧开始找
                self.nav.mode = "pause"
                self._prev_fail()
                return self._pause_loop()
            if self.nav.mode == "pause":
                return self._pause_loop()

        if self._apply_seek():
            if self.nav.mode == "pause":
                return self._pause_loop()

        if self.nav.mode == "scan":
            if not self.nav.decide(view, self.every):
                self.abort_frame = view.index
                return False
            if self.nav.mode == "pause":
                return self._pause_loop()
            return True
        if self.nav.mode == "pause":
            return self._pause_loop()

        self.nav.decide(view, self.every)
        if self.nav.should_render(view, self.every):
            self._render(view)
        return True

    def summary(self) -> dict:
        return {
            "enabled": True,
            "every": self.every,
            "buffer": self.buffer.capacity,
            "width": self.width,
            "viewed": self.viewed,
            "jumps": self.nav.jumps,
            "fp": self.fp,
            "miss": self.miss,
            "aborted": self.abort_frame is not None,
            "abort_frame": self.abort_frame,
        }

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            self.window.close()
