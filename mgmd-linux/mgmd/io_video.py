"""mgmd.io_video — 视频读取（SPEC §4.1）。

常驻 cv2.VideoCapture 顺序读帧（不要每帧 open/seek，慢且易错）。
帧号 1 基，与标注一致。
"""
from __future__ import annotations

from pathlib import Path

import cv2

from mgmd import config as C


class VideoReader:
    """顺序读帧器。frame_index 为 1 基帧号。"""

    def __init__(self, video: str):
        path = C.ARD_VIDEOS / f"{video}.mp4"
        if not path.is_file():
            raise C.MgmdError("E_DATA_MISSING", f"视频不存在: {path}",
                              "确认 60 视频集完整（SPEC §4.1）", C.EXIT_DATA)
        self.path = path
        self.video = video
        self._cap = cv2.VideoCapture(str(path))
        if not self._cap.isOpened():
            raise C.MgmdError("E_RUNTIME", f"无法打开视频: {path}",
                              "检查视频完整性与解码器", C.EXIT_RUNTIME)
        self.fps = float(self._cap.get(cv2.CAP_PROP_FPS) or 0.0)
        self.n_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    def read(self):
        """读下一帧，返回 (frame_index, bgr) 或 (None, None) 到末尾。"""
        ok, frame = self._cap.read()
        if not ok:
            return None, None
        idx = int(self._cap.get(cv2.CAP_PROP_POS_FRAMES))
        return idx, frame

    def seek(self, frame_index: int) -> None:
        """跳到指定 1 基帧号（下次 read 返回该帧）。"""
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index - 1)

    def iter_frames(self, start: int = 1, limit: int | None = None):
        """从 start（1 基）迭代至多 limit 帧，产出 (idx, frame)。"""
        self.seek(start)
        n = 0
        while True:
            if limit is not None and n >= limit:
                return
            idx, frame = self.read()
            if idx is None:
                return
            yield idx, frame
            n += 1

    def close(self) -> None:
        self._cap.release()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()
