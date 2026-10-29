"""mgmd_onnx — ONNX Runtime 版 LAC/LAD，接口与 torch 版完全一致（SPEC-ROS §13）。

为什么需要这一层：
    阶段一的 `mgmd/lac.py` 与 `mgmd/lad.py` 依赖 **torch 2.5 + Python 3.11**，
    而 ROS1 Noetic 绑死 **Python 3.8**（只能用 torch 2.4）。
    为避免双解释器（sidecar），把两个模型的**推理**换成 ONNX Runtime，
    其余逻辑（预处理、NMS、坐标反算）原样复用阶段一代码。

约束（SPEC-ROS V7）：**不得修改 `mgmd/`**。本包只做包装与替换。

用法：
    from mgmd_onnx import LacOnnx, LadOnnx
    lac = LacOnnx(device="cuda")          # 接口同 mgmd.lac.LacClassifier
    lad = LadOnnx(device="cuda")          # 接口同 mgmd.lad.LadDetector

    lac.mav_prob_batch(crops)  -> list[float]
    lad.refine(frame, center)  -> RefinedBox | None
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from mgmd import config as C

__all__ = ["LacOnnx", "LadOnnx", "ort_providers", "make_session"]


def ort_providers(device: str) -> list[str]:
    """按 device 选 provider；cuda 不可用时自动退回 CPU。"""
    import onnxruntime as ort
    avail = ort.get_available_providers()
    if device == "cuda" and "CUDAExecutionProvider" in avail:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def make_session(path: Path, device: str, *, enable_trt: bool = False):
    """建 ONNX 会话。TRT 默认关闭（engine 与版本强绑定，换机要重导）。"""
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.log_severity_level = 3                      # 只报 error，屏蔽 provider 噪声
    provs = ort_providers(device)
    if enable_trt and "TensorrtExecutionProvider" in ort.get_available_providers():
        provs = ["TensorrtExecutionProvider"] + provs
    if not Path(path).is_file():
        raise C.MgmdError("E_WEIGHTS_MISSING", f"ONNX 模型不存在: {path}",
                          "运行 scripts/export_onnx.py 生成",
                          C.EXIT_DATA)
    return ort.InferenceSession(str(path), so, providers=provs)


class LacOnnx:
    """ONNX 版 LAC。接口对齐 `mgmd.lac.LacClassifier`。"""

    def __init__(self, model: Path | None = None, input_size: int = 32,
                 device: str = "cpu"):
        self.model_path = Path(model) if model else (C.WEIGHTS_DIR / "lac.onnx")
        self.input_size = input_size
        self.device = device
        self.session = make_session(self.model_path, device)
        self._input_name = self.session.get_inputs()[0].name

    def _prep(self, crops: list[np.ndarray]) -> np.ndarray:
        """与 `LacClassifier.mav_prob_batch` 逐步对齐：BGR→resize→RGB→CHW/255。

        关键：`[::-1]` 是负步长视图，必须 `.copy()` 再 transpose，
        否则 `astype` 会读错数据（SPEC-ROS §13.5 踩坑 1）。
        """
        import cv2
        s = self.input_size
        arrs = []
        for c in crops:
            img = np.ascontiguousarray(c)
            if img.shape[0] != s or img.shape[1] != s:
                img = cv2.resize(img, (s, s))
            arrs.append(img[:, :, ::-1].copy())
        batch = np.stack(arrs)
        return np.ascontiguousarray(batch.transpose(0, 3, 1, 2)).astype(np.float32) / 255.0

    def mav_prob(self, crop_bgr: np.ndarray) -> float:
        return self.mav_prob_batch([crop_bgr])[0]

    def mav_prob_batch(self, crops: list[np.ndarray]) -> list[float]:
        if not crops:
            return []
        x = self._prep(list(crops))
        logits = self.session.run(None, {self._input_name: x})[0]
        logits = logits.astype(np.float64)
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = e / e.sum(axis=1, keepdims=True)
        return [float(p) for p in probs[:, 1]]


class LadOnnx:
    """ONNX 版 LAD。接口对齐 `mgmd.lad.LadDetector`。

    **只把 `model(x)` 换成 ONNX 推理**，NMS 与 scale_coords 仍用 yolov5 utils
    （ONNX 输出是原始预测 `(1,6300,6)`，不含 NMS，SPEC-ROS §13.2）。
    """

    def __init__(self, model: Path | None = None, crop_size: int = 320,
                 conf_thresh: float = 0.5, device: str = "cpu"):
        import torch
        from mgmd.lad import yolov5_dir, _inject_syspath
        _inject_syspath(yolov5_dir())

        self.model_path = Path(model) if model else (C.WEIGHTS_DIR / "yolov5s_GLAD-crop.onnx")
        self.crop_size = crop_size
        self.conf_thresh = conf_thresh
        self.device = device
        self.session = make_session(self.model_path, device)
        self._input_name = self.session.get_inputs()[0].name
        self._torch = torch
        # stride 用于 letterbox；YOLOv5s 320 输入固定 32
        self._stride = 32
        self.nc = 1

    def _preprocess(self, crop_bgr: np.ndarray) -> np.ndarray:
        """与 `LadDetector._preprocess` 完全一致，区别只是返回 numpy 而非 tensor。"""
        from utils.augmentations import letterbox
        img = letterbox(crop_bgr, (self.crop_size, self.crop_size),
                        stride=self._stride, auto=True)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1).copy()
        return np.ascontiguousarray(img, dtype=np.float32) / 255.0

    def infer(self, crop_bgr: np.ndarray) -> np.ndarray:
        """原始 ONNX 输出 (1,6300,6)，供测试对拍用。"""
        x = self._preprocess(crop_bgr)[None]
        return self.session.run(None, {self._input_name: x})[0]

    def refine(self, frame_bgr: np.ndarray, center):
        """候选中心裁剪推理，返回最佳精修框（原图坐标）。逻辑同 LadDetector.refine。"""
        import torch
        from utils.general import non_max_suppression, scale_coords
        from mgmd.lad import RefinedBox, crop_region

        h, w = frame_bgr.shape[:2]
        x1, y1, cw, ch = crop_region((h, w), center, self.crop_size)
        crop = frame_bgr[y1:y1 + ch, x1:x1 + cw]
        if crop.size == 0:
            return None

        x = self._preprocess(crop)[None]
        raw = self.session.run(None, {self._input_name: x})[0]
        det = non_max_suppression(torch.from_numpy(raw), self.conf_thresh, 0.45)[0]
        if det is None or len(det) == 0:
            return None

        det = det.detach().cpu().numpy()
        det[:, :4] = scale_coords(
            (self.crop_size, self.crop_size),
            torch.from_numpy(det[:, :4].copy()), crop.shape).numpy().round()
        # 取离裁剪中心最近的框（官方 detector3 语义）
        ccx, ccy = cw / 2.0, ch / 2.0
        cxs = (det[:, 0] + det[:, 2]) / 2.0
        cys = (det[:, 1] + det[:, 3]) / 2.0
        k = int(np.argmin(np.hypot(cxs - ccx, cys - ccy)))
        bx1, by1, bx2, by2, conf = (float(v) for v in det[k, :5])
        return RefinedBox(box=(bx1 + x1, by1 + y1, bx2 + x1, by2 + y1), conf=conf)

    def refine_batch(self, frame_bgr: np.ndarray, centers, batch_size: int = 8):
        """逐框调用 refine（ONNX 侧暂不做真批处理，接口保持一致）。"""
        return [self.refine(frame_bgr, c) for c in centers]
