"""mgmd.detect — LAD 局部精细检测器（SPEC §5.5）。

候选中心 320×320 裁剪（边界不足时平移窗口 clamp），yolov5 v6.0 加载
yolov5s_GLAD-crop.pt，conf>conf_thresh(0.5)，多框取离裁剪中心最近者（对齐官方
detector3.detect(region, x_center, y_center) 语义），坐标加回偏移并 ÷downscale
映射原图。

torch/yolov5 懒加载（FR-11）；torch≥2.6 默认 weights_only=True 与 v6.0 旧权重冲突，
本模块封装 torch.load(..., weights_only=False)（SPEC D9）——仅限 sha256 校验通过
的官方权重（SPEC §7）。权重缺失 → E_WEIGHTS_MISSING（退出码 3）。

yolov5 源码（SPEC §1.3 D3 修正）：v6.0 tag 无打包（无 setup.py），pip 不可装；
且权重 pickle 引用顶层 ``models.*``/``utils.*`` 模块路径（PyPI yolov5 7.x 的
``yolov5.models.*`` 不兼容）。故用 codeload tarball 落地 ``third_party/yolov5-v6.0/``
（scripts/fetch_yolov5_src.sh，sha256 预 pin），加载时 sys.path 注入其根目录。
可用环境变量 ``MGMD_YOLOV5_DIR`` 覆盖源码目录。
"""
from __future__ import annotations

import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mgmd.config import LadConfig, MgmdError
from mgmd.classify import resolve_device

_YOLOV5_DEFAULT_DIR = Path(__file__).resolve().parents[1] / "third_party" / "yolov5-v6.0"


def yolov5_dir() -> Path:
    """yolov5 v6.0 源码目录（MGMD_YOLOV5_DIR 覆盖，默认 third_party/yolov5-v6.0）。"""
    return Path(os.environ.get("MGMD_YOLOV5_DIR", str(_YOLOV5_DEFAULT_DIR)))


def _require_lad_deps() -> Path:
    try:
        import torch  # noqa: F401
    except ImportError as exc:
        raise MgmdError("E_DEP_MISSING",
                        f"LAD 依赖缺失: {exc.name}",
                        ".venv/bin/pip install -r env/requirements.txt（torch 组）",
                        4) from exc
    src = yolov5_dir()
    if not (src / "models" / "experimental.py").is_file():
        raise MgmdError("E_DEP_MISSING",
                        f"LAD 依赖缺失: yolov5 v6.0 源码目录不完整: {src}",
                        "bash scripts/fetch_yolov5_src.sh 获取（codeload tarball，sha256 预 pin）",
                        4)
    return src


def _inject_yolov5_syspath(src: Path) -> None:
    """把 yolov5 v6.0 仓库根注入 sys.path（顶层 models/utils 模块路径，unpickle 需要）。"""
    s = str(src)
    if s not in sys.path:
        sys.path.insert(0, s)


@dataclass
class RefinedBox:
    rect: tuple[float, float, float, float]  # (x, y, w, h)，原图尺度坐标
    conf: float


def crop_region(shape: tuple[int, int], center_xy: tuple[float, float],
                crop_size: int) -> tuple[int, int, int, int]:
    """以 center 为中心裁剪 crop_size×crop_size，边界平移 clamp（SPEC §5.5）。

    shape=(H, W)。帧尺寸不足 crop_size 方向取整幅（refine 内 letterbox 处理，
    SPEC §5.8：MVDT 不触发）。
    """
    h, w = shape
    cx, cy = center_xy
    size_w, size_h = min(crop_size, w), min(crop_size, h)
    x1 = int(round(cx - size_w / 2.0))
    y1 = int(round(cy - size_h / 2.0))
    x1 = max(0, min(x1, w - size_w))
    y1 = max(0, min(y1, h - size_h))
    return int(x1), int(y1), int(size_w), int(size_h)


class LadDetector:
    """LAD：320×320 裁剪区 YOLOv5s-crop 推理，输出精炼框。"""

    def __init__(self, cfg: LadConfig) -> None:
        src = _require_lad_deps()
        if not Path(cfg.weights).is_file():
            raise MgmdError("E_WEIGHTS_MISSING",
                            f"LAD 权重不存在: {cfg.weights}",
                            "bash scripts/fetch_mgmd_weights.sh 下载并校验官方权重",
                            3)
        _inject_yolov5_syspath(src)
        self.cfg = cfg
        self._model = self._load_model(cfg)
        stride = self._model.stride
        if hasattr(stride, "detach"):            # torch tensor（可能在 cuda 上）
            stride = stride.detach().cpu().numpy()
        self._stride = int(np.max(np.asarray(stride)))

    @staticmethod
    def _load_model(cfg: LadConfig):
        """attempt_load 等价物（SPEC D9：显式 weights_only=False，仅限已校验权重）。"""
        import torch

        _inject_yolov5_syspath(yolov5_dir())
        device = torch.device(resolve_device(cfg.device))
        ckpt = torch.load(cfg.weights, map_location=device, weights_only=False)
        model = ckpt["ema" if ckpt.get("ema") else "model"].float()
        model.fuse().eval()
        model.to(device)
        return model

    def _infer(self, img_tensor):
        """模型前向；v6.0 返回 pred 或 (pred, ...) 元组。"""
        import torch

        with torch.no_grad():
            pred = self._model(img_tensor)
        if isinstance(pred, (list, tuple)):
            pred = pred[0]
        return pred

    def _preprocess(self, crop_bgr: np.ndarray):
        """BGR crop → letterbox(imgsz) → CHW RGB float /255 batch 张量。"""
        import torch
        _inject_yolov5_syspath(yolov5_dir())
        from utils.augmentations import letterbox      # yolov5 v6.0 顶层模块路径

        img = letterbox(crop_bgr, self.cfg.imgsz, stride=self._stride, auto=True)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)     # BGR→RGB, HWC→CHW
        img = np.ascontiguousarray(img, dtype=np.float32) / 255.0
        device = next(self._model.parameters()).device
        return torch.from_numpy(img).unsqueeze(0).to(device)

    def refine(self, frame_bgr: np.ndarray, center_xy: tuple[float, float],
               downscale: float = 1.0) -> RefinedBox | None:
        """候选中心裁剪推理；无达标检测返回 None。坐标映射回原图尺度。"""
        _inject_yolov5_syspath(yolov5_dir())
        from utils.general import non_max_suppression, scale_coords  # yolov5 v6.0 顶层模块路径

        h, w = frame_bgr.shape[:2]
        x1, y1, cw, ch = crop_region((h, w), center_xy, self.cfg.crop_size)
        if cw < self.cfg.crop_size or ch < self.cfg.crop_size:
            warnings.warn(f"[mgmd] 帧 {w}x{h} 小于 crop_size={self.cfg.crop_size}，"
                          "letterbox 上采样（SPEC §5.8）", UserWarning, stacklevel=2)
        crop = frame_bgr[y1:y1 + ch, x1:x1 + cw]
        img = self._preprocess(crop)
        pred = self._infer(img)
        det = non_max_suppression(pred, self.cfg.conf_thresh, 0.45)[0]
        if det is None or len(det) == 0:
            return None
        det = det.detach().cpu().numpy()
        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], crop.shape).round()
        # 离裁剪中心最近者（官方 detector3 最近中心语义）
        ccx, ccy = cw / 2.0, ch / 2.0
        centers = np.stack([(det[:, 0] + det[:, 2]) / 2.0,
                            (det[:, 1] + det[:, 3]) / 2.0], axis=1)
        k = int(np.argmin(np.hypot(centers[:, 0] - ccx, centers[:, 1] - ccy)))
        bx1, by1, bx2, by2, conf = (float(v) for v in det[k, :5])
        s = 1.0 / downscale
        return RefinedBox(rect=((bx1 + x1) * s, (by1 + y1) * s,
                                (bx2 - bx1) * s, (by2 - by1) * s),
                          conf=conf)
