"""tests/test_mgmd_appearance.py — LAC + LAD 外观模块（SPEC §5.5，US-005）。

合成张量/mock 验证接口形状与数值范围，不依赖 GPU；torch 缺失时整体 skip（FR-11）。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="torch 未装：外观用例 skip（FR-11）")

from mgmd.config import LadConfig, LacConfig, MgmdError   # noqa: E402
from mgmd.classify import LacClassifier, resolve_device   # noqa: E402
from mgmd.detect import LadDetector, RefinedBox, crop_region, yolov5_dir  # noqa: E402

WEIGHTS_LAC = Path("data_raw/mgmd_weights/Net_best.pth")
WEIGHTS_LAD = Path("data_raw/mgmd_weights/yolov5s_GLAD-crop.pt")


def _yolov5_src_ok() -> bool:
    return (yolov5_dir() / "models" / "experimental.py").is_file()


requires_yolov5 = pytest.mark.skipif(
    not _yolov5_src_ok(),
    reason="yolov5 v6.0 源码缺失：bash scripts/fetch_yolov5_src.sh（FR-11 skip）")


def _lac_cfg(device: str = "cpu", weights: str = str(WEIGHTS_LAC)) -> LacConfig:
    return LacConfig(weights=weights, input_size=32, prob_thresh=0.5, device=device)


def _lad_cfg(device: str = "cpu", weights: str = str(WEIGHTS_LAD)) -> LadConfig:
    return LadConfig(weights=weights, crop_size=320, conf_thresh=0.5, imgsz=320,
                     device=device)


# ---------------- crop_region 数学（不依赖 torch 推理路径） ----------------

def test_crop_region_center():
    assert crop_region((1000, 1920), (960, 540), 320) == (800, 380, 320, 320)


def test_crop_region_border_shift_clamp():
    # 中心贴左上 → 平移 clamp，窗口仍 320×320
    assert crop_region((1080, 1920), (10, 20), 320) == (0, 0, 320, 320)
    # 中心贴右下
    assert crop_region((1080, 1920), (1919, 1079), 320) == (1600, 760, 320, 320)


def test_crop_region_small_frame_takes_whole():
    assert crop_region((200, 300), (150, 100), 320) == (0, 0, 300, 200)


# ---------------- LAC ----------------

def test_resolve_device():
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("cuda") == "cuda"
    assert resolve_device("auto") in ("cpu", "cuda")


@pytest.mark.skipif(not WEIGHTS_LAC.is_file(), reason="权重未下载：bash scripts/fetch_mgmd_weights.sh")
def test_lac_real_weights_prob_range():
    """真实 Net_best.pth：输出概率 ∈[0,1]，结构逐层对齐官方 Net。"""
    clf = LacClassifier(_lac_cfg())
    rng = np.random.default_rng(0)
    crop = rng.integers(0, 256, size=(64, 48, 3), dtype=np.uint8)  # 非 32×32，走 resize
    p = clf.mav_prob(crop)
    assert isinstance(p, float) and 0.0 <= p <= 1.0
    crop32 = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)
    assert 0.0 <= clf.mav_prob(crop32) <= 1.0


@pytest.mark.skipif(not WEIGHTS_LAC.is_file(), reason="权重未下载")
def test_lac_net_architecture_matches_official():
    """state_dict 键集合 == 官方 Net（conv1/conv2/fc1/fc2/fc3）。"""
    state = torch.load(WEIGHTS_LAC, map_location="cpu", weights_only=True)
    keys = set(state.keys())
    assert keys == {"conv1.weight", "conv1.bias", "conv2.weight", "conv2.bias",
                    "fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias",
                    "fc3.weight", "fc3.bias"}
    assert tuple(state["fc3.weight"].shape) == (2, 84)


def test_lac_missing_weights_exit_3():
    with pytest.raises(MgmdError) as ei:
        LacClassifier(_lac_cfg(weights="/nonexistent/Net_best.pth"))
    assert ei.value.exit_code == 3
    assert "fetch_mgmd_weights.sh" in str(ei.value)


# ---------------- LAD ----------------

@requires_yolov5
def test_lad_missing_weights_exit_3():
    with pytest.raises(MgmdError) as ei:
        LadDetector(_lad_cfg(weights="/nonexistent/yolov5s_GLAD-crop.pt"))
    assert ei.value.exit_code == 3
    assert "fetch_mgmd_weights.sh" in str(ei.value)


class _FakeYolo(torch.nn.Module):
    """合成推理：固定输出两个候选框（yolov5 v6.0 张量格式 [1,N,6]）。"""

    stride = torch.tensor([32.0])

    def __init__(self):
        super().__init__()
        self._p = torch.nn.Parameter(torch.zeros(1))

    def forward(self, img):
        # yolov5 NMS 输入格式 [cx, cy, w, h, obj_conf, cls_conf]：
        # 框A 中心 (160,160) 40×40 conf 0.9；框B conf 0.3（低于阈值被滤）
        pred = torch.tensor([[[160.0, 160.0, 40.0, 40.0, 0.90, 1.00],
                              [25.0, 25.0, 30.0, 30.0, 0.30, 1.00]]])
        return pred.to(img.device)


def _fake_detector() -> LadDetector:
    det = LadDetector.__new__(LadDetector)
    det.cfg = _lad_cfg()
    det._model = _FakeYolo()
    det._stride = 32
    return det


@requires_yolov5
def test_lad_refine_mock_nearest_center_and_mapping():
    det = _fake_detector()
    frame = np.zeros((1080, 1920, 3), np.uint8)
    out = det.refine(frame, (960.0, 540.0), downscale=1.0)
    assert isinstance(out, RefinedBox)
    # crop 原点 (800,380)；框A xyxy(140,140,180,180) → xywh(940,520,40,40)
    assert out.rect == pytest.approx((940.0, 520.0, 40.0, 40.0), abs=1.0)
    assert out.conf == pytest.approx(0.90, abs=1e-3)


@requires_yolov5
def test_lad_refine_mock_downscale_mapping():
    det = _fake_detector()
    frame = np.zeros((540, 960, 3), np.uint8)   # downscale 0.5 后的处理尺度
    out = det.refine(frame, (480.0, 270.0), downscale=0.5)
    # crop 原点 (320,110)；框A → ((140+320)/0.5, (140+110)/0.5, 40/0.5, 40/0.5)
    assert out.rect == pytest.approx((920.0, 500.0, 80.0, 80.0), abs=2.0)


@requires_yolov5
def test_lad_refine_no_detection():
    class _Empty(_FakeYolo):
        def forward(self, img):
            return torch.zeros((1, 0, 6)).to(img.device)

    det = _fake_detector()
    det._model = _Empty()
    frame = np.zeros((1080, 1920, 3), np.uint8)
    assert det.refine(frame, (960.0, 540.0)) is None


@requires_yolov5
@pytest.mark.skipif(not WEIGHTS_LAD.is_file(), reason="权重未下载")
def test_lad_real_weights_load():
    """真实 yolov5s_GLAD-crop.pt 加载（SPEC D9 weights_only=False 封装）+ 推理形状。"""
    det = LadDetector(_lad_cfg())
    assert det._stride >= 8
    frame = np.zeros((1080, 1920, 3), np.uint8)
    out = det.refine(frame, (960.0, 540.0))     # 全黑帧：无检测或合法框
    assert out is None or (0.0 <= out.conf <= 1.0 and out.rect[2] > 0)
