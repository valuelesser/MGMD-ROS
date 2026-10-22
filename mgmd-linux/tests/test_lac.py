"""tests.test_lac — LAC 验收（SPEC T07）。"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from mgmd.lac import LacClassifier, build_net, crop_box


def test_net_structure_forward_shape():
    net = build_net()
    out = net(torch.randn(2, 3, 32, 32))
    assert out.shape == (2, 2)


def test_net_param_count():
    """官方 Net 参数量 61,326（PRD 记录值）。"""
    net = build_net()
    n = sum(p.numel() for p in net.parameters())
    assert n == 61326, f"参数量应为 61326，实际 {n}"


def test_weight_loading_matches_official():
    """官方权重可加载且形状匹配。"""
    clf = LacClassifier(device="cpu")
    assert clf.weights.is_file()


def test_mav_prob_in_range():
    clf = LacClassifier(device="cpu")
    img = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
    p = clf.mav_prob(img)
    assert 0.0 <= p <= 1.0


def test_batch_matches_single():
    """批量推理与单张结果应一致。"""
    clf = LacClassifier(device="cpu")
    rng = np.random.RandomState(0)
    crops = [rng.randint(0, 255, (32, 32, 3), dtype=np.uint8) for _ in range(3)]
    batch = clf.mav_prob_batch(crops)
    single = [clf.mav_prob(c) for c in crops]
    for a, b in zip(batch, single):
        assert abs(a - b) < 1e-5, f"批/单不一致: {a} vs {b}"


def test_crop_box_clamps_to_image():
    img = np.zeros((100, 100, 3), np.uint8)
    c = crop_box(img, (-10, -10, 20, 20))
    assert c is not None
    assert c.shape[0] <= 32 and c.shape[1] <= 32


def test_crop_box_invalid_returns_none():
    img = np.zeros((100, 100, 3), np.uint8)
    assert crop_box(img, (200, 200, 210, 210)) is None


def test_empty_batch():
    clf = LacClassifier(device="cpu")
    assert clf.mav_prob_batch([]) == []
