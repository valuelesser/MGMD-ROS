"""pytest 共享 fixtures（SPEC §9.1）。"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import yaml

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


@pytest.fixture
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture
def configs_dir(repo_root: Path) -> Path:
    return repo_root / "configs"


@pytest.fixture
def write_yaml(tmp_path: Path):
    def _write(doc, name: str) -> Path:
        path = tmp_path / name
        path.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
        return path

    return _write


@pytest.fixture
def small_bgr_image() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(24, 32, 3), dtype=np.uint8)


@pytest.fixture
def small_gray_float() -> np.ndarray:
    rng = np.random.default_rng(1)
    return rng.random((24, 32), dtype=np.float32)
