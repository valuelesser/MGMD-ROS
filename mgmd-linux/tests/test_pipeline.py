"""tests.test_pipeline — 串联与模块子集（SPEC T06/F1）。"""
from __future__ import annotations

import pytest

from mgmd.pipeline import parse_modules


def test_parse_modules_valid_combinations():
    """Table 3 的 6 种组合应都能解析。"""
    for combo in ("mfe", "mfe,tf", "mfe,lac", "mfe,tf,lac",
                  "mfe,lad", "mfe,lac,lad", "mfe,tf,lac,lad"):
        mods = parse_modules(combo)
        assert "mfe" in mods


def test_parse_modules_rejects_unknown():
    with pytest.raises(Exception):
        parse_modules("mfe,bogus")


def test_parse_modules_requires_mfe():
    with pytest.raises(Exception):
        parse_modules("tf,lac")


def test_parse_modules_handles_spaces():
    mods = parse_modules(" mfe , tf ")
    assert mods == frozenset({"mfe", "tf"})


def test_parse_modules_empty():
    with pytest.raises(Exception):
        parse_modules("")
