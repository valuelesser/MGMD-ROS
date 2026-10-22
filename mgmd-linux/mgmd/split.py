"""mgmd.split — 固化官方 45/15 划分（SPEC §4.2）。

测试集 ID 取自 GLAD README（唯一可核验来源），训练/验证按视频 5:1 随机、seed 固定。
文件一旦生成不可覆盖（除非 --force），防止历史结果不可比。
"""
from __future__ import annotations

import json
import random

from mgmd import config as C


def build_split(seed: int = 0, val_ratio: float = 1 / 6) -> dict:
    """构造划分。返回 dict，不写盘。"""
    all_videos = C.list_annotated_videos()
    ids = {C.name_to_video_id(v) for v in all_videos}
    test_ids = set(C.TEST_IDS)

    missing = test_ids - ids
    if missing:
        raise C.MgmdError(
            "E_DATA_MISSING",
            f"测试集视频缺失: {sorted(missing)}",
            "确认 60 视频集完整（SPEC §4.1）", C.EXIT_DATA,
        )

    train_pool = sorted(ids - test_ids)
    rng = random.Random(seed)
    shuffled = train_pool[:]
    rng.shuffle(shuffled)
    n_val = max(1, round(len(shuffled) * val_ratio))
    val_ids = sorted(shuffled[:n_val])
    train_ids = sorted(shuffled[n_val:])

    assert not (set(train_ids) & set(val_ids)), "训练/验证交集非空"
    assert not ((set(train_ids) | set(val_ids)) & test_ids), "训练/验证与测试交集非空"

    return {
        "spec_version": "0.1.0",
        "source": "GLAD README (https://github.com/WindyLab/Global-Local-MAV-Detection)",
        "note": "论文正文写 50/10，README 写 45/15；采用 README（偏差 D-01）",
        "seed": seed,
        "val_ratio": round(val_ratio, 6),
        "test_ids": sorted(test_ids),
        "val_ids": val_ids,
        "train_ids": train_ids,
        "condition_groups": {k: list(v) for k, v in C.CONDITION_GROUPS.items()},
        "counts": {
            "train": len(train_ids),
            "val": len(val_ids),
            "test": len(test_ids),
            "total": len(ids),
        },
    }


def verify_split(split: dict) -> list[str]:
    """校验划分自洽性，返回问题列表（空=通过）。"""
    problems = []
    tr, va, te = set(split["train_ids"]), set(split["val_ids"]), set(split["test_ids"])
    if tr & va:
        problems.append(f"训练/验证交集: {sorted(tr & va)}")
    if tr & te:
        problems.append(f"训练/测试交集: {sorted(tr & te)}")
    if va & te:
        problems.append(f"验证/测试交集: {sorted(va & te)}")
    if te != set(C.TEST_IDS):
        problems.append(f"测试集与官方 ID 不一致: {sorted(te ^ set(C.TEST_IDS))}")

    expected = {C.name_to_video_id(v) for v in C.list_annotated_videos()}
    union = tr | va | te
    if union != expected:
        problems.append(f"并集不等于全集: 缺 {sorted(expected - union)}, 多 {sorted(union - expected)}")

    for group, gids in C.CONDITION_GROUPS.items():
        if not set(gids) <= te:
            problems.append(f"条件组 {group} 不在测试集内: {sorted(set(gids) - te)}")
    return problems


def write_split(split: dict, force: bool = False) -> None:
    C.SPLIT_FILE.parent.mkdir(parents=True, exist_ok=True)
    if C.SPLIT_FILE.exists() and not force:
        raise C.MgmdError(
            "E_CHECK",
            f"划分文件已存在: {C.SPLIT_FILE}",
            "如需重建，显式传 --force（会作废历史对比）", C.EXIT_CHECK,
        )
    C.SPLIT_FILE.write_text(
        json.dumps(split, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
