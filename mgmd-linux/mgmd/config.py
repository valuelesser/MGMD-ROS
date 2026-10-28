"""mgmd.config — 路径与常量唯一入口（SPEC §3/§4）。

所有其它模块只从这里取路径，禁止硬编码。数据源只读，产物写工作区。
"""
from __future__ import annotations

import json
from pathlib import Path

# ---------- 路径 ----------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 数据集根（NTFS 只读）。SPEC §4.1
# 注意：卷标会随挂载变化（0C1CFCAF1CFC94C4 -> 0C1CFCAF1CFC94C41，SPEC §10.7）。
# 因此按候选顺序探测，取第一个存在的卷，避免盘符变动导致整条链路失效。
_ARD_REL = "Users/gaodu/Desktop/毕业设计/mgmd/data/videos/视频/ARD-MAV"
_ARD_VOLUMES = ("0C1CFCAF1CFC94C41", "0C1CFCAF1CFC94C4")
_ard_env = __import__("os").environ.get("MGMD_ARD_ROOT")
if _ard_env:
    ARD_ROOT = Path(_ard_env)
else:
    ARD_ROOT = next(
        (Path("/media/gaoduan") / v / _ARD_REL for v in _ARD_VOLUMES
         if (Path("/media/gaoduan") / v / _ARD_REL).is_dir()),
        Path("/media/gaoduan") / _ARD_VOLUMES[0] / _ARD_REL,
    )
ARD_VIDEOS = ARD_ROOT / "videos"
ARD_ANNOTATIONS = ARD_ROOT / "Annotations"

# 官方权重（SPEC §2）
WEIGHTS_DIR = Path("/home/gaoduan/tracking/data_raw/mgmd_weights")

# 产物目录
OUTPUTS = PROJECT_ROOT / "outputs"
RUNS = OUTPUTS / "runs"
TABLES = OUTPUTS / "tables"
FIGURES = OUTPUTS / "figures"
LOGS = OUTPUTS / "logs"
DERIVED = PROJECT_ROOT / "data_derived"
GT_DIR = DERIVED / "gt"
SPLIT_FILE = PROJECT_ROOT / "configs" / "split_ard.json"

# ---------- 官方划分（GLAD README，SPEC §4.2）----------

# 15 个测试视频（不带 phantom 前缀的数字 ID）
TEST_IDS = (5, 8, 9, 10, 19, 30, 41, 43, 46, 47, 58, 63, 65, 70, 86)

# 论文 Table 1 的条件分组（官方 README）
CONDITION_GROUPS = {
    "Ordinary": (9, 10, 30, 47, 70),
    "Complex": (5, 8, 58, 65, 86),
    "Small": (19, 41, 43, 46, 63),
}

# ---------- 评测口径（SPEC §5）----------

IOU_THRESH = 0.25          # 论文值
MATCH_ALGO = "hungarian"   # 一对一，防重复计 TP
AP_MODES = ("11point", "allpoint")

# 分组报告的 GT 数分档（SPEC §5.3）
GT_COUNT_BUCKETS = ((1, 1), (2, 3), (4, 10**9))

# ---------- 错误码（SPEC §9）----------

EXIT_OK = 0
EXIT_CONFIG = 2
EXIT_DATA = 3
EXIT_RUNTIME = 4
EXIT_CHECK = 5
EXIT_INTERRUPT = 130


class MgmdError(RuntimeError):
    """带退出码与补救建议的错误（SPEC §9）。"""

    def __init__(self, code: str, msg: str, remediation: str, exit_code: int):
        self.code = code
        self.msg = msg
        self.remediation = remediation
        self.exit_code = exit_code
        super().__init__(f"[mgmd] {code}: {msg} → {remediation}")


# ---------- 工具 ----------

def video_id_to_name(vid: int) -> str:
    """5 → phantom05；101 → phantom101。"""
    return f"phantom{vid:02d}"


def name_to_video_id(name: str) -> int:
    """phantom05 → 5；phantom101 → 101。"""
    if not name.startswith("phantom"):
        raise ValueError(f"非法视频名: {name}")
    return int(name[len("phantom"):])


def list_annotated_videos() -> list[str]:
    """返回 Annotations/ 下有 XML 的视频名，按编号排序。"""
    if not ARD_ANNOTATIONS.is_dir():
        raise MgmdError("E_DATA_MISSING", f"标注目录不存在: {ARD_ANNOTATIONS}",
                        "确认 NTFS 分区已挂载（SPEC §2）", EXIT_DATA)
    names = [p.name for p in ARD_ANNOTATIONS.iterdir() if p.is_dir()]
    return sorted(names, key=name_to_video_id)


def load_split() -> dict:
    """读取固化的划分文件；不存在 → 明确的错误。"""
    if not SPLIT_FILE.is_file():
        raise MgmdError("E_DATA_MISSING", f"划分文件不存在: {SPLIT_FILE}",
                        "先运行 scripts/make_split.py（SPEC T02）", EXIT_DATA)
    return json.loads(SPLIT_FILE.read_text(encoding="utf-8"))


def ensure_dirs() -> None:
    for d in (OUTPUTS, RUNS, TABLES, FIGURES, LOGS, DERIVED, GT_DIR):
        d.mkdir(parents=True, exist_ok=True)
