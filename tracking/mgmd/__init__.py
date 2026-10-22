"""mgmd — MGMD（Motion-guided MAV Detection）在本仓库的移植（SPEC tasks/mgmd_mvdt_reproduction_spec.md）。

模块布局镜像 spa2d/：
- config:    配置 YAML 加载与 schema 校验（``[mgmd] <code>: <msg> → <remediation>``）
- io_mvdt:   包装 spa2d.io_mvdt.FrameSource（常驻 capture 顺序读帧）+ downscale
- align:     网格关键点 LK + RANSAC 单应帧对齐（issue-019）
- diff:      自适应阈值两帧差分 + 形态学 + 连通域候选（issue-019）
- track:     SORT + 论文 Algorithm 1 轨迹分类（issue-020）
- classify:  LAC 32×32 CNN（torch 懒加载，issue-021）
- detect:    LAD 320×320 裁剪 YOLOv5s-crop（torch/yolov5 懒加载，issue-021）
- pipeline:  逐帧编排 + WindowRecord 适配（issue-022）

退出码约定（仓库统一）：0 成功 / 2 配置非法 / 3 数据缺失 / 4 运行时 / 5 保留 / 130 中断。
"""

__version__ = "0.1.0"
