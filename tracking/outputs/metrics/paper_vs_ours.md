# Paper vs Ours

论文目标值来自 PRD §6（Jetson AGX Orin 50W）。本环境为 RTX 5060 Ti，FPS 不做绝对对照。
Tier A 需要作者 artifact（当前未获取）→ 下列 Tier A 列记为 N/A，结论按 **Tier B 协议复现**。

| config | metric | paper | ours | tier |
|---|---|---:|---:|---|
| D1_no_overlap | effective_frame_fps | 27.91 | 1.03 | B (A=N/A) |
| D1_no_overlap | valid_outcome | 99.5 | 0.00 | B (A=N/A) |
| D1_no_overlap | recall@100 | 100.0 | 0.00 | B (A=N/A) |
| D1_no_overlap | precision@100 | 98.99 | nan | B (A=N/A) |
| D1_overlap | effective_frame_fps | 9.49 | N/A | B (A=N/A) |
| D1_overlap | valid_outcome | 99.41 | N/A | B (A=N/A) |
| D1_overlap | recall@100 | 100.0 | N/A | B (A=N/A) |
| D1_overlap | precision@100 | 98.82 | N/A | B (A=N/A) |
| D2_no_overlap | effective_frame_fps | 16.53 | 2.80 | B (A=N/A) |
| D2_no_overlap | valid_outcome | 97.79 | 100.00 | B (A=N/A) |
| D2_no_overlap | recall@100 | 97.38 | 100.00 | B (A=N/A) |
| D2_no_overlap | precision@100 | 99.55 | 100.00 | B (A=N/A) |
| D2_overlap | effective_frame_fps | 5.36 | N/A | B (A=N/A) |
| D2_overlap | valid_outcome | 98.69 | N/A | B (A=N/A) |
| D2_overlap | recall@100 | 99.21 | N/A | B (A=N/A) |
| D2_overlap | precision@100 | 98.98 | N/A | B (A=N/A) |
| D3_overlap | effective_frame_fps | 10.57 | 0.31 | B (A=N/A) |
| D3_overlap | valid_outcome | 93.86 | 70.00 | B (A=N/A) |
| D3_overlap | recall@100 | 92.41 | 0.00 | B (A=N/A) |
| D3_overlap | precision@100 | 93.72 | 0.00 | B (A=N/A) |
| D4_overlap | effective_frame_fps | 1.93 | N/A | B (A=N/A) |
| D4_overlap | valid_outcome | 85.84 | N/A | B (A=N/A) |
| D4_overlap | recall@100 | 84.64 | N/A | B (A=N/A) |
| D4_overlap | precision@100 | 93.36 | N/A | B (A=N/A) |
