# 一帧的全部检测结果
std_msgs/Header header

# 图像尺寸，便于下游换算归一化坐标
uint32 image_width
uint32 image_height

Detection[] detections

# MFE 原始候选数量（诊断用，反映背景噪声水平）
uint32 n_candidates

# 本帧处理耗时（毫秒，分段）
float32 mfe_ms
float32 tf_ms
float32 lac_ms
float32 lad_ms
