# 单个检测结果（图像坐标系，像素）
std_msgs/Header header

# 边界框 (x1, y1, x2, y2)，左上/右下角
float32 x1
float32 y1
float32 x2
float32 y2

# 中心点（便于下游使用）
float32 cx
float32 cy

# 置信度 [0,1]：LAD conf 或 LAC P(MAV)。复用轨迹沿用最近 LAD conf，纯 TF 展示用 0 表示无模型评分。
float32 confidence

# 来源模块：mfe / lac / lad / tf
string source

# 该检测关联的轨迹 ID，-1 表示未关联
int32 track_id
