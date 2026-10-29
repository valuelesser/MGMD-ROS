# 单条轨迹
std_msgs/Header header

int32 track_id
string state                    # TENTATIVE / CONFIRMED / LOST

# 当前框
float32 x1
float32 y1
float32 x2
float32 y2
float32 cx
float32 cy

# 速度（像素/帧，注意不是像素/秒，与阶段一 KalmanBox 一致，SPEC-ROS §9.1.3）
float32 vx
float32 vy

# 年龄与命中统计
uint32 age
uint32 hits
float32 confidence

# 轨迹历史（最近 N 个中心点，扁平 [x0,y0,x1,y1,...]）
float32[] history_xy
