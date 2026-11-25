# Source from bash inside the configured mgmd-ros container.
source /opt/ros/noetic/setup.bash
source /workspace/MGMD-ROS/mgmd-linux/ros1_ws/devel/setup.bash
showcase_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
opencv_prefix=${MGMD_OPENCV_PREFIX:-$HOME/opt/opencv-cuda-4.12.0-cudacodec}
cuda_libs=$(find "$(python3 -m site --user-site)/nvidia" /usr/local/lib/python3.8/dist-packages/nvidia -maxdepth 2 -type d -name lib 2>/dev/null | paste -sd: || true)
export PYTHONPATH="$showcase_root:$opencv_prefix/python:$PYTHONPATH"
export LD_LIBRARY_PATH="$opencv_prefix/lib:/tmp/mgmd-g0-video-libs:$cuda_libs:$LD_LIBRARY_PATH"
export ROS_PACKAGE_PATH="$showcase_root/ros1_ws/src:/opt/ros/noetic/share"
export CMAKE_PREFIX_PATH=/opt/ros/noetic
export MGMD_YOLOV5_DIR=/workspace/MGMD-ROS/tracking/third_party/yolov5-v6.0
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
