cmake_minimum_required(VERSION 3.0.2)
project(mav_tracker_nodes)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  sensor_msgs
  geometry_msgs
  mav_tracker_msgs
)

catkin_package()

catkin_install_python(PROGRAMS
  scripts/video_source.py
  scripts/detector_node.py
  scripts/cuda_pipeline_node.py
  scripts/rqt_fullscreen.py
  scripts/visualize_node.py
  scripts/target_mover.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch config worlds
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
