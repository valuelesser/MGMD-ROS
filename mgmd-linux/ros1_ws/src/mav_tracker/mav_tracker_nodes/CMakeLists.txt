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
  scripts/tt_driver.py
  scripts/mav_node.py
  scripts/tracker_node.py
  scripts/viz_node.py
  scripts/mission_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
