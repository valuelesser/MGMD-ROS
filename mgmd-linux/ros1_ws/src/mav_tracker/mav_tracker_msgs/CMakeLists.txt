cmake_minimum_required(VERSION 3.0.2)
project(mav_tracker_msgs)

find_package(catkin REQUIRED COMPONENTS
  message_generation
  std_msgs
)

add_message_files(
  FILES
  Detection.msg
  DetectionArray.msg
  Track.msg
  TrackArray.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
)

catkin_package(
  CATKIN_DEPENDS message_runtime std_msgs
)
