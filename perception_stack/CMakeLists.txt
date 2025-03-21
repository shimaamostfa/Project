cmake_minimum_required(VERSION 3.0.2)
project(perception_stack)

find_package(catkin REQUIRED COMPONENTS
  rospy
  sensor_msgs
  std_msgs
  geometry_msgs
  cv_bridge
  message_generation
)

add_message_files(
  FILES
  TrackedObject.msg
  TrackedObjectList.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
  sensor_msgs
  geometry_msgs
)

catkin_package(
  CATKIN_DEPENDS rospy sensor_msgs std_msgs geometry_msgs cv_bridge message_runtime
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)
