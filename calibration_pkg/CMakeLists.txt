cmake_minimum_required(VERSION 3.0.2)
project(calibration_pkg)

find_package(catkin REQUIRED COMPONENTS
  rospy
  tf2_ros
  tf_conversions
  geometry_msgs
)

catkin_package()

catkin_install_python(PROGRAMS
  scripts/handeye_matrix_node.py
  scripts/hand_eye_calibration.py
  scripts/calibration_d415.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
