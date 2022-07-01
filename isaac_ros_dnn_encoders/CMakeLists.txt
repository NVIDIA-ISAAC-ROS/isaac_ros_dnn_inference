# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

cmake_minimum_required(VERSION 3.8)
project(isaac_ros_dnn_encoders LANGUAGES C CXX)

# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Default to Release build
if(NOT CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "")
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
endif()
message( STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}" )

execute_process(COMMAND uname -m COMMAND tr -d '\n'
  OUTPUT_VARIABLE ARCHITECTURE
)
message( STATUS "Architecture: ${ARCHITECTURE}" )

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

# Find VPI dependency
find_package(vpi REQUIRED)

# DNN Image Encoder Node
ament_auto_add_library(dnn_image_encoder_node SHARED src/dnn_image_encoder_node.cpp)
target_compile_definitions(dnn_image_encoder_node
  PRIVATE "COMPOSITION_BUILDING_DLL"
)
target_link_libraries(dnn_image_encoder_node)
rclcpp_components_register_nodes(dnn_image_encoder_node "nvidia::isaac_ros::dnn_inference::DnnImageEncoderNode")
set(node_plugins "${node_plugins}nvidia::isaac_ros::dnn_inference::DnnImageEncoderNode;$<TARGET_FILE:dnn_image_encoder_node>\n")

# Install config directory
install(
  DIRECTORY config
  DESTINATION share/${PROJECT_NAME}
)

install(TARGETS dnn_image_encoder_node
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)

  # Ignore copyright notices since we use custom NVIDIA Isaac ROS Software License
  set(ament_cmake_copyright_FOUND TRUE)

  ament_lint_auto_find_test_dependencies()

  find_package(launch_testing_ament_cmake REQUIRED)
  add_launch_test(test/isaac_ros_dnn_image_encoder_test.py)
endif()

ament_auto_package(INSTALL_TO_SHARE)
