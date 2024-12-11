# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.22.1)
project(isaac_ros_tensor_proc LANGUAGES C CXX CUDA)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

# Interleaved to planar
ament_auto_add_library(interleaved_to_planar_node SHARED src/interleaved_to_planar_node.cpp)
rclcpp_components_register_nodes(interleaved_to_planar_node
  "nvidia::isaac_ros::dnn_inference::InterleavedToPlanarNode")
set(node_plugins
  "${node_plugins}nvidia::isaac_ros::dnn_inference::InterleavedToPlanarNode;$<TARGET_FILE:interleaved_to_planar_node>\n")
set_target_properties(interleaved_to_planar_node PROPERTIES
  BUILD_WITH_INSTALL_RPATH TRUE
  BUILD_RPATH_USE_ORIGIN TRUE
  INSTALL_RPATH_USE_LINK_PATH TRUE)


# Reshape
ament_auto_add_library(reshape_node SHARED src/reshape_node.cpp)
rclcpp_components_register_nodes(reshape_node
  "nvidia::isaac_ros::dnn_inference::ReshapeNode")
set(node_plugins
  "${node_plugins}nvidia::isaac_ros::dnn_inference::ReshapeNode;$<TARGET_FILE:reshape_node>\n")
set_target_properties(reshape_node PROPERTIES
  BUILD_WITH_INSTALL_RPATH TRUE
  BUILD_RPATH_USE_ORIGIN TRUE
  INSTALL_RPATH_USE_LINK_PATH TRUE)

# Image to Tensor
ament_auto_add_library(image_to_tensor_node SHARED src/image_to_tensor_node.cpp)
rclcpp_components_register_nodes(image_to_tensor_node
  "nvidia::isaac_ros::dnn_inference::ImageToTensorNode")
set(node_plugins
  "${node_plugins}nvidia::isaac_ros::dnn_inference::ImageToTensorNode;$<TARGET_FILE:image_to_tensor_node>\n")
target_link_libraries(image_to_tensor_node cvcuda nvcv_types)
set_target_properties(image_to_tensor_node PROPERTIES
  BUILD_WITH_INSTALL_RPATH TRUE
  BUILD_RPATH_USE_ORIGIN TRUE
  INSTALL_RPATH_USE_LINK_PATH TRUE)

# Image tensor normalize node
ament_auto_add_library(image_tensor_normalize_node SHARED src/image_tensor_normalize_node.cpp)
  rclcpp_components_register_nodes(image_tensor_normalize_node
    "nvidia::isaac_ros::dnn_inference::ImageTensorNormalizeNode")
  set(node_plugins
    "${node_plugins}nvidia::isaac_ros::dnn_inference::ImageTensorNormalizeNode;$<TARGET_FILE:image_tensor_normalize_node>\n")
  target_link_libraries(image_tensor_normalize_node cvcuda nvcv_types)
  set_target_properties(image_tensor_normalize_node PROPERTIES
    BUILD_WITH_INSTALL_RPATH TRUE
    BUILD_RPATH_USE_ORIGIN TRUE
    INSTALL_RPATH_USE_LINK_PATH TRUE)

# Normalize
ament_auto_add_library(normalize_node SHARED src/normalize_node.cpp)
  rclcpp_components_register_nodes(normalize_node
    "nvidia::isaac_ros::dnn_inference::NormalizeNode")
  set(node_plugins
    "${node_plugins}nvidia::isaac_ros::dnn_inference::NormalizeNode;$<TARGET_FILE:normalize_node>\n")
  set_target_properties(normalize_node PROPERTIES
    BUILD_WITH_INSTALL_RPATH TRUE
    BUILD_RPATH_USE_ORIGIN TRUE
    INSTALL_RPATH_USE_LINK_PATH TRUE)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()

  # Gtest for interleaved_to_planar_node
  ament_add_gtest(interleaved_to_planar_node_test test/interleaved_to_planar_node_test.cpp)
  target_link_libraries(interleaved_to_planar_node_test interleaved_to_planar_node)
  target_include_directories(interleaved_to_planar_node_test PUBLIC include/isaac_ros_tensor_proc/)
  target_include_directories(interleaved_to_planar_node_test PUBLIC /usr/src/googletest/googlemock/include/)
  ament_target_dependencies(interleaved_to_planar_node_test rclcpp)
  ament_target_dependencies(interleaved_to_planar_node_test isaac_ros_nitros)
  # Gtest for normalize_node
  ament_add_gtest(normalize_node_test test/normalize_node_test.cpp)
  target_link_libraries(normalize_node_test normalize_node)
  target_include_directories(normalize_node_test PUBLIC include/isaac_ros_tensor_proc/)
  target_include_directories(normalize_node_test PUBLIC /usr/src/googletest/googlemock/include/)
  ament_target_dependencies(normalize_node_test rclcpp)
  ament_target_dependencies(normalize_node_test isaac_ros_nitros)

  # The FindPythonInterp and FindPythonLibs modules are removed
  if(POLICY CMP0148)
    cmake_policy(SET CMP0148 OLD)
  endif()

  find_package(launch_testing_ament_cmake REQUIRED)
  add_launch_test(test/isaac_ros_interleaved_to_planar_test.py)
  add_launch_test(test/isaac_ros_reshape_test.py)
  add_launch_test(test/isaac_ros_image_to_tensor.py)
  add_launch_test(test/isaac_ros_image_to_tensor_no_scale.py)
  add_launch_test(test/isaac_ros_image_to_tensor_no_scale_float.py)
  add_launch_test(test/isaac_ros_image_tensor_normalize_test.py)
  add_launch_test(test/isaac_ros_normalize_test.py)
endif()


# Embed versioning information into installed files
ament_index_get_resource(ISAAC_ROS_COMMON_CMAKE_PATH isaac_ros_common_cmake_path isaac_ros_common)
include("${ISAAC_ROS_COMMON_CMAKE_PATH}/isaac_ros_common-version-info.cmake")
generate_version_info(${PROJECT_NAME})

ament_auto_package(INSTALL_TO_SHARE config)
