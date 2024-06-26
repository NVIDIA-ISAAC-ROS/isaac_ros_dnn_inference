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
project(isaac_ros_tensor_rt LANGUAGES C CXX)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

# Dependencies
find_package(TENSORRT 8 MODULE REQUIRED)

# TensorRTNode
ament_auto_add_library(tensor_rt_node SHARED src/tensor_rt_node.cpp)
target_link_libraries(tensor_rt_node TENSORRT::nvonnxparser)
rclcpp_components_register_nodes(tensor_rt_node "nvidia::isaac_ros::dnn_inference::TensorRTNode")
set(node_plugins "${node_plugins}nvidia::isaac_ros::dnn_inference::TensorRTNode;$<TARGET_FILE:tensor_rt_node>\n")
set_target_properties(tensor_rt_node PROPERTIES
  BUILD_WITH_INSTALL_RPATH TRUE
  BUILD_RPATH_USE_ORIGIN TRUE
  INSTALL_RPATH_USE_LINK_PATH TRUE
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()

  # Add gtests
  ament_add_gtest(tensor_rt_node_test test/tensor_rt_node_test.cpp)
  target_link_libraries(tensor_rt_node_test tensor_rt_node)
  target_include_directories(tensor_rt_node_test PUBLIC include/isaac_ros_tensor_rt/)
  target_include_directories(tensor_rt_node_test PUBLIC /usr/src/googletest/googlemock/include/)
  ament_target_dependencies(tensor_rt_node_test rclcpp)
  ament_target_dependencies(tensor_rt_node_test isaac_ros_nitros)

  # The FindPythonInterp and FindPythonLibs modules are removed
  if(POLICY CMP0148)
    cmake_policy(SET CMP0148 OLD)
  endif()

  find_package(launch_testing_ament_cmake REQUIRED)
  add_launch_test(test/isaac_ros_tensor_rt_test.py TIMEOUT "300")
endif()

ament_auto_package(INSTALL_TO_SHARE config launch)