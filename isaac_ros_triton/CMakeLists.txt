# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
project(isaac_ros_triton LANGUAGES C CXX)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

# Triton Server is installed under /opt by the triton-server Debian dependency.
find_path(
  _TRITON_INC_DIR
  NAMES triton/core/tritonserver.h
  PATHS /opt/tritonserver/include
  NO_DEFAULT_PATH
)
find_library(
  _TRITONSERVER_SO
  NAMES tritonserver
  PATHS /opt/tritonserver/lib /opt/tritonserver/lib64
  NO_DEFAULT_PATH
)
if(NOT _TRITON_INC_DIR OR NOT _TRITONSERVER_SO)
  message(
    FATAL_ERROR
    "The triton-server Debian dependency is not installed under /opt/tritonserver"
  )
endif()
get_filename_component(_TRITON_LIB_DIR "${_TRITONSERVER_SO}" DIRECTORY)

# TritonNode
ament_auto_add_library(triton_node SHARED src/triton_node.cpp)
find_package(CUDAToolkit REQUIRED)

target_include_directories(triton_node PRIVATE ${_TRITON_INC_DIR})
add_library(tritonserver::tritonserver SHARED IMPORTED)
set_target_properties(tritonserver::tritonserver PROPERTIES
  IMPORTED_LOCATION "${_TRITONSERVER_SO}"
  INTERFACE_INCLUDE_DIRECTORIES "${_TRITON_INC_DIR}"
)
target_link_libraries(triton_node
  tritonserver::tritonserver
  dl
  pthread
  CUDA::cudart
  CUDA::cuda_driver
)
rclcpp_components_register_nodes(triton_node "nvidia::isaac_ros::dnn_inference::TritonNode")
set(
  node_plugins
  "${node_plugins}nvidia::isaac_ros::dnn_inference::TritonNode;$<TARGET_FILE:triton_node>\n"
)
# Keep the Triton Debian library directory in the installed RPATH so consumers
# such as triton_node_test can resolve libtritonserver.so directly.
set_target_properties(triton_node PROPERTIES
  BUILD_WITH_INSTALL_RPATH TRUE
  BUILD_RPATH_USE_ORIGIN TRUE
  INSTALL_RPATH_USE_LINK_PATH TRUE
  INSTALL_RPATH "${_TRITON_LIB_DIR}"
)

if(BUILD_TESTING)
  find_package(CUDAToolkit REQUIRED)
  find_package(ament_lint_auto REQUIRED)
  find_package(ament_cmake_gtest REQUIRED)
  ament_lint_auto_find_test_dependencies()

  # Force use of ROS2 vendor googletest to avoid version conflicts
  set(CMAKE_IGNORE_PATH "/usr/src/googletest" ${CMAKE_IGNORE_PATH})
  set(CMAKE_IGNORE_PATH "/usr/include/gtest" ${CMAKE_IGNORE_PATH})
  set(CMAKE_IGNORE_PATH "/usr/include/gmock" ${CMAKE_IGNORE_PATH})

  # Explicitly use ROS2 vendor googletest/gmock
  include_directories(BEFORE /opt/ros/$ENV{ROS_DISTRO}/src/gtest_vendor/include)
  include_directories(BEFORE /opt/ros/$ENV{ROS_DISTRO}/src/gmock_vendor/include)

  # Gtest for triton node
  ament_add_gtest(triton_node_test test/triton_node_test.cpp)
  target_link_libraries(triton_node_test triton_node CUDA::cudart ${rclcpp_TARGETS})
  target_include_directories(triton_node_test PUBLIC include/ ${_TRITON_INC_DIR})

  # The FindPythonInterp and FindPythonLibs modules are removed
  if(POLICY CMP0148)
    cmake_policy(SET CMP0148 OLD)
  endif()

  find_package(launch_testing_ament_cmake REQUIRED)
  add_launch_test(test/isaac_ros_triton_test_onnx.py TIMEOUT "300")
  add_launch_test(test/isaac_ros_triton_test_pytorch.py TIMEOUT "300")
endif()


# Embed versioning information into installed files
ament_index_get_resource(ISAAC_ROS_COMMON_CMAKE_PATH isaac_ros_common_cmake_path isaac_ros_common)
include("${ISAAC_ROS_COMMON_CMAKE_PATH}/isaac_ros_common-version-info.cmake")
generate_version_info(${PROJECT_NAME})

ament_auto_package(INSTALL_TO_SHARE launch)
