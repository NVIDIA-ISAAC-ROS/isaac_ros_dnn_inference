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
project(isaac_ros_triton LANGUAGES C CXX)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

# Triton Server integration - always enabled
include(FetchContent)

# Only prebuilt tarball is supported. Source builds are not allowed.
if(NOT TRITON_SERVER_TARBALL_URL)
  if(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
    set(_TRITON_BASE_URL "https://github.com/triton-inference-server/server/releases/download/v2.60.0/")
    set(TRITON_SERVER_TARBALL_URL "${_TRITON_BASE_URL}tritonserver2.60.0-agx.tar")
    message(STATUS "Using default Triton AGX tarball for aarch64: ${TRITON_SERVER_TARBALL_URL}")
  elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64" OR CMAKE_SYSTEM_PROCESSOR STREQUAL "AMD64")
    set(TRITON_SERVER_TARBALL_URL "https://urm.nvidia.com/artifactory/sw-isaac-ros-generic-local/triton/tritonserver.2.60.tar.gz")
    message(STATUS "Using default Triton x86_64 tarball: ${TRITON_SERVER_TARBALL_URL}")
  else()
    message(FATAL_ERROR "TRITON_SERVER_TARBALL_URL must be provided for ${CMAKE_SYSTEM_PROCESSOR}. Only prebuilt tarballs are supported.")
  endif()
endif()

set(TRITON_SERVER_TARBALL_URL "${TRITON_SERVER_TARBALL_URL}" CACHE STRING "URL to prebuilt Triton Server tarball")

# Download and extract the Triton server tarball
fetchcontent_declare(
  tritonserver
  URL ${TRITON_SERVER_TARBALL_URL}
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
fetchcontent_populate(tritonserver)

# Locate header and library recursively in the extracted content
file(GLOB_RECURSE _TRITON_HDR_CANDIDATES "${tritonserver_SOURCE_DIR}/*/tritonserver.h")
list(LENGTH _TRITON_HDR_CANDIDATES _HDR_COUNT)
if(_HDR_COUNT GREATER 0)
  list(GET _TRITON_HDR_CANDIDATES 0 _TRITONSERVER_H)
  # Use the include directory, not the triton/core subdirectory
  get_filename_component(_TRITON_CORE_DIR "${_TRITONSERVER_H}" DIRECTORY)
  get_filename_component(_TRITON_DIR "${_TRITON_CORE_DIR}" DIRECTORY)
  get_filename_component(_TRITON_INC_DIR "${_TRITON_DIR}" DIRECTORY)
  message(STATUS "Found tritonserver.h at: ${_TRITONSERVER_H}")
  message(STATUS "Setting include directory to: ${_TRITON_INC_DIR}")
else()
  message(FATAL_ERROR "Could not find tritonserver.h under ${tritonserver_SOURCE_DIR}")
endif()

file(GLOB_RECURSE _TRITON_LIB_CANDIDATES "${tritonserver_SOURCE_DIR}/*/libtritonserver.so")
list(LENGTH _TRITON_LIB_CANDIDATES _LIB_COUNT)
if(_LIB_COUNT GREATER 0)
  list(GET _TRITON_LIB_CANDIDATES 0 _TRITONSERVER_SO)
  get_filename_component(_TRITON_LIB_DIR "${_TRITONSERVER_SO}" DIRECTORY)
  message(STATUS "Found libtritonserver.so at: ${_TRITONSERVER_SO}")
else()
  message(FATAL_ERROR "Could not locate libtritonserver.so under ${tritonserver_SOURCE_DIR}")
endif()

# TritonNode
ament_auto_add_library(triton_node SHARED src/triton_node.cpp)
ament_target_dependencies(triton_node
  rclcpp
  rclcpp_components
  isaac_ros_common
  isaac_ros_managed_nitros
  isaac_ros_nitros_tensor_list_type
)

# Add Triton dependencies if found
if(_HDR_COUNT GREATER 0)
  target_include_directories(triton_node PRIVATE ${_TRITON_INC_DIR})
endif()
if(_LIB_COUNT GREATER 0)
  # Create an imported target for Triton
  add_library(tritonserver::tritonserver SHARED IMPORTED)
  set_target_properties(tritonserver::tritonserver PROPERTIES
    IMPORTED_LOCATION "${_TRITON_LIB_DIR}/libtritonserver.so"
    INTERFACE_INCLUDE_DIRECTORIES "${_TRITON_INC_DIR}"
  )
  target_link_libraries(triton_node tritonserver::tritonserver dl pthread)
endif()
rclcpp_components_register_nodes(triton_node "nvidia::isaac_ros::dnn_inference::TritonNode")
set(node_plugins "${node_plugins}nvidia::isaac_ros::dnn_inference::TritonNode;$<TARGET_FILE:triton_node>\n")
set_target_properties(triton_node PROPERTIES
  BUILD_WITH_INSTALL_RPATH TRUE
  BUILD_RPATH_USE_ORIGIN TRUE
  INSTALL_RPATH_USE_LINK_PATH TRUE
)

if(BUILD_TESTING)
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
  target_link_libraries(triton_node_test triton_node)
  target_include_directories(triton_node_test PUBLIC include/ ${_TRITON_INC_DIR})
  ament_target_dependencies(triton_node_test rclcpp)
  ament_target_dependencies(triton_node_test isaac_ros_nitros)

  # The FindPythonInterp and FindPythonLibs modules are removed
  if(POLICY CMP0148)
    cmake_policy(SET CMP0148 OLD)
  endif()

  find_package(launch_testing_ament_cmake REQUIRED)
  add_launch_test(test/isaac_ros_triton_test_onnx.py TIMEOUT "300")
endif()


# Embed versioning information into installed files
ament_index_get_resource(ISAAC_ROS_COMMON_CMAKE_PATH isaac_ros_common_cmake_path isaac_ros_common)
include("${ISAAC_ROS_COMMON_CMAKE_PATH}/isaac_ros_common-version-info.cmake")
generate_version_info(${PROJECT_NAME})

ament_auto_package(INSTALL_TO_SHARE launch)
