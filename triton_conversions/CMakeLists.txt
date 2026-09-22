# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
project(triton_conversions LANGUAGES C CXX)

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

# Header-only interface library. Triton and CUDA are omitted from the exported
# link interface so downstream packages are not forced to find_package for
# tritonserver/CUDAToolkit before loading this package's export; only the include
# directory and ROS package dependencies are propagated.
add_library(${PROJECT_NAME} INTERFACE)
target_include_directories(${PROJECT_NAME} INTERFACE
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
  "$<INSTALL_INTERFACE:include>")

install(DIRECTORY include/ DESTINATION include)
install(TARGETS ${PROJECT_NAME} EXPORT export_${PROJECT_NAME})

ament_export_targets(export_${PROJECT_NAME})
ament_export_include_directories(include)
ament_export_dependencies(cuda_buffer tensor_msgs)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # Use the custom NVIDIA Isaac ROS Software License headers.
  set(ament_cmake_copyright_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()

  find_package(ament_cmake_gtest REQUIRED)
  find_package(CUDAToolkit REQUIRED)

  # The header includes <triton/core/tritonserver.h>. Locate the Triton include
  # directory and libtritonserver.so installed by the triton-server dependency
  # so the test can compile and link (from_triton_data_type's error path calls
  # TRITONSERVER_DataTypeString, a real symbol in the Triton library).
  find_path(TRITON_INCLUDE_DIR
    NAMES triton/core/tritonserver.h
    PATH_SUFFIXES include
    PATHS /opt/tritonserver /usr/local /usr)
  find_library(TRITONSERVER_LIBRARY
    NAMES tritonserver
    PATH_SUFFIXES lib lib64
    PATHS /opt/tritonserver /usr/local /usr)

  if(TRITON_INCLUDE_DIR AND TRITONSERVER_LIBRARY)
    # Force use of ROS2 vendor googletest to avoid version conflicts
    set(CMAKE_IGNORE_PATH "/usr/src/googletest" ${CMAKE_IGNORE_PATH})
    set(CMAKE_IGNORE_PATH "/usr/include/gtest" ${CMAKE_IGNORE_PATH})
    set(CMAKE_IGNORE_PATH "/usr/include/gmock" ${CMAKE_IGNORE_PATH})
    include_directories(BEFORE /opt/ros/$ENV{ROS_DISTRO}/src/gtest_vendor/include)

    # Unit tests for the conversion logic: pure host maps/shape helpers and
    # accessor const-correctness, plus a GPU-backed suite (TritonTensorTest) that
    # acquires real cuda_buffer Read/WriteHandles and so needs a GPU at run time.
    ament_add_gtest(triton_conversions_test test/triton_conversions_test.cpp)
    target_include_directories(triton_conversions_test PRIVATE ${TRITON_INCLUDE_DIR})
    # This ROS distro drops ament_target_dependencies; link the modern imported
    # targets directly. ${<pkg>_TARGETS} carries transitive include dirs (e.g.
    # cuda_buffer pulls in rosidl_buffer), populated by ament_auto_find_build_dependencies.
    # cuda_buffer's error path calls the CUDA driver API (cuGetErrorName), so the
    # test links CUDA::cuda_driver alongside CUDA::cudart.
    target_link_libraries(triton_conversions_test
      ${PROJECT_NAME}
      ${TRITONSERVER_LIBRARY}
      CUDA::cudart
      CUDA::cuda_driver
      ${cuda_buffer_TARGETS}
      tensor_msgs::tensor_msgs
    )
  else()
    message(WARNING
      "Triton headers/library not found "
      "(TRITON_INCLUDE_DIR=${TRITON_INCLUDE_DIR}, TRITONSERVER_LIBRARY=${TRITONSERVER_LIBRARY}); "
      "skipping triton_conversions_test.")
  endif()
endif()

ament_auto_package()
