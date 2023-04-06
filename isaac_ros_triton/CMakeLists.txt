# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

cmake_minimum_required(VERSION 3.23.2)
project(isaac_ros_triton LANGUAGES C CXX)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

execute_process(COMMAND uname -m COMMAND tr -d '\n' OUTPUT_VARIABLE ARCHITECTURE)
message( STATUS "Architecture: ${ARCHITECTURE}" )

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

# TritonNode
ament_auto_add_library(triton_node SHARED src/triton_node.cpp)
rclcpp_components_register_nodes(triton_node "nvidia::isaac_ros::dnn_inference::TritonNode")
set(node_plugins "${node_plugins}nvidia::isaac_ros::dnn_inference::TritonNode;$<TARGET_FILE:triton_node>\n")

### Install extensions built from source

# Triton + dependencies
add_subdirectory(gxf/triton)
install(TARGETS gxf_triton_ext DESTINATION share/${PROJECT_NAME}/gxf/triton/)
if( ${ARCHITECTURE} STREQUAL "x86_64" )
  set(ARCH_GXF_PATH "gxf_x86_64_cuda_11_8")
  elseif( ${ARCHITECTURE} STREQUAL "aarch64" )
  set(ARCH_GXF_PATH "gxf_jetpack502")
endif()
install(DIRECTORY gxf/triton/nvds/lib/${ARCH_GXF_PATH}/
        DESTINATION share/${PROJECT_NAME}/gxf/triton)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()

  find_package(launch_testing_ament_cmake REQUIRED)
  add_launch_test(test/isaac_ros_triton_test_onnx.py TIMEOUT "300")
  add_launch_test(test/isaac_ros_triton_test_tf.py TIMEOUT "300")
endif()

ament_auto_package(INSTALL_TO_SHARE config launch)
