# Copyright 2026 Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

# 3.28 is the oldest CMake for which ExecuTorch's package offers imported targets.
cmake_minimum_required(VERSION 3.28)
project(isaac_ros_executorch LANGUAGES CXX)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

find_package(CUDAToolkit REQUIRED)
# Points executorch_DIR at the vendored ExecuTorch and sets EXECUTORCH_VENDOR_VERSION_RANGE.
find_package(executorch_cuda_vendor REQUIRED)
find_package(executorch ${EXECUTORCH_VENDOR_VERSION_RANGE} REQUIRED CONFIG
  COMPONENTS backend_cuda extension_cuda kernels_optimized)
# From the torch-tensorrt-executorch-runtime wheel, which has no ROS package yet; see the README.
find_package(executorch_backend_tensorrt REQUIRED CONFIG)

ament_auto_add_library(executorch_node SHARED src/executorch_node.cpp)
target_link_libraries(executorch_node
  CUDA::cudart
  executorch::extension_cuda
  # The node calls none of these. They register delegates and operators when
  # loaded, which is how a program finds its TensorRT and CUDA parts.
  executorch::backend_tensorrt
  executorch::backend_cuda
  executorch::kernels_optimized)
rclcpp_components_register_nodes(executorch_node "nvidia::isaac_ros::dnn_inference::ExecuTorchNode")

ament_auto_add_executable(tensor_list_client src/tensor_list_client.cpp NO_TARGET_LINK_LIBRARIES)

set_target_properties(executorch_node tensor_list_client PROPERTIES
  INSTALL_RPATH_USE_LINK_PATH TRUE)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()

  find_package(launch_testing_ament_cmake REQUIRED)
  add_launch_test(test/isaac_ros_executorch_test.py TIMEOUT "120")
endif()

# Embed versioning information into installed files
ament_index_get_resource(ISAAC_ROS_COMMON_CMAKE_PATH isaac_ros_common_cmake_path isaac_ros_common)
include("${ISAAC_ROS_COMMON_CMAKE_PATH}/isaac_ros_common-version-info.cmake")
generate_version_info(${PROJECT_NAME})

ament_auto_package(INSTALL_TO_SHARE launch scripts)
