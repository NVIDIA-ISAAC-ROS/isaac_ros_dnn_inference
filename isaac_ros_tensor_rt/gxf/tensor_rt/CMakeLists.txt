# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

project(gxf_tensor_rt LANGUAGES C CXX)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-fPIC -w)
endif()

# Dependencies
find_package(CUDAToolkit)
find_package(GXF ${ISAAC_ROS_GXF_VERSION} MODULE REQUIRED
    COMPONENTS
    core
    cuda
    std
)
find_package(TENSORRT 8 MODULE REQUIRED)
include(YamlCpp)

# TensorRT extension
add_library(gxf_tensor_rt SHARED
  tensor_rt_extension.cpp
  tensor_rt_inference.cpp
  tensor_rt_inference.hpp
)
target_link_libraries(gxf_tensor_rt
  PUBLIC
    CUDA::cudart
    GXF::std
    GXF::cuda
    TENSORRT::nvonnxparser
    yaml-cpp
)
