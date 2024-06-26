# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
project(gxf_isaac_triton LANGUAGES C CXX)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

# Determine the architecture
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
  set(GXF_EXT_LIB_PATH "${CMAKE_CURRENT_SOURCE_DIR}/lib/gxf_jetpack60")
  set(NVDS_LIB_PATH "${CMAKE_CURRENT_SOURCE_DIR}/nvds/lib/gxf_jetpack60")
  elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
  set(GXF_EXT_LIB_PATH "${CMAKE_CURRENT_SOURCE_DIR}/lib/gxf_x86_64_cuda_12_2")
  set(NVDS_LIB_PATH "${CMAKE_CURRENT_SOURCE_DIR}/nvds/lib/gxf_x86_64_cuda_12_2")
else()
  message(FATAL_ERROR "Unsupported architecture: ${CMAKE_SYSTEM_PROCESSOR}")
endif()

# Add an interface target to export dependencies
add_library(${PROJECT_NAME} INTERFACE)

# Install shared library
install(FILES "${GXF_EXT_LIB_PATH}/lib${PROJECT_NAME}.so"
        DESTINATION "${CMAKE_INSTALL_PREFIX}/share/${PROJECT_NAME}/gxf/lib")
set_property(TARGET ${PROJECT_NAME} PROPERTY
  INTERFACE_LINK_LIBRARIES
    "$<INSTALL_PREFIX>/share/${PROJECT_NAME}/gxf/lib/lib${PROJECT_NAME}.so"
)

# Install headers
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/gxf")
  install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/gxf/"
          DESTINATION "${CMAKE_INSTALL_PREFIX}/share/${PROJECT_NAME}/gxf/include")
  set_target_properties(${PROJECT_NAME} PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES
      "$<INSTALL_PREFIX>/share/${PROJECT_NAME}/gxf/include")
endif()

install(TARGETS ${PROJECT_NAME}
        EXPORT export_${PROJECT_NAME}
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin)
ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)

# Install nvds libraries
install(DIRECTORY ${NVDS_LIB_PATH}/
        DESTINATION share/${PROJECT_NAME}/gxf/lib)

ament_auto_package(INSTALL_TO_SHARE)
