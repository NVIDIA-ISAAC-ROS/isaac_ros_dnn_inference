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

project(gxf_triton_ext LANGUAGES C CXX)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-fPIC -w)
endif()

set(CMAKE_WARN_DEPRECATED OFF CACHE BOOL "" FORCE)
set(LAST_CMAKE_MESSAGE_LOG_LEVEL ${CMAKE_MESSAGE_LOG_LEVEL})
set(CMAKE_MESSAGE_LOG_LEVEL ERROR)

# Dependencies
include(FetchContent)
find_package(yaml-cpp)
find_package(GXF ${ISAAC_ROS_GXF_VERSION} MODULE REQUIRED
  COMPONENTS
  core
  std
)
# Lock version of Protocol buffers for compatibility with pre-built NVDS
set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
set(protobuf_BUILD_TESTS OFF)
set(protobuf_BUILD_EXPORT OFF)
set(protobuf_MSVC_STATIC_RUNTIME OFF)
set(Protobuf_USE_STATIC_LIBS ON)
set(Protobuf_BUILD_SHARED_LIBS OFF)

fetchcontent_declare(
  protobuf
  GIT_REPOSITORY https://github.com/protocolbuffers/protobuf.git
  GIT_TAG        v3.8.0
  SOURCE_SUBDIR  cmake
)
fetchcontent_makeavailable(protobuf)
find_package(Protobuf QUIET)
set(CMAKE_MESSAGE_LOG_LEVEL ${LAST_CMAKE_MESSAGE_LOG_LEVEL})

# Compile protocol buffers
file(GLOB ProtoFiles "${CMAKE_CURRENT_SOURCE_DIR}/nvds/include/*.proto")
PROTOBUF_GENERATE_CPP(ProtoSources ProtoHeaders ${ProtoFiles})
add_library(libgxf_triton_proto STATIC ${ProtoSources} ${ProtoHeaders})
target_link_libraries(libgxf_triton_proto PUBLIC protobuf::libprotobuf)

# NVDS pre-built
add_library(libs_triton::libnvbuf_fdmap SHARED IMPORTED)
add_library(libs_triton::libnvbufsurface SHARED IMPORTED)
add_library(libs_triton::libnvbufsurftransform SHARED IMPORTED)
add_library(libs_triton::libnvds_infer_server SHARED IMPORTED)
add_library(libs_triton::libnvds_inferlogger SHARED IMPORTED)
add_library(libs_triton::libnvds_inferutils SHARED IMPORTED)
add_library(libs_triton::libs_triton INTERFACE IMPORTED)
set_property(TARGET libs_triton::libs_triton PROPERTY
  INTERFACE_LINK_LIBRARIES
    libs_triton::libnvbuf_fdmap
    libs_triton::libnvbufsurface
    libs_triton::libnvbufsurftransform
    libs_triton::libnvds_infer_server
    libs_triton::libnvds_inferlogger
    libs_triton::libnvds_inferutils
)

execute_process(COMMAND uname -m COMMAND tr -d '\n' OUTPUT_VARIABLE ARCHITECTURE)
message( STATUS "Architecture: ${ARCHITECTURE}" )
if( ${ARCHITECTURE} STREQUAL "x86_64" )
  set(ARCH_GXF_PATH "gxf_x86_64_cuda_11_8")
  elseif( ${ARCHITECTURE} STREQUAL "aarch64" )
  set(ARCH_GXF_PATH "gxf_jetpack502")
endif()
set_property(TARGET libs_triton::libnvbuf_fdmap PROPERTY IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/nvds/lib/${ARCH_GXF_PATH}/libnvbuf_fdmap.so)
set_property(TARGET libs_triton::libnvbufsurface PROPERTY IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/nvds/lib/${ARCH_GXF_PATH}/libnvbufsurface.so)
set_property(TARGET libs_triton::libnvbufsurftransform PROPERTY IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/nvds/lib/${ARCH_GXF_PATH}/libnvbufsurftransform.so)
set_property(TARGET libs_triton::libnvds_infer_server PROPERTY IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/nvds/lib/${ARCH_GXF_PATH}/libnvds_infer_server.so)
set_property(TARGET libs_triton::libnvds_inferlogger PROPERTY IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/nvds/lib/${ARCH_GXF_PATH}/libnvds_inferlogger.so)
set_property(TARGET libs_triton::libnvds_inferutils PROPERTY IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/nvds/lib/${ARCH_GXF_PATH}/libnvds_inferutils.so)

# Triton extension
add_library(gxf_triton_ext SHARED
  extensions/triton/triton_server.cpp
  inferencers/triton_inferencer_impl.cpp
  triton_ext.cpp
  triton_inference_request.cpp
  triton_inference_response.cpp
  triton_scheduling_terms.cpp
)
set(CMAKE_INCLUDE_CURRENT_DIR TRUE)
target_include_directories(gxf_triton_ext PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/nvds/include
  ${CMAKE_CURRENT_SOURCE_DIR}/extensions/triton
)
target_link_libraries(gxf_triton_ext
  PUBLIC
    GXF::std
    libgxf_triton_proto
    libs_triton::libs_triton
    yaml-cpp
)
set_target_properties(gxf_triton_ext PROPERTIES BUILD_WITH_INSTALL_RPATH TRUE)
set_target_properties(gxf_triton_ext PROPERTIES INSTALL_RPATH "$ORIGIN")

