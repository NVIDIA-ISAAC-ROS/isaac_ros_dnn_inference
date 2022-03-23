# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

cmake_minimum_required(VERSION 3.5)
project(isaac_ros_dnn_inference_test LANGUAGES C CXX)

# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Default to Release build
if(NOT CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "")
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
endif()
message( STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}" )

execute_process(COMMAND uname -m COMMAND tr -d '\n'
  OUTPUT_VARIABLE ARCHITECTURE
)
message( STATUS "Architecture: ${ARCHITECTURE}" )

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

# test_tensor_publisher_node
ament_auto_add_library(test_tensor_publisher_node SHARED src/test_tensor_publisher_node.cpp)
target_compile_definitions(test_tensor_publisher_node
  PRIVATE "COMPOSITION_BUILDING_DLL"
)
target_link_libraries(test_tensor_publisher_node)
rclcpp_components_register_nodes(test_tensor_publisher_node "isaac_ros::dnn_inference::TestTensorPublisherNode")
set(node_plugins "${node_plugins}isaac_ros::dnn_inference::TestTensorPublisherNode;$<TARGET_FILE:test_tensor_publisher_node>\n")

# run test tensor publisher executable
ament_auto_add_executable("run_test_publisher"
  src/test_tensor_publisher_main.cpp
)

target_link_libraries("run_test_publisher" test_tensor_publisher_node)

install(TARGETS "run_test_publisher"
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)

  # Ignore copyright notices since we use custom NVIDIA Isaac ROS Software License
  set(ament_cmake_copyright_FOUND TRUE)

  ament_lint_auto_find_test_dependencies()

endif()

ament_auto_package()
