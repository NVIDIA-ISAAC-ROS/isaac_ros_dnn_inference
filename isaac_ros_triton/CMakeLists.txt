# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

cmake_minimum_required(VERSION 3.5)
project(isaac_ros_triton LANGUAGES C CXX)

# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

execute_process(COMMAND uname -m COMMAND tr -d '\n' OUTPUT_VARIABLE ARCHITECTURE)
message( STATUS "Architecture: ${ARCHITECTURE}" )

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

# triton_node
ament_auto_add_library(triton_node SHARED src/triton_node.cpp)
target_compile_definitions(triton_node
  PRIVATE "COMPOSITION_BUILDING_DLL"
)
target_link_libraries(triton_node)
rclcpp_components_register_nodes(triton_node "isaac_ros::dnn_inference::TritonNode")
set(node_plugins "${node_plugins}isaac_ros::dnn_inference::TritonNode;$<TARGET_FILE:triton_node>\n")

# Install config directory
install(
  DIRECTORY config
  DESTINATION share/${PROJECT_NAME}
)

# Install package executable
install(TARGETS triton_node
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)

  # Ignore copyright notices since we use custom JetPack EULA
  set(ament_cmake_copyright_FOUND TRUE)

  ament_lint_auto_find_test_dependencies()

  find_package(launch_testing_ament_cmake REQUIRED)
  add_launch_test(test/isaac_ros_triton_test_onnx.py TIMEOUT "300")
  add_launch_test(test/isaac_ros_triton_test_tf.py TIMEOUT "300")

endif()

ament_auto_package()
