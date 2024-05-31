// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// SPDX-License-Identifier: Apache-2.0

#include <gmock/gmock.h>
#include "tensor_rt_node.hpp"
#include "rclcpp/rclcpp.hpp"


TEST(tensor_rt_node_test, test_engine_file_path)
{
  rclcpp::init(0, nullptr);
  rclcpp::NodeOptions options;
  options.arguments(
  {
    "--ros-args",
    "-p", "engine_file_path:=''",
  });
  EXPECT_THROW(
  {
    try {
      nvidia::isaac_ros::dnn_inference::TensorRTNode trt_node(options);
    } catch (const std::invalid_argument & e) {
      EXPECT_THAT(e.what(), testing::HasSubstr("Empty engine_file_path"));
      throw;
    }
  }, std::invalid_argument);
  rclcpp::shutdown();
}


int main(int argc, char ** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
