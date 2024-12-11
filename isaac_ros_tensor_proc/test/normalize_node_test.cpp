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
#include "normalize_node.hpp"
#include "rclcpp/rclcpp.hpp"

// Objective: to cover code lines where exceptions are thrown
// Approach: send Invalid Arguments for node parameters to trigger the exception


TEST(normalize_node_test, test_invalid_input_image_width)
{
  rclcpp::init(0, nullptr);
  rclcpp::NodeOptions options;
  options.append_parameter_override("input_image_width", 0);
  // options.arguments(
  // {
  //   "--ros-args",
  //   "-p", "input_image_width:=0",
  // });
  EXPECT_THROW(
  {
    try {
      nvidia::isaac_ros::dnn_inference::NormalizeNode normalize_node(options);
    } catch (const std::invalid_argument & e) {
      EXPECT_THAT(e.what(), testing::HasSubstr("Invalid input_image_width"));
      throw;
    } catch (const rclcpp::exceptions::InvalidParameterValueException & e) {
      EXPECT_THAT(e.what(), testing::HasSubstr("No parameter value set"));
      throw;
    }
  }, std::invalid_argument);
  rclcpp::shutdown();
}

TEST(normalize_node_test, test_invalid_input_image_height)
{
  rclcpp::init(0, nullptr);
  rclcpp::NodeOptions options;
  options.append_parameter_override("input_image_width", 1);
  options.append_parameter_override("input_image_height", 0);
  // options.arguments(
  // {
  //   "--ros-args",
  //   "-p", "input_image_width:=1",
  //   // "-p", "input_image_height:=0",
  // });
  EXPECT_THROW(
  {
    try {
      nvidia::isaac_ros::dnn_inference::NormalizeNode normalize_node(options);
    } catch (const std::invalid_argument & e) {
      EXPECT_THAT(e.what(), testing::HasSubstr("Invalid input_image_height"));
      throw;
    } catch (const rclcpp::exceptions::InvalidParameterValueException & e) {
      EXPECT_THAT(e.what(), testing::HasSubstr("No parameter value set"));
      throw;
    }
  }, std::invalid_argument);
  rclcpp::shutdown();
}

TEST(normalize_node_test, test_invalid_image_channels)
{
  rclcpp::init(0, nullptr);
  rclcpp::NodeOptions options;
  options.append_parameter_override("input_image_width", 1);
  options.append_parameter_override("input_image_height", 1);
  options.append_parameter_override("image_mean", std::vector<double>{0.5, 0.5});
  // options.arguments(
  // {
  //   "--ros-args",
  //   "-p", "input_image_width:=1",
  //   "-p", "input_image_height:=1",
  //   "-p", "image_mean:=[0.5, 0.5]",
  // });
  EXPECT_THROW(
  {
    try {
      nvidia::isaac_ros::dnn_inference::NormalizeNode normalize_node(options);
    } catch (const std::invalid_argument & e) {
      EXPECT_THAT(
        e.what(),
        testing::HasSubstr("Did not receive 3 image mean channels or 3 image stddev channels"));
      throw;
    } catch (const rclcpp::exceptions::InvalidParameterValueException & e) {
      EXPECT_THAT(e.what(), testing::HasSubstr("No parameter value set"));
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
