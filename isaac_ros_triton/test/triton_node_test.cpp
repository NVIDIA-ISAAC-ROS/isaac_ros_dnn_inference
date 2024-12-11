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
#include "triton_node.hpp"
#include "rclcpp/rclcpp.hpp"

// Objective: to cover code lines where exceptions are thrown
// Approach: send Invalid Arguments for node parameters to trigger the exception

TEST(triton_node_test, test_empty_model_name)
{
  rclcpp::init(0, nullptr);
  rclcpp::NodeOptions options;
  options.append_parameter_override("model_name", "");
  EXPECT_THROW(
  {
    try {
      nvidia::isaac_ros::dnn_inference::TritonNode triton_node(options);
    } catch (const std::invalid_argument & e) {
      EXPECT_THAT(e.what(), testing::HasSubstr("Empty model_name"));
      throw;
    } catch (const rclcpp::exceptions::InvalidParameterValueException & e) {
      EXPECT_THAT(e.what(), testing::HasSubstr("No parameter value set"));
      throw;
    }
  }, std::invalid_argument);
  rclcpp::shutdown();
}

TEST(triton_node_test, test_empty_model_repository_paths)
{
  rclcpp::init(0, nullptr);
  rclcpp::NodeOptions options;
  options.append_parameter_override("model_name", "dummy_name");
  EXPECT_THROW(
  {
    try {
      nvidia::isaac_ros::dnn_inference::TritonNode triton_node(options);
    } catch (const std::invalid_argument & e) {
      EXPECT_THAT(e.what(), testing::HasSubstr("Empty model_repository_paths"));
      throw;
    } catch (const rclcpp::exceptions::InvalidParameterValueException & e) {
      EXPECT_THAT(e.what(), testing::HasSubstr("No parameter value set"));
      throw;
    }
  }, std::invalid_argument);
  rclcpp::shutdown();
}

TEST(triton_node_test, test_empty_input_tensor_names)
{
  rclcpp::init(0, nullptr);
  rclcpp::NodeOptions options;
  options.append_parameter_override("model_name", "dummy_name");
  options.append_parameter_override(
    "model_repository_paths",
    std::vector<std::string>{"dummy_path"});
  EXPECT_THROW(
  {
    try {
      nvidia::isaac_ros::dnn_inference::TritonNode triton_node(options);
    } catch (const std::invalid_argument & e) {
      EXPECT_THAT(e.what(), testing::HasSubstr("Empty input_tensor_names"));
      throw;
    } catch (const rclcpp::exceptions::InvalidParameterValueException & e) {
      EXPECT_THAT(e.what(), testing::HasSubstr("No parameter value set"));
      throw;
    }
  }, std::invalid_argument);
  rclcpp::shutdown();
}

TEST(triton_node_test, test_empty_input_binding_names)
{
  rclcpp::init(0, nullptr);
  rclcpp::NodeOptions options;
  options.append_parameter_override("model_name", "dummy_name");
  options.append_parameter_override(
    "model_repository_paths",
    std::vector<std::string>{"dummy_path"});
  options.append_parameter_override("input_tensor_names", std::vector<std::string>{"dummy"});
  EXPECT_THROW(
  {
    try {
      nvidia::isaac_ros::dnn_inference::TritonNode triton_node(options);
    } catch (const std::invalid_argument & e) {
      EXPECT_THAT(e.what(), testing::HasSubstr("Empty input_binding_names"));
      throw;
    } catch (const rclcpp::exceptions::InvalidParameterValueException & e) {
      EXPECT_THAT(e.what(), testing::HasSubstr("No parameter value set"));
      throw;
    }
  }, std::invalid_argument);
  rclcpp::shutdown();
}

TEST(triton_node_test, test_empty_output_tensor_names)
{
  rclcpp::init(0, nullptr);
  rclcpp::NodeOptions options;
  options.append_parameter_override("model_name", "dummy_name");
  options.append_parameter_override(
    "model_repository_paths",
    std::vector<std::string>{"dummy_path"});
  options.append_parameter_override("input_tensor_names", std::vector<std::string>{"dummy"});
  options.append_parameter_override("input_binding_names", std::vector<std::string>{"dummy"});
  EXPECT_THROW(
  {
    try {
      nvidia::isaac_ros::dnn_inference::TritonNode triton_node(options);
    } catch (const std::invalid_argument & e) {
      EXPECT_THAT(e.what(), testing::HasSubstr("Empty output_tensor_names"));
      throw;
    } catch (const rclcpp::exceptions::InvalidParameterValueException & e) {
      EXPECT_THAT(e.what(), testing::HasSubstr("No parameter value set"));
      throw;
    }
  }, std::invalid_argument);
  rclcpp::shutdown();
}

TEST(triton_node_test, test_empty_output_binding_names)
{
  rclcpp::init(0, nullptr);
  rclcpp::NodeOptions options;
  options.append_parameter_override("model_name", "dummy_name");
  options.append_parameter_override(
    "model_repository_paths",
    std::vector<std::string>{"dummy_path"});
  options.append_parameter_override("input_tensor_names", std::vector<std::string>{"dummy"});
  options.append_parameter_override("input_binding_names", std::vector<std::string>{"dummy"});
  options.append_parameter_override("output_tensor_names", std::vector<std::string>{"dummy"});
  EXPECT_THROW(
  {
    try {
      nvidia::isaac_ros::dnn_inference::TritonNode triton_node(options);
    } catch (const std::invalid_argument & e) {
      EXPECT_THAT(e.what(), testing::HasSubstr("Empty output_binding_names"));
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
