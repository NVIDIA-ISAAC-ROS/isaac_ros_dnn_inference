// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "isaac_ros_dnn_inference_test/test_tensor_publisher_node.hpp"

#include <filesystem>
#include <chrono>
#include <cstring>
#include <string>
#include <vector>

#include "isaac_ros_tensor_list_interfaces/msg/tensor_shape.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace dnn_inference
{

namespace
{
int calculate_length(const std::vector<int64_t> dimensions)
{
  int length{1};
  for (int dim : dimensions) {
    length = length * dim;
  }
  return length;
}
}  // namespace

using namespace std::chrono_literals;

TestTensorPublisherNode::TestTensorPublisherNode(
  const rclcpp::NodeOptions & options)
: Node("test_tensor_publisher", options),
  dimensions_(declare_parameter<std::vector<int64_t>>("dimensions", {1, 3, 224, 224})),
  tensor_name_(declare_parameter<std::string>("tensor_name", "input_tensor")),
  data_type_(declare_parameter<int>("data_type", 9)),
  length_(declare_parameter<int>("length", calculate_length(dimensions_))),
  rank_(declare_parameter<int>("rank", 4)),
  pub_channel_(declare_parameter<std::string>("pub_channel", "tensor_pub"))
{
  pub_ =
    this->create_publisher<isaac_ros_tensor_list_interfaces::msg::TensorList>(pub_channel_, 10);
  timer_ = this->create_wall_timer(
    1000ms, std::bind(&TestTensorPublisherNode::timer_callback, this));

  // Don't let negative dimensions be passed
  for (const auto & dim : dimensions_) {
    if (dim <= 0) {
      throw std::invalid_argument("Error: received negative or zero dimension!");
    }
  }
}

void TestTensorPublisherNode::timer_callback()
{
  auto tensor = isaac_ros_tensor_list_interfaces::msg::Tensor();
  tensor.name = tensor_name_;
  tensor.data_type = data_type_;
  tensor.strides = {};  // Not used in GXF so leave empty

  switch (data_type_) {
    case 9:
      setTensorData<float>(tensor, length_);
      break;
  }

  auto tensor_shape = isaac_ros_tensor_list_interfaces::msg::TensorShape();
  tensor_shape.rank = rank_;
  for (const int64_t dim : dimensions_) {
    tensor_shape.dims.push_back(static_cast<uint32_t>(dim));
  }
  tensor.shape = tensor_shape;

  auto tensor_list = isaac_ros_tensor_list_interfaces::msg::TensorList();
  tensor_list.tensors = {tensor};
  pub_->publish(tensor_list);
}

TestTensorPublisherNode::~TestTensorPublisherNode() {}

}  // namespace dnn_inference
}  // namespace isaac_ros
}  // namespace nvidia
