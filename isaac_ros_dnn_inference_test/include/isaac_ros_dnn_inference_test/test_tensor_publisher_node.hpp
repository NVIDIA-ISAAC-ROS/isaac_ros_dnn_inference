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

#ifndef ISAAC_ROS_DNN_INFERENCE_TEST__TEST_TENSOR_PUBLISHER_NODE_HPP_
#define ISAAC_ROS_DNN_INFERENCE_TEST__TEST_TENSOR_PUBLISHER_NODE_HPP_

#include <string>
#include <vector>
#include <utility>

#include "rclcpp/rclcpp.hpp"
#include "isaac_ros_tensor_list_interfaces/msg/tensor.hpp"
#include "isaac_ros_tensor_list_interfaces/msg/tensor_list.hpp"


namespace nvidia
{
namespace isaac_ros
{
namespace dnn_inference
{

// Sends test ROS tensors of specified dimension to specified channel
// By default this node will send input tensors compatible with mobilenetv2-1.0
// Tensors being sent in are of the dimension 1x3x224x224 as specified by mobilenet
class TestTensorPublisherNode : public rclcpp::Node
{
public:
  TestTensorPublisherNode(
    const rclcpp::NodeOptions & options);

  ~TestTensorPublisherNode();

  TestTensorPublisherNode(const TestTensorPublisherNode &) = delete;

  TestTensorPublisherNode & operator=(const TestTensorPublisherNode &) = delete;

private:
  // Dimensions of tensor (eg. {3, 255, 255} for RGB Image of 255x255)
  std::vector<int64_t> dimensions_;
  // Name of tensor
  const std::string tensor_name_;
  // Enum of type of data (check Tensor.msg for mappings)
  const int data_type_;
  // Length of tensor when flattened to 1D
  const int length_;
  // Rank of tensor
  const int rank_;
  // Publisher channel
  std::string pub_channel_;

  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::Publisher<isaac_ros_tensor_list_interfaces::msg::TensorList>::SharedPtr pub_;

  // Sets the tensor data based on size and type (supports only: float32)
  template<typename T>
  void setTensorData(isaac_ros_tensor_list_interfaces::msg::Tensor & tensor, int length)
  {
    std::vector<T> test_data(length);
    tensor.data.resize(test_data.size() * sizeof(T));
    memcpy(tensor.data.data(), test_data.data(), test_data.size() * sizeof(T));
  }

  void timer_callback();
};

}  // namespace dnn_inference
}  // namespace isaac_ros
}  // namespace nvidia

#endif  // ISAAC_ROS_DNN_INFERENCE_TEST__TEST_TENSOR_PUBLISHER_NODE_HPP_
